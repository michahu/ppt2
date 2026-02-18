"""
Model ladder with expansion from a small procedural pretraining checkpoint.

This script runs the standard OLMo-core model ladder but first expands a small
checkpoint (upper-left pasting) to each ladder size before training.

Usage examples:

    # Dry run
    python scripts/expand_ladder.py dry-run --size=190M --checkpoint=./wsw-step500/model.pt

    # Run locally
    torchrun --nproc-per-node=8 scripts/expand_ladder.py run --size=190M \\
        --checkpoint=./wsw-step500/model.pt

    # Launch on Beaker
    python scripts/expand_ladder.py launch --size=190M \\
        --checkpoint=weka://path/to/wsw-step500/model.pt --name=expand-ladder

    # Launch all sizes
    python scripts/expand_ladder.py launch-all --max-size=1B \\
        --checkpoint=weka://path/to/wsw-step500/model.pt --name=expand-ladder
"""

import argparse
import logging
import typing
from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
import olmo_core.train.callbacks as callbacks
from olmo_core.data import DataMix, TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    NumpyDocumentSourceMixConfig,
    set_composable_seed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.train import prepare_training_environment, teardown_training_environment

from olmo_core.internal.common import get_gpu_type, get_root_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# expand_model_weights -- inlined from procedural_pretraining/utils.py
# ---------------------------------------------------------------------------

def expand_model_weights(
    small_state_dict: dict[str, torch.Tensor],
    large_model_or_state: "torch.nn.Module | dict[str, torch.Tensor]",
    alpha: float = 1.0,
    use_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """Expand a small model's state dict to fit a larger model (upper-left pasting).

    For each parameter in the small state dict, paste its values into the upper-left
    corner of the corresponding large parameter. The result is interpolated with the
    large model's random init: final = alpha * expanded + (1 - alpha) * large_random.

    Args:
        small_state_dict: State dict from the smaller model.
        large_model_or_state: The larger model instance or its state dict (used for
            shape reference).
        alpha: Interpolation weight. 1.0 = fully expanded, 0.0 = fully random.
        use_embeddings: When False, skip embedding/lm_head keys (they keep random init
            since vocab sizes typically differ between small and large models).

    Returns:
        A new state dict compatible with large_model.
    """
    if isinstance(large_model_or_state, dict):
        result = {k: v.clone() for k, v in large_model_or_state.items()}
    else:
        result = {k: v.clone() for k, v in large_model_or_state.state_dict().items()}

    embed_patterns = ["embedding", "lm_head"]

    for name, small_tensor in small_state_dict.items():
        if not use_embeddings and any(p in name.lower() for p in embed_patterns):
            continue
        if name not in result:
            log.warning("Skipping key '%s': not found in large model", name)
            continue

        large_tensor = result[name]

        if small_tensor.ndim == 0:
            # scalar: final = alpha * small + (1 - alpha) * random
            large_tensor.mul_(1 - alpha).add_(small_tensor, alpha=alpha)
        elif small_tensor.ndim == 1:
            size_s = small_tensor.shape[0]
            # outside small region: scale by (1 - alpha)
            large_tensor[size_s:].mul_(1 - alpha)
            # inside small region: alpha * small + (1 - alpha) * random
            large_tensor[:size_s].mul_(1 - alpha).add_(small_tensor, alpha=alpha)
        elif small_tensor.ndim == 2:
            rows_s, cols_s = small_tensor.shape
            large_tensor[:rows_s, cols_s:].mul_(1 - alpha)
            large_tensor[rows_s:, :].mul_(1 - alpha)
            large_tensor[:rows_s, :cols_s].mul_(1 - alpha).add_(small_tensor, alpha=alpha)
        else:
            raise ValueError(
                f"Unsupported tensor dimension {small_tensor.ndim} for key '{name}'"
            )

    return result


# ---------------------------------------------------------------------------
# ExpandingModelLadder -- subclass that injects expansion before training
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class ExpandingModelLadder(ModelLadder):
    """A ModelLadder that expands a small checkpoint to each ladder size before training."""

    expand_from: Optional[str] = None
    """Path to the small checkpoint to expand from."""

    alpha: float = 1.0
    """Interpolation weight for expansion (1.0 = fully expanded, 0.0 = fully random)."""

    load_embeddings: bool = False
    """Whether to load embeddings from the small checkpoint (False by default since
    procedural pretraining uses a different vocabulary)."""

    def run(self, size_spec: str, for_benchmarking: bool = False):
        """
        Execute a ladder run with model expansion injected between building
        the trainer and calling trainer.fit().
        """
        if size_spec not in self.sizes:
            raise ValueError(f"Invalid size_spec '{size_spec}', must be one of {self.sizes}")
        prepare_training_environment(seed=self.seed, backend=self.backend)
        set_composable_seed(self.seed)

        # Configure model.
        model_config = self.get_model_config(size_spec)
        num_params = model_config.num_non_embedding_params

        # Configure global batch size.
        (
            global_batch_size,
            rank_microbatch_size,
            requested_devices,
            _,
        ) = self._configure_batch_size_and_num_devices(size_spec, num_params)
        if requested_devices != dist_utils.get_world_size():
            raise OLMoConfigurationError(
                f"Requested {requested_devices} devices for model of size '{size_spec}', "
                f"but {dist_utils.get_world_size()} are available."
            )

        # Configure optimizer and scheduler.
        optim_config = self.run_configurator.configure_optimizer(num_params, global_batch_size)
        scheduler = self.run_configurator.configure_lr_scheduler(num_params, global_batch_size)

        # Configure trainer.
        trainer_config = self._configure_trainer(size_spec, for_benchmarking=for_benchmarking)

        # Build instance sources and data loader.
        instance_sources = [
            source.build(work_dir=self.work_dir) for source in self.instance_sources
        ]
        data_loader = self.data_loader.build(
            *instance_sources,
            work_dir=self.work_dir,
            global_batch_size=global_batch_size,
            tokenizer=self.tokenizer,
        )
        if data_loader.sequence_length != self.sequence_length:
            raise OLMoConfigurationError(
                f"Data loader sequence of {data_loader.sequence_length} does not match "
                f"configured sequence length of {self.sequence_length}."
            )

        # Build train module.
        train_module = self.model_configurator.build_train_module(
            size_spec=size_spec,
            sequence_length=self.sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            model_config=model_config,
            optim_config=optim_config,
            scheduler=scheduler,
            device_type=self.device_type,
        )

        # Build trainer.
        trainer = trainer_config.build(train_module, data_loader)

        # --- Expansion step ---
        if self.expand_from:
            log.info(
                "Expanding weights from %s (alpha=%.2f, load_embeddings=%s)",
                self.expand_from, self.alpha, self.load_embeddings,
            )
            small_state = torch.load(self.expand_from, map_location="cpu", weights_only=True)
            model_state = get_model_state_dict(
                trainer.train_module.model,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            )
            expanded = expand_model_weights(
                small_state, model_state,
                alpha=self.alpha, use_embeddings=self.load_embeddings,
            )
            set_model_state_dict(
                trainer.train_module.model,
                model_state_dict=expanded,
                options=StateDictOptions(full_state_dict=False, cpu_offload=False, strict=False),
            )
            log.info("Model expansion complete.")

        # Record all configs.
        config_dict = {
            "seed": self.seed,
            "size": str(size_spec),
            "model": model_config.as_config_dict(),
            "optim": optim_config.as_config_dict(),
            "scheduler": scheduler.as_config_dict(),
            "data_loader": self.data_loader.as_config_dict(),
            "instance_sources": [s.as_config_dict() for s in self.instance_sources],
            "expand_from": self.expand_from,
            "alpha": self.alpha,
            "load_embeddings": self.load_embeddings,
        }
        typing.cast(
            callbacks.ConfigSaverCallback, trainer.callbacks["config_saver"]
        ).config = config_dict

        # Train.
        trainer.fit()

        teardown_training_environment()


# ---------------------------------------------------------------------------
# configure_model / configure_ladder / add_additional_args
# ---------------------------------------------------------------------------

def configure_model(args: argparse.Namespace) -> Olmo3ModelConfigurator:
    return Olmo3ModelConfigurator()


def configure_ladder(args: argparse.Namespace) -> ExpandingModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    instance_sources = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceMixConfig(
                    tokenizer=tokenizer,
                    mix=DataMix.OLMo_mix_0925,
                    mix_base_dir="gs://ai2-llm/",
                )
            ],
            sequence_length=args.sequence_length,
        ),
    ]
    ladder = ExpandingModelLadder(
        name=args.name,
        project=args.project,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=list(TransformerSize),
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=configure_model(args),
        run_configurator=WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple,
            lr_multiplier=args.lr_multiplier,
            stepped_schedule=args.stepped_schedule,
        ),
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8, instance_filter_config=InstanceFilterConfig()
        ),
        expand_from=args.checkpoint,
        alpha=args.alpha,
        load_embeddings=not args.no_load_embeddings,
    )
    return ladder


def add_additional_args(cmd: str, parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the small procedural pretraining checkpoint to expand from.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Interpolation weight for expansion (1.0 = fully expanded, 0.0 = fully random).",
    )
    parser.add_argument(
        "--no-load-embeddings",
        action="store_true",
        default=True,
        help="Do NOT load embeddings from the checkpoint (default: True, i.e. embeddings are "
             "not loaded since procedural pretraining uses a different vocabulary).",
    )


if __name__ == "__main__":
    main(
        configure_ladder=configure_ladder,
        add_additional_args=add_additional_args,
    )
