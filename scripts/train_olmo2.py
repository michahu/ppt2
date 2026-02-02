import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, cast

import rich
from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    SubCmd,
    build_common_components,
    launch,
    launch_prep,
    prep,
)
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all
from olmo_core.distributed.utils import get_rank, scatter_object
from olmo_core.io import normalize_path

USE_NOPE = True
SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 256 * SEQUENCE_LENGTH
WARMUP_STEPS = 1000
N_TOKENS = 50_000 * GLOBAL_BATCH_SIZE

DATA_ROOT = "/vast/myh2014/data".rstrip("/")


def _read_data_mix_file(filename: str) -> List[str]:
    """Read URLs from a data mix file in the data_mixes folder."""
    script_dir = Path(__file__).parent.parent  # Get project root
    data_mix_path = script_dir / "data_mixes" / filename

    if not data_mix_path.exists():
        log.warning(f"Data mix file not found: {data_mix_path}")
        return []

    paths = []
    with open(data_mix_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # replace http://olmo-data.org/ with DATA_ROOT
                paths.append(line.replace("http://olmo-data.org/", DATA_ROOT + "/"))

    return paths


DATA_PATHS = _read_data_mix_file("OLMo-mix-0625-150Bsample.txt")
EVAL_DATA_PATHS = _read_data_mix_file("v3-small-ppl-validation.txt")
DATA_WORK_DIR = "./data/"

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(Config):
    run_name: str
    launch: Optional[BeakerLaunchConfig]
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def get_tokenizer_config() -> TokenizerConfig:
    """
    Get tokenizer config, optionally with BOS token for NoPE.

    When using NoPE (No Positional Encoding), we need a BOS token at the start
    of each sequence. We use the padded vocab size as the BOS token ID, which
    means the model's vocab size needs to be increased by 1.
    """
    base_config = TokenizerConfig.dolma2()

    if USE_NOPE:
        # Use the padded vocab size as BOS token ID
        # This requires increasing vocab_size by 1 in the model
        padded_vocab = base_config.padded_vocab_size()
        return TokenizerConfig(
            vocab_size=base_config.vocab_size + 1,  # +1 for BOS token
            eos_token_id=base_config.eos_token_id,
            pad_token_id=base_config.pad_token_id,
            bos_token_id=padded_vocab,  # BOS is at the padded vocab size index
            identifier=base_config.identifier,
        )
    return base_config


def build_model_config(
    common: CommonComponents, model_size: str = "190M"
) -> TransformerConfig:
    # When using NoPE with BOS, we need vocab_size = padded_vocab_size + 1
    # to accommodate the new BOS token
    if USE_NOPE:
        vocab_size = common.tokenizer.padded_vocab_size() + 1
    else:
        vocab_size = common.tokenizer.padded_vocab_size()

    if model_size == "190M":
        config = TransformerConfig.olmo2_190M(
            vocab_size=vocab_size,
            dtype=DType.bfloat16,
        )
    elif model_size == "1B":
        config = TransformerConfig.olmo2_1B_v2(
            vocab_size=vocab_size,
            dtype=DType.bfloat16,
        )
    else:
        raise ValueError(f"Invalid model size: {model_size}. Must be '190M' or '1B'")

    # Disable RoPE for NoPE (No Positional Encoding)
    if USE_NOPE:
        config.block.attention.rope = None

    config.block.attention.sliding_window = SlidingWindowAttentionConfig(
        force_full_attention_on_first_layer=False,
        force_full_attention_on_last_layer=True,
        pattern=[4096, 4096, 4096, -1],
    )
    config.block.attention.use_flash = True
    return config


def _set_beaker_execution_units(config: ExperimentConfig):
    # When running on Augusta with hostname constraints enabled, setting more beaker
    # execution units than model replicas may result in the replicas being split across
    # Augusta hardware blocks.
    if (
        config.launch
        and config.launch.use_hostname_constraints
        and any("augusta" in cluster for cluster in config.launch.clusters)
        and (dp_config := config.train_module.dp_config) is not None
    ):
        if dp_config.num_replicas is not None:
            num_model_replicas = dp_config.num_replicas
        elif dp_config.shard_degree is not None:
            nodes_per_replica = max(1, dp_config.shard_degree // config.launch.num_gpus)
            num_model_replicas = config.launch.num_nodes // nodes_per_replica
        else:
            return

        if config.launch.num_execution_units is None:
            log.info(f"Setting number of execution units to {num_model_replicas}.")
            config.launch.num_execution_units = num_model_replicas
        elif config.launch.num_execution_units > num_model_replicas:
            log.warning(
                f"Number of execution units {config.launch.num_execution_units} exceeds number of model replicas {num_model_replicas}. "
                "On Augusta, this may result in suboptimal performance due to model replicas being split "
                "across hardware blocks. To resolve, decrease num_execution_units in beaker launch config, "
                "increase number of model replicas or disable use_hostname_constraints in beaker launch config."
            )


def build_train_module_config(
    common: CommonComponents, model_size: str = "190M"
) -> TransformerTrainModuleConfig:
    rank_microbatch_size = 8 * SEQUENCE_LENGTH
    if common.launch is not None:
        gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
        if all("B200" in g for g in gpus):
            rank_microbatch_size *= 2

    # Set learning rate based on model size
    # For 190M: use higher LR (4e-3), for 1B: use standard LR (8e-4)
    if model_size == "190M":
        learning_rate = 4e-3
        rank_microbatch_size *= 4
    elif model_size == "1B":
        learning_rate = 4e-4 * 2  # 8e-4
    else:
        raise ValueError(f"Invalid model size: {model_size}. Must be '190M' or '1B'")

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=learning_rate,
            weight_decay=0.033,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                )
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=WARMUP_STEPS),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    cancel_check_interval = 50

    run_name = (
        f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%z')}"
    )

    return (
        TrainerConfig(
            save_folder=f"./runs/{common.run_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(int(N_TOKENS)),
            hard_stop=Duration.tokens(
                int(2.5e12 + GLOBAL_BATCH_SIZE * (WARMUP_STEPS / 2))
            ),  # After this, we switch to a longer cosine to reach 6T.
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=5000,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                group=common.run_name,
                entity="ai2-llm",
                project="willm-ppt2",
                enabled=True,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=DATA_ROOT,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=common.tokenizer,
                    work_dir=DATA_WORK_DIR,
                ),
                eval_interval=5000,
            ),
        )
    )


def build_config(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    checkpoint: Optional[str],
    overrides: List[str],
    *,
    common_config_builder: Callable[..., CommonComponents] = build_common_components,
    model_config_builder: Callable[[CommonComponents, str], TransformerConfig],
    train_module_config_builder: Callable[
        [CommonComponents, str], TransformerTrainModuleConfig
    ],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
    model_size: str = "190M",
    tokenizer: Optional[TokenizerConfig] = None,
    init_seed: int = 12536,
    global_batch_size: int = GLOBAL_BATCH_SIZE,
    sequence_length: int = SEQUENCE_LENGTH,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
    **kwargs,
) -> ExperimentConfig:
    effective_tokenizer = tokenizer if tokenizer is not None else get_tokenizer_config()

    # Create CLI context for the new API
    cli_context = CliContext(
        script=script,
        cmd=cmd,
        run_name=run_name,
        cluster=cluster,
        overrides=overrides,
    )

    # Build common components with new API
    common = common_config_builder(
        cli_context,
        tokenizer=effective_tokenizer,
        global_batch_size=global_batch_size,
        max_sequence_length=sequence_length,
        beaker_image=beaker_image,
        num_nodes=num_nodes,
        beaker_workspace=beaker_workspace,
    )

    model = model_config_builder(common, model_size)

    dataset = NumpyFSLDatasetConfig(
        # @willm might be called data_paths
        paths=DATA_PATHS,
        work_dir=DATA_WORK_DIR,
        tokenizer=effective_tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
    )

    # Build data loader config directly
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=34521,
        num_workers=4,
    )

    trainer = trainer_config_builder(common)

    config = ExperimentConfig(
        run_name=run_name,
        launch=common.launch,
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module_config_builder(common, model_size),
        trainer=trainer,
        init_seed=init_seed,
    )

    config = config.merge(overrides)

    _set_beaker_execution_units(config)

    if finalize_config is not None:
        finalize_config(config)

    return config


def load_checkpoint_with_options(
    trainer,
    dir,
    *,
    load_trainer_state: Optional[bool] = None,
    load_optim_state: Optional[bool] = None,
    load_embeddings: bool = True,
    alpha: float = 1.0,
):
    """
    Load a checkpoint with optional control over which components to load.

    :param trainer: The Trainer instance.
    :param dir: The path/URL to a checkpoint or a folder of checkpoints.
    :param load_trainer_state: Load trainer state (data loader state, RNG states, and other bookkeeping).
    :param load_optim_state: Load optimizer state in the train module.
    :param load_embeddings: If False, skip loading embedding weights (useful for vocab changes).
                           This handles mismatched embedding sizes.
    :param alpha: Interpolation factor between checkpoint and current model weights.
                  Final weights = alpha * checkpoint + (1 - alpha) * current_model.
                  alpha=1.0 (default) means fully load checkpoint weights.
                  alpha=0.0 means keep current model weights (no loading).
    """
    import torch
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict,
        get_model_state_dict,
        StateDictOptions,
    )
    from torch.distributed import checkpoint as dist_cp
    from olmo_core.distributed.checkpoint import (
        get_checkpoint_metadata,
        RemoteFileSystemReader,
    )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    dir = normalize_path(dir)

    # NOTE: to avoid making a ton of client requests (S3 or otherwise) we only make those
    # requests from rank 0 then scatter the result to the other ranks.
    if get_rank() == 0 and not trainer.checkpointer.dir_is_checkpoint(dir):
        # Try to find the latest checkpoint in the directory.
        dir = trainer.checkpointer.latest_checkpoint(dir)
    dir = scatter_object(dir)

    log.info(f"Loading checkpoint from '{dir}'...")
    if alpha < 1.0:
        log.info(f"Using interpolation: alpha={alpha} (final = {alpha}*checkpoint + {1-alpha}*current)")

    # If alpha is 0, skip loading entirely
    if alpha == 0.0:
        log.info("alpha=0.0, keeping current model weights")
        return

    if load_embeddings and alpha == 1.0:
        # Standard loading path (no interpolation needed)
        trainer_state = trainer.checkpointer.load(
            dir,
            trainer.train_module,
            load_trainer_state=load_trainer_state,
            load_optim_state=load_optim_state,
        )
        if trainer_state is not None:
            trainer.load_state_dict(cast(dict, trainer_state))
    else:
        # Custom loading path that supports:
        # - Skipping embeddings (for size mismatches)
        # - Interpolation between checkpoint and current weights
        if not load_embeddings:
            log.info("Skipping embedding weights during checkpoint load (supports size mismatch)")

        model = trainer.train_module.model

        # Save current model state for interpolation
        if alpha < 1.0:
            current_state = get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=False, cpu_offload=False),
            )
            # Deep copy the tensors
            current_state = {k: v.clone() for k, v in current_state.items()}

        # Get the model state dict structure (this gives us the keys we need)
        model_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        )

        # Filter out embedding and lm_head keys if not loading embeddings
        # (lm_head typically has the same vocab dimension as embeddings)
        if not load_embeddings:
            exclude_patterns = ["embedding", "lm_head"]
            keys_to_exclude = [
                k for k in list(model_state.keys())
                if any(pattern in k.lower() for pattern in exclude_patterns)
            ]
            for key in keys_to_exclude:
                del model_state[key]
                if alpha < 1.0 and key in current_state:
                    del current_state[key]
                log.info(f"Excluding from checkpoint load: {key}")

        # Determine checkpoint directory (could be model_and_optim subdir or root)
        train_module_dir = f"{dir}/model_and_optim"
        try:
            get_checkpoint_metadata(train_module_dir)
        except FileNotFoundError:
            train_module_dir = dir

        # Load checkpoint weights
        dist_cp.load(
            state_dict={"model": model_state},
            storage_reader=RemoteFileSystemReader(train_module_dir),
        )

        # Apply interpolation if alpha < 1.0
        if alpha < 1.0:
            log.info(f"Interpolating weights with alpha={alpha}")
            for key in model_state.keys():
                if key in current_state:
                    # final = alpha * checkpoint + (1 - alpha) * current
                    model_state[key] = alpha * model_state[key] + (1 - alpha) * current_state[key]

        # Apply loaded (and possibly interpolated) weights to model
        set_model_state_dict(
            model,
            model_state_dict=model_state,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False, strict=False),
        )

        if alpha < 1.0:
            log.info(f"Loaded checkpoint with interpolation (alpha={alpha})")
        else:
            log.info("Loaded checkpoint excluding embedding weights")

    for callback in trainer.callbacks.values():
        if hasattr(callback, "post_checkpoint_loaded"):
            callback.post_checkpoint_loaded(dir)


def train(config: ExperimentConfig, checkpoint: Optional[str], load_embeddings: bool = True, alpha: float = 1.0):
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    if checkpoint is not None:
        load_checkpoint_with_options(
            trainer,
            checkpoint,
            load_trainer_state=False,
            load_embeddings=load_embeddings,
            alpha=alpha,
        )

    # Record the config to W&B/Comet and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


def main(
    *,
    global_batch_size: int,
    common_config_builder: Callable[..., CommonComponents] = build_common_components,
    model_config_builder: Callable[[CommonComponents, str], TransformerConfig],
    train_module_config_builder: Callable[
        [CommonComponents, str], TransformerTrainModuleConfig
    ],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
    sequence_length: int = 4096,
    include_default_evals: bool = True,
    intra_document_masking: bool = False,
    include_instance_filter: bool = False,
    beaker_image: str = OLMoCoreBeakerImage.stable,
    num_nodes: int = 1,
    beaker_workspace: str = "ai2/OLMo-core",
    use_hostname_constraints: bool = False,
    num_execution_units: Optional[int] = None,
    tokenizer: Optional[TokenizerConfig] = None,
):
    USAGE = f"""
PPT Phase 1.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]launch|train|dry_run[/] [i b]RUN_NAME CLUSTER [MODEL_SIZE] [PRETRAIN_CHECKPOINT][/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use the [b magenta]launch[/] cmd to submit it to Beaker or run it via torchrun if you know what you're doing.
[b magenta]dry_run:[/]     Print the config for debugging.

[b]Model Size[/]
Optional positional argument after CLUSTER. Must be "190M" or "1B" (default: "190M")

[b]Seed[/]
Use --seed=N to set the initialization seed (default: 12536)

[b]Embeddings[/]
Use --no-load-embeddings to skip loading embedding weights from checkpoint (default: load embeddings)

[b]Interpolation[/]
Use --alpha=N to interpolate between checkpoint and random init: final = alpha*checkpoint + (1-alpha)*random
(default: 1.0, i.e., fully load checkpoint)

[b]Examples[/]
$ [i]python {sys.argv[0]} launch run01 ai2/jupiter-cirrascale-2 190M gs://ai2-llm/checkpoints/peteish32/step419000 --launch.num_nodes=2[/]
$ [i]python {sys.argv[0]} launch run01 ai2/jupiter-cirrascale-2 1B --launch.num_nodes=2[/]
$ [i]python {sys.argv[0]} launch run02 ai2/jupiter-cirrascale-2 --launch.num_nodes=2[/]
""".strip()

    # Parse command line arguments.
    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        rich.get_console().print(USAGE, highlight=False)
        sys.exit(1)

    script, cmd, run_name, cluster, *rest = sys.argv

    # Parse optional model_size and checkpoint arguments
    model_size = "190M"  # default
    checkpoint = None
    overrides = []
    seed = 12536  # default seed
    load_embeddings = True  # default
    alpha = 1.0  # default

    if rest:
        # Check if first arg is model_size (190M or 1B)
        if rest[0] in ["190M", "1B"]:
            model_size = rest[0]
            rest = rest[1:]

        # Check if next arg is checkpoint (doesn't start with --)
        if rest and not rest[0].startswith("--"):
            checkpoint = rest[0]
            overrides = rest[1:]
        else:
            overrides = rest

    # Extract --seed=N from overrides
    seed_overrides = [o for o in overrides if o.startswith("--seed=")]
    if seed_overrides:
        seed = int(seed_overrides[-1].split("=")[1])
        overrides = [o for o in overrides if not o.startswith("--seed=")]

    # Extract --no-load-embeddings from overrides
    if "--no-load-embeddings" in overrides:
        load_embeddings = False
        overrides = [o for o in overrides if o != "--no-load-embeddings"]

    # Extract --alpha=N from overrides
    alpha_overrides = [o for o in overrides if o.startswith("--alpha=")]
    if alpha_overrides:
        alpha = float(alpha_overrides[-1].split("=")[1])
        overrides = [o for o in overrides if not o.startswith("--alpha=")]

    cmd = SubCmd(cmd)

    # Use custom tokenizer if provided, otherwise use default with NoPE BOS token
    effective_tokenizer = tokenizer if tokenizer is not None else get_tokenizer_config()

    config = build_config(
        script,
        cmd,
        run_name,
        cluster,
        checkpoint,
        overrides,
        global_batch_size=global_batch_size,
        common_config_builder=common_config_builder,
        model_config_builder=model_config_builder,
        train_module_config_builder=train_module_config_builder,
        trainer_config_builder=trainer_config_builder,
        finalize_config=finalize_config,
        sequence_length=sequence_length,
        include_default_evals=include_default_evals,
        intra_document_masking=intra_document_masking,
        include_instance_filter=include_instance_filter,
        beaker_image=beaker_image,
        num_nodes=num_nodes,
        beaker_workspace=beaker_workspace,
        model_size=model_size,
        tokenizer=effective_tokenizer,
        init_seed=seed,
    )

    cmd.prepare_environment(config)

    # need to move these out of SubCmd to use our custom train method
    if get_local_rank() == 0:
        print(config)
        print(
            "\n"
            f"[b blue]Total parameters:[/]         {config.model.num_params:,d} ({config.model.num_active_params:,d} active)\n"
            f"[b blue]Non-embedding parameters:[/] {config.model.num_non_embedding_params:,d} ({config.model.num_active_non_embedding_params:,d} active)"
        )

    if cmd == SubCmd.launch:
        launch(config)
    elif cmd == SubCmd.dry_run:
        pass
    elif cmd == SubCmd.train:
        try:
            train(config, checkpoint, load_embeddings=load_embeddings, alpha=alpha)
        finally:
            teardown_training_environment()
    elif cmd == SubCmd.train_single:
        if config.train_module.dp_config is not None:
            log.warning(
                "'dp_config' is set to %s, but you can't use data parallelism when running on a single node. Disabling.",
                config.train_module.dp_config,
            )
            config.train_module.dp_config = None
        if config.train_module.tp_config is not None:
            log.warning(
                "'tp_config' is set to %s, but you can't use tensor parallelism when running on a single node. Disabling.",
                config.train_module.dp_config,
            )
            config.train_module.tp_config = None
        try:
            train(config, checkpoint, load_embeddings=load_embeddings, alpha=alpha)
        finally:
            teardown_training_environment()
    elif cmd == SubCmd.prep:
        prep(config)
    elif cmd == SubCmd.launch_prep:
        launch_prep(config)
    else:
        raise NotImplementedError(cmd)


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,
        include_default_evals=False,  # Can't use default evals on Greene
        intra_document_masking=False,
    )
