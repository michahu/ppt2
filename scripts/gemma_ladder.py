"""
Gemma-like model ladder with weight expansion from a small procedural checkpoint.

This is based on the gemma_like_ladder from OLMo-core PR #610, adapted to inject
an upper-left pasting expansion step before training. The small procedural pretraining
checkpoint is expanded into each Gemma-like ladder model size.

Usage examples:

    # Dry run
    python scripts/expand_ladder.py dry_run gl-v2-260m ai2/jupiter \\
        --checkpoint=./path/to/small.pt

    # Launch on Beaker
    python scripts/expand_ladder.py launch gl-v2-260m ai2/jupiter \\
        --checkpoint=weka://path/to/model.pt

    # Launch with expansion interpolation and LR sweep
    python scripts/expand_ladder.py launch gl-v2-260m ai2/jupiter \\
        --checkpoint=weka://path/to/model.pt --alpha=0.5 --lr-multiplier=2.0

    # Single-rank debug run (CPU/MacBook, no torchrun needed)
    python scripts/expand_ladder.py train_single gl-v2-260m ai2/jupiter \\
        --checkpoint=./path/to/small.pt

    # Override config values directly
    python scripts/expand_ladder.py launch gl-v2-8b ai2/jupiter \\
        --checkpoint=weka://path/to/model.pt \\
        --train_module.optim.lr=0.001 \\
        --launch.num_nodes=2
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, cast

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from olmo_core.config import DType, StrEnum
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.eval.task_groups import TASK_GROUPS
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import build_launch_config, get_root_dir, get_work_dir
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import CliContext, ExperimentConfig, SubCmd
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GatedDeltaNetConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig, LMLossImplementation
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import (
    CosWithWarmupAndLinearDecay,
    OptimGroupOverride,
    SchedulerUnits,
    SkipStepAdamWConfig,
)
from olmo_core.train import Duration, TrainerConfig, teardown_training_environment
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

try:
    from olmo_core.train.callbacks import StabilityMonitorCallback

    _has_stability_monitor = True
except ImportError:
    _has_stability_monitor = False

log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 8192

# ---------------------------------------------------------------------------
# Module-level expansion params (populated during build_experiment_config,
# consumed during _train_with_expansion)
# ---------------------------------------------------------------------------

_expand_from: Optional[str] = None
_alpha: float = 1.0
_load_embeddings: bool = False


# ---------------------------------------------------------------------------
# expand_model_weights -- upper-left pasting
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
        large_model_or_state: The larger model instance or its state dict (shape reference).
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
            large_tensor.mul_(1 - alpha).add_(small_tensor, alpha=alpha)
        elif small_tensor.ndim == 1:
            size_s = small_tensor.shape[0]
            large_tensor[size_s:].mul_(1 - alpha)
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
# GemmaLikeTransformerConfig (from OLMo-core PR #610)
# ---------------------------------------------------------------------------


@dataclass
class GemmaLikeTransformerConfig(TransformerConfig):
    @classmethod
    def v1(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """The v1 model series baseline."""
        return cls.gemma3_like(
            vocab_size=vocab_size,
            n_kv_heads=6,
            head_dim=128,
            global_layer_interval=5,
            activation=ActivationFunction.silu,
            gate=GateConfig(
                granularity=GateGranularity.elementwise,
                full_precision=True,
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_3),
            **kwargs,
        )

    @classmethod
    def v1_250M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 250M model config.

        251,359,360 total params
        187,134,080 non-embedding params
        """
        return cls.v1(
            d_model=640,
            hidden_size=640 * 8,
            n_layers=10,
            n_heads=6,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_680M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 680M model config.

        677,446,400 total params
        574,685,952 non-embedding params
        """
        return cls.v1(
            d_model=1024,
            hidden_size=1024 * 8,
            n_layers=15,
            n_heads=12,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_1p2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1.2B model config.

        1,200,728,320 total params
        1,072,277,760 non-embedding params
        """
        return cls.v1(
            d_model=1280,
            hidden_size=1280 * 8,
            n_layers=20,
            n_heads=12,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 2B model config.

        2,048,423,680 total params
        1,894,283,008 non-embedding params
        """
        return cls.v1(
            d_model=1536,
            hidden_size=1536 * 8,
            n_layers=25,
            n_heads=18,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 4B model config.

        4,091,799,040 total params
        3,886,278,144 non-embedding params
        """
        return cls.v1(
            d_model=2048,
            hidden_size=2048 * 8,
            n_layers=30,
            n_heads=24,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B model config.

        8,142,615,040 total params
        7,885,713,920 non-embedding params
        """
        return cls.v1(
            d_model=2560,
            hidden_size=2560 * 8,
            n_layers=40,
            n_heads=30,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_14B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 14B model config.

        14,301,109,760 total params
        13,992,828,416 non-embedding params
        """
        return cls.v1(
            d_model=3072,
            hidden_size=3072 * 8,
            n_layers=50,
            n_heads=36,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v1_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B model config.

        32,311,906,560 total params
        31,900,864,768 non-embedding params
        """
        return cls.v1(
            d_model=4096,
            hidden_size=4096 * 8,
            n_layers=65,
            n_heads=48,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model: int = kwargs.pop("d_model")
        n_layers: int = kwargs.pop("n_layers")
        n_heads: int = kwargs.pop("n_heads")
        hidden_size: int = kwargs.pop("hidden_size")
        n_kv_heads = 8
        head_dim = 128
        global_layer_interval = 5
        global_rope_theta = 1_000_000
        layer_norm_eps = 1e-6
        dtype = DType.float32
        attn_backend = kwargs.pop("attn_backend", AttentionBackendName.flash_3)
        lm_head_loss_impl = kwargs.pop("lm_head_loss_impl", LMLossImplementation.default)
        use_gdn = kwargs.pop("use_gdn", False)

        local_rope_theta = 10_000
        local_window_size = 1024

        layer_norm = LayerNormConfig(
            name=LayerNormType.rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        feed_forward = FeedForwardConfig(
            hidden_size=hidden_size,
            bias=False,
            dtype=dtype,
            activation=ActivationFunction.silu,
        )

        if use_gdn:
            block = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=GatedDeltaNetConfig(
                    n_heads=n_heads,
                    n_v_heads=n_heads,
                    head_dim=head_dim,
                    expand_v=1.0,
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )
        else:
            block = TransformerBlockConfig(
                name=TransformerBlockType.peri_norm,
                sequence_mixer=AttentionConfig(
                    name=AttentionType.default,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    bias=False,
                    rope=RoPEConfig(name=RoPEType.default, theta=local_rope_theta),
                    gate=GateConfig(
                        granularity=GateGranularity.elementwise,
                        full_precision=True,
                    ),
                    qk_norm=layer_norm,
                    use_head_qk_norm=True,
                    backend=attn_backend,
                    sliding_window=SlidingWindowAttentionConfig(
                        pattern=[local_window_size] * (global_layer_interval - 1) + [-1],
                        force_full_attention_on_first_layer=False,
                        force_full_attention_on_last_layer=False,
                    ),
                    dtype=dtype,
                ),
                feed_forward=feed_forward,
                layer_norm=layer_norm,
            )

        block_overrides: Dict[int, TransformerBlockConfig] = {}
        for layer_idx in range(n_layers):
            if layer_idx % global_layer_interval == (global_layer_interval - 1):
                global_block = TransformerBlockConfig(
                    name=TransformerBlockType.peri_norm,
                    sequence_mixer=AttentionConfig(
                        name=AttentionType.default,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        bias=False,
                        rope=RoPEConfig(name=RoPEType.default, theta=global_rope_theta),
                        gate=GateConfig(
                            granularity=GateGranularity.elementwise,
                            full_precision=True,
                        ),
                        qk_norm=layer_norm,
                        use_head_qk_norm=True,
                        backend=attn_backend,
                        dtype=dtype,
                    ),
                    feed_forward=feed_forward,
                    layer_norm=layer_norm,
                )
                block_overrides[layer_idx] = global_block

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(
                loss_implementation=lm_head_loss_impl,
                layer_norm=layer_norm,
                bias=False,
                dtype=dtype,
            ),
            dtype=dtype,
            block_overrides=block_overrides if block_overrides else None,
            embed_scale=math.sqrt(d_model),
            **kwargs,
        )

    @classmethod
    def v2_260M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 260M model config.

        259,551,360 total params
        195,326,080 non-embedding params
        """
        return cls.v2(
            d_model=640,
            hidden_size=640 * 8,
            n_layers=10,
            n_heads=8,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_709M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 709M model config.

        708,903,680 total params
        606,143,232 non-embedding params
        """
        return cls.v2(
            d_model=1024,
            hidden_size=1024 * 8,
            n_layers=15,
            n_heads=16,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_1p3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1.3B model config.

        1,253,157,120 total params
        1,124,706,560 non-embedding params
        """
        return cls.v2(
            d_model=1280,
            hidden_size=1280 * 8,
            n_layers=20,
            n_heads=16,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_2B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 2.2B model config.

        2,156,558,080 total params
        2,002,417,408 non-embedding params
        """
        return cls.v2(
            d_model=1536,
            hidden_size=1536 * 8,
            n_layers=25,
            n_heads=24,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 4.3B model config.

        4,312,000,000 total params
        4,106,479,104 non-embedding params
        """
        return cls.v2(
            d_model=2048,
            hidden_size=2048 * 8,
            n_layers=30,
            n_heads=32,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B model config.

        8,588,259,840 total params
        8,331,358,720 non-embedding params
        """
        return cls.v2(
            d_model=2560,
            hidden_size=2560 * 8,
            n_layers=40,
            n_heads=40,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_15B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 15B model config.

        15,087,541,760 total params
        14,779,260,416 non-embedding params
        """
        return cls.v2(
            d_model=3072,
            hidden_size=3072 * 8,
            n_layers=50,
            n_heads=48,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def v2_34B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 34B model config.

        34,084,000,000 total params
        33,672,958,208 non-embedding params
        """
        return cls.v2(
            d_model=4096,
            hidden_size=4096 * 8,
            n_layers=65,
            n_heads=64,
            vocab_size=vocab_size,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Model size registry
# ---------------------------------------------------------------------------


@dataclass
class _ModelSizeSettings:
    """Training settings for a specific model size."""

    size: str
    num_nodes: int
    batch_size_round_nearest: int
    activation_memory_budget: float


class GemmaLikeOlmoV2(StrEnum):
    GL_260M = "260M"
    GL_709M = "709M"
    GL_1p3B = "1.3B"
    GL_2B = "2B"
    GL_4B = "4B"
    GL_8B = "8B"
    GL_15B = "15B"
    GL_34B = "34B"

    def get_settings(
        self, vocab_size: int, use_gdn: bool = False
    ) -> Tuple[TransformerConfig, _ModelSizeSettings]:
        """Get the model config and all settings for this model size."""
        settings_map = {
            GemmaLikeOlmoV2.GL_260M: _ModelSizeSettings("260M", 1, 16, 1.0),
            GemmaLikeOlmoV2.GL_709M: _ModelSizeSettings("709M", 2, 16, 1.0),
            GemmaLikeOlmoV2.GL_1p3B: _ModelSizeSettings("1p3B", 3, 16, 1.0),
            GemmaLikeOlmoV2.GL_2B: _ModelSizeSettings("2B", 8, 16, 1.0),
            GemmaLikeOlmoV2.GL_4B: _ModelSizeSettings("4B", 9, 32, 1.0),
            GemmaLikeOlmoV2.GL_8B: _ModelSizeSettings("8B", 14, 64, 0.9),
            GemmaLikeOlmoV2.GL_15B: _ModelSizeSettings("15B", 16, 128, 0.4),
            GemmaLikeOlmoV2.GL_34B: _ModelSizeSettings("34B", 16, 16, 0.1),
        }
        if self not in settings_map:
            raise ValueError(
                f"Model not in list! Valid models: {[m.name for m in GemmaLikeOlmoV2]}\n\n"
            )

        settings = settings_map[self]
        config_method = getattr(GemmaLikeTransformerConfig, f"v2_{settings.size}")
        model_config = config_method(vocab_size, use_gdn=use_gdn)
        return model_config, settings


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def handle_custom_args(
    overrides: list[str],
) -> tuple[list[str], argparse.Namespace]:
    """Extract custom override values using argparse and remove them from the list."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix-base-dir", type=str, default="gs://ai2-llm")
    parser.add_argument("--root-dir", type=str, default="")
    parser.add_argument("--work-dir", type=str, default="")
    parser.add_argument("--save-folder", type=str, default="")
    parser.add_argument("--lr-multiplier", type=float, default=1.0)
    parser.add_argument("--batch-multiplier", type=float, default=1.0)
    parser.add_argument("--chinchilla-multiple", type=float, default=4.0)
    parser.add_argument("--no-beaker-launch", action="store_true", default=False)
    parser.add_argument("--use-gdn", action="store_true", default=False)
    parser.add_argument(
        "--data-mix",
        type=DataMix,
        choices=list(DataMix),
        default=str(DataMix.OLMo_mix_0925),
    )
    # Expansion-specific args
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
        "--load-embeddings",
        action="store_true",
        default=False,
        help="Load embeddings from the checkpoint (default: False, i.e. embeddings keep "
        "random init since procedural pretraining uses a different vocabulary).",
    )

    arg_prefixes: List[str] = []
    boolean_flags: List[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._StoreAction):
            arg_prefixes.extend(action.option_strings)
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            boolean_flags.extend(action.option_strings)

    custom_args_list = []
    remaining = []
    for override in overrides:
        matched = False
        if any(override.startswith(f"{prefix}=") for prefix in arg_prefixes):
            key, value = override.split("=", 1)
            custom_args_list.extend([key, value])
            matched = True
        elif override in boolean_flags:
            custom_args_list.append(override)
            matched = True

        if not matched:
            remaining.append(override)

    args = parser.parse_args(custom_args_list)
    return remaining, args


# ---------------------------------------------------------------------------
# Hyperparameter computation (Li 2025 step law)
# ---------------------------------------------------------------------------


def get_learning_rate(model_params: int, training_tokens: int) -> float:
    """
    Get optimal learning rate using step law from Li 2025.
    https://arxiv.org/pdf/2503.04715v1
    """
    n = model_params
    d = training_tokens
    lr = 1.79 * pow(n, -0.713) * pow(d, 0.307)
    print(f"Model size: {n}, training tokens: {d}, opt_lr: {lr}")
    return lr


def get_global_batch_size(
    model_params: int,
    training_tokens: int,
    sequence_length: int,
    round_nearest: int,
    batch_size_multiplier: float = 1.0,
) -> int:
    """
    Get optimal global batch size in tokens using step law from Li 2025.
    https://arxiv.org/pdf/2503.04715v1
    """
    n = model_params
    d = training_tokens
    global_bsz = 0.58 * pow(d, 0.571)
    print(f"Model size: {n}, training tokens: {d}, opt_global_bsz: {global_bsz}")
    if batch_size_multiplier != 1.0:
        print(f"Applying bsz multiplier: {batch_size_multiplier}")
        global_bsz = global_bsz * batch_size_multiplier

    instance_bsz = global_bsz / sequence_length
    rounded_instance_bsz = int(math.ceil(instance_bsz / round_nearest) * round_nearest)
    print(f"Rounding instance bsz from {instance_bsz} to {rounded_instance_bsz}")
    rounded_global_bsz = sequence_length * rounded_instance_bsz
    print(f"Rounding global bsz from {global_bsz} to {rounded_global_bsz}")
    return rounded_global_bsz


def parse_model_size(run_name: str) -> GemmaLikeOlmoV2:
    """
    Parse model size from run name.
    The run name must contain one of the enum values (e.g., "260M", "1.3B", "8B").
    Examples: "260m", "gl-v2-260m", "1.3b", "1p3b" (normalized to "1.3b").
    """
    normalized = run_name.lower().strip().replace("1p3b", "1.3b").replace("1p3", "1.3")
    for size in sorted(GemmaLikeOlmoV2, key=lambda s: len(s.value), reverse=True):
        if size.value.lower() in normalized:
            return size
    raise ValueError(
        f"Could not parse model size from run name '{run_name}'. "
        f"Valid sizes: {[s.value for s in GemmaLikeOlmoV2]}. "
        f"Examples: '260m', 'gl-v2-260m', '1.3b'"
    )


# ---------------------------------------------------------------------------
# Experiment config builder
# ---------------------------------------------------------------------------


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build experiment config for the Gemma-like expand ladder.

    Model size is parsed from the run name (e.g. "gl-v2-260m" → 260M).
    Hyperparameters are computed using StepFun optimal schedules [Li 2025].

    Standard config overrides:
        --train_module.optim.lr=0.001          Override learning rate
        --data_loader.global_batch_size=1000   Override batch size
        --launch.num_nodes=2                   Override node count

    Convenience multipliers:
        --lr-multiplier=2.0                    Multiply computed learning rate
        --batch-multiplier=0.5                 Multiply computed batch size
        --chinchilla-multiple=8                Multiply Chinchilla training tokens

    Expansion args:
        --checkpoint=path/to/small.pt          Small checkpoint to expand from
        --alpha=0.5                            Expansion interpolation weight
        --load-embeddings                      Also expand embedding weights
    """
    global _expand_from, _alpha, _load_embeddings

    model = parse_model_size(cli_context.run_name)
    print(f"Parsed model size: {model} from run name: {cli_context.run_name}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name_with_timestamp = f"{cli_context.run_name}-{timestamp}"

    overrides = list(cli_context.overrides)
    overrides, custom_args = handle_custom_args(overrides)

    mix_base_dir = custom_args.mix_base_dir
    data_mix = custom_args.data_mix
    lr_multiplier = custom_args.lr_multiplier
    batch_multiplier = custom_args.batch_multiplier
    chinchilla_multiple = custom_args.chinchilla_multiple
    no_beaker_launch = custom_args.no_beaker_launch
    use_gdn = custom_args.use_gdn

    # Expansion params -- store in module state for use during training
    _expand_from = custom_args.checkpoint
    _alpha = custom_args.alpha
    _load_embeddings = custom_args.load_embeddings

    sequence_length = DEFAULT_SEQUENCE_LENGTH
    root_dir = custom_args.root_dir or get_root_dir(cli_context.cluster)
    work_dir = custom_args.work_dir or get_work_dir(root_dir)
    save_folder = custom_args.save_folder or f"{root_dir}/checkpoints/{cli_context.run_name}"

    print(f"mix_base_dir (dataset location): {mix_base_dir}")
    print(f"root_dir (checkpoint location): {root_dir}")
    print(f"work_dir (local path for temp files): {work_dir}")
    print(f"save_folder (checkpoint location): {save_folder}")
    if _expand_from:
        print(f"expand_from: {_expand_from} (alpha={_alpha}, load_embeddings={_load_embeddings})")

    tokenizer_config = TokenizerConfig.dolma2()
    model_config, model_size_settings = model.get_settings(
        tokenizer_config.padded_vocab_size(), use_gdn=use_gdn
    )

    model_active_params = model_config.num_non_embedding_params
    train_duration = Duration.chinchilla_tokens(
        chinchilla_multiple, model_params=model_active_params
    )
    training_tokens = train_duration.value

    learning_rate = get_learning_rate(model_active_params, training_tokens)
    global_batch_size = get_global_batch_size(
        model_params=model_active_params,
        training_tokens=training_tokens,
        sequence_length=sequence_length,
        round_nearest=model_size_settings.batch_size_round_nearest,
        batch_size_multiplier=batch_multiplier,
    )
    adjusted_learning_rate = learning_rate * lr_multiplier
    if lr_multiplier != 1.0:
        print(
            f"Applied LR multiplier: {lr_multiplier}, "
            f"LR: {learning_rate} -> {adjusted_learning_rate}"
        )

    beaker_launch_config: Optional[BeakerLaunchConfig] = None
    if not no_beaker_launch:
        beaker_launch_config = build_launch_config(
            name=cli_context.run_name,
            cmd=cli_context.remote_cmd,
            cluster=cli_context.cluster,
            root_dir=root_dir,
            workspace="ai2/oe-t-ladder",
            num_nodes=model_size_settings.num_nodes,
            nccl_debug=True,
            step_soft_timeout=None,
        )

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        mix=data_mix,
        tokenizer=tokenizer_config,
        mix_base_dir=mix_base_dir,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=work_dir,
        instance_filter_config=InstanceFilterConfig(),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size, seed=34521, num_workers=8
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=adjusted_learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=CosWithWarmupAndLinearDecay(
            units=SchedulerUnits.tokens,
            warmup=2000 * global_batch_size,
            decay=2000 * global_batch_size,
            decay_fraction=None,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=model_size_settings.activation_memory_budget,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            work_dir=work_dir,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=train_duration,
        )
        .with_callbacks(configure_required_callbacks(cli_context.run_name))
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
    )

    if _has_stability_monitor:
        trainer_config = trainer_config.with_callback(
            "stability_monitor", StabilityMonitorCallback(enabled=True)  # type: ignore[name-defined]
        )

    trainer_config = (
        trainer_config
        .with_callback(
            "comet",
            CometCallback(
                name=cli_context.run_name,
                project="gl-olmo-v1",
                workspace="oe-t-ladder",
                cancel_check_interval=10,
                auto_resume=False,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_timestamp,
                group=cli_context.run_name,
                project="oe-t-ladder",
                entity="ai2-llm",
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=mix_base_dir,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_on_finish=True,
                log_interval=10,
                eval_interval=2_500,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(TASK_GROUPS["fast"]),
                tokenizer=tokenizer_config,
                eval_on_finish=True,
                eval_interval=5_000,
                lazy=True,
            ),
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
    )

    return experiment_config.merge(overrides)


# ---------------------------------------------------------------------------
# Custom train function with expansion injected before trainer.fit()
# ---------------------------------------------------------------------------


def _train_with_expansion(config: ExperimentConfig) -> None:
    """Run training with an optional weight-expansion step before trainer.fit()."""
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)

    # Build data loader (NumpyDataLoaderConfig path used by this script)
    assert isinstance(config.data_loader, NumpyDataLoaderConfig), type(config.data_loader)
    assert isinstance(config.dataset, NumpyDatasetConfig), type(config.dataset)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )

    trainer = config.trainer.build(train_module, data_loader)

    # Expansion step: paste small checkpoint into the large model
    if _expand_from:
        log.info(
            "Expanding weights from %s (alpha=%.2f, load_embeddings=%s)",
            _expand_from,
            _alpha,
            _load_embeddings,
        )
        small_state = torch.load(_expand_from, map_location="cpu", weights_only=True)
        model_state = get_model_state_dict(
            trainer.train_module.model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True),
        )
        expanded = expand_model_weights(
            small_state,
            model_state,
            alpha=_alpha,
            use_embeddings=_load_embeddings,
        )
        set_model_state_dict(
            trainer.train_module.model,
            model_state_dict=expanded,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False, strict=False),
        )
        log.info("Model expansion complete.")

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    trainer.fit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich import get_console

    usage = (
        f"[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] "
        f"[i b magenta]{' | '.join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]\n\n"
        "[b]Subcommands[/]\n"
        "[b magenta]launch:[/]       Launch the script on Beaker with the [b magenta]train[/] subcommand.\n"
        "[b magenta]train:[/]        Run the trainer (usually via torchrun, not directly).\n"
        "[b magenta]train_single:[/] Run the trainer on a single device (for debugging on CPU/MacBook).\n"
        "[b magenta]prep:[/]         Prepare the dataset ahead of training.\n"
        "[b magenta]launch_prep:[/]  Launch dataset prep on Beaker.\n"
        "[b magenta]dry_run:[/]      Pretty print the config and exit.\n\n"
        "[b]Examples[/]\n"
        f"$ [i]python {sys.argv[0]} dry_run gl-v2-260m ai2/jupiter --checkpoint=./small.pt[/]\n"
        f"$ [i]python {sys.argv[0]} launch gl-v2-260m ai2/jupiter --checkpoint=weka://path/to/small.pt --alpha=0.5[/]"
    )

    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd_str, run_name, cluster, *overrides = sys.argv
    cmd = SubCmd(cmd_str)
    cli_context = CliContext(script, cmd, run_name, cluster, overrides)

    config = build_experiment_config(cli_context)
    cmd.prepare_environment(config)

    if cmd in (SubCmd.train, SubCmd.train_single):
        if cmd == SubCmd.train_single:
            if (dp := getattr(config.train_module, "dp_config", None)) is not None:
                log.warning(
                    "'dp_config' is set to %s, but you can't use data parallelism when "
                    "running on a single device. Disabling.",
                    dp,
                )
                config.train_module.dp_config = None  # type: ignore[assignment]
            if (tp := getattr(config.train_module, "tp_config", None)) is not None:
                log.warning(
                    "'tp_config' is set to %s, but you can't use tensor parallelism when "
                    "running on a single device. Disabling.",
                    tp,
                )
                config.train_module.tp_config = None  # type: ignore[assignment]
        _train_with_expansion(config)
        teardown_training_environment()
    else:
        cmd.run(config)
