import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import rich
from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.internal.common import CLUSTER_TO_GPU_TYPE
from olmo_core.internal.experiment import (
    CommonComponents,
    SubCmd,
    build_common_components,
)
from olmo_core.launch.beaker import BeakerLaunchConfig, OLMoCoreBeakerImage
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 32 * SEQUENCE_LENGTH
WARMUP_STEPS = 1000
N_TOKENS = 30_000 * GLOBAL_BATCH_SIZE


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
                paths.append(line)

    return paths


DATA_ROOT = "/vast/myh2014/data".rstrip("/")
DATA_PATHS = _read_data_mix_file("OLMo-mix-0625-150Bsample.txt")
EVAL_DATA_PATHS = _read_data_mix_file("v3-small-ppl-validation.txt")
DATA_WORK_DIR = "scratch/myh2014/ppt2/data/"

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(Config):
    run_name: str
    launch: Optional[BeakerLaunchConfig]
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    backend: Optional[str]
    init_seed: int = 12536


def build_model_config(common: CommonComponents) -> TransformerConfig:
    config = TransformerConfig.olmo2_1B_v2(
        vocab_size=common.tokenizer.padded_vocab_size()
    )
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


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    rank_microbatch_size = 4 * SEQUENCE_LENGTH
    if common.launch is not None:
        gpus = {CLUSTER_TO_GPU_TYPE.get(c, "unknown") for c in common.launch.clusters}
        if all("B200" in g for g in gpus):
            rank_microbatch_size *= 2

    return TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=common.dataset.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=4e-4 * 2,
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

    if common.launch is None:
        cluster = "local"
    else:
        assert len(common.launch.clusters) == 1
        cluster = common.launch.clusters[0]

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
                save_interval=250,  # willm: 500 corresponds to original paper
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
                eval_dataset=NumpyDatasetConfig(
                    paths=EVAL_DATA_PATHS,
                    name=NumpyDatasetType.padded_fsl,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=common.tokenizer,
                    work_dir=DATA_WORK_DIR,
                ),
                eval_interval=250,
                eval_duration=Duration.steps(50),
            ),
        )
        .with_recommended_evals(
            common.tokenizer,
            SEQUENCE_LENGTH,
            cluster,
            task_set="fast",
            eval_interval=1000,
        )
    )


def build_config(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    checkpoint: str,
    overrides: List[str],
    *,
    common_config_builder: Callable[..., CommonComponents] = build_common_components,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    train_module_config_builder: Callable[
        [CommonComponents], TransformerTrainModuleConfig
    ],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    finalize_config: Optional[Callable[[ExperimentConfig], None]] = None,
    **kwargs,
) -> ExperimentConfig:
    common = common_config_builder(script, cmd, run_name, cluster, overrides, **kwargs)

    model = model_config_builder(common)

    dataset = NumpyDatasetConfig(
        # @willm might be called data_paths
        paths=DATA_PATHS,
        name=NumpyDatasetType.fsl,
        work_dir=DATA_WORK_DIR,
        tokenizer=common.tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
    )

    trainer = trainer_config_builder(common)
    for name, cb in common.callbacks.items():
        if name not in trainer.callbacks:
            trainer.add_callback(name, cb)

    config = ExperimentConfig(
        run_name=run_name,
        launch=common.launch,
        model=model,
        dataset=dataset,
        data_loader=common.data_loader,
        train_module=train_module_config_builder(common),
        trainer=trainer,
    )

    config = config.merge(overrides)

    _set_beaker_execution_units(config)

    if finalize_config is not None:
        finalize_config(config)

    if not trainer.maybe_load_checkpoint(checkpoint):
        log.info(f"Initializing model from checkpoint: {checkpoint}")
        config.train_module.load_checkpoint = checkpoint

    return config


def main(
    *,
    global_batch_size: int,
    common_config_builder: Callable[..., CommonComponents] = build_common_components,
    model_config_builder: Callable[[CommonComponents], TransformerConfig],
    train_module_config_builder: Callable[
        [CommonComponents], TransformerTrainModuleConfig
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
):
    USAGE = f"""
PPT Phase 1.

[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]launch|train|dry_run[/] [i b]RUN_NAME PRETRAIN_CHECKPOINT CLUSTER[/] [i][OVERRIDES...][/]

[b]Subcommands[/]
[b magenta]launch:[/]      Launch the script on Beaker with the [b magenta]train[/] subcommand.
[b magenta]train:[/]       Run the trainer. You usually shouldn't invoke the script with this subcommand directly.
             Instead use the [b magenta]launch[/] cmd to submit it to Beaker or run it via torchrun if you know what you're doing.
[b magenta]dry_run:[/]     Print the config for debugging.

[b]Examples[/]
$ [i]python {sys.argv[0]} launch run01 gs://ai2-llm/checkpoints/peteish32/step419000 ai2/jupiter-cirrascale-2 --launch.num_nodes=2[/]
""".strip()

    # Parse command line arguments.
    if len(sys.argv) < 5 or sys.argv[1] not in ("launch", "train", "dry_run"):
        rich.get_console().print(USAGE, highlight=False)
        sys.exit(1)

    script, cmd, run_name, checkpoint, cluster, *overrides = sys.argv

    cmd = SubCmd(cmd)

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
        # myhu: @willm you might need to uncomment these
        # use_hostname_constraints=use_hostname_constraints,
        # num_execution_units=num_execution_units,
    )

    cmd.prepare_environment(config)
    config.trainer.load_checkpoint(checkpoint, load_trainer_state=False)
    cmd.run(config)


if __name__ == "__main__":
    main(
        global_batch_size=GLOBAL_BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_instance_filter=False,
        include_default_evals=False,
        intra_document_masking=False,
    )
