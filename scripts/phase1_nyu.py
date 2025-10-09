import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

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
    main,
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

# === willm: taken from https://arxiv.org/abs/2502.19249 ===
SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 32 * SEQUENCE_LENGTH
WARMUP_STEPS = 1000
N_TOKENS = 500 * GLOBAL_BATCH_SIZE  # 35M tokens
# === willm: original values ===
# SEQUENCE_LENGTH = 8 * 1024
# GLOBAL_BATCH_SIZE = 4 * 1024 * 1024
# WARMUP_STEPS = 2000




def _read_data_mix_file(filename: str) -> List[str]:
    """Read URLs from a data mix file in the data_mixes folder."""
    script_dir = Path(__file__).parent.parent  # Get project root
    data_mix_path = script_dir / "data_mixes" / filename
    
    if not data_mix_path.exists():
        log.warning(f"Data mix file not found: {data_mix_path}")
        return []
    
    paths = []
    with open(data_mix_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                paths.append(line)
    
    return paths

DATA_ROOT = "/scratch/myh2014/ppt2/data".rstrip("/")
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
    init_seed: int = 12536
    backend: Optional[str] = "cpu:gloo,cuda:nccl"


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
            save_folder=f"gs://ai2-llm/checkpoints/{common.run_name}/",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.tokens(
                int(10 * N_TOKENS)
            ),  # willm: 1 * N_TOKENS is original
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


@dataclass
class AnnealingConfig(Config):
    """
    Custom config class for the annealing run.

    Making config classes isn't strictly necessary for OLMo-core, but it gives us a nice way to
    capture all of the hyperparameters for a run and an easy way to override those options from
    the command line without configuring a complicated command line parser.
    """

    run_name: str
    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536

    @classmethod
    def build(
        cls,
        *,
        script: str,
        cmd: str,
        run_name: str,
        checkpoint: str,
        cluster: str,
        overrides: List[str],
    ) -> "AnnealingConfig":
        root_dir = get_root_dir(cluster)

        tokenizer_config = TokenizerConfig.dolma2()

        # Get step number and max steps to infer where the learning rate left off.
        train_state = torch.load(
            resource_path(f"{checkpoint}/train", "rank0.pt"), weights_only=False
        )
        last_pretrain_step: int = train_state["global_step"]
        max_pretrain_steps: int = train_state.get("max_steps", 774861)  # default found in logs
        log.info(
            f"Will anneal from checkpoint at step {last_pretrain_step:,d} of {max_pretrain_steps:,d}"
        )

        # Now infer the learning rate.
        with resource_path(checkpoint, "config.json").open() as f:
            config = json.load(f)
        base_lr = config["optim"]["lr"]
        scheduler_config = config["trainer"]["callbacks"]["lr_scheduler"]["scheduler"]
        assert scheduler_config.pop("_CLASS_") == CosWithWarmup.__name__
        scheduler = CosWithWarmup(**scheduler_config)
        starting_lr = float(scheduler.get_lr(base_lr, last_pretrain_step, max_pretrain_steps))

        run_name = f"peteish32-from{last_pretrain_step}-{run_name}"

        config = AnnealingConfig(
            run_name=run_name,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[script, cmd, run_name, checkpoint, cluster, *overrides],
                cluster=cluster,
                nccl_debug=False,
            ),
            model=TransformerConfig.olmo2_32B(vocab_size=tokenizer_config.padded_vocab_size()),
            dataset=NumpyDatasetConfig.from_data_mix(
                AnnealingDataMix.dolmino100,
                tokenizer=tokenizer_config,
                mix_base_dir=root_dir,
                sequence_length=4096,
                work_dir=get_work_dir(root_dir),
            ),
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=2048 * 4096,  # NOTE: this is specified in TOKENS, not instances.
                seed=34521,  # NOTE: can update this to change data order.
                num_workers=4,
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=2 * 4096,  # NOTE: again this is specified in tokens.
                max_sequence_length=4096,
                z_loss_multiplier=1e-5,
                compile_model=True,
                optim=SkipStepAdamWConfig(
                    lr=starting_lr,
                    weight_decay=0.1,
                    betas=(0.9, 0.95),
                    group_overrides=[
                        OptimGroupOverride(
                            params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                        )
                    ],
                    compile=True,
                ),
                # dp_config=TransformerDataParallelConfig(
                #     name=DataParallelType.fsdp,
                #     param_dtype=DType.bfloat16,
                #     reduce_dtype=DType.float32,
                # ),
                dp_config=TransformerDataParallelConfig(
                    name=DataParallelType.hsdp,
                    param_dtype=DType.bfloat16,
                    reduce_dtype=DType.float32,
                    num_replicas=128 // 32,  # common.launch.num_nodes // 2,
                ),
                ac_config=TransformerActivationCheckpointingConfig(
                    mode=TransformerActivationCheckpointingMode.selected_modules,
                    modules=["blocks.*.feed_forward"],
                ),
                # ac_config=TransformerActivationCheckpointingConfig(
                #    mode=TransformerActivationCheckpointingMode.full
                # ),
                scheduler=LinearWithWarmup(
                    warmup_steps=0,
                    alpha_f=0.0,
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"gs://ai2-llm/checkpoints/peteish32-anneal/{run_name}",
                load_strategy=LoadStrategy.always,
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.tokens(int(100e9)),
            )
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=1000,
                    ephemeral_save_interval=500,
                    save_async=True,
                ),
            )
            .with_callback(
                "comet",
                CometCallback(
                    name=run_name,
                    workspace="ai2",
                    project="peteish32",
                    enabled=True,
                    cancel_check_interval=10,
                ),
            )
            .with_callback(
                "gpu_monitor",
                GPUMemoryMonitorCallback(),
            )
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "downstream_evaluator",
                DownstreamEvaluatorCallbackConfig(
                    tasks=[
                        # MMLU for backwards compatibility
                        "mmlu_stem_mc_5shot",
                        "mmlu_humanities_mc_5shot",
                        "mmlu_social_sciences_mc_5shot",
                        "mmlu_other_mc_5shot",
                        # MMLU test
                        "mmlu_stem_mc_5shot_test",
                        "mmlu_humanities_mc_5shot_test",
                        "mmlu_social_sciences_mc_5shot_test",
                        "mmlu_other_mc_5shot_test",
                        ## Core 12 tasks for backwards compatibility
                        # "arc_challenge",
                        # "arc_easy",
                        # "basic_arithmetic",
                        # "boolq",
                        # "commonsense_qa",
                        # "copa",
                        # "hellaswag",
                        # "openbook_qa",
                        # "piqa",
                        # "sciq",
                        # "social_iqa",
                        # "winogrande",
                        ## Core 12 tasks 5-shot
                        # "arc_challenge_rc_5shot",
                        # "arc_easy_rc_5shot",
                        ## "basic_arithmetic_rc_5shot",  # doesn't exist
                        ## "boolq_rc_5shot",  # we don't like it
                        # "csqa_rc_5shot",
                        ## "copa_rc_5shot",  # doesn't exist
                        # "hellaswag_rc_5shot",
                        # "openbookqa_rc_5shot",
                        # "piqa_rc_5shot",
                        ## "sciq_rc_5shot",  # doesn't exist
                        # "socialiqa_rc_5shot",
                        # "winogrande_rc_5shot",
                        ## New in-loop evals
                        # "arc_challenge_val_rc_5shot",
                        # "arc_challenge_val_mc_5shot",
                        "arc_challenge_test_rc_5shot",
                        # "arc_challenge_test_mc_5shot",
                        # "arc_easy_val_rc_5shot",
                        # "arc_easy_val_mc_5shot",
                        "arc_easy_test_rc_5shot",
                        # "arc_easy_test_mc_5shot",
                        # "boolq_val_rc_5shot",
                        # "boolq_val_mc_5shot",
                        "csqa_val_rc_5shot",
                        # "csqa_val_mc_5shot",
                        "hellaswag_val_rc_5shot",
                        # "hellaswag_val_mc_5shot",
                        # "openbookqa_val_rc_5shot",
                        # "openbookqa_val_mc_5shot",
                        "openbookqa_test_rc_5shot",
                        # "openbookqa_test_mc_5shot",
                        "piqa_val_rc_5shot",
                        # "piqa_val_mc_5shot",
                        "socialiqa_val_rc_5shot",
                        # "socialiqa_val_mc_5shot",
                        # "winogrande_val_rc_5shot",
                        # "winogrande_val_mc_5shot",
                        # "mmlu_stem_val_rc_5shot",
                        # "mmlu_stem_val_mc_5shot",
                        # "mmlu_humanities_val_rc_5shot",
                        # "mmlu_humanities_val_mc_5shot",
                        # "mmlu_social_sciences_val_rc_5shot",
                        # "mmlu_social_sciences_val_mc_5shot",
                        # "mmlu_other_val_rc_5shot",
                        # "mmlu_other_val_mc_5shot",
                    ],
                    tokenizer=tokenizer_config,
                    eval_interval=1000,
                    enabled=False,
                ),
            ),
        ).merge(overrides)

        # Make sure this is an 'AnnealingDataMix' instance.
        config.dataset.mix = AnnealingDataMix(config.dataset.mix)
        return config


def build_config(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
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

    return config



if __name__ == "__main__":
    USAGE = f"""
Anneal the 32B model.

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

    # Prepare the environment for the given command.
    if cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif cmd == "train":
        prepare_training_environment()
    else:
        raise NotImplementedError(cmd)

    # Build the config, applying any overrides.
    config = AnnealingConfig.build(
        script=script,
        cmd="train",
        run_name=run_name,
        checkpoint=checkpoint,
        cluster=cluster,
        overrides=overrides,
    )

    # Print the config for debugging and then execute the command.
    if get_local_rank() == 0:
        print(config)

    if cmd == "dry_run":
        pass
    elif cmd == "launch":
        config.launch.launch(follow=True)
    elif cmd == "train":
        try:
            train(checkpoint, config)
        finally:
            teardown_training_environment()
    else:
        raise NotImplementedError(cmd)
