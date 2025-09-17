"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/scripts/train/ppt2/phase0_launch.py run01 ai2/jupiter-cirrascale-2 [OVERRIDES...]
"""

import sys
from typing import List

from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.utils import generate_uuid, prepare_cli_environment
from olmo_core.internal.common import build_launch_config


def get_launch_config(run_name, cluster) -> BeakerLaunchConfig:

    launch_config = build_launch_config(
        cmd=["python", "src/scripts/train/ppt2/phase0_ai2.py", "train", run_name],
        name=run_name,
        cluster=cluster,
        workspace="ai2/willm-ppt2",
        root_dir="/weka/oe-training-default/ai2-llm",
    )
    launch_config.allow_dirty = True
    return launch_config


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} run_name cluster [OVERRIDES...]")
        sys.exit(1)

    run_name, cluster, *overrides = sys.argv[1:]

    prepare_cli_environment()

    get_launch_config(run_name, cluster).launch(follow=True)