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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} run_name cluster [OVERRIDES...]")
        sys.exit(1)

    run_name, cluster, *overrides = sys.argv[1:]

    prepare_cli_environment()

    root_dir = "/weka/oe-training-default/ai2-llm"
    cmd = sys.argv
    build_launch_config(run_name, root_dir, cmd, cluster).launch(follow=True)