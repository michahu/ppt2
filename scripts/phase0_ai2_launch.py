"""
An example of how to launch the training script on Beaker.
Run this with:

    python src/scripts/train/ppt2/phase0_launch.py run01 ai2/jupiter-cirrascale-2 [OVERRIDES...]
"""

from functools import lru_cache
import sys
from typing import List, Optional

from beaker import Beaker, BeakerError, SecretNotFound

from olmo_core.launch.beaker import BeakerLaunchConfig, BeakerEnvSecret
from olmo_core.utils import generate_uuid, prepare_cli_environment
from olmo_core.internal.common import build_launch_config


# See https://github.com/allenai/OLMo-modular/blob/main/src/olmo_modular/internal/experiment.py
# Authenticate in order to clone private repo.
GITHUB_AUTH_STEPS = [
    "conda install gh --channel conda-forge",
    # assumes that conda is installed, which is true for our beaker images. # TODO: add to image
    "echo $GITHUB_TOKEN | gh auth login --with-token",
    # this is possibly redundant?
]


@lru_cache()
def get_beaker_client() -> Optional[Beaker]:
    try:
        return Beaker.from_env(check_for_upgrades=False)
    except BeakerError:
        return None


@lru_cache()
def get_beaker_username() -> Optional[str]:
    beaker = get_beaker_client()
    if beaker is not None:
        return beaker.account.whoami().name
    else:
        return None


def get_launch_config(run_name, cluster) -> BeakerLaunchConfig:
    launch_config = build_launch_config(
        # Match the command that would be created with the normal launcher.
        cmd=["src/scripts/train/ppt2/phase0_ai2.py", "train", run_name, cluster],
        name=run_name,
        cluster=cluster,
        workspace="ai2/willm-ppt2",
        root_dir="/weka/oe-training-default/ai2-llm",
    )

    # Configure GitHub token secret.
    beaker_user = get_beaker_username()
    github_token = BeakerEnvSecret(name="GITHUB_TOKEN", secret=f"{beaker_user}_GITHUB_TOKEN")
    launch_config.env_secrets.append(github_token)

    # Add initial setup steps to authenticate GitHub CLI.
    launch_config.setup_steps = GITHUB_AUTH_STEPS + launch_config.setup_steps

    # Allow dirty state.
    launch_config.allow_dirty = True

    return launch_config


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} run_name cluster [OVERRIDES...]")
        sys.exit(1)

    run_name, cluster, *overrides = sys.argv[1:]

    prepare_cli_environment()

    get_launch_config(run_name, cluster).launch(follow=True)