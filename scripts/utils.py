"""
Utility functions and classes for PPT2 training scripts.
"""

from dataclasses import dataclass
from typing import Optional, Union

from olmo_core.data import TokenizerConfig
from olmo_core.internal.experiment import (
    CliContext,
    CommonComponents,
    build_common_components,
)
from olmo_core.launch.beaker import BeakerLaunchConfig


@dataclass
class LocalCommonComponents:
    """
    Minimal CommonComponents-compatible class for local/NYU cluster execution.
    Bypasses olmo_core's cluster detection which only recognizes AI2's Beaker clusters.
    """
    run_name: str
    launch: Optional[BeakerLaunchConfig]
    tokenizer: TokenizerConfig
    global_batch_size: int
    sequence_length: int


def is_local_cluster(cluster: str) -> bool:
    """
    Check if the cluster is a local/non-AI2 cluster that needs special handling.
    This includes "local" and NYU HPC cluster hostnames.
    """
    if cluster == "local":
        return True
    # NYU Greene/HPC cluster hostnames (e.g., gl039.hpc.nyu.edu)
    if ".hpc.nyu.edu" in cluster or ".nyu.edu" in cluster:
        return True
    # Add other non-AI2 clusters here as needed
    return False


def build_common_components_local(
    cli_context: CliContext,
    tokenizer: TokenizerConfig,
    global_batch_size: int,
    max_sequence_length: int,
    **kwargs,
) -> Union[LocalCommonComponents, CommonComponents]:
    """
    Custom builder for local/NYU cluster execution that bypasses olmo_core's
    cluster detection which only recognizes AI2's Beaker clusters.
    """
    if is_local_cluster(cli_context.cluster):
        # For local/NYU execution, create a minimal CommonComponents-compatible object
        return LocalCommonComponents(
            run_name=cli_context.run_name,
            launch=None,  # No Beaker launch for local execution
            tokenizer=tokenizer,
            global_batch_size=global_batch_size,
            sequence_length=max_sequence_length,
        )
    else:
        # For recognized AI2 clusters, use the standard builder
        return build_common_components(
            cli_context,
            tokenizer=tokenizer,
            global_batch_size=global_batch_size,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )
