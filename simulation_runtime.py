"""Select simulation precision and repeatable Warp execution at startup."""

import os
import argparse
import sys
import warp as wp


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
simulation_precision = _parser.parse_known_args()[0].precision


def configure_precision(precision: str) -> None:
    """Select precision before importing simulation structs and kernels."""
    global simulation_precision
    assert precision in ("fp32", "fp64"), precision
    assert "warp_mpm.precision" not in sys.modules or simulation_precision == precision, (
        "Use a fresh process to change simulation precision")
    simulation_precision = precision


def configure_simulation(deterministic: bool) -> None:
    """Choose one execution mode per process without editing native kernels."""
    if deterministic:
        # Native vertex volumes/normals use Torch index_add_ on CUDA.
        # Set cuBLAS's workspace before any CUDA initialization.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    wp.config.deterministic = (wp.DeterministicMode.GPU_TO_GPU if deterministic
                               else wp.DeterministicMode.NOT_GUARANTEED)
    print(f"Warp simulation execution: {wp.config.deterministic.name}", flush=True)
