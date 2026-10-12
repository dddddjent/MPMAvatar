"""Select repeatable tensor preparation and Warp atomic ordering at startup."""

import os
import warp as wp


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
