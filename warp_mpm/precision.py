"""One floating-point precision for all simulation buffers and kernels."""

import torch
import warp as wp

from simulation_runtime import simulation_precision

scalar = wp.float64 if simulation_precision == "fp64" else wp.float32
vec2 = wp.vec2d if simulation_precision == "fp64" else wp.vec2
vec3 = wp.vec3d if simulation_precision == "fp64" else wp.vec3
mat22 = wp.mat22d if simulation_precision == "fp64" else wp.mat22
mat33 = wp.mat33d if simulation_precision == "fp64" else wp.mat33
torch_dtype = torch.float64 if simulation_precision == "fp64" else torch.float32


@wp.func
def scalar_index(index: int) -> scalar:
    # Keep Warp 1.17's float64 constant conversion from changing loop index types.
    return scalar(index)
