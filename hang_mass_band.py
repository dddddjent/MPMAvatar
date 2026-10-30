"""Explicit C2 grid-mass-band control for the hanging-cloth experiments.

# python probe_hang_mass_band.py --baseline output/hang_heavy_E_control_h1e-6_r10 --output output/hang_heavy_E_mass_band
"""
import warp as wp

from warp_mpm.mpm_solver import MPMModelStruct, MPMStateStruct
from warp_mpm.precision import scalar


@wp.func
def mass_activation(mass: scalar, lower: scalar, upper: scalar) -> scalar:
    activation = scalar(0.0)
    if mass >= upper:
        activation = scalar(1.0)
    elif mass > lower:
        t = (mass - lower) / (upper - lower)
        activation = t * t * t * (scalar(10.0) + t * (scalar(-15.0) + scalar(6.0) * t))
    return activation


@wp.kernel
def grid_normalization_mass_band(
    state: MPMStateStruct, model: MPMModelStruct, dt: scalar, lower: scalar, upper: scalar,
) -> None:
    i, j, k = wp.tid()
    mass = state.grid_m[i, j, k]
    # zero_grid already set v_out to zero; skip the division below the band.
    if mass > lower:
        activation = mass_activation(mass, lower, upper)
        velocity = state.grid_v_in[i, j, k] * (scalar(1.0) / mass)
        velocity = velocity + dt * model.gravitational_accelaration
        state.grid_v_out[i, j, k] = activation * velocity
