"""Diagnostic cloth-only control: release transverse shear when r33 > 1.

Native solver code is unchanged. Used only by trace_hang_e.py's explicit control flag.
# python trace_hang_e.py --config output/hang_heavy_E_trace/config.json --output output/hang_heavy_E_trace/release_control --frames 75 --stride 10 --step-size 1e-12 --direction log --offsets -1 1 --substeps 400 --snapshot-stride 400 --release-tension-shear
"""
import warp as wp
from warp_mpm.mpm_solver import MPMModelStruct, MPMStateStruct, anisotropy_return_mapping, kirchoff_stress_Anisotropy
from warp_mpm.precision import scalar, mat33


@wp.func
def release_mapping(d: mat33, model: MPMModelStruct, p: int) -> mat33:
    q, r = wp.qr3(d)
    sign0 = scalar(1.0)
    sign1 = scalar(1.0)
    if r[0, 0] < scalar(0.0):
        sign0 = scalar(-1.0)
    if r[1, 1] < scalar(0.0):
        sign1 = scalar(-1.0)
    if sign0 * sign1 * r[2, 2] > scalar(1.0):
        sign2 = sign0 * sign1
        return mat33(d[0, 0], d[0, 1], sign2 * q[0, 2],
                     d[1, 0], d[1, 1], sign2 * q[1, 2],
                     d[2, 0], d[2, 1], sign2 * q[2, 2])
    return anisotropy_return_mapping(d, model, p)


@wp.kernel
def compute_cloth_stress(state: MPMStateStruct, model: MPMModelStruct, dt: scalar) -> None:
    p = wp.tid()
    if state.particle_selection[p] == 0:
        state.particle_d[p] = release_mapping(state.particle_d[p], model, p)
        state.particle_stress[p] = kirchoff_stress_Anisotropy(
            state.particle_R_inv[p], state.particle_d[p], state.faces[p], state.particle_vol[p], state,
            model.mu[p], model.lam[p], model.gamma[p], model.kappa[p])
