"""Independent float64, CPU-only cloth equations for derivative diagnostics.

Used by check_differentiable.py; this is not a production simulation path.
"""

import torch
import warp as wp

from warp_mpm.differentiable import ClothState, DifferentiableMPM


def reference_step(
    state: ClothState, parameters: torch.Tensor, vertices: torch.Tensor,
    faces: torch.Tensor, volumes: torch.Tensor, dt: float, inv_dx: float,
    friction: float, rpic: float, gravity: torch.Tensor, grid_lim: float,
) -> ClothState:
    """Free-cloth APIC transfer, native anisotropic stress and return mapping."""
    x, v, affine, d = state
    ne = len(faces)
    d1, d2, d3 = d.unbind(dim=2)
    q1 = d1 / d1.norm(dim=1, keepdim=True)
    q3 = torch.linalg.cross(d1, d2)
    q3 = q3 / q3.norm(dim=1, keepdim=True)
    q2 = torch.linalg.cross(q3, q1)
    q = torch.stack((q1, q2, q3), dim=2)
    r13, r23, r33 = (q.transpose(1, 2) @ d3[:, :, None]).squeeze(2).unbind(1)
    normal_force = 500. * (1. - r33).square()
    shear_force = 500. * torch.stack((r13, r23), dim=1).norm(dim=1)
    sliding = shear_force > friction * normal_force
    scale = torch.where(sliding, friction * normal_force / shear_force, torch.ones_like(shear_force))
    r13 = torch.where(r33 > 1., torch.zeros_like(r13), r13 * scale)
    r23 = torch.where(r33 > 1., torch.zeros_like(r23), r23 * scale)
    r33 = r33.clamp(max=1.)
    d3 = (q @ torch.stack((r13, r23, r33), dim=1)[:, :, None]).squeeze(2)
    # Match the second QR decomposition in the native stress function.
    r13, r23, r33 = (q.transpose(1, 2) @ d3[:, :, None]).squeeze(2).unbind(1)
    scaled = vertices * torch.stack((parameters.new_tensor(1.), parameters[2], parameters.new_tensor(1.)))
    e1, e2 = scaled[faces[:, 1]] - scaled[faces[:, 0]], scaled[faces[:, 2]] - scaled[faces[:, 0]]
    rest11 = e1.norm(dim=1)
    rest12 = (e1 * e2).sum(1) / rest11
    rest22 = (e2 - (rest12 / rest11)[:, None] * e1).norm(dim=1)
    i11, i12, i22 = 1. / rest11, -rest12 / (rest11 * rest22), 1. / rest22
    f11 = d1.norm(dim=1) * i11
    f12 = d1.norm(dim=1) * i12 + (q1 * d2).sum(1) * i22
    f22 = (q2 * d2).sum(1) * i22
    zero = torch.zeros_like(f11)
    length = ((f11 + f22).square() + f12.square()).sqrt()
    cosine, sine = (f11 + f22) / length, f12 / length
    youngs, nu = 100. * parameters[1], .3
    mu, lam = youngs / (2. * (1. + nu)), youngs * nu / ((1. + nu) * (1. - 2. * nu))
    j = f11 * f22
    dr11 = 2. * mu * (f11 - cosine) + lam * (j - 1.) * f22
    dr12 = 2. * mu * (f12 - sine)
    dr22 = 2. * mu * (f22 - cosine) + lam * (j - 1.) * f11
    dr = torch.stack((dr11, dr12, 500. * r13, zero, dr22, 500. * r23,
                      zero, zero, -500. * (1. - r33).square()), dim=1).reshape(ne, 3, 3)
    ridt = torch.stack((f11, zero, zero, f12, f22, zero, r13, r23, r33), dim=1).reshape(ne, 3, 3)
    k3 = dr @ ridt
    k3_sym = k3.triu() + k3.triu(1).transpose(1, 2)
    p = q @ k3_sym @ torch.linalg.inv(ridt)
    f2 = -volumes[:ne, None] * (i11[:, None] * p[:, :, 0] + i12[:, None] * p[:, :, 1])
    f3 = -volumes[:ne, None] * i22[:, None] * p[:, :, 1]
    forces = torch.stack((-f2 - f3, f2, f3), dim=1)
    vertex_force = torch.zeros_like(x[ne:]).index_add(0, faces.flatten(), forces.reshape(-1, 3))
    stress = volumes[:ne, None, None] * p[:, :, 2, None] * d3[:, None, :]
    grid_pos = x * inv_dx
    base = (grid_pos - .5).to(torch.int64)
    fx = grid_pos - base
    shifts = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3))
    weights = torch.stack((.5 * (1.5 - fx).square(), .75 - (fx - 1.).square(), .5 * (fx - .5).square()), dim=2)
    derivatives = torch.stack((fx - 1.5, -2. * (fx - 1.), fx - .5), dim=2)
    wx, wy, wz = (weights[:, axis, shifts[:, axis]] for axis in range(3))
    weight = wx * wy * wz
    dwx, dwy, dwz = (derivatives[:, axis, shifts[:, axis]] for axis in range(3))
    dweight = inv_dx * torch.stack((dwx * wy * wz, wx * dwy * wz, wx * wy * dwz), dim=2)
    dpos = shifts[None] - fx[:, None]
    ng = int(grid_lim * inv_dx)
    nodes = base[:, None] + shifts
    indices = nodes[:, :, 0] * ng * ng + nodes[:, :, 1] * ng + nodes[:, :, 2]
    unique, inverse = indices.unique(return_inverse=True)
    affine_transfer = (1. - rpic) * affine + rpic / 2. * (affine - affine.transpose(1, 2))
    if rpic < -.001:
        affine_transfer = torch.zeros_like(affine_transfer)
    mass = parameters[0] * volumes
    transfer_force = torch.cat((-(stress[:, None] @ dweight[:ne, :, :, None]).squeeze(3),
                                weight[ne:, :, None] * vertex_force[:, None]))
    momentum = weight[:, :, None] * mass[:, None, None] * (v[:, None] +
               (affine_transfer[:, None] @ (dpos / inv_dx)[:, :, :, None]).squeeze(3)) + dt * transfer_force
    grid_mass = torch.zeros(len(unique), dtype=x.dtype).index_add(0, inverse.flatten(), (weight * mass[:, None]).flatten())
    grid_momentum = torch.zeros((len(unique), 3), dtype=x.dtype).index_add(0, inverse.flatten(), momentum.reshape(-1, 3))
    active = grid_mass > 1e-15
    grid_v = torch.where(active[:, None], grid_momentum / grid_mass.clamp(min=1e-15)[:, None] + dt * gravity,
                         torch.zeros_like(grid_momentum))
    gathered = grid_v[inverse]
    new_v = (gathered * weight[:, :, None]).sum(1)
    new_c = 4. * inv_dx * (gathered[:, :, :, None] * dpos[:, :, None, :] * weight[:, :, None, None]).sum(1)
    new_f = (gathered[:ne, :, :, None] * dweight[:ne, :, None, :]).sum(1)
    vertex_x = (x[ne:] + dt * new_v[ne:]).clamp(min=2. / inv_dx, max=grid_lim - 2. / inv_dx)
    positions = torch.cat((vertex_x[faces].mean(1), vertex_x))
    velocities = torch.cat((new_v[ne:][faces].mean(1), new_v[ne:]))
    evolved3 = ((torch.eye(3, dtype=x.dtype)[None] + dt * new_f) @ d3[:, :, None]).squeeze(2)
    directions = torch.stack((vertex_x[faces[:, 1]] - vertex_x[faces[:, 0]],
                              vertex_x[faces[:, 2]] - vertex_x[faces[:, 0]], evolved3), dim=2)
    return positions, velocities, new_c, directions


def reference_rollout(
    engine: DifferentiableMPM, initial: ClothState, parameters: torch.Tensor,
    vertices: torch.Tensor, substeps: int,
) -> ClothState:
    """Float64 reference for the free-cloth CPU fixture only."""
    assert engine.device.is_cpu and not engine.colliders and not engine.solver.num_joint_v
    assert not engine.solver.num_joint_f and engine.model.grid_v_damping_scale >= 1.
    assert torch.allclose(wp.to_torch(engine.model.nu), torch.full_like(wp.to_torch(engine.model.nu), .3))
    assert (wp.to_torch(engine.model.gamma) == 500.).all() and (wp.to_torch(engine.model.kappa) == 500.).all()
    faces = torch.as_tensor(engine.work.faces.numpy(), dtype=torch.int64)
    volumes = wp.to_torch(engine.work.particle_vol).double()
    gravity = torch.tensor(list(engine.model.gravitational_accelaration), dtype=torch.float64)
    state = tuple(value.double() for value in initial)
    for _ in range(substeps):
        state = reference_step(state, parameters, vertices.double(), faces, volumes,
                               float(torch.tensor(engine.dt)), engine.model.inv_dx,
                               engine.model.friction_coeff, engine.model.rpic_damping, gravity, engine.model.grid_lim)
    return state
