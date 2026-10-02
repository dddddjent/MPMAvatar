"""Geometry-only CUDA MPM rollouts on a fixed local patch grid.

The native cloth stress and particle/grid transfers retain their simulation units.
Only vertex G2P omits the native domain clipping; escaping particles fail clearly.
Torch, Warp, and the native solver are imported only when constructing a runner.
"""
# Template (activate the MPMAvatar environment first):
# python local_patch_fit.py --stage fit --patches PATH --output PATH --padding-cells 4 --substeps 400 --iterations 200 --density 1 --initial-E 100 --initial-H 1 --fd-log-E .05 --fd-H .005 --learning-rate-E .03 --learning-rate-H .003 --E-bounds 50 2500 --H-bounds .5 1.2 --device cuda:0

from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from local_patch_data import LocalPatch


@dataclass(frozen=True)
class PatchRollout:
    """World-meter predictions and errors; aggregate errors exclude frame zero."""

    vertices_m: np.ndarray
    mean_xyz_mse_m2: float
    mean_vertex_error_mm: float
    vertex_rmse_mm: float
    boundary_max_error_mm: float
    per_frame_xyz_mse_m2: np.ndarray
    per_frame_vertex_error_mm: np.ndarray
    per_frame_vertex_rmse_mm: np.ndarray
    per_frame_boundary_max_error_mm: np.ndarray
    grid_leakage_fraction: float
    per_frame_grid_leakage_fraction: np.ndarray


def score_rollout(patch: "LocalPatch", vertices_m: np.ndarray) -> PatchRollout:
    """Measure scored vertices only, preserving actual collar drift separately."""
    predicted = np.asarray(vertices_m, dtype=np.float64)
    reference = np.asarray(patch.reference_vertices_m, dtype=np.float64)
    assert predicted.shape == reference.shape
    assert reference.shape[0] >= 2
    assert patch.scored_vertex_ids.size > 0
    assert patch.boundary_vertex_ids.size > 0
    assert np.isfinite(predicted).all(), "Nonfinite local patch trajectory"
    errors = predicted[:, patch.scored_vertex_ids] - reference[:, patch.scored_vertex_ids]
    squared_vertex_errors = np.sum(errors * errors, axis=-1)
    per_frame_mse = np.mean(squared_vertex_errors, axis=1) / 3.0
    per_frame_mean = np.mean(np.sqrt(squared_vertex_errors), axis=1) * 1000.0
    per_frame_rmse = np.sqrt(np.mean(squared_vertex_errors, axis=1)) * 1000.0
    boundary_errors = predicted[:, patch.boundary_vertex_ids] - reference[:, patch.boundary_vertex_ids]
    per_frame_boundary = np.max(np.linalg.norm(boundary_errors, axis=-1), axis=1) * 1000.0
    return PatchRollout(
        vertices_m=predicted,
        mean_xyz_mse_m2=float(np.mean(per_frame_mse[1:])),
        mean_vertex_error_mm=float(np.mean(per_frame_mean[1:])),
        vertex_rmse_mm=float(np.sqrt(np.mean(squared_vertex_errors[1:])) * 1000.0),
        boundary_max_error_mm=float(np.max(per_frame_boundary)),
        per_frame_xyz_mse_m2=per_frame_mse,
        per_frame_vertex_error_mm=per_frame_mean,
        per_frame_vertex_rmse_mm=per_frame_rmse,
        per_frame_boundary_max_error_mm=per_frame_boundary,
        grid_leakage_fraction=0.0,
        per_frame_grid_leakage_fraction=np.zeros(len(reference), dtype=np.float64),
    )


def cloth_directions_and_volume(
    vertices_sim: np.ndarray, faces: np.ndarray, thickness_sim: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Native triangle directions and centroid/vertex volume quadrature."""
    assert thickness_sim > 0.0 and np.isfinite(vertices_sim).all()
    d1 = vertices_sim[faces[:, 1]] - vertices_sim[faces[:, 0]]
    d2 = vertices_sim[faces[:, 2]] - vertices_sim[faces[:, 0]]
    normal = np.cross(d1, d2)
    twice_area = np.linalg.norm(normal, axis=-1)
    assert np.all(twice_area > 0.0), "Degenerate patch triangle"
    directions = np.stack((d1, d2, normal / twice_area[:, None]), axis=-1)
    element_volume = 0.25 * thickness_sim * 0.5 * twice_area
    vertex_volume = np.zeros(len(vertices_sim), dtype=vertices_sim.dtype)
    np.add.at(vertex_volume, faces.reshape(-1), np.repeat(element_volume, 3))
    assert np.all(vertex_volume > 0.0), "Patch contains an unused vertex"
    return directions, np.concatenate((element_volume, vertex_volume))


def rest_direction_inverse(rest_vertices_sim: np.ndarray, faces: np.ndarray, H: float) -> np.ndarray:
    """Native QR inverse after scaling frame-11 y edges, independent of origin."""
    assert H > 0.0 and np.isfinite(H) and np.isfinite(rest_vertices_sim).all()
    axis_scale = np.asarray((1.0, H, 1.0), dtype=rest_vertices_sim.dtype)
    d1 = (rest_vertices_sim[faces[:, 1]] - rest_vertices_sim[faces[:, 0]]) * axis_scale
    d2 = (rest_vertices_sim[faces[:, 2]] - rest_vertices_sim[faces[:, 0]]) * axis_scale
    r11 = np.linalg.norm(d1, axis=-1)
    assert np.all(r11 > 0.0), "Degenerate rest edge"
    r12 = np.sum(d1 * d2, axis=-1) / r11
    r22 = np.linalg.norm(d2 - (r12 / r11)[:, None] * d1, axis=-1)
    assert np.all(r22 > 0.0), "Degenerate transformed rest triangle"
    return np.stack((1.0 / r11, -r12 / (r11 * r22), 1.0 / r22), axis=-1)


def _make_kernels(wp: Any, structures: Any) -> dict[str, Any]:
    """Define patch-only Warp kernels lazily, without importing a renderer."""
    State = structures.MPMStateStruct
    Model = structures.MPMModelStruct

    @wp.kernel
    def unclipped_vertex_g2p(state: State, model: Model, dt: float, offset: int) -> None:
        p = wp.tid() + offset
        grid_pos = state.particle_x[p] * model.inv_dx
        bx = wp.int(grid_pos[0] - 0.5)
        by = wp.int(grid_pos[1] - 0.5)
        bz = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )
        velocity = wp.vec3(0.0)
        affine = wp.mat33(0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    grid_v = state.grid_v_out[bx + i, by + j, bz + k]
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    velocity = velocity + grid_v * weight
                    affine = affine + wp.outer(grid_v, dpos) * (weight * model.inv_dx * 4.0)
        state.particle_v[p] = velocity
        state.particle_x[p] = state.particle_x[p] + dt * velocity
        state.particle_C[p] = affine

    @wp.kernel
    def scatter_boundary_grid(
        state: State,
        model: Model,
        particle_ids: wp.array(dtype=wp.int32),
        velocities: wp.array(dtype=wp.vec3),
        weights: wp.array(dtype=float, ndim=3),
        momentum: wp.array(dtype=wp.vec3, ndim=3),
    ) -> None:
        q = wp.tid()
        p = particle_ids[q]
        grid_pos = state.particle_x[p] * model.inv_dx
        bx = wp.int(grid_pos[0] - 0.5)
        by = wp.int(grid_pos[1] - 0.5)
        bz = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    wp.atomic_add(weights, bx + i, by + j, bz + k, weight)
                    wp.atomic_add(momentum, bx + i, by + j, bz + k, weight * velocities[q])

    @wp.kernel
    def prescribe_boundary_grid(
        state: State,
        weights: wp.array(dtype=float, ndim=3),
        momentum: wp.array(dtype=wp.vec3, ndim=3),
    ) -> None:
        i, j, k = wp.tid()
        weight = weights[i, j, k]
        if weight > 1.0e-15:
            state.grid_v_out[i, j, k] = momentum[i, j, k] / weight

    @wp.kernel
    def prescribe_boundary_particles(
        state: State,
        ids: wp.array(dtype=wp.int32),
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        offset: int,
    ) -> None:
        q = wp.tid()
        p = ids[q] + offset
        state.particle_x[p] = positions[q]
        state.particle_v[p] = velocities[q]
        state.particle_C[p] = wp.mat33(0.0)

    @wp.kernel
    def detect_scored_grid_leakage(
        state: State,
        model: Model,
        scored_ids: wp.array(dtype=wp.int32),
        mover_weights: wp.array(dtype=float, ndim=3),
        hits: wp.array(dtype=wp.int32),
        offset: int,
    ) -> None:
        q = wp.tid()
        p = scored_ids[q] + offset
        grid_pos = state.particle_x[p] * model.inv_dx
        bx = wp.int(grid_pos[0] - 0.5)
        by = wp.int(grid_pos[1] - 0.5)
        bz = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(wp.float(bx), wp.float(by), wp.float(bz))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.matrix_from_cols(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.75) - wp.cw_mul(wb, wb),
            wp.cw_mul(wc, wc) * 0.5,
        )
        hit = int(0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    if weight > 0.0 and mover_weights[bx + i, by + j, bz + k] > 1.0e-15:
                        hit = int(1)
        hits[q] = hit

    @wp.kernel
    def refresh_elements(state: State, offset: int) -> None:
        p = wp.tid()
        face = state.faces[p]
        a = int(face[0]) + offset
        b = int(face[1]) + offset
        c = int(face[2]) + offset
        state.particle_x[p] = (state.particle_x[a] + state.particle_x[b] + state.particle_x[c]) / 3.0
        state.particle_v[p] = (state.particle_v[a] + state.particle_v[b] + state.particle_v[c]) / 3.0
        d1 = state.particle_x[b] - state.particle_x[a]
        d2 = state.particle_x[c] - state.particle_x[a]
        d = state.particle_d[p]
        d3 = wp.vec3(d[0, 2], d[1, 2], d[2, 2])
        state.particle_d[p] = wp.matrix_from_cols(d1, d2, d3)

    return {
        "vertex_g2p": unclipped_vertex_g2p,
        "scatter_boundary": scatter_boundary_grid,
        "boundary_grid": prescribe_boundary_grid,
        "boundary_particles": prescribe_boundary_particles,
        "refresh_elements": refresh_elements,
        "detect_leakage": detect_scored_grid_leakage,
    }


class PatchSimulator:
    """Native cloth MPM on a meter-sized local domain, with prescribed collar."""

    def __init__(
        self,
        patch: "LocalPatch",
        substeps: int,
        device: str,
        nu: float = 0.3,
        gamma: float = 500.0,
        kappa: float = 500.0,
        gravity: tuple[float, float, float] = (0.0, -9.8, 0.0),
        damping: float = 1.1,
        friction_angle: float = 40.0,
        padding_cells: int = 4,
        cell_size_factor: float = 1.0,
    ) -> None:
        import torch
        import warp as wp
        from local_patch_data import local_grid_domain

        assert torch.device(device).type == "cuda", "Patch MPM requires an explicit CUDA device"
        assert torch.cuda.is_available(), "Patch MPM requires CUDA; no CPU fallback is provided"
        assert substeps > 0 and isinstance(substeps, int)
        assert 0.0 < cell_size_factor <= 1.0
        assert -1.0 < nu < 0.5
        assert gamma >= 0.0 and kappa >= 0.0
        assert np.isfinite(gravity).all() and np.isfinite(damping)
        assert 0.0 <= friction_angle < 90.0
        assert np.all(np.diff(patch.frame_ids) == 1), "Patch frames must be consecutive at 25 Hz"
        assert patch.reference_vertices_m.shape[0] >= 2
        assert patch.faces.shape[0] > 0 and patch.boundary_vertex_ids.size > 0
        assert patch.scored_vertex_ids.size > 0
        assert np.intersect1d(patch.scored_vertex_ids, patch.boundary_vertex_ids).size == 0
        assert patch.simulation_scale > 0.0 and np.isfinite(patch.simulation_scale)

        native_directory = Path(__file__).resolve().parent / "warp_mpm"
        assert native_directory.is_dir()
        sys.path.insert(0, str(native_directory))
        structures = import_module("mpm_data_structure")
        native = import_module("mpm_utils")
        solver_module = import_module("mpm_solver")
        wp.init()
        self.wp = wp
        self.torch = torch
        self.structures = structures
        self.native = native
        self.kernels = _make_kernels(wp, structures)
        self.patch = patch
        self.device = str(torch.device(device))
        self.substeps = substeps
        self.dt = (1.0 / 25.0) / substeps
        self.scale = float(patch.simulation_scale)
        self.cell_size_m = patch.cell_size_m * cell_size_factor
        self.origin_m, self.world_side_m, self.n_grid = local_grid_domain(patch, cell_size_factor, padding_cells)
        self.grid_lim = self.world_side_m * self.scale
        self.cell_dx_sim = self.cell_size_m * self.scale
        self.nu = float(nu)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.gravity = gravity
        self.damping = float(damping)
        self.friction_angle = float(friction_angle)
        self.n_elements = len(patch.faces)
        self.n_vertices = len(patch.vertex_ids)
        self.n_particles = self.n_elements + self.n_vertices
        self.grid_shape = (self.n_grid,) * 3

        self.faces_t = torch.as_tensor(patch.faces, dtype=torch.long, device=self.device)
        self.boundary_t = torch.as_tensor(patch.boundary_vertex_ids, dtype=torch.long, device=self.device)
        self.boundary_ids_wp = wp.from_numpy(np.asarray(patch.boundary_vertex_ids, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.reference_t = torch.as_tensor(
            (np.asarray(patch.reference_vertices_m, dtype=np.float64) - self.origin_m) * self.scale,
            dtype=torch.float32,
            device=self.device,
        )
        # Rest frame 11 remains global and fixed across all observation windows.
        self.rest_sim = np.asarray(np.asarray(patch.rest_vertices_m) * self.scale, dtype=np.float32)
        initial_vertices = self.reference_t[0]
        initial_velocities = (self.reference_t[1] - initial_vertices) * 25.0
        directions, volume = self._directions_and_volume(initial_vertices)
        self.initial_directions = directions
        self.initial_particles = torch.cat((initial_vertices[self.faces_t].mean(dim=1), initial_vertices), dim=0)
        self.initial_velocities = torch.cat((initial_velocities[self.faces_t].mean(dim=1), initial_velocities), dim=0)
        self.initial_volume = volume
        boundary_mask = np.zeros(self.n_vertices, dtype=bool)
        boundary_mask[patch.boundary_vertex_ids] = True
        self.boundary_face_ids = np.flatnonzero(np.all(boundary_mask[patch.faces], axis=1))
        self.boundary_faces_t = torch.as_tensor(self.boundary_face_ids, dtype=torch.long, device=self.device)
        mover_ids = np.concatenate((patch.boundary_vertex_ids + self.n_elements, self.boundary_face_ids)).astype(np.int32)
        self.mover_ids_wp = wp.from_numpy(mover_ids, dtype=wp.int32, device=self.device)
        self.mover_count = len(mover_ids)
        self.scored_ids_wp = wp.from_numpy(np.asarray(patch.scored_vertex_ids, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.leakage_hits_wp = wp.zeros(len(patch.scored_vertex_ids), dtype=wp.int32, device=self.device)
        self.mover_weights = wp.zeros(self.grid_shape, dtype=float, device=self.device)
        self.mover_momentum = wp.zeros(self.grid_shape, dtype=wp.vec3, device=self.device)
        self.state = structures.MPMStateStruct()
        self.state.init(self.n_particles, self.n_elements, self.n_vertices, device=self.device, requires_grad=False)
        traditional = np.zeros(self.n_particles, dtype=np.int32)
        vertices = np.zeros(self.n_particles, dtype=np.int32)
        vertices[self.n_elements:] = 1
        elements = np.zeros(self.n_particles, dtype=np.int32)
        elements[:self.n_elements] = 1
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self.state.from_torch(
                self.initial_particles,
                self.initial_volume,
                torch.linalg.inv(directions),
                self._rest_inverse(1.0),
                self.faces_t,
                traditional,
                vertices,
                elements,
                tensor_velocity=self.initial_velocities,
                n_grid=self.n_grid,
                grid_lim=self.grid_lim,
                device=self.device,
                requires_grad=False,
            )
        self.model = structures.MPMModelStruct()
        self.model.init(self.n_particles, device=self.device, requires_grad=False)
        self.model.init_other_params(n_grid=self.n_grid, grid_lim=self.grid_lim, device=self.device)
        self.model.n_particles = self.n_particles
        assert np.isclose(self.model.dx, self.cell_dx_sim), "Domain must preserve requested physical cell size"
        # Omitting mesh arguments and collider registration explicitly disables body contact.
        self.solver = solver_module.MPMWARP(
            self.n_particles, self.n_elements, self.n_vertices,
            n_grid=self.n_grid, grid_lim=self.grid_lim, device=self.device,
        )
        assert len(self.solver.mesh_colliders) == 0

    def _directions_and_volume(self, vertices: Any) -> tuple[Any, Any]:
        directions, volume = cloth_directions_and_volume(vertices.detach().cpu().numpy(), self.patch.faces)
        return (
            self.torch.as_tensor(directions, dtype=vertices.dtype, device=self.device),
            self.torch.as_tensor(volume, dtype=vertices.dtype, device=self.device),
        )

    def _rest_inverse(self, H: float) -> Any:
        inverse = rest_direction_inverse(self.rest_sim, self.patch.faces, H)
        return self.torch.as_tensor(inverse, dtype=self.torch.float32, device=self.device)

    def _assert_safe_state(self) -> None:
        """Fail before unchecked transfers can index outside the fixed grid."""
        torch = self.torch
        for name in ("particle_x", "particle_v", "particle_C", "particle_d"):
            values = self.wp.to_torch(getattr(self.state, name))
            assert bool(torch.isfinite(values).all().item()), f"Nonfinite {name} in patch {self.patch.name}"
        positions = self.wp.to_torch(self.state.particle_x)
        lower = 2.0 * self.model.dx
        upper = self.grid_lim - 3.0 * self.model.dx
        assert bool(((positions >= lower) & (positions <= upper)).all().item()), (
            f"Patch {self.patch.name} left the stencil-safe local domain at t={self.solver.time:.6f}s; "
            f"simulation min={positions.min(dim=0).values.detach().cpu().tolist()}, "
            f"simulation max={positions.max(dim=0).values.detach().cpu().tolist()}, "
            f"safe interval=[{lower}, {upper}], "
            f"maximum speed={self.torch.linalg.vector_norm(self.wp.to_torch(self.state.particle_v), dim=1).max().item() / self.scale:.6g} m/s; "
            "check timestep stability or increase explicit domain padding, without clipping positions"
        )

    def _step(self, boundary_positions: Any, vertex_velocity: Any) -> float:
        """Use native constitutive/transfers and exact prescribed collar motion."""
        wp = self.wp
        state, model = self.state, self.model
        self._assert_safe_state()
        wp.launch(self.native.zero_grid, dim=self.grid_shape, inputs=[state, model], device=self.device)
        wp.launch(self.native.set_vec3_to_zero, dim=self.n_vertices, inputs=[state.vertex_force], device=self.device)
        wp.launch(self.native.compute_stress_from_F_trial, dim=self.n_elements, inputs=[state, model, self.dt], device=self.device)
        wp.launch(self.native.p2g_apic_with_stress, dim=self.n_particles, inputs=[state, model, self.dt, self.n_elements], device=self.device)
        wp.launch(self.native.grid_normalization_and_gravity, dim=self.grid_shape, inputs=[state, model, self.dt], device=self.device)
        if self.damping < 1.0:
            wp.launch(self.native.add_damping_via_grid, dim=self.grid_shape, inputs=[state, self.damping], device=self.device)
        boundary_velocity = vertex_velocity[self.boundary_t].contiguous()
        face_velocity = vertex_velocity[self.faces_t[self.boundary_faces_t]].mean(dim=1)
        mover_velocity = self.torch.cat((boundary_velocity, face_velocity), dim=0).contiguous()
        mover_velocity_wp = wp.from_torch(mover_velocity, dtype=wp.vec3, requires_grad=False)
        self.mover_weights.zero_()
        self.mover_momentum.zero_()
        wp.launch(self.kernels["scatter_boundary"], dim=self.mover_count,
                  inputs=[state, model, self.mover_ids_wp, mover_velocity_wp, self.mover_weights, self.mover_momentum], device=self.device)
        wp.launch(self.kernels["boundary_grid"], dim=self.grid_shape,
                  inputs=[state, self.mover_weights, self.mover_momentum], device=self.device)
        # Test the exact G2P support of each scored particle against both mover kinds.
        wp.launch(self.kernels["detect_leakage"], dim=len(self.patch.scored_vertex_ids),
                  inputs=[state, model, self.scored_ids_wp, self.mover_weights, self.leakage_hits_wp, self.n_elements], device=self.device)
        leakage_fraction = float(self.torch.count_nonzero(wp.to_torch(self.leakage_hits_wp)).item() / len(self.patch.scored_vertex_ids))
        assert leakage_fraction == 0.0, (
            f"Patch {self.patch.name} has prescribed-grid support at {leakage_fraction:.1%} of scored particles "
            f"at t={self.solver.time:.6f}s; revise the fixed scored core or collar before fitting"
        )
        wp.launch(self.kernels["vertex_g2p"], dim=self.n_vertices, inputs=[state, model, self.dt, self.n_elements], device=self.device)
        wp.launch(self.native.g2p_e, dim=self.n_elements, inputs=[state, model, self.dt, self.n_elements], device=self.device)
        positions_wp = wp.from_torch(boundary_positions.contiguous(), dtype=wp.vec3, requires_grad=False)
        velocities_wp = wp.from_torch(boundary_velocity, dtype=wp.vec3, requires_grad=False)
        wp.launch(self.kernels["boundary_particles"], dim=len(self.patch.boundary_vertex_ids),
                  inputs=[state, self.boundary_ids_wp, positions_wp, velocities_wp, self.n_elements], device=self.device)
        # Native g2p_e ran before collar projection; refresh edges and centers now.
        wp.launch(self.kernels["refresh_elements"], dim=self.n_elements, inputs=[state, self.n_elements], device=self.device)
        self.solver.time += self.dt
        self._assert_safe_state()
        prescribed = wp.to_torch(state.particle_x)[self.n_elements:][self.boundary_t]
        assert self.torch.equal(prescribed, boundary_positions), "Prescribed collar position drifted"
        return leakage_fraction

    def rollout(self, D: float, E: float, H: float) -> PatchRollout:
        """Reset all state and roll consecutive reference frames at 25 Hz.

        E is the actual exported Young modulus, with no trainer *100 multiplier.
        """
        assert np.isfinite((D, E, H)).all() and D > 0.0 and E > 0.0 and H > 0.0
        torch, wp = self.torch, self.wp
        with torch.no_grad(), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self.solver.time = 0.0
            self.solver.time_profile.clear()
            self.state.reset_state(
                self.n_vertices,
                self.initial_particles.clone(),
                self.initial_directions,
                tensor_velocity=self.initial_velocities,
                tensor_R_inv=self._rest_inverse(float(H)),
                device=self.device,
                requires_grad=False,
            )
            self.state.grid_m.zero_()
            self.state.grid_v_in.zero_()
            self.state.grid_v_out.zero_()
            self.state.particle_selection.zero_()
            self.solver.set_parameters_dict(self.model, self.state, {
                "material": "cloth", "g": self.gravity, "density": float(D),
                "grid_v_damping_scale": self.damping, "friction_angle": self.friction_angle,
            }, device=self.device)
            self.solver.set_E_nu(self.model, float(E), self.nu, self.gamma, self.kappa, device=self.device)
            self.solver.prepare_mu_lam(self.model, self.state, device=self.device)
            self._assert_safe_state()
            predicted = [wp.to_torch(self.state.particle_x)[self.n_elements:].detach().cpu().numpy().copy()]
            per_frame_leakage = np.zeros(len(self.patch.frame_ids), dtype=np.float64)
            for frame in range(1, len(self.patch.frame_ids)):
                start = self.reference_t[frame - 1]
                end = self.reference_t[frame]
                vertex_velocity = ((end - start) * 25.0).contiguous()
                for step in range(1, self.substeps + 1):
                    fraction = step / self.substeps
                    target = (start[self.boundary_t] + fraction * (end[self.boundary_t] - start[self.boundary_t])).contiguous()
                    leakage = self._step(target, vertex_velocity)
                    per_frame_leakage[frame] = max(per_frame_leakage[frame], leakage)
                predicted.append(wp.to_torch(self.state.particle_x)[self.n_elements:].detach().cpu().numpy().copy())
        vertices_m = np.asarray(predicted, dtype=np.float64) / self.scale + self.origin_m
        return replace(score_rollout(self.patch, vertices_m),
                       grid_leakage_fraction=float(per_frame_leakage.max()),
                       per_frame_grid_leakage_fraction=per_frame_leakage)
