"""Cloth-only MPM: frame checkpoints and a reusable one-substep adjoint.

The material vector is (D, E / 100, H), matching the native optimizer.
Body motion is prescribed; attachment interpolation remains differentiable
with respect to the evolving cloth positions.
"""

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp
from warp_mpm.precision import scalar, vec2, vec3, mat22, mat33, torch_dtype

from warp_mpm.mpm_solver import (
    MPMWARP, MPMModelStruct, MPMStateStruct, MPMSmallStateStruct, SimulationMesh,
    Mesh_collider, Particle_mover, anisotropy_return_mapping,
    kirchoff_stress_Anisotropy, p2g_apic_with_stress,
    grid_normalization_and_gravity, g2p_v_differentiable, g2p_e_differentiable,
)

STATE_KEYS = ("particle_x", "particle_v", "particle_C", "particle_d")
STATE_DTYPES = (vec3, vec3, mat33, mat33)
ClothState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class FrameInputs:
    mesh_x: torch.Tensor
    mesh_v: torch.Tensor
    joint_verts_v: torch.Tensor
    joint_faces_v: torch.Tensor


@wp.kernel
def prepare_material(
    parameters: wp.array(dtype=scalar), rest_vertices: wp.array(dtype=vec3),
    state: MPMStateStruct, model: MPMModelStruct, n_elements: int,
) -> None:
    p = wp.tid()
    density = parameters[0]
    youngs = scalar(100.0) * parameters[1]
    nu = model.nu[p]
    model.E[p] = youngs
    model.mu[p] = youngs / (scalar(2.0) * (scalar(1.0) + nu))
    model.lam[p] = youngs * nu / ((scalar(1.0) + nu) * (scalar(1.0) - scalar(2.0) * nu))
    state.particle_mass[p] = density * state.particle_vol[p]
    if p < n_elements:
        face = state.faces[p]
        a = rest_vertices[int(face[0])]
        b = rest_vertices[int(face[1])]
        c = rest_vertices[int(face[2])]
        height = parameters[2]
        e1 = vec3(b[0] - a[0], height * (b[1] - a[1]), b[2] - a[2])
        e2 = vec3(c[0] - a[0], height * (c[1] - a[1]), c[2] - a[2])
        r11 = wp.length(e1)
        r12 = wp.dot(e1, e2) / r11
        r22 = wp.length(e2 - (r12 / r11) * e1)
        state.particle_R_inv[p] = vec3(scalar(1.0) / r11, -r12 / (r11 * r22), scalar(1.0) / r22)


@wp.kernel
def cloth_stress(current: MPMSmallStateStruct, work: MPMStateStruct, model: MPMModelStruct) -> None:
    p = wp.tid()
    mapped = anisotropy_return_mapping(current.particle_d[p], model, p)
    work.particle_d[p] = mapped
    work.particle_stress[p] = kirchoff_stress_Anisotropy(
        work.particle_R_inv[p], mapped, work.faces[p], work.particle_vol[p],
        work, model.mu[p], model.lam[p], model.gamma[p], model.kappa[p],
    )


@wp.kernel
def body_collision(
    velocity: wp.array3d(dtype=vec3), result: wp.array3d(dtype=vec3),
    param: Mesh_collider,
) -> None:
    i, j, k = wp.tid()
    v = velocity[i, j, k]
    if param.weight[i, j, k] > scalar(1e-15):
        relative = v - param.mesh_v_out[i, j, k]
        n = wp.normalize(param.mesh_normal[i, j, k])
        normal = wp.dot(relative, n)
        projected = relative - wp.min(normal, scalar(0.0)) * n
        if normal < scalar(0.0) and wp.length(projected) > scalar(1e-20):
            friction = wp.max(scalar(0.0), wp.length(projected) + normal * param.friction) * wp.normalize(projected)
        else:
            friction = projected
        result[i, j, k] = friction + param.mesh_v_out[i, j, k]
    else:
        result[i, j, k] = v


@wp.kernel
def attachment_velocity(
    velocity: wp.array3d(dtype=vec3), result: wp.array3d(dtype=vec3),
    param: Particle_mover,
) -> None:
    i, j, k = wp.tid()
    if param.weight[i, j, k] > scalar(1e-15):
        result[i, j, k] = param.velocity[i, j, k] / param.weight[i, j, k]
    else:
        result[i, j, k] = velocity[i, j, k]


@wp.kernel
def damp_velocity(
    velocity: wp.array3d(dtype=vec3), result: wp.array3d(dtype=vec3), scale: scalar,
) -> None:
    i, j, k = wp.tid()
    v = velocity[i, j, k]
    result[i, j, k] = v - (scalar(1.0) - scale) * v


class DifferentiableMPM:
    """Own private buffers so replay does not mutate the native simulator."""

    def __init__(
        self, solver: MPMWARP, model: MPMModelStruct, template: MPMStateStruct,
        rest_vertices: torch.Tensor, substeps: int, frame_dt: float = 1.0 / 25.0,
        verify_replay: bool = False,
    ) -> None:
        assert substeps > 0 and frame_dt > 0
        assert solver.n_particles == solver.n_elements + solver.n_vertices, "Only cloth particles are supported"
        assert model.material == 7 and not model.update_cov_with_F
        assert not solver.pre_p2g_operations and not solver.particle_velocity_modifiers
        assert not solver.grid_postprocess, "Additional boundary kernels need an explicit differentiable implementation"
        assert not template.particle_selection.numpy().any(), "Frozen particles need an explicit state-copy kernel"
        assert rest_vertices.shape == (solver.n_vertices, 3)
        self.solver = solver
        self.device = template.particle_x.device
        self.substeps = substeps
        self.dt = frame_dt / substeps
        self.grid = (model.n_grid,) * 3
        self.parameters = wp.zeros(3, dtype=scalar, device=self.device, requires_grad=True)
        self.rest_vertices = wp.from_torch(rest_vertices.detach().contiguous(), dtype=vec3, requires_grad=False)
        self.model = MPMModelStruct()
        for key in MPMModelStruct.vars:
            setattr(self.model, key, getattr(model, key))
        for key in ("E", "mu", "lam"):
            setattr(self.model, key, wp.zeros(solver.n_particles, dtype=scalar, device=self.device, requires_grad=True))
        self.current = MPMSmallStateStruct()
        self.result = MPMSmallStateStruct()
        for small in (self.current, self.result):
            small.init(solver.n_particles, solver.n_elements, solver.n_vertices, device=self.device, requires_grad=True)
        self.work = MPMStateStruct()
        for key in MPMStateStruct.vars:
            setattr(self.work, key, getattr(template, key))
        for key in ("particle_x", "particle_v", "particle_C"):
            setattr(self.work, key, getattr(self.current, key))
        for key, count, dtype in (
            ("particle_mass", solver.n_particles, scalar),
            ("particle_R_inv", solver.n_elements, vec3),
            ("particle_d", solver.n_elements, mat33),
            ("particle_stress", solver.n_elements, mat33),
            ("vertex_force", solver.n_vertices, vec3),
        ):
            setattr(self.work, key, wp.zeros(count, dtype=dtype, device=self.device, requires_grad=True))
        self.work.grid_m = wp.zeros(self.grid, dtype=scalar, device=self.device, requires_grad=True)
        self.work.grid_v_in = wp.zeros(self.grid, dtype=vec3, device=self.device, requires_grad=True)
        self.velocities = [wp.zeros(self.grid, dtype=vec3, device=self.device, requires_grad=True)
                           for _ in range(1 + int(model.grid_v_damping_scale < 1.0) + len(solver.mesh_colliders)
                                          + int(solver.num_joint_v + solver.num_joint_f > 0))]
        self.work.grid_v_out = self.velocities[0]
        self.transport = MPMStateStruct()
        for key in MPMStateStruct.vars:
            setattr(self.transport, key, getattr(self.work, key))
        self.transport.grid_v_out = self.velocities[-1]
        self.mesh = SimulationMesh()
        self.mesh.points = wp.clone(solver.mesh.points)
        self.mesh.velocities = wp.clone(solver.mesh.velocities)
        self.mesh.indices = solver.mesh.indices
        self.colliders: list[Mesh_collider] = []
        for native in solver.mesh_collider_params:
            collider = Mesh_collider()
            assert native.mesh.points.ptr == solver.mesh.points.ptr, "Collider must use the prescribed body mesh"
            collider.mesh, collider.friction = self.mesh, native.friction
            for key, dtype in (("weight", scalar), ("mesh_v_in", vec3),
                               ("mesh_v_out", vec3), ("mesh_normal", vec3)):
                setattr(collider, key, wp.zeros(self.grid, dtype=dtype, device=self.device))
            self.colliders.append(collider)
        self.mover = Particle_mover()
        self.mover.weight = wp.zeros(self.grid, dtype=scalar, device=self.device, requires_grad=True)
        self.mover.velocity = wp.zeros(self.grid, dtype=vec3, device=self.device, requires_grad=True)
        self.joint_vertices = wp.zeros(solver.num_joint_v, dtype=vec3, device=self.device)
        self.joint_faces = wp.zeros(solver.num_joint_f, dtype=vec3, device=self.device)
        self.verify_replay = verify_replay
        self.replay_error = 0.0
        self.replay_errors = dict.fromkeys(STATE_KEYS, 0.0)
        self.replay_relative_errors = dict.fromkeys(STATE_KEYS, 0.0)
        wp.synchronize_device(self.device)

    def scope(self) -> wp.ScopedStream:
        if self.device.is_cuda:
            return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(wp.device_to_torch(self.device))))
        return wp.ScopedStream(None)

    def load(self, state: ClothState, parameters: torch.Tensor) -> None:
        assert parameters.shape == (3,) and parameters.dtype == torch_dtype
        for key, dtype, tensor in zip(STATE_KEYS, STATE_DTYPES, state):
            wp.copy(getattr(self.current, key), wp.from_torch(tensor.detach().contiguous(), dtype=dtype, requires_grad=False))
        wp.copy(self.parameters, wp.from_torch(parameters.detach().contiguous(), requires_grad=False))

    def snapshot(self, state: MPMSmallStateStruct) -> ClothState:
        return tuple(wp.to_torch(getattr(state, key)).detach().clone() for key in STATE_KEYS)

    def driving(self, frame: FrameInputs, substep: int) -> None:
        """Prescribed body buffers are regenerated outside the local tape."""
        points = frame.mesh_x + (substep * self.dt) * frame.mesh_v
        wp.copy(self.mesh.points, wp.from_torch(points.contiguous(), dtype=vec3, requires_grad=False))
        wp.copy(self.mesh.velocities, wp.from_torch(frame.mesh_v.contiguous(), dtype=vec3, requires_grad=False))
        for kernels, collider in zip(self.solver.mesh_colliders, self.colliders):
            wp.launch(kernels[0], self.grid, inputs=[collider], device=self.device)
            wp.launch(kernels[1], self.solver.num_mesh_f, inputs=[collider, self.model], device=self.device)
            wp.launch(kernels[2], self.grid, inputs=[collider], device=self.device)
        if self.solver.num_joint_v:
            wp.copy(self.joint_vertices, wp.from_torch(frame.joint_verts_v.contiguous(), dtype=vec3, requires_grad=False))
        if self.solver.num_joint_f:
            wp.copy(self.joint_faces, wp.from_torch(frame.joint_faces_v.contiguous(), dtype=vec3, requires_grad=False))

    def clear(self) -> None:
        self.work.grid_m.zero_()
        self.work.grid_v_in.zero_()
        self.velocities[0].zero_()
        self.work.vertex_force.zero_()
        self.mover.weight.zero_()
        self.mover.velocity.zero_()

    def step(self) -> None:
        """No nonlinear array overwrite on the local differentiation tape."""
        solver, model, work = self.solver, self.model, self.work
        wp.launch(prepare_material, solver.n_particles,
                  inputs=[self.parameters, self.rest_vertices, work, model, solver.n_elements], device=self.device)
        wp.launch(cloth_stress, solver.n_elements, inputs=[self.current, work, model], device=self.device)
        wp.launch(p2g_apic_with_stress, solver.n_particles,
                  inputs=[work, model, self.dt, solver.n_elements], device=self.device)
        index = 0
        wp.launch(grid_normalization_and_gravity, self.grid, inputs=[work, model, self.dt], device=self.device)
        if model.grid_v_damping_scale < 1.0:
            wp.launch(damp_velocity, self.grid,
                      inputs=[self.velocities[index], self.velocities[index + 1], model.grid_v_damping_scale], device=self.device)
            index += 1
        for collider in self.colliders:
            wp.launch(body_collision, self.grid,
                      inputs=[self.velocities[index], self.velocities[index + 1], collider], device=self.device)
            index += 1
        if solver.num_joint_v + solver.num_joint_f:
            assert len(solver.particle_movers) == 1
            kernels = solver.particle_movers[0]
            if solver.num_joint_v:
                wp.launch(kernels[2], solver.num_joint_v,
                          inputs=[self.joint_vertices, self.mover, work, model, solver.n_elements], device=self.device)
            if solver.num_joint_f:
                wp.launch(kernels[3], solver.num_joint_f,
                          inputs=[self.joint_faces, self.mover, work, model], device=self.device)
            wp.launch(attachment_velocity, self.grid,
                      inputs=[self.velocities[index], self.velocities[index + 1], self.mover], device=self.device)
            index += 1
        wp.launch(g2p_v_differentiable, solver.n_vertices,
                  inputs=[self.transport, self.result, model, self.dt, solver.n_elements], device=self.device)
        wp.launch(g2p_e_differentiable, solver.n_elements,
                  inputs=[self.transport, self.result, model, self.dt, solver.n_elements], device=self.device)

    def forward_frame(self, state: ClothState, parameters: torch.Tensor, frame: FrameInputs) -> ClothState:
        self.load(state, parameters)
        for substep in range(self.substeps):
            self.driving(frame, substep)
            self.clear()
            self.step()
            for key in STATE_KEYS:
                wp.copy(getattr(self.current, key), getattr(self.result, key))
        return self.snapshot(self.current)

    def backward_frame(
        self, state: ClothState, parameters: torch.Tensor, frame: FrameInputs,
        gradients: ClothState,
    ) -> tuple[ClothState, torch.Tensor]:
        checkpoints = [state]
        for substep in range(self.substeps):
            self.load(checkpoints[-1], parameters)
            self.driving(frame, substep)
            self.clear()
            self.step()
            checkpoints.append(self.snapshot(self.result))
        parameter_gradient = torch.zeros_like(parameters)
        for substep in reversed(range(self.substeps)):
            self.load(checkpoints[substep], parameters)
            self.driving(frame, substep)
            self.clear()
            tape = wp.Tape()
            with tape:
                self.step()
            if self.verify_replay:
                # CUDA atomic accumulation order differs between runs, so compare
                # relative to each field's magnitude (CPU replay is bitwise). Replay
                # restarts at each frame checkpoint; on 4D-DRESS 190 the divergence
                # reached ~1e-5 relative (particle_C) within 12 frames.
                for key, a, b in zip(STATE_KEYS, self.snapshot(self.result), checkpoints[substep + 1]):
                    error = float((a - b).abs().max())
                    relative = error / max(float(b.abs().max()), 1e-30)
                    self.replay_errors[key] = max(self.replay_errors[key], error)
                    self.replay_relative_errors[key] = max(self.replay_relative_errors[key], relative)
                    assert relative <= 1e-4, f"Substep replay diverged: {key} {error} ({relative} relative)"
                self.replay_error = max(self.replay_errors.values())
            seeds = {getattr(self.result, key): wp.from_torch(value.contiguous(), dtype=dtype, requires_grad=False)
                     for key, dtype, value in zip(STATE_KEYS, STATE_DTYPES, gradients)}
            tape.backward(grads=seeds)
            gradients = tuple(wp.to_torch(getattr(self.current, key).grad).clone() for key in STATE_KEYS)
            parameter_gradient.add_(wp.to_torch(self.parameters.grad))
            tape.zero()
            checkpoints.pop()
        return gradients, parameter_gradient

    def frame(self, state: ClothState, parameters: torch.Tensor, frame: FrameInputs) -> ClothState:
        return _CheckpointFrame.apply(*state, parameters, self, frame)


class _CheckpointFrame(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, v: torch.Tensor, affine: torch.Tensor,
        direction: torch.Tensor, parameters: torch.Tensor,
        engine: DifferentiableMPM, frame: FrameInputs,
    ) -> ClothState:
        ctx.save_for_backward(x, v, affine, direction, parameters)
        ctx.engine, ctx.frame = engine, frame
        ctx.set_materialize_grads(True)
        with engine.scope():
            return engine.forward_frame((x, v, affine, direction), parameters, frame)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[Any, ...]:
        x, v, affine, direction, parameters = ctx.saved_tensors
        with ctx.engine.scope():
            state_gradients, parameter_gradient = ctx.engine.backward_frame(
                (x, v, affine, direction), parameters, ctx.frame, gradients,
            )
        return (*state_gradients, parameter_gradient, None, None)
