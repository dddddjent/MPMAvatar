"""Native-forward and central-FD checks for checkpointed cloth MPM (CPU or CUDA)."""

import argparse
import json
from pathlib import Path

import torch
import warp as wp

# Template command (activate mpmavatar; run from MPMAvatar):
# CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python check_differentiable.py --substeps 1 8 40 400 --frames 2 --heights 0.94 1.0 --epsilon 0.1 0.05 0.03 0.01 0.003 0.001 --rtol 0.03 --atol 0.00000001 --full-tape-check --float64-reference --device cpu --output output/differentiable_cpu/report.json
# GPU (the float64 reference is CPU-only): python check_differentiable.py --substeps 1 8 40 400 --frames 2 --heights 0.94 1.0 --epsilon 0.1 0.05 0.03 0.01 0.003 0.001 --rtol 0.03 --atol 0.00000001 --full-tape-check --device cuda:0 --output output/differentiable_gpu/report.json

wp.config.kernel_cache_dir = "/tmp/mpmavatar_differentiable_cache"

from warp_mpm.differentiable import DifferentiableMPM, FrameInputs, ClothState, STATE_KEYS
from warp_mpm.mpm_solver import MPMWARP, MPMModelStruct, MPMStateStruct, in_plane_rotation

# Set by main(); torch factories use the same default device.
DEVICE = "cpu"


@wp.kernel
def rotation_probe(
    values: wp.array(dtype=wp.vec3), analytical: wp.array(dtype=wp.mat22),
    svd: wp.array(dtype=wp.mat22),
) -> None:
    p = wp.tid()
    a = values[p]
    analytical[p] = in_plane_rotation(a[0], a[1], a[2])
    f = wp.mat33(a[0], a[1], 0.0, 0.0, a[2], 0.0, 0.0, 0.0, 0.0)
    u = wp.mat33(0.0)
    v = wp.mat33(0.0)
    sigma = wp.vec3(0.0)
    wp.svd3(f, u, sigma, v)
    r = u * wp.transpose(v)
    svd[p] = wp.mat22(r[0, 0], r[0, 1], r[1, 0], r[1, 1])


def check_rotation() -> dict[str, object]:
    values = torch.tensor([[1., 0., 1.], [1.3, .2, .7], [.5, -.4, 1.6], [1., 1e-6, 1.], [2., 3., .1]])
    source = wp.from_torch(values.clone(), dtype=wp.vec3, requires_grad=True)
    analytical = wp.zeros(len(values), dtype=wp.mat22, device=DEVICE, requires_grad=True)
    svd = wp.zeros(len(values), dtype=wp.mat22, device=DEVICE)
    seed = torch.tensor([[.3, .7], [-.2, .4]]).expand(len(values), -1, -1).contiguous()
    tape = wp.Tape()
    with tape:
        wp.launch(rotation_probe, len(values), inputs=[source, analytical, svd], device=DEVICE)
    forward_error = float((wp.to_torch(analytical).detach() - wp.to_torch(svd)).abs().max())
    tape.backward(grads={analytical: wp.from_torch(seed, dtype=wp.mat22, requires_grad=False)})
    ad = wp.to_torch(source.grad).clone()
    errors = []
    for column in range(3):
        minus, plus = values.clone(), values.clone()
        minus[:, column] -= .001
        plus[:, column] += .001
        objectives = []
        for inputs in (minus, plus):
            wp.launch(rotation_probe, len(values),
                      inputs=[wp.from_torch(inputs, dtype=wp.vec3, requires_grad=False), analytical, svd], device=DEVICE)
            objectives.append((wp.to_torch(analytical).detach().double() * seed).sum((1, 2)))
        fd = (objectives[1] - objectives[0]) / (plus[:, column].double() - minus[:, column].double())
        errors.append(float((fd - ad[:, column]).abs().max()))
    result = {"svd_forward_max_error": forward_error, "ad_fd_max_absolute_error": max(errors),
              "passed": forward_error < 2e-6 and max(errors) < 1e-4}
    assert result["passed"], result
    return result


def check_state_vjp(yield_surface: bool = False) -> dict[str, object]:
    engine, state, frame, vertices = fixture(True, True, 8)
    parameters = torch.tensor([2., .8, .94])
    # Check a late state as well as the initial state. A complete state adjoint
    # is needed for checkpoint chaining even when the objective uses only x.
    for _ in range(30):
        state = engine.forward_frame(state, parameters, frame)
    d1, d2, d3 = state[3].double().unbind(dim=2)
    normal = torch.linalg.cross(d1, d2)
    normal = normal / normal.norm(dim=1, keepdim=True)
    r33 = (d3 * normal).sum(1)
    shear = (d3 - r33[:, None] * normal).norm(dim=1)
    margin = engine.model.friction_coeff * (1. - r33).square() - shear
    if not yield_surface:
        # Move away from the plastic yield surface for a smooth central-FD test.
        # The separate yield-surface diagnostic retains the actual late state.
        state = (*state[:3], state[3].clone())
        state[3][:, :, 2].mul_(.9)
    leaves = tuple(value.detach().clone().requires_grad_() for value in state)
    predicted = engine.frame(leaves, parameters, frame)
    objective = loss(predicted, vertices, False)
    objective.backward()
    rows = []
    for field, value in enumerate(leaves):
        assert value.grad is not None
        direction = torch.linspace(-.5, .7, value.numel()).reshape(value.shape)
        scale = .001 if field in (0, 3) else .01
        direction = direction * scale
        ad = float((value.grad * direction).sum())
        derivatives = []
        for epsilon in (.1, .03, .01):
            states = []
            for sign in (-1., 1.):
                varied = tuple(x.detach().clone() for x in leaves)
                varied[field].add_(sign * epsilon * direction)
                result = engine.forward_frame(varied, parameters, frame)
                states.append(float(loss(result, vertices, False)))
            fd = (states[1] - states[0]) / (2. * epsilon)
            error = abs(fd - ad)
            derivatives.append({"epsilon": epsilon, "fd": fd, "error": error,
                                "passed": error <= 1e-8 + .03 * max(abs(fd), abs(ad))})
        rows.append({"field": STATE_KEYS[field], "ad": ad, "fd_sweep": derivatives,
                     "passed": sum(bool(row["passed"]) for row in derivatives) >= 2})
    result = {"late_state_substeps": 240, "yield_surface": yield_surface,
              "original_friction_margins": margin.tolist(), "gradients": rows,
              "passed": all(bool(row["passed"]) for row in rows)}
    print(json.dumps({"state_vjp": result}), flush=True)
    return result


def check_full_tape() -> dict[str, object]:
    engine, initial, frame, vertices = fixture(False, False, 400)
    parameters = torch.tensor([2., .8, .94], requires_grad=True)
    state = initial
    for index in range(2):
        inputs = FrameInputs(frame.mesh_x + index * .04 * frame.mesh_v, frame.mesh_v,
                             frame.joint_verts_v, frame.joint_faces_v)
        state = engine.frame(state, parameters, inputs)
    loss(state, vertices, True).backward()
    assert parameters.grad is not None
    checkpoint_gradient = parameters.grad.detach().clone()
    shared_parameters = wp.from_torch(parameters.detach().clone(), requires_grad=True)
    tape = wp.Tape()
    stages = []
    for substep in range(800):
        stage = DifferentiableMPM(engine.solver, engine.model, engine.work, vertices, 1, frame_dt=1e-4)
        stage.driving(frame, substep)
        stage.clear()
        if substep == 0:
            stage.load(initial, parameters.detach())
        with tape:
            wp.copy(stage.parameters, shared_parameters)
            if substep > 0:
                for key in STATE_KEYS:
                    wp.copy(getattr(stage.current, key), getattr(stages[-1].result, key))
            stage.step()
        stages.append(stage)
    leaves = tuple(value.requires_grad_() for value in stages[-1].snapshot(stages[-1].result))
    objective = loss(leaves, vertices, True) + sum(value.double().sum() * 0.0 for value in leaves)
    seeds = torch.autograd.grad(objective, leaves)
    tape.backward(grads={getattr(stages[-1].result, key): wp.from_torch(seed.contiguous(), dtype=dtype, requires_grad=False)
                         for key, dtype, seed in zip(STATE_KEYS, (wp.vec3, wp.vec3, wp.mat33, wp.mat33), seeds)})
    full_gradient = wp.to_torch(shared_parameters.grad).clone()
    error = float((full_gradient - checkpoint_gradient).abs().max())
    result = {"substeps": 800, "checkpointed_ad": checkpoint_gradient.tolist(), "full_tape_ad": full_gradient.tolist(),
              "max_absolute_error": error, "passed": bool(torch.allclose(full_gradient, checkpoint_gradient, rtol=1e-4, atol=1e-8))}
    print(json.dumps({"full_tape": result}), flush=True)
    return result


def rest_metric(vertices: torch.Tensor, faces: torch.Tensor, height: float) -> torch.Tensor:
    scaled = vertices * torch.tensor([1.0, height, 1.0])
    e1 = scaled[faces[:, 1]] - scaled[faces[:, 0]]
    e2 = scaled[faces[:, 2]] - scaled[faces[:, 0]]
    r11 = e1.norm(dim=1)
    r12 = (e1 * e2).sum(1) / r11
    r22 = (e2 - (r12 / r11)[:, None] * e1).norm(dim=1)
    return torch.stack((1.0 / r11, -r12 / (r11 * r22), 1.0 / r22), dim=1)


def fixture(contact: bool, attachments: bool, substeps: int, face_attachments: bool = False) -> tuple[DifferentiableMPM, ClothState, FrameInputs, torch.Tensor]:
    vertices = torch.tensor([[.82, .97, .97], [1.04, .99, 1.01],
                             [.86, 1.19, .98], [1.09, 1.18, 1.04]])
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64)
    n_elements, n_vertices = len(faces), len(vertices)
    positions = torch.cat((vertices[faces].mean(1), vertices))
    e1, e2 = vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    normal = torch.linalg.cross(e1, e2)
    area = .5 * normal.norm(dim=1)
    normal = normal / normal.norm(dim=1, keepdim=True)
    directions = torch.stack((e1, e2, .93 * normal + .012 * e1 / e1.norm(dim=1, keepdim=True)), dim=2)
    element_volumes = .25 * 1e-5 * area
    vertex_volumes = torch.zeros(n_vertices)
    vertex_volumes.index_add_(0, faces.flatten(), element_volumes[:, None].expand(-1, 3).reshape(-1))
    volumes = torch.cat((element_volumes, vertex_volumes))
    vertex_velocity = torch.tensor([[.12, -.08, .07], [.08, -.12, .05],
                                    [.10, -.06, -.03], [.14, -.09, .04]])
    velocities = torch.cat((vertex_velocity[faces].mean(1), vertex_velocity))
    state = MPMStateStruct()
    state.init(n_elements + n_vertices, n_elements, n_vertices, device=DEVICE)
    state.from_torch(positions, volumes, torch.linalg.inv(directions), rest_metric(vertices, faces, 1.0), faces,
                     torch.zeros(len(positions), dtype=torch.int32).cpu().numpy(),
                     torch.tensor([0] * n_elements + [1] * n_vertices, dtype=torch.int32).cpu().numpy(),
                     torch.tensor([1] * n_elements + [0] * n_vertices, dtype=torch.int32).cpu().numpy(),
                     tensor_velocity=velocities, n_grid=16, grid_lim=2.0, device=DEVICE, requires_grad=False)
    affine = .08 * torch.eye(3).expand(len(positions), -1, -1).clone()
    state.continue_from_torch(positions, tensor_velocity=velocities, tensor_d=directions,
                             tensor_C=affine, device=DEVICE, requires_grad=False)
    model = MPMModelStruct()
    model.init(len(positions), device=DEVICE)
    model.init_other_params(n_grid=16, grid_lim=2.0, device=DEVICE)
    body = torch.tensor([[.65, .95, .75], [1.30, .95, .75], [.65, .95, 1.35], [1.30, .95, 1.35]])
    body_faces = torch.tensor([[0, 2, 1], [1, 2, 3]], dtype=torch.int32)
    joints = 1 if attachments else 0
    solver = MPMWARP(len(positions), n_elements, n_vertices, n_grid=16, grid_lim=2.0,
                     mesh_vertices=body.cpu().numpy(), mesh_faces=body_faces.cpu().numpy(), num_joint_v=joints,
                     num_joint_f=int(face_attachments), device=DEVICE)
    solver.set_parameters_dict(model, state, {"material": "cloth", "density": 1., "g": [0., -9.8, 0.],
                                             "friction_angle": 25.0, "grid_v_damping_scale": 1.1}, device=DEVICE)
    solver.set_E_nu(model, E=80., nu=.3, gamma=500., kappa=500., device=DEVICE)
    if contact:
        solver.add_mesh_collider(solver.mesh.id, 16, friction=.35, device=DEVICE)
    if attachments:
        solver.add_particle_mover(16, device=DEVICE)
    engine = DifferentiableMPM(solver, model, state, vertices, substeps, frame_dt=substeps * 1e-4, verify_replay=True)
    initial = tuple(wp.to_torch(getattr(state, key)).detach().clone() for key in STATE_KEYS)
    frame = FrameInputs(body, torch.tensor([[.015, .01, 0.]]).expand_as(body).contiguous(),
                        torch.tensor([[.06, .015, .02]])[:joints].contiguous(),
                        torch.tensor([[.045, .01, .015]])[:int(face_attachments)].contiguous())
    return engine, initial, frame, vertices


def loss(state: ClothState, vertices: torch.Tensor, geometry_only: bool) -> torch.Tensor:
    x, v, affine, direction = state
    result = (x[-len(vertices):].double() - vertices.double() - torch.tensor([.008, -.012, .006])).square().mean()
    if not geometry_only:
        # Signed, nonuniform output seeds exercise every state adjoint without
        # cancellation of the equal-and-opposite internal forces in mean velocity.
        result = result + (v.double() * torch.linspace(-.6, .7, v.numel()).reshape(v.shape)).mean()
        result = result + .02 * (affine.double() * torch.linspace(-.7, .4, affine.numel()).reshape(affine.shape)).mean()
        d3 = direction[:, :, 2]
        result = result + .01 * (d3.double() * torch.linspace(-.3, .8, d3.numel()).reshape(d3.shape)).mean()
    return result


def native_rollout(
    engine: DifferentiableMPM, initial: ClothState, parameters: torch.Tensor,
    frame: FrameInputs, vertices: torch.Tensor, frames: int,
) -> ClothState:
    solver, model = engine.solver, engine.model
    state = MPMStateStruct()
    for key in MPMStateStruct.vars:
        setattr(state, key, getattr(engine.work, key))
    state.init(solver.n_particles, solver.n_elements, solver.n_vertices, device=DEVICE)
    # Use the native template's topology/volumes, and private mutable buffers.
    for key in ("particle_vol", "particle_D_inv", "faces", "particle_traditional",
                "particle_vertices", "particle_elements", "particle_selection"):
        setattr(state, key, getattr(engine.work, key))
    state.init_grid(16, device=DEVICE)
    state.continue_from_torch(initial[0].clone(), tensor_velocity=initial[1], tensor_C=initial[2], tensor_d=initial[3],
                             tensor_R_inv=rest_metric(vertices, torch.as_tensor(state.faces.numpy(), dtype=torch.int64),
                                                      float(parameters[2])), device=DEVICE, requires_grad=False)
    state.reset_density(torch.full((solver.n_particles,), float(parameters[0])), device=DEVICE, update_mass=True)
    solver.set_E_nu(model, E=float(parameters[1]) * 100., nu=.3, gamma=500., kappa=500., device=DEVICE)
    solver.prepare_mu_lam(model, state, device=DEVICE)
    solver.time = 0.0
    for index in range(frames):
        moved = frame.mesh_x + index * engine.substeps * engine.dt * frame.mesh_v
        for substep in range(engine.substeps):
            solver.p2g2p(model, state, engine.dt, mesh_x=moved + substep * engine.dt * frame.mesh_v,
                         mesh_v=frame.mesh_v, joint_verts_v=frame.joint_verts_v,
                         joint_faces_v=frame.joint_faces_v, device=DEVICE)
    return tuple(wp.to_torch(getattr(state, key)).detach().clone() for key in STATE_KEYS)


def run_case(
    contact: bool, attachments: bool, substeps: int, frames: int,
    epsilons: list[float], rtol: float, atol: float, height: float = .94, face_attachments: bool = False,
) -> dict[str, object]:
    engine, initial, frame, vertices = fixture(contact, attachments, substeps, face_attachments)
    parameters = torch.tensor([2., .8, height], requires_grad=True)
    geometry_only = substeps >= 40

    def rollout(values: torch.Tensor) -> ClothState:
        state = initial
        for index in range(frames):
            driving = FrameInputs(frame.mesh_x + index * engine.substeps * engine.dt * frame.mesh_v,
                                  frame.mesh_v, frame.joint_verts_v, frame.joint_faces_v)
            state = engine.frame(state, values, driving)
        return state

    predicted = rollout(parameters)
    objective = loss(predicted, vertices, geometry_only)
    objective.backward()
    assert parameters.grad is not None and torch.isfinite(parameters.grad).all()
    ad = parameters.grad.detach().clone()
    reference = native_rollout(engine, initial, parameters.detach(), frame, vertices, frames)
    forward_errors = {key: float((a.detach() - b).abs().max()) for key, a, b in zip(STATE_KEYS, predicted, reference)}
    forward_passed = all(error <= 1e-6 + 1e-4 * float(b.abs().max())
                         for error, b in zip(forward_errors.values(), reference))
    rows: list[dict[str, object]] = []
    passed = forward_passed
    for parameter, name in enumerate(("D", "E_internal", "H")):
        derivatives = []
        for epsilon in epsilons:
            offset = epsilon * float(parameters.detach()[parameter])
            minus, plus = parameters.detach().clone(), parameters.detach().clone()
            minus[parameter] -= offset
            plus[parameter] += offset
            # FD uses the native forward solver, independently of the new adjoint.
            low = loss(native_rollout(engine, initial, minus, frame, vertices, frames), vertices, geometry_only)
            high = loss(native_rollout(engine, initial, plus, frame, vertices, frames), vertices, geometry_only)
            fd = float((high - low) / (plus[parameter].double() - minus[parameter].double()))
            error = abs(fd - float(ad[parameter]))
            accepted = error <= atol + rtol * max(abs(fd), abs(float(ad[parameter])))
            derivatives.append({"relative_epsilon": epsilon, "fd": fd, "absolute_error": error,
                                "relative_error": error / max(abs(fd), abs(float(ad[parameter])), 1e-30),
                                "passed": accepted})
        accepted = sum(bool(row["passed"]) for row in derivatives) >= min(2, len(epsilons))
        rows.append({"parameter": name, "ad": float(ad[parameter]), "fd_sweep": derivatives, "passed": accepted})
        passed = passed and accepted
    result = {"contact": contact, "attachments": attachments, "face_attachments": face_attachments, "substeps_per_frame": substeps,
              "frames": frames, "geometry_only": geometry_only, "loss": float(objective.detach()),
              "initial_H": height,
              "native_forward_max_errors": forward_errors, "replay_max_error": engine.replay_error,
              "native_forward_passed": forward_passed,
              "gradients": rows, "passed": passed}
    print(json.dumps(result), flush=True)
    return result


def check_float64_reference(rtol: float, atol: float) -> list[dict[str, object]]:
    from differentiable_reference import reference_rollout

    results = []
    for substeps in (1, 8, 400):
        for height in (.94, 1.):
            engine, initial, frame, vertices = fixture(False, False, substeps)
            parameters = torch.tensor([2., .8, height], requires_grad=True)
            state = initial
            for _ in range(2):
                state = engine.frame(state, parameters, frame)
            geometry = substeps >= 40
            loss(state, vertices, geometry).backward()
            assert parameters.grad is not None
            warp_gradient = parameters.grad.detach().clone()
            reference_parameters = parameters.detach().double().requires_grad_()
            reference = reference_rollout(engine, initial, reference_parameters, vertices, 2 * substeps)
            loss(reference, vertices, geometry).backward()
            assert reference_parameters.grad is not None
            reference_gradient = reference_parameters.grad.detach().clone()
            rows = []
            for p, name in enumerate(("D", "E_internal", "H")):
                derivatives = []
                for epsilon in (.05, .01, .001, .0001):
                    low, high = reference_parameters.detach().clone(), reference_parameters.detach().clone()
                    low[p] -= epsilon * reference_parameters.detach()[p]
                    high[p] += epsilon * reference_parameters.detach()[p]
                    low_loss = loss(reference_rollout(engine, initial, low, vertices, 2 * substeps), vertices, geometry)
                    high_loss = loss(reference_rollout(engine, initial, high, vertices, 2 * substeps), vertices, geometry)
                    fd = float((high_loss - low_loss) / (high[p] - low[p]))
                    derivatives.append({"relative_epsilon": epsilon, "fd": fd,
                        "reference_ad_error": abs(fd - float(reference_gradient[p])),
                        "warp_ad_error": abs(fd - float(warp_gradient[p])),
                        "reference_passed": abs(fd - float(reference_gradient[p])) <= atol + rtol * max(abs(fd), abs(float(reference_gradient[p]))),
                        "warp_passed": abs(fd - float(warp_gradient[p])) <= atol + rtol * max(abs(fd), abs(float(warp_gradient[p])))})
                rows.append({"parameter": name, "warp_ad": float(warp_gradient[p]),
                    "reference_ad": float(reference_gradient[p]), "fd_sweep": derivatives,
                    "reference_passed": sum(bool(d["reference_passed"]) for d in derivatives) >= 2,
                    "warp_passed": sum(bool(d["warp_passed"]) for d in derivatives) >= 2})
            result = {"substeps_per_frame": substeps, "frames": 2, "H": height,
                "forward_max_errors": {key: float((a.detach() - b.detach()).abs().max())
                                       for key, a, b in zip(STATE_KEYS, state, reference)},
                "gradients": rows,
                "reference_ad_fd_passed": all(bool(g["reference_passed"]) for g in rows),
                "warp_ad_reference_fd_passed": all(bool(g["warp_passed"]) for g in rows)}
            results.append(result)
            print(json.dumps({"float64_reference": result}), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 8, 40, 400])
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--heights", type=float, nargs="+", default=[.94, 1.0])
    parser.add_argument("--epsilon", type=float, nargs="+", default=[.1, .05, .03, .01, .003, .001])
    parser.add_argument("--rtol", type=float, default=.03)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--full-tape-check", action="store_true", help="Compare 800 steps of checkpointed AD with one full CPU tape")
    parser.add_argument("--float64-reference", action="store_true", help="Investigate long-rollout float32 sensitivity with independent float64 cloth equations")
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device such as cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.frames >= 1 and min(args.substeps) >= 1 and min(args.epsilon) > 0
    assert not (args.float64_reference and args.device != "cpu"), "The float64 reference is CPU-only"
    global DEVICE
    DEVICE = args.device
    torch.set_num_threads(1)
    torch.set_default_device(DEVICE)
    wp.init()
    wp.set_device(DEVICE)
    rotation = check_rotation()
    results = [run_case(contact, attachments, substeps, args.frames, args.epsilon, args.rtol, args.atol, height)
               for substeps in args.substeps for height in args.heights
               for contact, attachments in ((False, False), (True, False), (True, True))]
    face_mover = run_case(True, True, 8, 2, args.epsilon, args.rtol, args.atol, face_attachments=True)
    state_vjp = check_state_vjp()
    yield_surface = check_state_vjp(yield_surface=True)
    report = {"device": wp.get_device(DEVICE).name if DEVICE != "cpu" else "cpu", "warp": wp.__version__, "rtol": args.rtol, "atol": args.atol,
              "finite_difference": "Native forward solver; central differences with an epsilon sweep",
              "rotation": rotation, "state_vjp": state_vjp, "yield_surface_diagnostic": yield_surface,
              "face_attachment_case": face_mover, "cases": results,
              "passed": all(bool(case["passed"]) for case in results) and bool(state_vjp["passed"]) and bool(face_mover["passed"])}
    if args.full_tape_check:
        report["full_tape"] = check_full_tape()
        report["passed"] = bool(report["passed"]) and bool(report["full_tape"]["passed"])
    if args.float64_reference:
        report["float64_reference"] = check_float64_reference(args.rtol, args.atol)
        report["passed"] = bool(report["passed"]) and all(bool(case["warp_ad_reference_fd_passed"])
                                                       for case in report["float64_reference"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"], f"AD/FD disagreement; see {args.output}"


if __name__ == "__main__":
    main()
