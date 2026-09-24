"""Independent held-out scoring and selectable comparison-video rollouts."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm
import warp as wp

from dataset_input import load_manifest
from warp_mpm.initial_state import estimate_velocity, particle_velocities

if TYPE_CHECKING:
    from train_material_params import Trainer


def load_driving_surface(
    path: str,
    frame_ids: list[int],
    joint_ids: torch.Tensor,
    human_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose only prescribed vertices; free cloth observations are scoring data."""
    with np.load(path, allow_pickle=False) as data:
        vertices = data["vertices"]
        joints = vertices[np.ix_(frame_ids, joint_ids.cpu().numpy())]
        human = vertices[np.ix_(frame_ids, human_ids.cpu().numpy())]
    assert np.isfinite(joints).all() and np.isfinite(human).all()
    return (torch.as_tensor(joints, dtype=torch.float32, device=joint_ids.device),
            torch.as_tensor(human, dtype=torch.float32, device=human_ids.device))


@torch.no_grad()
def continuous_predictions(trainer: "Trainer") -> list[torch.Tensor]:
    """Initialize at the training start, then advance without observation resets."""
    device = str(trainer.particle_init_position.device)
    density, youngs, poisson, gamma, kappa = trainer.get_material_params(device)
    state, model, solver = trainer.mpm_state, trainer.mpm_model, trainer.mpm_solver
    rest_vertices = trainer.vertices_init_position.clone()
    rest_vertices[:, 1] *= trainer.torch_param["H"].to(device)
    rest_inv = trainer.compute_rest_dir_inv_from_vf(rest_vertices, trainer.new_cloth_faces)
    state.reset_state(trainer.n_vertices, trainer.particle_init_position.clone(),
                      trainer.particle_init_dir.clone(),
                      tensor_velocity=trainer.particle_init_velo.clone(),
                      tensor_R_inv=rest_inv, device=device, requires_grad=False)
    state.reset_density(density.clone(), None, device, update_mass=True)
    solver.set_E_nu_from_torch(model, youngs.clone(), poisson.clone(), gamma.clone(), kappa.clone(), device)
    solver.prepare_mu_lam(model, state, device)
    solver.time = 0.0
    solver.time_profile.clear()
    return _advance_predictions(trainer, 0, trainer.scene.test_frame_index, trainer.first_frame_verts)


@torch.no_grad()
def _advance_predictions(trainer: "Trainer", start_frame: int, frames: list[int],
                         initial_vertices: torch.Tensor) -> list[torch.Tensor]:
    """Advance a reset state without reading subsequent free-cloth observations."""
    state, model, solver = trainer.mpm_state, trainer.mpm_model, trainer.mpm_solver
    device = str(trainer.particle_init_position.device)
    substeps = trainer.args.substep
    assert substeps > 0
    dt = 1.0 / (25.0 * substeps)
    assert frames == list(range(frames[0], frames[-1] + 1))
    assert 0 <= start_frame <= frames[0]
    assert trainer.motion_frame_ids == list(range(frames[-1] + 1))
    predictions = []
    if frames[0] == start_frame:
        predictions.append(initial_vertices.clone())
    for frame in tqdm(range(start_frame, frames[-1]), desc=f"Cloth prediction from frame {start_frame}"):
        body = trainer.body_motion[frame]
        body_velocity = trainer.body_motion_velocity[frame] * trainer.scale
        mesh_x = trainer.wld2sim(body)
        joint_velocity = (trainer.prescribed_joint_positions[frame + 1]
                          - trainer.prescribed_joint_positions[frame]) * 25.0 * trainer.scale
        face_velocity = joint_velocity[trainer.new_cloth_faces[:trainer.num_joint_f].long()].mean(1)
        for substep in range(substeps):
            solver.p2g2p(model, state, dt, mesh_x=mesh_x + dt * substep * body_velocity,
                         mesh_v=body_velocity, joint_traditional_v=None,
                         joint_verts_v=joint_velocity, joint_faces_v=face_velocity, device=device)
        particles = wp.to_torch(state.particle_x)
        assert torch.isfinite(particles).all(), f"Nonfinite cloth at frame {frame + 1}"
        if frame + 1 >= frames[0]:
            vertices = torch.zeros_like(trainer.first_frame_verts)
            vertices[trainer.reordered_cloth_v_idx.long()] = trainer.sim2wld(
                particles[trainer.n_elements + trainer.n_traditional:]
            )
            vertices[trainer.reordered_human_v_idx.long()] = trainer.prescribed_human_positions[frame + 1]
            predictions.append(vertices)
    assert len(predictions) == len(frames)
    return predictions


@torch.no_grad()
def evaluation_predictions(trainer: "Trainer", frames: list[int]) -> list[torch.Tensor]:
    """Reset at the first held-out mesh; estimate its velocity from the first two."""
    assert len(frames) >= 2 and frames[0] > trainer.scene.train_frame_index[-1]
    assert trainer.n_traditional == 0
    device = str(trainer.particle_init_position.device)
    with np.load(trainer.args.prescribed_surface_path, allow_pickle=False) as data:
        observed = torch.as_tensor(data["vertices"][frames[:2]], dtype=torch.float32, device=device)
    assert torch.isfinite(observed).all()
    cloth_ids = trainer.reordered_cloth_v_idx.long()
    faces = trainer.new_cloth_faces.long()
    initial_vertices = trainer.wld2sim(observed[0, cloth_ids])
    particle_positions = torch.cat((initial_vertices[faces].mean(1), initial_vertices), dim=0)
    velocity = estimate_velocity(observed[:, cloth_ids], frames[:2])
    particle_velocity = particle_velocities(velocity, faces, trainer.scale)
    directions, _, _, _ = trainer.compute_dir_vol(initial_vertices, faces, thickness=1e-5)

    # H retains the fitted rest reference; resetting position does not redefine it.
    rest_vertices = trainer.vertices_init_position.clone()
    rest_vertices[:, 1] *= trainer.torch_param["H"].to(device)
    rest_inv = trainer.compute_rest_dir_inv_from_vf(rest_vertices, faces)
    state, model, solver = trainer.mpm_state, trainer.mpm_model, trainer.mpm_solver
    state.reset_state(trainer.n_vertices, particle_positions, directions,
                      tensor_velocity=particle_velocity, tensor_R_inv=rest_inv,
                      device=device, requires_grad=False)
    density, youngs, poisson, gamma, kappa = trainer.get_material_params(device)
    state.reset_density(density.clone(), None, device, update_mass=True)
    solver.set_E_nu_from_torch(model, youngs.clone(), poisson.clone(), gamma.clone(), kappa.clone(), device)
    solver.prepare_mu_lam(model, state, device)
    solver.time = 0.0
    solver.time_profile.clear()
    return _advance_predictions(trainer, frames[0], frames, observed[0])


@torch.no_grad()
def score_evaluation(trainer: "Trainer", manifest: dict[str, Any]) -> list[torch.Tensor]:
    """Always score the same rollout initialized at the evaluation boundary."""
    frames = manifest["evaluation_frame_ids"]
    predictions = evaluation_predictions(trainer, frames)
    predicted = torch.stack(predictions).cpu().numpy()
    output = Path(trainer.output_path)
    np.savez_compressed(output / "evaluation_predictions.npz", frame_ids=np.asarray(frames), vertices=predicted)
    experiment = manifest.get("body_shape_experiment", {})
    fitted_body = experiment.get("prediction_body") == "fitted_smplx"

    # Beyond the two initialization observations, free cloth is scoring-only.
    with np.load(trainer.args.prescribed_surface_path, allow_pickle=False) as data:
        target = data["vertices"][frames]
    free_ids = trainer.reordered_cloth_v_idx[trainer.num_joint_v:].cpu().numpy()
    assert len(free_ids) > 0
    assert np.isfinite(target[:, free_ids]).all()
    error = predicted[:, free_ids] - target[:, free_ids]
    v2v = np.linalg.norm(error, axis=-1).mean(axis=1)
    mse = np.square(error).mean(axis=(1, 2))
    report = {
        "protocol": "from_first_evaluation_frame",
        "material_checkpoint": str(Path(trainer.args.init_params_path).resolve()),
        "material_parameters": {key: float(trainer.torch_param[key]) * (100.0 if key == "E" else 1.0)
                                for key in ("D", "E", "H")},
        "fitting_frame_ids": trainer.scene.train_frame_index,
        "appearance_frame_ids": manifest["train_frame_ids"],
        "evaluation_frame_ids": frames,
        "evaluation_scope": "held_out",
        "initial_geometry_frame_id": frames[0],
        "initial_velocity_frame_ids": frames[:2],
        "initial_velocity_policy": "evaluation_forward_difference_25hz",
        "initial_frame_included_in_metrics": True,
        "rest_shape": "Training initial cloth with fitted H; unchanged by evaluation reset",
        "render_frame_ids": trainer.scene.test_frame_index,
        "render_protocol": ("continuous_from_training_start" if trainer.scene.test_frame_index[0] == 0
                            else "from_first_evaluation_frame"),
        "scored_predictions": "evaluation_predictions.npz",
        "driving_inputs": (["raw body mesh collider" if manifest.get("body_model", "smplx") == "raw"
                            else "fitted body collider",
                            "fitted SMPL-X body for prediction rendering" if fitted_body
                            else "original body surface for appearance"]
                           + (["recorded attachment motion"] if trainer.num_joint_v else [])),
        "scored_vertices": ("free cloth; prescribed attachment vertices excluded"
                            if trainer.num_joint_v else "all cloth vertices; no attachment constraints"),
        "free_cloth_vertex_count": len(free_ids),
        "mean_v2v_m": float(v2v.mean()),
        "mean_v2v_mm": float(v2v.mean() * 1000),
        "mean_xyz_mse_m2": float(mse.mean()),
        "per_frame_v2v_m": v2v.tolist(),
        "per_frame_xyz_mse_m2": mse.tolist(),
        "simulation_domain": manifest["simulation_domain"],
    }
    if fitted_body:
        report["prediction_body"] = "predicted_body.npz: exact simulation collider for render_frame_ids"
        report["body_shape_experiment"] = experiment
        report["predictions_archive_body"] = "Original-topology placeholder; render predicted_body.npz instead"
    if hasattr(trainer, "body_shape_result"):
        report["optimized_body"] = trainer.body_shape_result
    (output / "geometry_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "geometry_metrics.md").write_text(
        f"# Held-out cloth evaluation\n\n"
        f"- Mean vertex error: {report['mean_v2v_mm']:.6f} mm\n"
        f"- Mean XYZ squared error: {report['mean_xyz_mse_m2']:.9g} m²\n"
        f"- Scored frames: {frames[0]}–{frames[-1]} ({len(frames)} frames)\n"
        f"- Simulation initialized from observed cloth at frame {frames[0]}; "
        f"velocity from frames {frames[0]} and {frames[1]}.\n"
        "- The initialized frame is included in the average and has zero error.\n"
        "- Only free cloth is scored; the rendered frame range does not change this evaluation.\n"
    )
    print(f"Held-out evaluation (reset at frame {frames[0]}): "
          f"V2V {report['mean_v2v_mm']:.6f} mm, XYZ MSE {report['mean_xyz_mse_m2']:.9g} m^2\n"
          f"Report: {output / 'geometry_metrics.json'}", flush=True)
    return predictions


@torch.no_grad()
def simulate_future(trainer: "Trainer") -> list[torch.Tensor]:
    """Score the held-out reset rollout, then save the selected video rollout."""
    manifest = load_manifest(Path(trainer.scene.dataset_dir))
    frames = trainer.scene.test_frame_index
    assert frames in (manifest["evaluation_frame_ids"], manifest["frame_ids"])
    assert trainer.scene.train_frame_index == manifest["train_frame_ids"][:trainer.scene.train_frame_num]
    scored = score_evaluation(trainer, manifest)
    predictions = scored if frames == manifest["evaluation_frame_ids"] else continuous_predictions(trainer)
    output = Path(trainer.output_path)
    mesh_dir = output / "uvmesh"
    assert not mesh_dir.exists(), mesh_dir
    mesh_dir.mkdir()
    uv_faces = [line for line in Path(trainer.scene.uv_path).read_text().splitlines(keepends=True)
                if line.startswith(("vt ", "f "))]
    predicted = torch.stack(predictions).cpu().numpy()
    for index, vertices in enumerate(predicted):
        with (mesh_dir / f"{index:03d}.obj").open("w") as stream:
            stream.writelines(f"v {x} {y} {z}\n" for x, y, z in vertices)
            stream.writelines(uv_faces)
    np.savez_compressed(output / "predictions.npz", frame_ids=np.asarray(frames), vertices=predicted)
    if manifest.get("body_shape_experiment", {}).get("prediction_body") == "fitted_smplx":
        np.savez_compressed(
            output / "predicted_body.npz", frame_ids=np.asarray(frames),
            vertices=trainer.body_motion[frames].cpu().numpy(),
            faces=trainer.collider_faces.cpu().numpy(),
        )
    return predictions
