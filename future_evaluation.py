"""Continuous cloth prediction with recorded body and cut-boundary motion."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm
import warp as wp

from dataset_input import load_manifest

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

    substeps = trainer.args.substep
    assert substeps > 0
    dt = 1.0 / (25.0 * substeps)
    evaluation_ids = trainer.scene.test_frame_index
    assert evaluation_ids[0] == 0 or evaluation_ids[0] > trainer.scene.train_frame_index[-1]
    assert evaluation_ids == list(range(evaluation_ids[0], evaluation_ids[-1] + 1))
    assert trainer.motion_frame_ids == list(range(evaluation_ids[-1] + 1))
    predictions = []
    if evaluation_ids[0] == 0:
        predictions.append(trainer.first_frame_verts.clone())
    for frame in tqdm(range(evaluation_ids[-1]), desc="Continuous cloth prediction"):
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
        if frame + 1 >= evaluation_ids[0]:
            vertices = torch.zeros_like(trainer.first_frame_verts)
            vertices[trainer.reordered_cloth_v_idx.long()] = trainer.sim2wld(
                particles[trainer.n_elements + trainer.n_traditional:]
            )
            vertices[trainer.reordered_human_v_idx.long()] = trainer.prescribed_human_positions[frame + 1]
            predictions.append(vertices)
    assert len(predictions) == len(evaluation_ids)
    return predictions


@torch.no_grad()
def simulate_future(trainer: "Trainer") -> list[torch.Tensor]:
    """Save selected predictions and score them after the simulation finishes."""
    manifest = load_manifest(Path(trainer.scene.dataset_dir))
    frames = trainer.scene.test_frame_index
    assert frames in (manifest["evaluation_frame_ids"], manifest["frame_ids"])
    assert trainer.scene.train_frame_index == manifest["train_frame_ids"][:trainer.scene.train_frame_num]
    predictions = continuous_predictions(trainer)
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

    # Read free-cloth targets only after all predictions are complete.
    with np.load(trainer.args.prescribed_surface_path, allow_pickle=False) as data:
        target = data["vertices"][frames]
    free_ids = trainer.reordered_cloth_v_idx[trainer.num_joint_v:].cpu().numpy()
    assert len(free_ids) > 0
    assert np.isfinite(target[:, free_ids]).all()
    error = predicted[:, free_ids] - target[:, free_ids]
    v2v = np.linalg.norm(error, axis=-1).mean(axis=1)
    mse = np.square(error).mean(axis=(1, 2))
    report = {
        "protocol": "continuous_from_training_start",
        "material_checkpoint": str(Path(trainer.args.init_params_path).resolve()),
        "material_parameters": {key: float(trainer.torch_param[key]) * (100.0 if key == "E" else 1.0)
                                for key in ("D", "E", "H")},
        "fitting_frame_ids": trainer.scene.train_frame_index,
        "appearance_frame_ids": manifest["train_frame_ids"],
        "evaluation_frame_ids": frames,
        "evaluation_scope": "full_sequence" if frames == manifest["frame_ids"] else "held_out",
        "initial_velocity_frame_ids": trainer.scene.train_frame_index[:2],
        "driving_inputs": (["raw body mesh collider" if manifest.get("body_model", "smplx") == "raw"
                            else "fitted body collider", "original body surface for appearance"]
                           + (["recorded attachment motion"] if trainer.num_joint_v else [])),
        "scored_vertices": ("free cloth; prescribed attachment vertices excluded"
                            if trainer.num_joint_v else "all cloth vertices; no attachment constraints"),
        "free_cloth_vertex_count": len(free_ids),
        "mean_v2v_m": float(v2v.mean()),
        "mean_xyz_mse_m2": float(mse.mean()),
        "per_frame_v2v_m": v2v.tolist(),
        "per_frame_xyz_mse_m2": mse.tolist(),
        "simulation_domain": manifest["simulation_domain"],
    }
    (output / "geometry_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return predictions
