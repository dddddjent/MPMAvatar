"""Evaluate frozen local-fit materials on the original native training objective."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Template command (workspace root, mpmavatar environment, allocated GPU):
# python MPMAvatar/local_patch_native_training.py --parameters MPMAvatar/output/local_fits/s185_lower_native_grid/initial_param.npz MPMAvatar/output/local_fits/s185_lower_native_grid/best_param.npz --report-dir MPMAvatar/output/local_campaign/s185_lower_native_grid/native_training --save_name native_grid_transfer --trained_model_path data/MPMAvatar/4DDress/examples/s185_t1/output/tracking/s185_t1_11_100 --model_path data/MPMAvatar/4DDress/examples/s185_t1/model/s185_t1 --dataset_dir data/MPMAvatar/4DDress/examples/s185_t1/data --output_dir MPMAvatar/output/local_campaign/s185_lower_native_grid/native_setup --smplx_gender female --subject 185 --train_take 1 --test_take 1 --verts_start_idx 11 --split_idx_path data/MPMAvatar/4DDress/examples/s185_t1/data/s185_t1/split_idx_lower.npz --dataset_type 4ddress --uv_path data/MPMAvatar/4DDress/examples/s185_t1/data/s185_t1/mesh_processed.obj --test_camera_index 0 --train_frame_start_num 45 12 --test_frame_start_num 45 2 --grid_size 200 --substep 400 --iterations 1


def native_training_rollout(
    trainer: Any, density: float, modulus: float, height: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Follow train_one_step's unperturbed rollout without optimizer updates."""
    import torch
    import warp as wp

    assert np.isfinite((density, modulus, height)).all() and min(density, modulus, height) > 0
    device = "cuda:0"
    dt = (1.0 / 25.0) / trainer.args.substep
    with torch.no_grad(), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device))):
        trainer.mpm_solver.time = 0.0
        trainer.mpm_solver.time_profile.clear()
        rest = trainer.vertices_init_position.clone()
        rest[:, 1] *= height
        inverse = trainer.compute_rest_dir_inv_from_vf(rest, trainer.new_cloth_faces)
        trainer.mpm_state.reset_state(
            trainer.n_vertices, trainer.particle_init_position.clone(), trainer.particle_init_dir.clone(),
            tensor_velocity=trainer.particle_init_velo.clone(), tensor_R_inv=inverse,
            device=device, requires_grad=False,
        )
        trainer.mpm_state.set_require_grad(False)
        densities = torch.full_like(trainer.particle_init_position[:, 0], density)
        moduli = torch.full_like(densities, modulus)
        trainer.mpm_state.reset_density(densities, None, device, update_mass=True)
        trainer.mpm_solver.set_E_nu_from_torch(
            trainer.mpm_model, moduli, trainer.poisson_ratio.detach().clone(),
            trainer.gamma.detach().clone(), trainer.kappa.detach().clone(), device,
        )
        trainer.mpm_solver.prepare_mu_lam(trainer.mpm_model, trainer.mpm_state, device)
        trajectory = [trainer.sim2wld(trainer.particle_init_position[trainer.n_elements:]).cpu().numpy().copy()]
        for frame in range(trainer.scene.train_frame_num - 1):
            mesh_x = trainer.wld2sim(trainer.train_frame_collider[frame].clone())
            mesh_v = trainer.train_frame_collider_velo[frame].clone() * trainer.scale
            joint_v = trainer.train_frame_verts_velo[frame, trainer.joint_v_idx].clone() * trainer.scale
            joint_face_v = joint_v[trainer.new_cloth_faces[:trainer.num_joint_f]].mean(1).clone()
            for step in range(trainer.args.substep):
                trainer.mpm_solver.p2g2p(
                    trainer.mpm_model, trainer.mpm_state, dt, mesh_x=mesh_x + dt * step * mesh_v,
                    mesh_v=mesh_v, joint_traditional_v=None, joint_verts_v=joint_v,
                    joint_faces_v=joint_face_v, device=device,
                )
            vertices = trainer.sim2wld(wp.to_torch(trainer.mpm_state.particle_x)[trainer.n_elements:])
            trajectory.append(vertices.detach().cpu().numpy().copy())
            print(f"Native training frame {frame + 1}/{trainer.scene.train_frame_num - 1}", flush=True)
        predicted = np.asarray(trajectory, dtype=np.float64)
        reference = trainer.train_frame_verts[:, trainer.reordered_cloth_v_idx].detach().cpu().numpy()
    assert np.isfinite(predicted).all()
    squared = np.sum((predicted - reference)**2, axis=-1)
    mse = float(np.mean(squared[1:]) / 3.0)
    metrics = {
        "D": density, "E": modulus, "H": height,
        "mean_xyz_mse_m2": mse, "vertex_rmse_mm": float(1000 * np.sqrt(3 * mse)),
        "mean_vertex_error_mm": float(1000 * np.sqrt(squared[1:]).mean()),
        "per_frame_vertex_rmse_mm": (1000 * np.sqrt(squared.mean(axis=1))).tolist(),
        "frame_ids": list(trainer.scene.train_frame_index), "cloth_vertex_count": trainer.n_vertices,
        "grid_size": trainer.args.grid_size, "cell_size_m": float(trainer.mpm_model.dx / trainer.scale),
        "substeps": trainer.args.substep, "initialization_frame_scored": False,
        "prediction_scope": "original full lower garment with native body contact and attachment movers",
        "scoring_scope": "all lower garment vertices, including native attachments, matching history.csv",
    }
    return metrics, predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, nargs="+", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    options, native_arguments = parser.parse_known_args()
    assert not options.report_dir.exists(), "Native comparison output must be fresh"
    assert all(path.is_file() for path in options.parameters)
    from train_material_params import Trainer, parse_args

    sys.argv = [sys.argv[0], *native_arguments]
    args, opt, pipe, run_eval, _, _, _ = parse_args()
    assert not run_eval and not args.resume and not args.checkpoint_material
    assert args.dataset_type == "4ddress" and args.subject == 185
    assert args.train_frame_start_num == [45, 12] and args.grid_size == 200 and args.substep == 400
    assert "lower" in args.split_idx_path
    trainer = Trainer(args, opt, pipe, False)
    options.report_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for path in options.parameters:
        label = path.parent.name if path.name == "best_param.npz" else path.stem
        destination = options.report_dir / label
        assert not destination.exists(), f"Duplicate material name: {label}"
        destination.mkdir()
        with np.load(path, allow_pickle=False) as data:
            metrics, predicted = native_training_rollout(trainer, *(float(data[key]) for key in ("D", "E", "H")))
        metrics.update(name=label, parameters=str(path.resolve()))
        np.savez_compressed(destination / "trajectory.npz", vertices_m=predicted,
                            frame_ids=np.asarray(trainer.scene.train_frame_index),
                            vertex_ids=trainer.reordered_cloth_v_idx.cpu().numpy())
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        records.append(metrics)
        (options.report_dir / "metrics.json").write_text(json.dumps({"materials": records}, indent=2) + "\n")
        print(f"{label}: native training vertex RMSE {metrics['vertex_rmse_mm']:.5f} mm", flush=True)


if __name__ == "__main__":
    main()
