"""Replay one fitted material over the 4D-DRESS fitting window and save cloth positions.

Builds the material-fitting Trainer, overwrites D/E/H with the checkpoint's
evaluated_* values (the parameters its stored loss was computed at), and
repeats body_shape.geometry_loss while recording the cloth after every frame.
Output `positions` is (frames, cloth vertices, 3) in world meters and reordered
cloth order, starting with the initial fitting frame.

Template (mpmavatar environment, from MPMAvatar):
python replay_material_window.py --params ../data/MPMAvatar/4DDress/examples/s185_t1/output/phys/s185_t1_lower/seed0/best_param_00081.npz --replay_output ../WarpXPBD/output/185_lower/phase0/mpmavatar_best/trajectory.npz --save_name s185_t1_lower --trained_model_path ../data/MPMAvatar/4DDress/examples/s185_t1/output/tracking/s185_t1_11_100 --model_path ../data/MPMAvatar/4DDress/examples/s185_t1/model/s185_t1 --dataset_dir ../data/MPMAvatar/4DDress/examples/s185_t1/data --output_dir ../data/MPMAvatar/4DDress/examples/s185_t1/output/phys_replay --smplx_gender female --subject 185 --train_take 1 --test_take 7 --verts_start_idx 11 --split_idx_path ../data/MPMAvatar/4DDress/examples/s185_t1/data/s185_t1/split_idx_lower.npz --dataset_type 4ddress --uv_path ../data/MPMAvatar/4DDress/examples/s185_t1/data/s185_t1/mesh_processed.obj --test_camera_index 0 --train_frame_start_num 45 12 --test_frame_start_num 45 2 --iterations 200 --init_params_path ''
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from train_material_params import Trainer


@torch.no_grad()
def replay(trainer: "Trainer") -> tuple[np.ndarray, float]:
    """body_shape.geometry_loss with per-frame cloth positions recorded."""
    import warp as wp

    device = str(trainer.particle_init_position.device)
    state, model, solver = trainer.mpm_state, trainer.mpm_model, trainer.mpm_solver
    rest = trainer.vertices_init_position.clone()
    rest[:, 1] *= trainer.torch_param["H"].to(device)
    rest_inv = trainer.compute_rest_dir_inv_from_vf(rest, trainer.new_cloth_faces)
    state.reset_state(trainer.n_vertices, trainer.particle_init_position.clone(),
                      trainer.particle_init_dir.clone(), tensor_velocity=trainer.particle_init_velo.clone(),
                      tensor_R_inv=rest_inv, device=device, requires_grad=False)
    density, youngs, poisson, gamma, kappa = trainer.get_material_params(device)
    state.reset_density(density.clone(), None, device, update_mass=True)
    solver.set_E_nu_from_torch(model, youngs.clone(), poisson.clone(), gamma.clone(), kappa.clone(), device)
    solver.prepare_mu_lam(model, state, device)
    solver.time = 0.0
    solver.time_profile.clear()
    substeps = trainer.args.substep
    dt = 1.0 / (25.0 * substeps)
    start = trainer.n_elements + trainer.n_traditional
    frames = [trainer.sim2wld(trainer.particle_init_position[start:].to(device))]
    loss = torch.zeros((), device=device)
    for frame in range(trainer.scene.train_frame_num - 1):
        mesh_x = trainer.wld2sim(trainer.train_frame_collider[frame])
        mesh_v = trainer.train_frame_collider_velo[frame] * trainer.scale
        joint_v = trainer.train_frame_verts_velo[frame, trainer.joint_v_idx] * trainer.scale
        face_v = joint_v[trainer.new_cloth_faces[:trainer.num_joint_f]].mean(1)
        for substep in range(substeps):
            solver.p2g2p(model, state, dt, mesh_x=mesh_x + dt * substep * mesh_v,
                         mesh_v=mesh_v, joint_traditional_v=None, joint_verts_v=joint_v,
                         joint_faces_v=face_v, device=device)
        positions = wp.to_torch(state.particle_x)
        assert torch.isfinite(positions).all(), f"Nonfinite simulation at training frame {frame + 1}"
        cloth = trainer.sim2wld(positions[start:])
        frames.append(cloth.clone())
        loss += torch.nn.functional.mse_loss(cloth, trainer.train_frame_verts[frame + 1, trainer.reordered_cloth_v_idx])
        print(f"frame {frame + 1} loss {float(loss) / (frame + 1):.6e}", flush=True)
    return torch.stack(frames).cpu().numpy(), float(loss / (trainer.scene.train_frame_num - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--replay_output", type=Path, required=True)
    replay_args, remaining = parser.parse_known_args()
    assert replay_args.params.is_file(), replay_args.params
    sys.argv = [sys.argv[0], *remaining]
    from train_material_params import parse_args, Trainer

    args, opt, pipe, run_eval, *_ = parse_args()
    assert not run_eval and not args.checkpoint_material and not args.resume and not args.init_params_path
    trainer = Trainer(args, opt, pipe, run_eval)
    selected = dict(np.load(replay_args.params, allow_pickle=False))
    trainer.torch_param["D"] = torch.tensor(float(selected["evaluated_D"]))
    trainer.torch_param["E"] = torch.tensor(float(selected["evaluated_E"]) / 100.0)
    trainer.torch_param["H"] = torch.tensor(float(selected["evaluated_H"]))
    positions, loss = replay(trainer)
    rest = trainer.vertices_init_position.clone()
    rest[:, 1] *= trainer.torch_param["H"].to(rest.device)
    idx = trainer.reordered_cloth_v_idx
    replay_args.replay_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(replay_args.replay_output, positions=positions.astype(np.float32),
             target=trainer.train_frame_verts[:, idx].cpu().numpy().astype(np.float32),
             rest_H=trainer.sim2wld(rest).cpu().numpy().astype(np.float32),
             faces=trainer.new_cloth_faces.cpu().numpy(), num_joints=int(trainer.num_joint_v),
             frame_ids=np.asarray(trainer.scene.train_frame_index), D=float(selected["evaluated_D"]),
             E=float(selected["evaluated_E"]), H=float(selected["evaluated_H"]),
             loss=loss, checkpoint_loss=float(selected["loss"]))
    print(f"replay loss {loss:.6e} checkpoint loss {float(selected['loss']):.6e} -> {replay_args.replay_output}", flush=True)


if __name__ == "__main__":
    main()
