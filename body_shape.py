"""Fit the one perturbed SMPL-X beta through fixed-material cloth simulation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from train_material_params import Trainer

# Template command (mpmavatar environment, from clothes-reconstruction):
# python MPMAvatar/run.py --data /path/to/tweaked/export --output /path/to/run --stage body --checkpoint /path/to/material.npz --body-iterations 100 --beta-lr 0.01 --beta-epsilon 0.01 --body-batch-size 8


class SingleBetaBody:
    """Regenerate the selected export with its poses and nine other betas fixed."""

    def __init__(self, source_root: Path, device: torch.device, batch_size: int = 8) -> None:
        import smplx

        self.source_root = source_root.resolve()
        manifest = json.loads((self.source_root / "manifest.json").read_text())
        config = json.loads((self.source_root / "config.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["component"] == "dgarments_mpmavatar_export"
        assert "body_shape_experiment" in manifest, "Select an export made by dgarments_avatar.tweak_body"
        experiment = manifest["body_shape_experiment"]
        assert experiment["perturbation"]["indexing"] == "zero_based"
        self.beta_index = int(experiment["perturbation"]["beta_index"])
        assert 0 <= self.beta_index < 10 and batch_size > 0
        self.batch_size = batch_size
        with np.load(self.source_root / experiment["body_motion"], allow_pickle=False) as data:
            self.source_betas = torch.as_tensor(data["betas"], dtype=torch.float32, device=device).clone()
            self.pose = torch.as_tensor(data["full_pose"], dtype=torch.float32, device=device).clone()
            self.translation = torch.as_tensor(data["transl"], dtype=torch.float32, device=device).clone()
            self.faces = data["faces"].copy()
            source_vertices = torch.as_tensor(data["vertices"], dtype=torch.float32, device=device)
            np.testing.assert_array_equal(data["frame_ids"], manifest["frame_ids"])
            np.testing.assert_array_equal(data["frame_ids"], np.arange(len(self.pose)))
            assert np.allclose(np.diff(data["timestamps"]), 1.0 / 25.0)
        assert self.source_betas.shape == (1, 10)
        assert self.pose.shape == (len(self.translation), 165)
        assert self.translation.shape == (len(self.pose), 3)
        np.testing.assert_array_equal(self.source_betas.cpu().numpy().reshape(-1), experiment["fixed_betas"])
        model_path = Path(config["smplx_model"])
        assert model_path.is_file(), model_path
        self.model = smplx.SMPLX(str(model_path), gender=config["gender"], ext="npz", num_betas=10,
                                 use_pca=False, flat_hand_mean=True).to(device)
        self.model.requires_grad_(False)
        np.testing.assert_array_equal(self.model.faces, self.faces)
        self.initial_beta = float(self.source_betas[0, self.beta_index])
        reconstructed = self.vertices(self.initial_beta)
        assert torch.allclose(reconstructed, source_vertices, atol=2e-5, rtol=0), "Source SMPL-X parameters do not reproduce its collider"

    @torch.no_grad()
    def vertices(self, beta_value: float) -> torch.Tensor:
        from smplx.lbs import lbs

        assert np.isfinite(beta_value)
        betas = self.source_betas.clone()
        betas[0, self.beta_index] = beta_value
        batches = []
        for start in range(0, len(self.pose), self.batch_size):
            pose = self.pose[start:start + self.batch_size]
            vertices, _ = lbs(betas.expand(len(pose), -1), pose, self.model.v_template,
                              self.model.shapedirs[:, :, :10], self.model.posedirs,
                              self.model.J_regressor, self.model.parents, self.model.lbs_weights)
            batches.append(vertices + self.translation[start:start + len(pose), None, :])
        result = torch.cat(batches)
        assert torch.isfinite(result).all(), f"Nonfinite body at beta={beta_value}"
        return result

    def apply(self, trainer: Trainer, beta_value: float) -> None:
        vertices = self.vertices(beta_value)
        np.testing.assert_array_equal(trainer.collider_faces.cpu().numpy(), self.faces)
        frames = trainer.motion_frame_ids
        assert frames == list(range(len(frames))) and len(frames) <= len(vertices)
        trainer.body_motion = vertices[:len(frames)]
        trainer.body_motion_velocity = (trainer.body_motion[1:] - trainer.body_motion[:-1]) * 25.0
        trainer.train_frame_collider = trainer.body_motion[trainer.scene.train_frame_index]
        trainer.train_frame_collider_velo = trainer.body_motion_velocity[trainer.scene.train_frame_index[:-1]]
        trainer.test_frame_collider = trainer.body_motion[trainer.scene.test_frame_index]
        trainer.test_frame_collider_velo = trainer.body_motion_velocity[trainer.scene.test_frame_index[:-1]]


@torch.no_grad()
def geometry_loss(trainer: Trainer) -> float:
    """Repeat material fitting's frame-averaged cloth MSE from the same state."""
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
    assert substeps > 0 and trainer.scene.train_frame_num > 1
    dt = 1.0 / (25.0 * substeps)
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
        cloth = trainer.sim2wld(positions[trainer.n_elements + trainer.n_traditional:])
        loss += torch.nn.functional.mse_loss(cloth, trainer.train_frame_verts[frame + 1, trainer.reordered_cloth_v_idx])
    result = float(loss / (trainer.scene.train_frame_num - 1))
    assert np.isfinite(result), "Nonfinite body-fitting loss"
    return result


def _atomic_json(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_progress(trainer: Trainer, body: SingleBetaBody, state: dict[str, Any],
                   optimizer: torch.optim.Optimizer) -> None:
    root = Path(trainer.output_path)
    context = state["context"]
    for name in ("best", "last"):
        selected = state[name]
        betas = body.source_betas.cpu().numpy().copy()
        betas[0, body.beta_index] = selected["beta"]
        values = {"beta_index": body.beta_index, "beta_value": selected["beta"], "betas": betas,
                  "loss": selected["loss"], "step": selected["step"],
                  "source_dataset": context["source_dataset"], "material_checkpoint": context["material_checkpoint"],
                  **context["material"], "dataset_dir": context["source_dataset"],
                  "fitting_frame_ids": np.asarray(trainer.scene.train_frame_index),
                  "initial_velocity_policy": "training_forward_difference_25hz",
                  "initial_velocity_frame_ids": np.asarray(trainer.scene.train_frame_index[:2]),
                  "initial_cloth_velocity_world_m_s": trainer.initial_cloth_velocity.cpu().numpy(),
                  "initial_tension_model": "original_rest_height_H"}
        target = root / f"{name}_body_shape.npz"
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **values)
        temporary.replace(target)
    state["optimizer"] = optimizer.state_dict()
    temporary = root / "body_shape_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(root / "body_shape_state.pt")
    summary = {**context, "completed_iterations": state["next_step"],
               "initial_beta": body.initial_beta, "best": state["best"], "last": state["last"],
               "objective": "Mean of cloth vertex coordinate MSE over noninitial training frames",
               "fixed": "Material D/E/H, all other betas, poses, translations and prescribed cloth attachments"}
    _atomic_json(root / "beta_summary.json", summary)
    with (root / "beta_history.csv.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state["history"][0]))
        writer.writeheader()
        writer.writerows(state["history"])
    (root / "beta_history.csv.tmp").replace(root / "beta_history.csv")
    from fit_curves import render_body
    render_body(root / "beta_history.csv", root / "fit_curves.png", body.beta_index)
    text = (f"Optimized beta index (zero based): {body.beta_index}\n"
            f"Source dataset: {context['source_dataset']}\nMaterial checkpoint: {context['material_checkpoint']}\n"
            f"Frozen material: {context['material']}\nCompleted iterations: {state['next_step']}\n"
            f"Initial beta: {body.initial_beta:.9g}\n"
            f"Best beta: {state['best']['beta']:.9g}; loss: {state['best']['loss']:.9g}; step: {state['best']['step']}\n"
            f"Last beta: {state['last']['beta']:.9g}; loss: {state['last']['loss']:.9g}\n"
            "Only this beta was optimized; all other betas, poses and translations remained fixed.\n")
    temporary = root / "beta_summary.txt.tmp"
    temporary.write_text(text)
    temporary.replace(root / "beta_summary.txt")


@torch.no_grad()
def fit_body_shape(trainer: Trainer, source_root: Path, material_path: Path, *, iterations: int,
                   learning_rate: float, finite_difference: float, batch_size: int = 8,
                   resume: bool = False) -> None:
    """Use central finite differences and AdamW on the one recorded beta."""
    assert iterations > 0 and learning_rate > 0 and finite_difference > 0
    assert trainer.accelerator.num_processes == 1, "Body fitting requires one process"
    assert source_root.resolve() == Path(trainer.scene.dataset_dir).resolve()
    assert material_path.is_file(), material_path
    root = Path(trainer.output_path)
    root.mkdir(parents=True, exist_ok=True)
    body = SingleBetaBody(source_root, trainer.train_frame_collider.device, batch_size)
    with np.load(material_path, allow_pickle=False) as data:
        material = {key: float(data[key]) for key in ("D", "E", "H")}
    assert all(np.isfinite(value) and value > 0 for value in material.values())
    for key, value in material.items():
        trainer.torch_param[key].fill_(value / (100.0 if key == "E" else 1.0))
    context = {"source_dataset": str(source_root.resolve()), "material_checkpoint": str(material_path.resolve()),
               "material": material, "beta_index": body.beta_index, "source_betas": body.source_betas.cpu().tolist(),
               "learning_rate": learning_rate, "finite_difference": finite_difference,
               "optimizer": "AdamW", "weight_decay": 0.0,
               "simulation": {key: value for key, value in trainer.resume_context.items() if key != "iterations"}}
    beta = torch.tensor(body.initial_beta, dtype=torch.float32)
    optimizer = torch.optim.AdamW([beta], lr=learning_rate, weight_decay=0.0)
    state_path = root / "body_shape_state.pt"
    if resume:
        assert state_path.is_file(), state_path
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        assert state["context"] == context, "Body resume settings or input selection changed"
        beta.fill_(state["last"]["beta"])
        optimizer.load_state_dict(state["optimizer"])
        _save_progress(trainer, body, state, optimizer)
    else:
        assert not state_path.exists(), "Use resume for the existing body fit"
        body.apply(trainer, float(beta))
        initial = {"step": 0, "beta": float(beta), "loss": geometry_loss(trainer)}
        state = {"context": context, "next_step": 0, "best": initial.copy(), "last": initial.copy(),
                 "history": [{**initial, "beta_before": float(beta), "loss_minus": "", "loss_plus": "", "gradient": ""}]}
        _save_progress(trainer, body, state, optimizer)
        print(f"Body beta[{body.beta_index}] initial={float(beta):.9g} loss={initial['loss']:.9g}", flush=True)
    for step in range(state["next_step"], iterations):
        before = float(beta)
        minus, plus = float(np.float32(before - finite_difference)), float(np.float32(before + finite_difference))
        assert minus < before < plus, "Finite-difference epsilon is too small at this beta"
        body.apply(trainer, minus)
        loss_minus = geometry_loss(trainer)
        body.apply(trainer, plus)
        loss_plus = geometry_loss(trainer)
        gradient = (loss_plus - loss_minus) / (plus - minus)
        assert np.isfinite(gradient), "Nonfinite beta derivative"
        optimizer.zero_grad(set_to_none=True)
        beta.grad = torch.tensor(gradient, dtype=beta.dtype)
        optimizer.step()
        body.apply(trainer, float(beta))
        evaluated = {"step": step + 1, "beta": float(beta), "loss": geometry_loss(trainer)}
        state["last"] = evaluated
        if evaluated["loss"] < state["best"]["loss"]:
            state["best"] = evaluated.copy()
        state["next_step"] = step + 1
        state["history"].append({**evaluated, "beta_before": before, "loss_minus": loss_minus,
                                 "loss_plus": loss_plus, "gradient": gradient})
        _save_progress(trainer, body, state, optimizer)
        print(f"Body step {step + 1}/{iterations}: beta[{body.beta_index}]={float(beta):.9g} "
              f"loss={evaluated['loss']:.9g} gradient={gradient:.9g} "
              f"best_beta={state['best']['beta']:.9g}", flush=True)
    body.apply(trainer, state["best"]["beta"])
    print(f"Body fit: best beta[{body.beta_index}]={state['best']['beta']:.9g}; {root / 'beta_summary.txt'}", flush=True)


def load_body_shape(trainer: Trainer, checkpoint_path: Path, source_root: Path, batch_size: int = 8) -> None:
    """Apply a chosen fitted beta to the existing continuous evaluation pipeline."""
    assert checkpoint_path.is_file(), checkpoint_path
    body = SingleBetaBody(source_root, trainer.train_frame_collider.device, batch_size)
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        assert Path(str(checkpoint["source_dataset"])).resolve() == source_root.resolve()
        assert int(checkpoint["beta_index"]) == body.beta_index
        np.testing.assert_array_equal(checkpoint["fitting_frame_ids"], trainer.scene.train_frame_index)
        expected = body.source_betas.cpu().numpy().copy()
        expected[0, body.beta_index] = float(checkpoint["beta_value"])
        np.testing.assert_array_equal(expected, checkpoint["betas"])
        with np.load(trainer.args.init_params_path, allow_pickle=False) as material:
            for key in ("D", "E", "H"):
                assert float(checkpoint[key]) == float(material[key]), f"Evaluation material {key} differs from the body fit"
        body.apply(trainer, float(checkpoint["beta_value"]))
        trainer.body_shape_result = {
            "checkpoint": str(checkpoint_path.resolve()), "beta_index": body.beta_index,
            "beta_value": float(checkpoint["beta_value"]), "betas": checkpoint["betas"].reshape(-1).tolist(),
            "source_dataset": str(checkpoint["source_dataset"]),
            "material_checkpoint": str(checkpoint["material_checkpoint"]),
            "material": {key: float(checkpoint[key]) for key in ("D", "E", "H")},
        }
        print(f"Loaded body beta[{body.beta_index}]={float(checkpoint['beta_value']):.9g} from {checkpoint_path}", flush=True)
