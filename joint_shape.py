"""Fit one SMPL-X beta and D/E/H together using training-cloth simulation loss."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from body_shape import SingleBetaBody, geometry_loss
from fit_curves import render_body, render_material

if TYPE_CHECKING:
    from train_material_params import Trainer

# Template commands (activate mpmavatar; run from clothes-reconstruction):
# python MPMAvatar/run.py --data /path/to/tweaked/export --output /path/to/joint/run --stage joint --appearance-model /path/to/original/appearance --checkpoint /path/to/initial/material.npz --body-iterations 100 --beta-lr 0.01 --beta-epsilon 0.01 --body-batch-size 8 --material-frames all --grid-size 200 --substeps 400
# python MPMAvatar/run.py --data /path/to/tweaked/export --output /path/to/joint/run --stage evaluate --appearance-model /path/to/original/appearance --checkpoint /path/to/joint/run/joint/seed0/best_joint_shape.npz --body-checkpoint /path/to/joint/run/joint/seed0/best_joint_shape.npz --material-frames all --grid-size 200 --substeps 400 --body-batch-size 8 --render-appearance gt_lighting


def _material_values(trainer: Trainer) -> dict[str, float]:
    return {key: float(value) * (100.0 if key == "E" else 1.0)
            for key, value in trainer.torch_param.items()}


def _save_progress(trainer: Trainer, body: SingleBetaBody, state: dict[str, Any],
                   optimizer: torch.optim.Optimizer) -> None:
    root = Path(trainer.output_path)
    context = state["context"]
    for name in ("best", "last"):
        selected = state[name]
        betas = body.source_betas.cpu().numpy().copy()
        betas[0, body.beta_index] = selected["beta"]
        values = {**selected, "beta_index": body.beta_index, "beta_value": selected["beta"],
                  "betas": betas, "optimization_kind": "joint_beta_material",
                  "source_dataset": context["source_dataset"],
                  "material_checkpoint": context["initial_material_checkpoint"],
                  "dataset_dir": context["source_dataset"],
                  "fitting_frame_ids": np.asarray(trainer.scene.train_frame_index),
                  "initial_velocity_policy": "training_forward_difference_25hz",
                  "initial_velocity_frame_ids": np.asarray(trainer.scene.train_frame_index[:2]),
                  "initial_cloth_velocity_world_m_s": trainer.initial_cloth_velocity.cpu().numpy(),
                  "initial_tension_model": "original_rest_height_H"}
        target = root / f"{name}_joint_shape.npz"
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **values)
        temporary.replace(target)
    state["beta_optimizer"] = optimizer.state_dict()
    state["material_optimizer"] = trainer.optimizer.state_dict()
    state["material_scheduler"] = trainer.scheduler.state_dict()
    temporary = root / "joint_shape_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(root / "joint_shape_state.pt")
    summary = {**context, "completed_iterations": state["next_step"],
               "planned_iterations": state["planned_iterations"], "stop_after_iterations": state["stop_after_iterations"],
               "initial_beta": body.initial_beta, "best": state["best"], "last": state["last"],
               "objective": "Mean cloth vertex coordinate MSE over noninitial training frames",
               "loss_note": "Loss is evaluated at the listed beta and D/E/H after their simultaneous update.",
               "fixed": "Other betas, poses, translations and prescribed cloth attachments"}
    temporary = root / "joint_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(root / "joint_summary.json")
    for filename in ("history.csv", "beta_history.csv"):
        temporary = root / (filename + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(state["history"][0]))
            writer.writeheader()
            writer.writerows(state["history"])
        temporary.replace(root / filename)
    render_material(root / "history.csv", root / "material_fit_curves.png", loss_at_parameters=True)
    render_body(root / "beta_history.csv", root / "body_fit_curves.png", body.beta_index)


@torch.no_grad()
def fit_joint_shape(trainer: Trainer, source_root: Path, material_path: Path, *, iterations: int,
                    learning_rate: float, finite_difference: float, batch_size: int = 8,
                    resume: bool = False, stop_after: int = 0) -> None:
    """Take simultaneous beta/material updates with derivatives at one common state."""
    assert iterations > 0 and learning_rate > 0 and finite_difference > 0
    assert 0 <= stop_after <= iterations, "Joint stop-after must be within the planned iteration target"
    end_step = stop_after or iterations
    assert trainer.accelerator.num_processes == 1, "Joint fitting requires one process"
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
    context = {"source_dataset": str(source_root.resolve()),
               "initial_material_checkpoint": str(material_path.resolve()), "initial_material": material,
               "beta_index": body.beta_index, "source_betas": body.source_betas.cpu().tolist(),
               "beta_learning_rate": learning_rate, "beta_finite_difference": finite_difference,
               "material_finite_differences": {"D": 0.05, "E": 0.05, "H": 0.005},
               "beta_optimizer": "AdamW", "beta_weight_decay": 0.0,
               "material_optimizer": "Adam",
               "material_learning_rates": [group["initial_lr"] for group in trainer.optimizer.param_groups],
               "simulation": trainer.resume_context.copy()}
    beta = torch.tensor(body.initial_beta, dtype=torch.float32)
    optimizer = torch.optim.AdamW([beta], lr=learning_rate, weight_decay=0.0)
    state_path = root / "joint_shape_state.pt"
    if resume:
        assert state_path.is_file(), state_path
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        assert state["context"] == context, "Joint resume settings or input selection changed"
        beta.fill_(state["last"]["beta"])
        for key, value in trainer.torch_param.items():
            value.fill_(state["last"][key] / (100.0 if key == "E" else 1.0))
        trainer.scheduler.load_state_dict(state["material_scheduler"])
        trainer.optimizer.load_state_dict(state["material_optimizer"])
        optimizer.load_state_dict(state["beta_optimizer"])
    else:
        assert not state_path.exists(), "Use resume for the existing joint fit"
        body.apply(trainer, float(beta))
        initial = {"step": 0, "beta": float(beta), **_material_values(trainer), "loss": geometry_loss(trainer)}
        state = {"context": context, "next_step": 0, "best": initial.copy(), "last": initial.copy(),
                 "history": [{**initial, "beta_before": float(beta), "loss_before": initial["loss"],
                              "loss_minus": "", "loss_plus": "", "gradient_beta": "",
                              "gradient_D": "", "gradient_E": "", "gradient_H": ""}]}
    state["planned_iterations"], state["stop_after_iterations"] = iterations, stop_after
    _save_progress(trainer, body, state, optimizer)
    for step in range(state["next_step"], end_step):
        before = float(beta)
        body.apply(trainer, before)
        loss_before = geometry_loss(trainer)
        gradients = {}
        for key, epsilon in context["material_finite_differences"].items():
            parameter = trainer.torch_param[key]
            original = float(parameter)
            parameter.fill_(original + epsilon)
            perturbed = float(parameter)
            assert perturbed > original, f"Material finite difference is too small for {key}"
            gradients[key] = (geometry_loss(trainer) - loss_before) / (perturbed - original)
            parameter.fill_(original)
        minus = float(np.float32(before - finite_difference))
        plus = float(np.float32(before + finite_difference))
        assert minus < before < plus, "Beta finite-difference epsilon is too small"
        body.apply(trainer, minus)
        loss_minus = geometry_loss(trainer)
        body.apply(trainer, plus)
        loss_plus = geometry_loss(trainer)
        beta_gradient = (loss_plus - loss_minus) / (plus - minus)
        assert all(np.isfinite(value) for value in (*gradients.values(), beta_gradient)), "Nonfinite joint derivative"
        optimizer.zero_grad(set_to_none=True)
        trainer.optimizer.zero_grad(set_to_none=True)
        beta.grad = torch.tensor(beta_gradient, dtype=beta.dtype)
        for key, parameter in trainer.torch_param.items():
            parameter.grad = torch.tensor(gradients[key], dtype=parameter.dtype, device=parameter.device)
        optimizer.step()
        trainer.optimizer.step()
        trainer.scheduler.step()
        for key, parameter in trainer.torch_param.items():
            parameter.clamp_(min=float(trainer.param_ranges[key][0]), max=float(trainer.param_ranges[key][-1]))
        body.apply(trainer, float(beta))
        evaluated = {"step": step + 1, "beta": float(beta), **_material_values(trainer), "loss": geometry_loss(trainer)}
        state["last"] = evaluated
        if evaluated["loss"] < state["best"]["loss"]:
            state["best"] = evaluated.copy()
        state["next_step"] = step + 1
        state["history"].append({**evaluated, "beta_before": before, "loss_before": loss_before,
                                 "loss_minus": loss_minus, "loss_plus": loss_plus, "gradient_beta": beta_gradient,
                                 **{f"gradient_{key}": value for key, value in gradients.items()}})
        _save_progress(trainer, body, state, optimizer)
        print(f"Joint step {step + 1}/{iterations}: beta[{body.beta_index}]={float(beta):.9g} "
              f"D={evaluated['D']:.9g} E={evaluated['E']:.9g} H={evaluated['H']:.9g} "
              f"loss={evaluated['loss']:.9g} best_beta={state['best']['beta']:.9g}", flush=True)
    body.apply(trainer, state["best"]["beta"])
    for key, parameter in trainer.torch_param.items():
        parameter.fill_(state["best"][key] / (100.0 if key == "E" else 1.0))
    print(f"Joint completed {state['next_step']}/{iterations} planned updates (stop-after {end_step})", flush=True)
    print(f"Joint fit: best beta[{body.beta_index}]={state['best']['beta']:.9g}; {root / 'joint_summary.json'}", flush=True)
