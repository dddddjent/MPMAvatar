"""Save native tracking at frame boundaries without storing its dense Laplacian."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Template command (mpmavatar, prepared subject's preprocess directory):
# python /projects/bivb/junlinl6/Documents/clothes-reconstruction/MPMAvatar/preprocess/train_mesh_lbs_4ddress.py --exp_name tracking --seq s191_t2 --save_name s191_t2_21_100 --start_idx 21 --num_frames 100 --labels 3 --data_path ../data/4D-DRESS/00191_Inner/Inner/Take2 --resume

PARAMETER_KEYS = ("vertices", "rgb_colors", "logit_opacities", "log_scales", "cam_m", "cam_c")
MUTABLE_VARIABLE_KEYS = (
    "unnorm_rotations", "max_2D_radius", "face_neighbors", "neighbor_weight", "neighbor_dist",
)


def training_context(args: argparse.Namespace) -> dict[str, Any]:
    logging_keys = {"resume", "wandb", "wandb_proj", "wandb_entity", "wandb_name"}
    return {key: value for key, value in vars(args).items() if key not in logging_keys}


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    next_t: int,
    params: dict[str, torch.Tensor],
    variables: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    beta: torch.Tensor | None,
) -> None:
    end_t = args.start_idx + args.num_frames
    assert args.start_idx <= next_t <= end_t
    state = {
        "context": training_context(args), "next_t": next_t, "end_t": end_t,
        "params": {key: params[key].detach().cpu().clone() for key in PARAMETER_KEYS},
        "variables": {key: variables[key].detach().cpu().clone() for key in MUTABLE_VARIABLE_KEYS},
        "optimizer": optimizer.state_dict(),
        "beta": None if beta is None else beta.detach().cpu().clone(),
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if params["vertices"].is_cuda else [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(
    path: Path,
    args: argparse.Namespace,
    params: dict[str, torch.Tensor],
    variables: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> tuple[int, torch.Tensor | None]:
    assert path.is_file(), f"Frame-boundary checkpoint required: {path}"
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["context"] == training_context(args), "Tracking options changed since the checkpoint"
    end_t = args.start_idx + args.num_frames
    assert state["end_t"] == end_t
    next_t = state["next_t"]
    assert args.start_idx <= next_t <= end_t
    with torch.no_grad():
        for key in PARAMETER_KEYS:
            params[key].copy_(state["params"][key])
    for key in MUTABLE_VARIABLE_KEYS:
        variables[key] = state["variables"][key].to(variables[key].device)
    optimizer.load_state_dict(state["optimizer"])
    beta = state["beta"]
    if next_t > args.start_idx:
        assert isinstance(beta, torch.Tensor), "Completed frames require fixed SMPL-X betas"
        beta = beta.to(params["vertices"].device)
    else:
        assert beta is None
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if params["vertices"].is_cuda:
        assert state["rng"]["cuda"], "CUDA random state is required for GPU continuation"
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    else:
        assert not state["rng"]["cuda"]
    return next_t, beta
