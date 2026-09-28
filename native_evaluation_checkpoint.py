"""Preserve native cloth MPM evaluation at completed frame boundaries."""
from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
    from warp_mpm.mpm_data_structure import MPMStateStruct
    from warp_mpm.mpm_solver import MPMWARP

# Template command (MPMAvatar directory, allocated A40, mpmavatar environment;
# workspace slurm/native_bin on PATH for Blender):
# python train_material_params.py --save_name s190_t2 --trained_model_path ../data/MPMAvatar/4DDress/examples/s190_t2/output/tracking/s190_t2_11_100 --model_path ../data/MPMAvatar/4DDress/examples/s190_t2/model/s190_t2 --dataset_dir ../data/MPMAvatar/4DDress/examples/s190_t2/data --output_dir ../data/MPMAvatar/4DDress/examples/s190_t2/output/phys --smplx_gender female --subject 190 --train_take 2 --test_take 5 --verts_start_idx 11 --split_idx_path ../data/MPMAvatar/4DDress/examples/s190_t2/data/s190_t2/split_idx.npz --dataset_type 4ddress --uv_path ../data/MPMAvatar/4DDress/examples/s190_t2/data/s190_t2/mesh_processed.obj --test_camera_index 0 --train_frame_start_num 19 2 --test_frame_start_num 11 100 --grid_size 200 --substep 400 --init_params_path ../data/MPMAvatar/4DDress/examples/s190_t2/output/phys/s190_t2/seed0/best_param_00199.npz --run_eval --checkpoint_eval

PARTICLE_KEYS = ("particle_x", "particle_v", "particle_C", "particle_d", "particle_R_inv")


def _validate_context(context: dict[str, Any], solver: MPMWARP) -> None:
    assert context["dataset_type"] == "4ddress"
    assert context["num_processes"] == 1
    assert context["n_traditional"] == 0
    assert context["num_frames"] >= 1
    assert solver.n_particles == solver.n_elements + solver.n_vertices


def save_evaluation_state(
    path: Path,
    next_frame: int,
    context: dict[str, Any],
    state: MPMStateStruct,
    solver: MPMWARP,
) -> None:
    """Save the state after the OBJ for ``next_frame - 1`` was committed."""
    _validate_context(context, solver)
    assert 1 <= next_frame <= context["num_frames"]
    wp.synchronize_device(state.particle_x.device)
    particles = {
        key: wp.to_torch(getattr(state, key)).detach().cpu().clone()
        for key in PARTICLE_KEYS
    }
    device = wp.to_torch(state.particle_x).device
    checkpoint = {
        "context": context,
        "next_frame": next_frame,
        "time": solver.time,
        "particles": particles,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_evaluation_state(
    path: Path,
    context: dict[str, Any],
    state: MPMStateStruct,
    solver: MPMWARP,
) -> int:
    """Restore persistent cloth state, then restore all RNG state last."""
    _validate_context(context, solver)
    assert path.is_file(), f"Native evaluation checkpoint required: {path}"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["context"] == context, "Native evaluation context changed"
    next_frame = checkpoint["next_frame"]
    assert 1 <= next_frame <= context["num_frames"]
    wp.synchronize_device(state.particle_x.device)
    device = wp.to_torch(state.particle_x).device
    particles = checkpoint["particles"]
    for key in PARTICLE_KEYS:
        current = wp.to_torch(getattr(state, key))
        assert particles[key].shape == current.shape, key
        assert particles[key].dtype == current.dtype, key
    state.continue_from_torch(
        tensor_x=particles["particle_x"].to(device),
        tensor_velocity=particles["particle_v"].to(device),
        tensor_d=particles["particle_d"].to(device),
        tensor_C=particles["particle_C"].to(device),
        tensor_R_inv=particles["particle_R_inv"].to(device),
        device=str(device),
        requires_grad=False,
    )
    solver.time = checkpoint["time"]
    wp.synchronize_device(state.particle_x.device)
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if device.type == "cuda":
        assert len(rng["cuda"]) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all(rng["cuda"])
    else:
        assert rng["cuda"] == []
    return next_frame
