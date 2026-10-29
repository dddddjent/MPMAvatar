"""Replay two saved hang states through the native kernels and compare each stage.

Activate mpmavatar; from MPMAvatar (E arguments use the exposed parameter):
# python replay_hang_e.py --config output/hang_heavy_E_trace/config.json --state-a output/hang_heavy_E_trace/trace/run_00_k+0/state_000100.npz --state-b output/hang_heavy_E_trace/trace/run_03_k-1/state_000100.npz --E-a 100 --E-b 99.9999999999 --substeps 400 --steps 1 --output output/hang_heavy_E_trace/replay
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hang_local_sweep import HangSimulation, save

FIELDS = ("particle_x", "particle_v", "particle_C", "particle_d", "particle_stress", "vertex_force",
          "grid_m", "grid_v_in", "grid_v_out")


def differences(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    delta = a - b
    return {"rms": float(np.sqrt(np.mean(delta * delta))), "max_abs": float(np.abs(delta).max()),
            "different_components": int(np.count_nonzero(a != b)), "max_abs_a": float(np.abs(a).max()),
            "max_abs_b": float(np.abs(b).max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-a", type=Path, required=True)
    parser.add_argument("--state-b", type=Path, required=True)
    parser.add_argument("--E-a", type=float, required=True)
    parser.add_argument("--E-b", type=float, required=True)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.config.is_file() and args.state_a.is_file() and args.state_b.is_file()
    assert args.steps > 0 and args.substeps > 0
    config = json.loads(args.config.read_text())
    config["frames"] = 0
    config["native"]["substeps"] = args.substeps
    checkpoints = []
    for path in (args.state_a, args.state_b):
        with np.load(path) as archive:
            checkpoints.append(dict(archive))
    assert checkpoints[0]["step"] == checkpoints[1]["step"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    simulation = HangSimulation(config)
    import torch
    import warp as wp

    dt = 1.0 / (config["native"]["frame_rate"] * args.substeps)
    # Every grid node read by either replay's particle support during the first substep.
    shifts = np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"), axis=-1).reshape(-1, 3)
    positions = np.concatenate([checkpoint["particle_x"] for checkpoint in checkpoints])
    bases = (positions * simulation.model.inv_dx - 0.5).astype(np.int32)
    nodes = np.unique((bases[:, None, :] + shifts).reshape(-1, 3), axis=0)
    assert nodes.min() >= 0 and nodes.max() < simulation.native["grid_size"]
    indices = torch.tensor(nodes, dtype=torch.long, device=simulation.device)

    def sample() -> dict[str, np.ndarray]:
        values = {}
        for name in FIELDS:
            tensor = wp.to_torch(getattr(simulation.state, name))
            if name.startswith("grid_"):
                tensor = tensor[indices[:, 0], indices[:, 1], indices[:, 2]]
            values[name] = tensor.cpu().numpy().copy()
        return values

    stage_reference: list[dict[str, np.ndarray]] = []
    step_reference: list[dict[str, np.ndarray]] = []
    stage_rows, step_rows = [], []
    launch_native = wp.launch
    for run, (checkpoint, young) in enumerate(zip(checkpoints, (args.E_a, args.E_b))):
        simulation.rollout({**config["center"], "E": young})
        tensors = {name: torch.tensor(checkpoint[name], device=simulation.device)
                   for name in ("particle_x", "particle_v", "particle_C", "particle_d")}
        simulation.state.continue_from_torch(tensors["particle_x"], tensors["particle_v"], tensors["particle_d"],
                                             tensors["particle_C"], device=simulation.device, requires_grad=False)
        simulation.solver.time = float(checkpoint["time"])
        stage_index = 0

        def launch(*launch_args: Any, **kwargs: Any) -> Any:
            nonlocal stage_index
            result = launch_native(*launch_args, **kwargs)
            values = sample()
            kernel = kwargs["kernel"]
            if run == 0:
                stage_reference.append(values)
            else:
                for name in FIELDS:
                    stage_rows.append({"stage": stage_index, "kernel": kernel.key, "field": name,
                                       **differences(stage_reference[stage_index][name], values[name])})
            stage_index += 1
            return result

        with torch.no_grad():
            for step in range(args.steps + 1):
                values = {name: getattr(simulation.state, name).numpy()
                          for name in ("particle_x", "particle_v", "particle_C", "particle_d")}
                if run == 0:
                    step_reference.append(values)
                else:
                    for name in values:
                        step_rows.append({"step": int(checkpoint["step"]) + step, "field": name,
                                          **differences(step_reference[step][name], values[name])})
                if step == args.steps:
                    break
                # Observe only the first substep; native kernel code and inputs are unchanged.
                wp.launch = launch if step == 0 else launch_native
                simulation.solver.p2g2p(simulation.model, simulation.state, dt,
                                        joint_verts_v=torch.zeros((simulation.n_pinned, 3), dtype=torch.float64,
                                                                  device=simulation.device),
                                        joint_faces_v=torch.zeros((0, 3), dtype=torch.float64,
                                                                  device=simulation.device), device=simulation.device)
                wp.launch = launch_native
        save(args.output / f"result_{run}.npz", step=int(checkpoint["step"]) + args.steps,
             time=simulation.solver.time, E=young, **values)
    for name, rows in (("stages", stage_rows), ("steps", step_rows)):
        with (args.output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Compared {len(stage_reference)} native launches, {len(nodes)} grid nodes, {args.steps} substeps", flush=True)
    for row in stage_rows:
        if row["max_abs"] > 1e-6:
            print(row, flush=True)


if __name__ == "__main__":
    main()
