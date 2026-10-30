"""Replay two saved hang states through the native kernels and compare each stage.

Activate mpmavatar; from MPMAvatar (E arguments use the exposed parameter):
# python replay_hang_e.py --config output/hang_heavy_E_trace/config.json --state-a output/hang_heavy_E_trace/trace/run_00_k+0/state_000100.npz --state-b output/hang_heavy_E_trace/trace/run_03_k-1/state_000100.npz --E-a 100 --E-b 99.9999999999 --substeps 400 --steps 1 --output output/hang_heavy_E_trace/replay
# python replay_hang_e.py --config output/hang_heavy_E_control_h1e-6_r10/config.json --state-a output/hang_heavy_E_control_h1e-6_r10/runs/k+00/state_000400.npz --state-b output/hang_heavy_E_control_h1e-6_r10/runs/k+01/state_000400.npz --E-a 100 --E-b 100.00010000005 --substeps 400 --steps 1 --output output/hang_heavy_E_control_h1e-6_r10/replay --release-tension-shear --capture-stages 2 3 4 8 12 13 14
# python replay_hang_e.py --config output/hang_heavy_E_control_h1e-6_r10/config.json --state-a output/hang_heavy_E_control_h1e-6_r10/diagnosis/pair_0_1_to2553/result_0.npz --state-b output/hang_heavy_E_control_h1e-6_r10/diagnosis/pair_0_1_to2553/result_1.npz --E-a 100 --E-b 100.00010000005 --substeps 400 --steps 1 --output output/hang_heavy_E_mass_band/default/pair_0_1 --release-tension-shear --mass-band 5e-16 2e-15 --capture-stages 4 13 14
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
    parser.add_argument("--release-tension-shear", action="store_true", help="Replay the existing controlled-shear model")
    parser.add_argument("--capture-stages", nargs="*", type=int, default=[], help="Save raw first-substep stage arrays")
    parser.add_argument("--mass-band", nargs=2, type=float, metavar=("LOWER", "UPPER"),
                        help="Use the explicit C2 grid-mass band instead of the native cutoff")
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
    from warp_mpm.mpm_solver import compute_stress_from_F_trial

    stress_kernel = compute_stress_from_F_trial
    if args.release_tension_shear:
        from hang_mapping_control import compute_cloth_stress

        stress_kernel = compute_cloth_stress

    from warp_mpm.mpm_solver import grid_normalization_and_gravity

    normalization_kernel = grid_normalization_and_gravity
    if args.mass_band is not None:
        assert 0 < args.mass_band[0] < args.mass_band[1]
        from hang_mass_band import grid_normalization_mass_band

        normalization_kernel = grid_normalization_mass_band

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
    band_launches = 0

    def launch_solver(*launch_args: Any, **kwargs: Any) -> Any:
        nonlocal band_launches
        if kwargs["kernel"] is compute_stress_from_F_trial:
            kwargs["kernel"] = stress_kernel
        if kwargs["kernel"] is grid_normalization_and_gravity:
            kwargs["kernel"] = normalization_kernel
            if args.mass_band is not None:
                kwargs["inputs"] = [*kwargs["inputs"], *args.mass_band]
                band_launches += 1
        return launch_native(*launch_args, **kwargs)
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
            kernel = stress_kernel if kwargs["kernel"] is compute_stress_from_F_trial else kwargs["kernel"]
            if kernel is grid_normalization_and_gravity:
                kernel = normalization_kernel
            result = launch_solver(*launch_args, **kwargs)
            values = sample()
            if stage_index in args.capture_stages:
                save(args.output / f"stage_{stage_index:02d}_run{run}.npz", nodes=nodes, **values)
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
                wp.launch = launch if step == 0 else launch_solver
                simulation.solver.p2g2p(simulation.model, simulation.state, dt,
                                        joint_verts_v=torch.zeros((simulation.n_pinned, 3), dtype=torch.float64,
                                                                  device=simulation.device),
                                        joint_faces_v=torch.zeros((0, 3), dtype=torch.float64,
                                                                  device=simulation.device), device=simulation.device)
                wp.launch = launch_native
        save(args.output / f"result_{run}.npz", step=int(checkpoint["step"]) + args.steps,
             time=simulation.solver.time, E=young, **values)
    assert args.mass_band is None or band_launches == 2 * args.steps, band_launches
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
