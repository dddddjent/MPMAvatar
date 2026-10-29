"""Compare native hang-heavy substeps with identical repeats and tiny E changes.

Activate mpmavatar; from MPMAvatar:
# python trace_hang_e.py --config output/hang_heavy_E_trace/config.json --output output/hang_heavy_E_trace/trace --frames 75 --stride 10 --step-size 1e-12 --direction log --offsets 0 -2 -1 1 2 --substeps 400 --snapshot-stride 400
# python trace_hang_e.py --config output/hang_heavy_E_trace/config.json --output output/hang_heavy_E_trace/release_control --frames 75 --stride 10 --step-size 1e-12 --direction log --offsets -1 1 --substeps 400 --snapshot-stride 400 --release-tension-shear
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hang_local_sweep import HangSimulation, save


def plot(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for path in sorted(output.glob("run_*/trace.csv")):
        data = np.genfromtxt(path, delimiter=",", names=True)
        label = path.parent.name
        axes[0].semilogy(data["time"], np.maximum(data["rms_x_m"], 1e-18), label=label)
        axes[1].semilogy(data["time"], np.maximum(data["max_v_m_s"], 1e-18), label=label)
    axes[0].set_ylabel("RMS vertex separation from baseline (m)")
    axes[1].set_ylabel("Maximum vertex speed (m/s)")
    for axis in axes:
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(output / f"divergence.{extension}", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--step-size", type=float, required=True)
    parser.add_argument("--direction", choices=("log", "absolute"), required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--snapshot-stride", type=int, required=True)
    parser.add_argument("--release-tension-shear", action="store_true", help="Diagnostic continuous return-map control")
    args = parser.parse_args()
    assert args.config.is_file() and args.frames > 0 and args.stride > 0 and args.snapshot_stride > 0
    config = json.loads(args.config.read_text())
    config["frames"] = args.frames
    config["native"]["substeps"] = args.substeps
    steps = args.frames * config["native"]["substeps"]
    assert steps % args.stride == 0
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({**config, "trace": vars(args)}, default=str, indent=2) + "\n")
    simulation = HangSimulation(config)
    import torch
    import warp as wp

    assert simulation.init_pos.dtype == torch.float64
    if args.release_tension_shear:
        from hang_mapping_control import compute_cloth_stress
        from warp_mpm.mpm_solver import compute_stress_from_F_trial

        native_launch = wp.launch

        def launch(*launch_args: Any, **kwargs: Any) -> Any:
            if kwargs["kernel"] is compute_stress_from_F_trial:
                kwargs["kernel"] = compute_cloth_stress
            return native_launch(*launch_args, **kwargs)

        wp.launch = launch
    (args.output / "runtime.json").write_text(json.dumps({"torch": torch.__version__, "warp": wp.__version__,
                                                        "device": torch.cuda.get_device_name(),
                                                        "deterministic": wp.config.deterministic.name,
                                                        "position_dtype": str(simulation.state.particle_x.dtype),
                                                        "E_dtype": str(simulation.model.E.dtype)}, indent=2) + "\n")
    shape = (steps // args.stride + 1, simulation.n_vertices, 3)
    reference_x = np.lib.format.open_memmap(args.output / "baseline_x.npy", mode="w+", dtype=np.float64, shape=shape)
    reference_v = np.lib.format.open_memmap(args.output / "baseline_v.npy", mode="w+", dtype=np.float64, shape=shape)
    dt = 1.0 / (config["native"]["frame_rate"] * config["native"]["substeps"])
    summary: list[dict[str, Any]] = []
    reference_trajectory = np.empty((0,))
    for index, offset in enumerate([0, *args.offsets]):
        young = config["center"]["E"]
        young = young * math.exp(offset * args.step_size) if args.direction == "log" else young + offset * args.step_size
        run = args.output / f"run_{index:02d}_k{offset:+d}"
        run.mkdir(exist_ok=True)
        rows: list[dict[str, Any]] = []
        print(f"Starting {run.name}: E={young!r}, internal E={young * 100!r}", flush=True)
        with (run / "trace.csv").open("w", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "time", "rms_x_m", "max_x_m", "rms_v_m_s",
                                                       "max_v_m_s", "different_x_components"])
            writer.writeheader()

            def observe(step: int) -> None:
                if step % args.snapshot_stride == 0:
                    save(run / f"state_{step:06d}.npz", step=step, time=simulation.solver.time, E=young,
                         **{name: getattr(simulation.state, name).numpy()
                            for name in ("particle_x", "particle_v", "particle_C", "particle_d")})
                if step % args.stride != 0:
                    return
                x = wp.to_torch(simulation.state.particle_x)[simulation.n_elements:].cpu().numpy()
                v = wp.to_torch(simulation.state.particle_v)[simulation.n_elements:].cpu().numpy()
                slot = step // args.stride
                if index == 0:
                    reference_x[slot], reference_v[slot] = x, v
                dx = (x - reference_x[slot]) / simulation.scale
                dv = (v - reference_v[slot]) / simulation.scale
                row = {"step": step, "time": step * dt, "rms_x_m": float(np.sqrt(np.mean(np.sum(dx * dx, axis=1)))),
                       "max_x_m": float(np.linalg.norm(dx, axis=1).max()),
                       "rms_v_m_s": float(np.sqrt(np.mean(np.sum(dv * dv, axis=1)))),
                       "max_v_m_s": float(np.linalg.norm(v, axis=1).max() / simulation.scale),
                       "different_x_components": int(np.count_nonzero(x != reference_x[slot]))}
                writer.writerow(row)
                rows.append(row)
                if step % config["native"]["substeps"] == 0:
                    print(f"{run.name} step {step}/{steps}: dx RMS {row['rms_x_m']:.6g} m, "
                          f"speed {row['max_v_m_s']:.6g} m/s", flush=True)

            trajectory = simulation.rollout({**config["center"], "E": young}, observer=observe)
        if index == 0:
            reference_trajectory = trajectory
            reference_x.flush()
            reference_v.flush()
        save(run / "trajectory.npz", trajectory=trajectory, E=young, offset=offset, device=torch.cuda.get_device_name())
        entry = {"run": run.name, "E": young, "internal_E": young * 100, "delta_E": young - config["center"]["E"],
                 "loss_center": float(np.mean((trajectory - reference_trajectory) ** 2)),
                 "max_sampled_rms_x_m": max(row["rms_x_m"] for row in rows),
                 "max_sampled_x_m": max(row["max_x_m"] for row in rows),
                 "different_samples": sum(row["different_x_components"] > 0 for row in rows)}
        summary.append(entry)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        plot(args.output)
        print(entry, flush=True)


if __name__ == "__main__":
    main()
