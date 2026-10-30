"""Local controlled-shear hang-heavy sweep with substep traces and replay checkpoints.

Activate mpmavatar; from MPMAvatar:
# python sweep_hang_e.py --stage center --output output/hang_heavy_E_control_h1e-6_r10 --offsets 0 --stride 10 --snapshot-stride 400
# python sweep_hang_e.py --stage target --output output/hang_heavy_E_control_h1e-6_r10 --offsets 0 --stride 10 --snapshot-stride 400
# python sweep_hang_e.py --stage points --output output/hang_heavy_E_control_h1e-6_r10 --offsets -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 1 2 3 4 5 6 7 8 9 10 --stride 10 --snapshot-stride 400
# python sweep_hang_e.py --stage plot --output output/hang_heavy_E_control_h1e-6_r10 --offsets 0 --stride 10 --snapshot-stride 400
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from hang_local_sweep import HangSimulation, save


def case_path(output: Path, k: int) -> Path:
    return output / "runs" / f"k{k:+03d}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert rows
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cutoff_figure(path: Path) -> Any:
    import matplotlib.pyplot as plt

    data = json.loads(path.read_text())
    original, coarse, fine = data["brackets"]
    figure = plt.figure(figsize=(12, 8), facecolor="white")
    figure.text(0.055, 0.94, "Remaining small steps: the grid-mass cutoff", fontsize=21, weight="bold")
    figure.text(0.055, 0.885,
                f"Controlled shear release; substep {data['step']} (t = {data['time']:.4f} s); "
                f"grid node {tuple(data['grid_node'])}.", fontsize=13)
    figure.text(0.055, 0.825, r"$\mathbf{v}_i^+=\mathbf{p}_i/m_i+\Delta t\,\mathbf{g}\qquad(m_i>m_c)$", fontsize=20)
    figure.text(0.055, 0.765, r"$\mathbf{v}_i^+=\mathbf{0}\qquad(m_i\leq m_c),\qquad m_c=10^{-15}$", fontsize=20)
    figure.text(0.055, 0.715, "The grid is zeroed first. The cutoff decides whether P2G momentum is normalized.", fontsize=12)
    figure.text(0.055, 0.655,
                f"k=0 / +1 node masses: {original['mass_a']:.12e} / {original['mass_b']:.12e}", fontsize=13)
    figure.text(0.055, 0.61,
                "Node velocity: (0, 0, 0) -> (0.004922, -0.175880, -0.054632) simulation units/s.", fontsize=13)
    figure.text(0.055, 0.555, "Narrowing the input bracket retains the finite output jump", fontsize=15, weight="bold")
    axis = figure.add_axes((0.055, 0.325, 0.89, 0.19))
    axis.set_axis_off()
    rows = [[r["bracket"], f"{r['delta_alpha']:.2g}", f"{r['delta_E']:.3e}",
             f"{r['grid_v_jump_max']:.9f}", f"{r['output_C_max']:.7f}", f"{r['output_x_max_sim']:.3e}"]
            for r in (coarse, fine)]
    table = axis.table(cellText=rows,
                       colLabels=["Bracket", "alpha span", "E span", "Max grid velocity jump", "Max APIC C difference", "Max position difference"],
                       loc="center", cellLoc="center", colWidths=[.1, .12, .14, .22, .22, .2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    figure.text(0.055, 0.285, "Velocity and position use simulation units; APIC C has units 1/s.", fontsize=11)
    figure.text(0.055, 0.23,
                "Local bracket: interpolate incoming x/v/C/d and E between the measured k=0/+1 states.\n"
                "This is a one-step continuity test, not another full E sweep.", fontsize=12, linespacing=1.6)
    figure.text(0.055, 0.14,
                f"The alpha span shrinks 1000x. Input position difference shrinks from {coarse['input_x_max_sim']:.2e}\n"
                f"to {fine['input_x_max_sim']:.2e}; the grid-velocity and APIC-gradient jumps persist.",
                fontsize=12, linespacing=1.6)
    figure.text(0.055, 0.085,
                "Also confirmed in k=-4/-3: substeps 4019 / 4199; grid velocity switches of 0.531 / 0.569 sim units/s.",
                fontsize=10.5)
    figure.text(0.055, 0.045,
                "Source: warp_mpm/mpm_utils.py, grid_normalization_and_gravity. No cutoff or model fix applied.",
                fontsize=11, color="0.35")
    return figure


def plot(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    config = json.loads((output / "config.json").read_text())
    with np.load(case_path(output, 0) / "trajectory.npz") as archive:
        center = archive["trajectory"]
    with np.load(output / "target" / "trajectory.npz") as archive:
        target = archive["trajectory"]
    radius, step = config["radius"], config["step_sizes"][0]
    trajectories, rows = {}, []
    for k in range(-radius, radius + 1):
        for path in case_path(output, k).glob("trajectory.npz"):
            with np.load(path) as archive:
                x, young = archive["trajectory"], float(archive["E"])
            trajectories[k] = x
            rows.append({"k": k, "ln_E_offset": k * step, "E": young,
                         "loss_center": float(np.mean((x - center) ** 2)),
                         "loss_target": float(np.mean((x - target) ** 2)),
                         "max_vertex_center_mm": float(1000 * np.linalg.norm(x - center, axis=-1).max())})
    write_csv(output / "points.csv", rows)
    row_by_k = {row["k"]: row for row in rows}
    adjacent = []
    for k in range(-radius, radius):
        if k in trajectories and k + 1 in trajectories:
            distance = np.linalg.norm(trajectories[k + 1] - trajectories[k], axis=-1)
            adjacent.append({"left": k, "right": k + 1,
                             "trajectory_rms_mm": float(1000 * np.sqrt(np.mean(distance ** 2))),
                             "peak_frame_rms_mm": float(1000 * np.sqrt(np.mean(distance ** 2, axis=1)).max()),
                             "max_vertex_mm": float(1000 * distance.max()),
                             "delta_loss_target": row_by_k[k + 1]["loss_target"] - row_by_k[k]["loss_target"]})
    if adjacent:
        write_csv(output / "adjacent.csv", adjacent)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ks = np.arange(-radius, radius + 1)
    for axis, metric, title in ((axes[0, 0], "loss_target", "Loss to target (E = 200)"),
                                (axes[0, 1], "loss_center", "Loss to center (E = 100)")):
        values = np.full(len(ks), np.nan)
        for row in rows:
            values[row["k"] + radius] = row[metric]
        axis.plot(ks, values, "o-", markersize=4)
        axis.set_title(title)
        axis.set_ylabel("Mean xyz squared distance (m²)")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
        axis.set_xlabel("k")
    values = np.full(2 * radius, np.nan)
    for row in adjacent:
        values[row["left"] + radius] = 1000 * row["trajectory_rms_mm"]
    axes[1, 0].plot(ks[:-1] + 0.5, values, "o-", markersize=4)
    axes[1, 0].set_title("Adjacent trajectories")
    axes[1, 0].set_xlabel("Pair midpoint k + 0.5")
    axes[1, 0].set_ylabel("Trajectory RMS separation (µm)")
    for row in sorted(adjacent, key=lambda r: r["trajectory_rms_mm"], reverse=True)[:3]:
        a, b = row["left"], row["right"]
        distance = trajectories[b] - trajectories[a]
        rms = np.sqrt(np.mean(np.sum(distance ** 2, axis=-1), axis=1))
        axes[1, 1].plot(np.arange(len(rms)) / config["native"]["frame_rate"], 1e6 * rms,
                        label=f"{a:+d} / {b:+d}")
    axes[1, 1].set_title("Largest adjacent separations")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Frame RMS separation (µm)")
    if adjacent:
        axes[1, 1].legend()
    if "comparison_baseline" in config:
        baseline = Path(config["comparison_baseline"])
        with (baseline / "points.csv").open() as stream:
            reference_points = list(csv.DictReader(stream))
        with (baseline / "adjacent.csv").open() as stream:
            reference_pairs = list(csv.DictReader(stream))
        for axis, metric in ((axes[0, 0], "loss_target"), (axes[0, 1], "loss_center")):
            axis.lines[0].set_label("C2 band")
            reference_values = np.array([float(row[metric]) for row in reference_points])
            if metric == "loss_target":
                axis.lines[0].set_ydata(axis.lines[0].get_ydata() - row_by_k[0][metric])
                reference_values -= next(float(row[metric]) for row in reference_points if int(row["k"]) == 0)
            axis.plot([int(row["k"]) for row in reference_points],
                      reference_values, "--", color="0.45", label="Hard cutoff")
            axis.legend()
        axes[0, 0].relim()
        axes[0, 0].autoscale_view()
        axes[0, 0].set_title("Target-loss change from each model's k=0")
        axes[0, 0].set_ylabel("Change in mean xyz squared distance (m²)")
        axes[1, 0].lines[0].set_label("C2 band")
        axes[1, 0].plot([int(row["left"]) + .5 for row in reference_pairs],
                        [1000 * float(row["trajectory_rms_mm"]) for row in reference_pairs],
                        "--", color="0.45", label="Hard cutoff")
        axes[1, 0].legend()
        with np.load(baseline / "runs/k-04/trajectory.npz") as a, np.load(baseline / "runs/k-03/trajectory.npz") as b:
            delta = b["trajectory"] - a["trajectory"]
        rms = np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=1))
        axes[1, 1].plot(np.arange(len(rms)) / config["native"]["frame_rate"], 1e6 * rms,
                        "--", color="0.45", label="Hard -4/-3")
        axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    normalization = f"C2 mass band {config['mass_band']}" if "mass_band" in config else "hard mass cutoff"
    figure.suptitle(f"Controlled-shear hang-heavy: ln(E) = ln(100) + k × {step:g}, |k| ≤ {radius}\n"
                   f"{len(rows)}/{2 * radius + 1} points; fp64 deterministic; 400 substeps/frame; {normalization}")
    temporary = output / f"sweep.{os.getpid()}.tmp.png"
    figure.savefig(temporary, dpi=170)
    temporary.replace(output / "sweep.png")
    temporary = output / f"sweep.{os.getpid()}.tmp.pdf"
    with PdfPages(temporary) as pdf:
        pdf.savefig(figure)
        if "mass_band" in config:
            from probe_hang_mass_band import center_figure, equation_figure

            equation = equation_figure()
            pdf.savefig(equation)
            plt.close(equation)
            centers = center_figure(output.parent.parent, Path(config["comparison_baseline"]))
            pdf.savefig(centers)
            plt.close(centers)
        for path in (output / "diagnosis").glob("mass_cutoff.json"):
            diagnostic = cutoff_figure(path)
            pdf.savefig(diagnostic)
            diagnostic.savefig(output / "diagnosis" / "mass_cutoff.png", dpi=170)
            plt.close(diagnostic)
    temporary.replace(output / "sweep.pdf")
    plt.close(figure)
    print(f"Plot updated: {len(rows)}/{2 * radius + 1} points, {len(adjacent)} adjacent pairs", flush=True)


def run(output: Path, stage: str, ks: list[int], stride: int, snapshot_stride: int) -> None:
    config = json.loads((output / "config.json").read_text())
    assert len(config["step_sizes"]) == 1 and config["precision"] == "fp64" and config["deterministic"]
    assert config["mapping"] == "release_tension_shear"
    simulation = HangSimulation(config)
    import torch
    import warp as wp
    from hang_mapping_control import compute_cloth_stress
    from warp_mpm.mpm_solver import compute_stress_from_F_trial
    from warp_mpm.mpm_solver import grid_normalization_and_gravity

    band = config.get("mass_band", [])
    assert not band or len(band) == 2 and 0 < band[0] < band[1]
    if band:
        from hang_mass_band import grid_normalization_mass_band

    native_launch = wp.launch
    band_launches = 0

    def launch(*launch_args: Any, **kwargs: Any) -> Any:
        nonlocal band_launches
        if kwargs["kernel"] is compute_stress_from_F_trial:
            kwargs["kernel"] = compute_cloth_stress
        if band and kwargs["kernel"] is grid_normalization_and_gravity:
            kwargs["kernel"] = grid_normalization_mass_band
            kwargs["inputs"] = [*kwargs["inputs"], *band]
            band_launches += 1
        return native_launch(*launch_args, **kwargs)

    wp.launch = launch

    assert simulation.init_pos.dtype == torch.float64
    (output / f"runtime_{os.getpid()}.json").write_text(json.dumps({"torch": torch.__version__,
        "warp": wp.__version__, "device": torch.cuda.get_device_name(), "mode": wp.config.deterministic.name,
        "precision": "fp64", "mapping": config["mapping"], "mass_band": band, "stage": stage, "offsets": ks}, indent=2) + "\n")
    steps = config["frames"] * config["native"]["substeps"]
    assert steps % stride == 0
    dt = 1.0 / (config["native"]["frame_rate"] * config["native"]["substeps"])
    if stage == "center":
        shape = (steps // stride + 1, simulation.n_vertices, 3)
        reference_x = np.lib.format.open_memmap(output / "center_x.npy", mode="w+", dtype=np.float64, shape=shape)
        reference_v = np.lib.format.open_memmap(output / "center_v.npy", mode="w+", dtype=np.float64, shape=shape)
    elif stage == "points":
        assert (case_path(output, 0) / "trajectory.npz").is_file()
        reference_x = np.load(output / "center_x.npy", mmap_mode="r")
        reference_v = np.load(output / "center_v.npy", mmap_mode="r")
    for k in ks:
        assert abs(k) <= config["radius"]
        directory = output / "target" if stage == "target" else case_path(output, k)
        assert not (directory / "trajectory.npz").exists(), directory
        directory.mkdir(parents=True, exist_ok=True)
        young = config["target"]["E"] if stage == "target" else config["center"]["E"] * math.exp(k * config["step_sizes"][0])
        print(f"Controlled shear {directory.name}: E={young!r}", flush=True)
        with (directory / "trace.csv").open("w", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "time", "rms_x_m", "max_x_m", "rms_v_m_s", "max_v_m_s"])
            writer.writeheader()

            def observe(step: int) -> None:
                if stage != "target" and step % snapshot_stride == 0:
                    save(directory / f"state_{step:06d}.npz", step=step, time=simulation.solver.time, E=young,
                         **{name: getattr(simulation.state, name).numpy()
                            for name in ("particle_x", "particle_v", "particle_C", "particle_d")})
                if step % stride != 0:
                    return
                x = wp.to_torch(simulation.state.particle_x)[simulation.n_elements:].cpu().numpy()
                v = wp.to_torch(simulation.state.particle_v)[simulation.n_elements:].cpu().numpy()
                slot = step // stride
                if stage == "center":
                    reference_x[slot], reference_v[slot] = x, v
                if stage != "target":
                    dx, dv = (x - reference_x[slot]) / simulation.scale, (v - reference_v[slot]) / simulation.scale
                row = {"step": step, "time": step * dt,
                       "rms_x_m": float(np.sqrt(np.mean(np.sum(dx * dx, axis=1)))) if stage != "target" else "",
                       "max_x_m": float(np.linalg.norm(dx, axis=1).max()) if stage != "target" else "",
                       "rms_v_m_s": float(np.sqrt(np.mean(np.sum(dv * dv, axis=1)))) if stage != "target" else "",
                       "max_v_m_s": float(np.linalg.norm(v, axis=1).max() / simulation.scale)}
                writer.writerow(row)
                if step % config["native"]["substeps"] == 0:
                    print(f"{directory.name} step {step}/{steps}: "
                          f"speed {row['max_v_m_s']:.6g} m/s", flush=True)

            launches_before = band_launches
            trajectory = simulation.rollout({**config["center"], "E": young}, observer=observe)
            assert not band or band_launches - launches_before == steps
        if stage == "center":
            reference_x.flush()
            reference_v.flush()
        save(directory / "trajectory.npz", trajectory=trajectory, E=young, k=k, mapping=config["mapping"], mass_band=band)
        print(f"Completed {directory.name}", flush=True)
        if stage == "points":
            with (output / "plot.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                plot(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("center", "target", "points", "plot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", nargs="+", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--snapshot-stride", type=int, required=True)
    args = parser.parse_args()
    assert (args.output / "config.json").is_file() and args.stride > 0 and args.snapshot_stride > 0
    if args.stage == "plot":
        with (args.output / "plot.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            plot(args.output)
    else:
        assert args.stage == "points" or args.offsets == [0]
        run(args.output, args.stage, args.offsets, args.stride, args.snapshot_stride)


if __name__ == "__main__":
    main()
