"""Counter-rotate two opposite edges of a square, then reverse the same motion.

Activate mpmavatar; from MPMAvatar:
# python twist_square.py --stage run --settings output/trough20_E_smooth_g200_s200/config.json --output output/square_twist --resolution 64 --degrees 360 --twist-seconds 3 --hold-seconds 1 --settle-seconds 1 --grid-size 200 --substeps 200
# MPLCONFIGDIR=/tmp/robustdiffcloth-matplotlib python twist_square.py --stage render --output output/square_twist --ffmpeg /usr/bin/ffmpeg --playback-fps 25
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def square(resolution: int) -> tuple[NDArray, NDArray, NDArray]:
    axis = np.linspace(-0.5, 0.5, resolution)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack([x.ravel(), np.zeros(resolution**2), z.ravel()], axis=1)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            faces.extend([(a, a + 1, a + resolution + 1), (a, a + resolution + 1, a + resolution)])
    grips = np.concatenate([np.arange(resolution), np.arange(resolution * (resolution - 1), resolution**2)])
    return vertices, np.asarray(faces, dtype=np.int32), grips


def angle_at(time: float, config: dict[str, Any]) -> tuple[float, str]:
    twist, hold = config["twist_seconds"], config["hold_seconds"]
    if time < twist:
        t, phase = time / twist, "Twist"
    elif time < twist + hold:
        t, phase = 1.0, "Hold twisted"
    elif time < 2 * twist + hold:
        t, phase = 1 - (time - twist - hold) / twist, "Untwist"
    else:
        t, phase = 0.0, "Hold restored grips"
    t = min(1.0, max(0.0, t))
    return math.radians(config["degrees"]) * (t**3 * (10 + t * (-15 + 6 * t))), phase


def grip_positions(rest: NDArray, grips: NDArray, angle: float) -> NDArray:
    positions = rest[grips].copy()
    angles = np.repeat([-angle, angle], len(grips) // 2)
    positions[:, 1] = -rest[grips, 2] * np.sin(angles)
    positions[:, 2] = rest[grips, 2] * np.cos(angles)
    return positions


def proximity(vertices: NDArray, rest: NDArray, cell: float) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    pairs = cKDTree(vertices).query_pairs(4 * cell, output_type="ndarray")
    far = np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1) > 0.15
    pairs = pairs[far]
    distances = np.linalg.norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]], axis=1)
    return {"nonlocal_pairs_4cells": len(pairs), "nonlocal_pairs_2cells": int((distances < 2 * cell).sum()),
            "nonlocal_pairs_1cell": int((distances < cell).sum()),
            "min_nonlocal_distance_m": float(distances.min()) if len(distances) else ""}


def run(args: argparse.Namespace) -> None:
    assert args.settings.is_file() and not args.output.exists()
    assert args.resolution >= 8 and args.degrees > 0 and args.twist_seconds > 0
    source = json.loads(args.settings.read_text())
    config = {name: source[name] for name in ("center", "native", "precision", "deterministic", "mapping", "mass_band")}
    config["settings_source"] = str(args.settings.resolve())
    config["native"].update(grid_size=args.grid_size, substeps=args.substeps)
    config.update(resolution=args.resolution, degrees=args.degrees, twist_seconds=args.twist_seconds,
                  hold_seconds=args.hold_seconds, settle_seconds=args.settle_seconds,
                  sim_scale=0.8, sim_shift=[1.0, 1.0, 1.0], experiment="square_twist")
    config["gravity_sim"] = source["gravity_sim"] * 0.8 / source["sim_scale"]
    duration = 2 * args.twist_seconds + args.hold_seconds + args.settle_seconds
    fps, substeps = config["native"]["frame_rate"], args.substeps
    frame_count, dt = round(duration * fps), 1 / (fps * substeps)
    assert abs(frame_count / fps - duration) < 1e-12
    config["frames"] = frame_count
    assert config["mapping"] == "release_tension_shear" and config["mass_band"] == [0.5e-15, 2e-15]
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rest, original_faces, grips = square(args.resolution)
    np.savez(args.output / "scene.npz", rest=rest, faces=original_faces, grips=grips)

    from simulation_runtime import configure_precision, configure_simulation

    configure_precision(config["precision"])
    configure_simulation(config["deterministic"])
    import torch
    import warp as wp
    from hang_mapping_control import compute_cloth_stress
    from hang_mass_band import grid_normalization_mass_band
    from train_material_params import Trainer
    from twist_boundary import rotating_grips
    from warp_mpm.mpm_data_structure import MPMModelStruct, MPMStateStruct
    from warp_mpm.mpm_solver import MPMWARP, compute_stress_from_F_trial, g2p_e, grid_normalization_and_gravity
    from warp_mpm.precision import torch_dtype, vec3

    device, native = "cuda:0", config["native"]
    order = np.concatenate([grips, np.setdiff1d(np.arange(len(rest)), grips)])
    inverse = np.argsort(order)
    vertices = torch.tensor(rest[order] * config["sim_scale"] + config["sim_shift"], dtype=torch_dtype, device=device)
    faces = torch.tensor(inverse[original_faces], dtype=torch.int32, device=device)
    n_elements, n_vertices, n_grips = len(faces), len(rest), len(grips)
    count = n_elements + n_vertices
    directions, rest_directions, element_vol, vertex_vol = Trainer.compute_dir_vol(
        None, vertices, faces, thickness=native["thickness"])
    positions = torch.cat([vertices[faces.long()].mean(1), vertices])
    vertex_flags = np.zeros(count, dtype=np.int32)
    vertex_flags[n_elements:] = 1
    wp.init()
    state = MPMStateStruct()
    state.init(count, n_elements, n_vertices, device=device, requires_grad=False)
    state.from_torch(positions.clone(), torch.cat([element_vol, vertex_vol]), torch.linalg.inv(directions),
                     Trainer.compute_rest_dir_inv(None, rest_directions), faces, np.zeros(count, dtype=np.int32),
                     vertex_flags, 1 - vertex_flags, torch.zeros((n_elements, 6), dtype=torch_dtype, device=device),
                     device=device, requires_grad=False, n_grid=args.grid_size, grid_lim=2.0)
    model = MPMModelStruct()
    model.init(count, device=device, requires_grad=False)
    model.init_other_params(n_grid=args.grid_size, grid_lim=2.0, device=device)
    solver = MPMWARP(count, n_elements, n_vertices, n_grid=args.grid_size, grid_lim=2.0,
                     num_joint_v=n_grips, device=device)
    state.reset_state(n_vertices, positions.clone(), directions.clone(), tensor_velocity=torch.zeros_like(positions),
                      tensor_R_inv=Trainer.compute_rest_dir_inv(None, rest_directions),
                      device=device, requires_grad=False)
    solver.set_parameters_dict(model, state, {"material": "cloth", "g": [0, config["gravity_sim"], 0],
        "density": config["center"]["D"], "grid_v_damping_scale": native["grid_v_damping_scale"],
        "friction_angle": native["friction_angle"]})
    ones = torch.ones(count, dtype=torch_dtype, device=device)
    state.reset_density(ones * config["center"]["D"], None, device, update_mass=True)
    solver.set_E_nu_from_torch(model, ones * config["center"]["E"] * 100, ones * native["nu"],
                               ones * native["gamma"], ones * native["kappa"], device)
    solver.prepare_mu_lam(model, state, device)
    solver.add_particle_mover(n_grid=args.grid_size)
    signs = wp.from_numpy(np.repeat([-1, 1], n_grips // 2).astype(np.int32), dtype=wp.int32, device=device)
    rest_array = wp.from_torch(vertices, dtype=vec3)
    origin = vec3(*config["sim_shift"])
    native_launch = wp.launch
    angle, next_angle, band_launches = 0.0, 0.0, 0

    def enforce(end: int) -> None:
        native_launch(rotating_grips, dim=n_grips,
                      inputs=[state, rest_array, signs, n_elements, origin, angle, next_angle, dt, end], device=device)

    def launch(*launch_args: Any, **kwargs: Any) -> Any:
        nonlocal band_launches
        if kwargs["kernel"] is compute_stress_from_F_trial:
            kwargs["kernel"] = compute_cloth_stress
        if kwargs["kernel"] is grid_normalization_and_gravity:
            kwargs["kernel"] = grid_normalization_mass_band
            kwargs["inputs"] = [*kwargs["inputs"], *config["mass_band"]]
            band_launches += 1
        if kwargs["kernel"] is g2p_e:
            enforce(1)  # End-step edge positions must enter the native element-direction update.
        return native_launch(*launch_args, **kwargs)

    wp.launch = launch
    (args.output / "runtime.json").write_text(json.dumps({"torch": torch.__version__, "warp": wp.__version__,
        "device": torch.cuda.get_device_name(), "mode": wp.config.deterministic.name,
        "grid_size": args.grid_size, "substeps": substeps, "boundary": "exact two-edge rotations + native grid mover",
        "self_contact": "native shared-grid transfer and constitutive mapping; no pairwise contact added"}, indent=2) + "\n")
    trajectories, rows = [rest.copy()], []
    cell = 2 / args.grid_size / config["sim_scale"]
    zero_faces = torch.zeros((0, 3), dtype=torch_dtype)
    with torch.no_grad(), (args.output / "metrics.csv").open("w", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame", "time", "phase", "edge_angle_deg", "grip_error_m",
            "nonlocal_pairs_4cells", "nonlocal_pairs_2cells", "nonlocal_pairs_1cell", "min_nonlocal_distance_m"])
        writer.writeheader()
        for frame in range(frame_count):
            for substep in range(substeps):
                time = (frame * substeps + substep) * dt
                angle, _ = angle_at(time, config)
                next_angle, phase = angle_at(time + dt, config)
                enforce(0)
                a = grip_positions(rest, grips, angle)
                b = grip_positions(rest, grips, next_angle)
                velocity = torch.from_numpy((b - a) * config["sim_scale"] / dt)
                solver.p2g2p(model, state, dt, joint_verts_v=velocity, joint_faces_v=zero_faces, device=device)
            for name in ("particle_x", "particle_v", "particle_C", "particle_d"):
                assert torch.isfinite(wp.to_torch(getattr(state, name))).all(), (frame + 1, name)
            sim = state.particle_x.numpy()[n_elements:]
            assert sim.min() > 0.03 and sim.max() < 1.97, f"Grid boundary reached at frame {frame + 1}"
            x = ((sim - config["sim_shift"]) / config["sim_scale"])[inverse]
            error = np.linalg.norm(x[grips] - grip_positions(rest, grips, next_angle), axis=1).max()
            assert error < 1e-12, error
            trajectories.append(x)
            row = {"frame": frame + 1, "time": (frame + 1) / fps, "phase": phase,
                   "edge_angle_deg": math.degrees(next_angle), "grip_error_m": float(error),
                   **proximity(x, rest, cell)}
            rows.append(row)
            writer.writerow(row)
            np.savez(args.output / "trajectory.npz", trajectory=np.asarray(trajectories), rest=rest,
                     faces=original_faces, grips=grips)
            if (frame + 1) % 10 == 0:
                print(f"Frame {frame + 1}/{frame_count}: {phase}, edge ±{row['edge_angle_deg']:.2f}°, "
                      f"nonlocal pairs within 2 cells: {row['nonlocal_pairs_2cells']}", flush=True)
    assert band_launches == frame_count * substeps
    np.savez(args.output / "final_state.npz", **{name: getattr(state, name).numpy()
        for name in ("particle_x", "particle_v", "particle_C", "particle_d")})
    distances = [r["min_nonlocal_distance_m"] for r in rows if r["min_nonlocal_distance_m"] != ""]
    summary = {"frames": len(trajectories), "all_frames_finite": True, "cell_size_m": cell,
        "max_grip_error_m": max(r["grip_error_m"] for r in rows),
        "peak_nonlocal_pairs_2cells": max(r["nonlocal_pairs_2cells"] for r in rows),
        "min_nonlocal_distance_m": min(distances) if distances else None,
        "final_vertex_rms_from_rest_m": float(np.sqrt(np.mean(np.sum((trajectories[-1] - rest)**2, axis=1)))),
        "proximity_definition": "vertex pairs >15 cm apart in rest; proximity does not prove absence of triangle crossings"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def render(output: Path, ffmpeg: Path, playback_fps: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    assert ffmpeg.is_file() and playback_fps > 0
    config = json.loads((output / "config.json").read_text())
    with np.load(output / "trajectory.npz") as archive:
        trajectory, rest, faces, grips = (archive[name] for name in ("trajectory", "rest", "faces", "grips"))
    assert len(trajectory) == config["frames"] + 1 and np.isfinite(trajectory).all()
    cloth = trajectory[..., [0, 2, 1]]
    middle = (cloth.reshape(-1, 3).min(0) + cloth.reshape(-1, 3).max(0)) / 2
    radius = np.ptp(cloth.reshape(-1, 3), axis=0).max() * 0.56
    rest_centers = rest[faces].mean(1)
    checkers = (np.floor((rest_centers[:, 0] + 0.5) * 8) + np.floor((rest_centers[:, 2] + 0.5) * 8)) % 2
    frames = output / "video/frames"
    frames.mkdir(parents=True, exist_ok=True)
    fps = config["native"]["frame_rate"]
    for index, vertices in enumerate(cloth):
        figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
        for panel, (elev, azim, title) in enumerate([(25, -55, "Oblique view"), (0, 0, "Along the twist axis")], start=1):
            axis = figure.add_subplot(1, 2, panel, projection="3d")
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            assert (lengths > 0).all(), f"Degenerate rendered triangle at frame {index}"
            normals /= lengths
            camera = np.array([math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
                               math.cos(math.radians(elev)) * math.sin(math.radians(azim)), math.sin(math.radians(elev))])
            colors = np.where((normals @ camera > 0)[:, None], [0.95, 0.51, 0.2], [0.27, 0.61, 0.82])
            colors *= (0.75 + 0.25 * np.abs(normals @ camera))[:, None] * (0.85 + 0.15 * checkers[:, None])
            axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors="none", antialiased=False))
            for ids, color in ((grips[:len(grips)//2], "crimson"), (grips[len(grips)//2:], "purple")):
                axis.plot(*vertices[ids].T, color=color, linewidth=3)
            for setter, low, high in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), middle - radius, middle + radius):
                setter(low, high)
            axis.set_box_aspect((1, 1, 1), zoom=1.1)
            axis.set_proj_type("ortho")
            axis.view_init(elev=elev, azim=azim)
            axis.set_axis_off()
            axis.set_title(title, fontsize=13)
        angle, phase = angle_at(index / fps, config)
        figure.suptitle("MPMAvatar | square cloth twist–untwist", fontsize=19, y=0.97)
        figure.text(0.5, 0.91, f"{phase} | t = {index / fps:.2f} s | edges: ±{math.degrees(angle):.1f}° | "
                    f"relative twist: {2 * math.degrees(angle):.1f}°", ha="center", fontsize=13)
        figure.text(0.5, 0.06, f"{config['resolution']}×{config['resolution']} cloth | "
                    f"{config['native']['grid_size']}³ grid | {config['native']['substeps']} substeps/frame | "
                    "latest shear release + C2 mass band", ha="center", fontsize=11)
        figure.text(0.5, 0.025, "Orange/blue: two cloth sides | red/purple: prescribed edge grips | "
                    f"{playback_fps / fps:g}× playback", ha="center", fontsize=10)
        figure.subplots_adjust(left=0, right=1, bottom=0.11, top=0.85, wspace=0)
        figure.savefig(frames / f"frame_{index:04d}.png")
        if index in (round(config["twist_seconds"] * fps), len(cloth) - 1):
            figure.savefig(output / ("twisted.png" if index < len(cloth) - 1 else "untwisted.png"))
        plt.close(figure)
        if index % 20 == 0 or index == len(cloth) - 1:
            print(f"Rendered {index + 1}/{len(cloth)} frames", flush=True)
    subprocess.run([str(ffmpeg), "-y", "-framerate", str(playback_fps), "-i", str(frames / "frame_%04d.png"),
                    "-frames:v", str(len(cloth)), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output / "twist_untwist.mp4")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "render"), required=True)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--degrees", type=float, default=360)
    parser.add_argument("--twist-seconds", type=float, default=3)
    parser.add_argument("--hold-seconds", type=float, default=1)
    parser.add_argument("--settle-seconds", type=float, default=1)
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=200)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--playback-fps", type=float, default=25)
    args = parser.parse_args()
    if args.stage == "run":
        run(args)
    else:
        render(args.output, args.ffmpeg, args.playback_fps)


if __name__ == "__main__":
    main()
