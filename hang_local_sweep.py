"""MPMAvatar native solver on DiffCloth's hanging square with the heavy static sphere: dense ln E local sweep.

Scene (robustdiffcloth/DiffCloth/data/hang/square_1m_sphere_heavy): 92 x 92 grid, 1 m square, flat in y = 0, corners
0 and 91 pinned (MPMAvatar joint vertices, zero velocity), 1280-face icosphere as a static MPMAvatar mesh collider
(its own grid contact; DiffCloth's smoothed normals do not exist here). Metres -> sim: x * SIM_SCALE + shift, which
puts the DiffCloth heavy center trajectory and the sphere (2.1 m along z) inside the [0, 2]^3 grid. MPMAvatar's native
material/solver: D 1, E 100 (internal 1e4), H 1, nu .3, gamma/kappa 500, mesh friction .5, friction angle 40,
gravity = DiffCloth's (9.8 units/s^2 / 6 units/m = 1.633 m/s^2, x SIM_SCALE in sim units), grid 200, 400 substeps
per 1/25 s frame, 75 frames (3 s, as DiffCloth), fp64, deterministic.

Runs: "center" (E 100) and "target" (E x 2 = 200; E x 5 blows up at frame 51 with DiffCloth gravity) references;
one line per step h with ln E = ln 100 + k h, k = -radius..radius (its own k = 0). A line is bound to the first GPU
model that starts it; points of one line are claimed one at a time (flock), so several GPUs of that model share a
line. Loss = mean over all 76 frames, vertices and xyz of the squared difference in metres (DiffCloth's definition):
loss_target vs target, loss_center vs the line's own k = 0.

Template commands (activate mpmavatar; from MPMAvatar):
# python hang_local_sweep.py --stage prepare --data ../../robustdiffcloth/DiffCloth/data/hang/square_1m_sphere_heavy --output output/hang_sphere_heavy_E_sweep --step-sizes 1e-12 1e-6 1e-4 1e-2 --radius 50
# python hang_local_sweep.py --stage run --output output/hang_sphere_heavy_E_sweep   # inside a GPU allocation
# python hang_local_sweep.py --stage summarize --output output/hang_sphere_heavy_E_sweep
# python hang_local_sweep.py --stage video --output output/hang_sphere_heavy_E_sweep --baseline ../../robustdiffcloth/DiffCloth/output/hang_sphere_heavy/square/center.npz --ffmpeg /work/nvme/bivb/junlinl6/conda/envs/mpmavatar/bin/ffmpeg
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

CENTER = {"D": 1.0, "E": 100.0, "H": 1.0}
TARGET_E_FACTOR = 2.0
NATIVE = {"nu": 0.3, "gamma": 500.0, "kappa": 500.0, "mesh_friction": 0.5, "friction_angle": 40.0,
          "grid_size": 200, "substeps": 400, "frame_rate": 25.0, "thickness": 1e-5, "grid_v_damping_scale": 1.1}
SIM_SCALE = 0.8  # sim units per metre
GARMENT = "square"


def offsets(radius: int) -> list[int]:
    """Center first, then outward so partial lines stay symmetric."""
    return [0] + [sign * k for k in range(1, radius + 1) for sign in (1, -1)]


def line_dir(output: Path, step: float) -> Path:
    return output / GARMENT / "E" / f"s{step:.0e}"


def load_scene(data: Path) -> dict[str, np.ndarray]:
    assert (data / "scene.npz").is_file() and (data / "rest.obj").is_file(), data
    rest = np.array([[float(v) for v in line.split()[1:]] for line in (data / "rest.obj").read_text().splitlines()
                     if line.startswith("v ")])
    with np.load(data / "scene.npz", allow_pickle=False) as archive:
        scene = dict(archive)
    scale = float(scene["scale"])  # DiffCloth units per metre
    scene["rest_m"] = rest / scale
    scene["body_m"] = scene["body_vertices"] / scale
    return scene


def prepare(data: Path, output: Path, step_sizes: list[float], radius: int) -> None:
    assert not output.exists(), f"Choose a fresh output: {output}"
    scene = load_scene(data)
    extent = np.concatenate([scene["rest_m"], scene["body_m"]])
    # Box center: x/y middle of cloth + sphere, z middle of the full swing range (pinned edge z = -0.5 m, 1 m arm).
    center = np.array([(extent[:, 0].min() + extent[:, 0].max()) / 2, -0.55, -0.36])
    shift = 1.0 - SIM_SCALE * center
    config = {"data": str(data.resolve()), "center": CENTER, "target": {**CENTER, "E": CENTER["E"] * TARGET_E_FACTOR},
              "step_sizes": step_sizes, "radius": radius, "direction": "E", "log_direction": True, "native": NATIVE,
              "sim_scale": SIM_SCALE, "sim_shift": shift.tolist(), "frames": int(scene["frames"]),
              "gravity_sim": float(scene["gravity_y"]) / float(scene["scale"]) * SIM_SCALE,
              "precision": "fp64", "deterministic": True}
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared 2 references + {len(step_sizes) * (2 * radius + 1)} rollouts: {output}", flush=True)


class HangSimulation:
    """setup_simulation of train_material_params.Trainer for the hanging square (no body motion, static pins)."""

    def __init__(self, config: dict[str, Any]) -> None:
        from simulation_runtime import configure_precision, configure_simulation

        configure_precision(config["precision"])
        configure_simulation(config["deterministic"])
        import torch
        import warp as wp
        from train_material_params import Trainer
        from warp_mpm.mpm_data_structure import MPMModelStruct, MPMStateStruct
        from warp_mpm.mpm_solver import MPMWARP
        from warp_mpm.precision import torch_dtype

        self.config, self.native = config, config["native"]
        device = "cuda:0"
        scene = load_scene(Path(config["data"]))
        pinned = scene["attachment_ids"].astype(np.int64)
        n_vertices = len(scene["rest_m"])
        # Joint vertices come first in MPMAvatar's vertex order.
        self.order = np.concatenate([pinned, np.setdiff1d(np.arange(n_vertices), pinned)])
        inverse = np.empty_like(self.order)
        inverse[self.order] = np.arange(n_vertices)
        self.inverse = inverse
        self.scale, self.shift = config["sim_scale"], torch.tensor(config["sim_shift"], dtype=torch_dtype, device=device)
        verts = self.wld2sim(torch.tensor(scene["rest_m"][self.order], dtype=torch_dtype, device=device))
        faces = torch.tensor(inverse[scene["faces"]], dtype=torch.int32, device=device)
        self.faces, self.rest_sim = faces, verts
        n_elements = faces.shape[0]
        n_particles = n_elements + n_vertices
        init_dir, rest_dir, element_vol, vertex_vol = Trainer.compute_dir_vol(None, verts, faces,
                                                                              thickness=self.native["thickness"])
        self.init_pos = torch.cat([verts[faces.long()].mean(1), verts], 0)
        self.init_dir = init_dir
        self.init_velo = torch.zeros_like(self.init_pos)
        particle_vertices = np.zeros(n_particles, dtype=np.int32)
        particle_vertices[n_elements:] = 1
        particle_elements = 1 - particle_vertices
        wp.init()
        state = MPMStateStruct()
        state.init(n_particles, n_elements, n_vertices, device=device, requires_grad=True)
        grid = self.native["grid_size"]
        state.from_torch(self.init_pos.clone(), torch.cat([element_vol, vertex_vol]).clone(),
                         torch.linalg.inv(init_dir).clone(), Trainer.compute_rest_dir_inv(None, rest_dir).clone(),
                         faces.clone(), np.zeros(n_particles, dtype=np.int32), particle_vertices, particle_elements,
                         torch.zeros((n_elements, 6)).to(device), device=device, requires_grad=True, n_grid=grid,
                         grid_lim=2.0)
        model = MPMModelStruct()
        model.init(n_particles, device=device, requires_grad=True)
        model.init_other_params(n_grid=grid, grid_lim=2.0, device=device)
        body = self.wld2sim(torch.tensor(scene["body_m"], dtype=torch_dtype, device=device)).cpu().numpy()
        solver = MPMWARP(n_particles, n_elements, n_vertices, n_grid=grid, grid_lim=2.0, mesh_vertices=body,
                         mesh_faces=scene["body_faces"].astype(np.int32), num_joint_t=0, num_joint_v=len(pinned),
                         num_joint_f=0, device=device)
        solver.set_parameters_dict(model, state, {"material": "cloth", "g": [0.0, config["gravity_sim"], 0.0],
                                                  "density": 1.0,
                                                  "grid_v_damping_scale": self.native["grid_v_damping_scale"],
                                                  "friction_angle": self.native["friction_angle"]})
        ones = torch.ones_like(self.init_pos[:, 0])
        state.reset_density(ones.clone(), torch.ones_like(ones).type(torch.int), device, update_mass=True)
        solver.set_E_nu_from_torch(model, ones * CENTER["E"] * 100, ones * self.native["nu"],
                                   ones * self.native["gamma"], ones * self.native["kappa"], device)
        solver.prepare_mu_lam(model, state, device)
        solver.add_mesh_collider(solver.mesh, n_grid=model.n_grid, friction=self.native["mesh_friction"])
        solver.add_particle_mover(n_grid=model.n_grid)
        self.state, self.model, self.solver, self.device = state, model, solver, device
        self.n_elements, self.n_vertices, self.n_pinned = n_elements, n_vertices, len(pinned)
        self.compute_rest_dir_inv_from_vf = lambda v, f: Trainer.compute_rest_dir_inv_from_vf(None, v, f)

    def wld2sim(self, p: Any) -> Any:
        return p * self.scale + self.shift

    def sim2wld(self, p: Any) -> Any:
        return (p - self.shift) / self.scale

    def rollout(self, values: dict[str, float]) -> np.ndarray:
        """(frames + 1, vertices, 3) metres in DiffCloth vertex order; frame 0 is the flat rest state."""
        import torch
        import warp as wp

        with torch.no_grad():
            device, solver, state, model = self.device, self.solver, self.state, self.model
            rest = self.rest_sim.clone()
            rest[:, 1] *= values["H"]
            state.reset_state(self.n_vertices, self.init_pos.clone(), self.init_dir.clone(),
                              tensor_velocity=self.init_velo.clone(),
                              tensor_R_inv=self.compute_rest_dir_inv_from_vf(rest, self.faces), device=device,
                              requires_grad=False)
            ones = torch.ones_like(self.init_pos[:, 0])
            state.reset_density(ones * values["D"], None, device, update_mass=True)
            solver.set_E_nu_from_torch(model, ones * values["E"] * 100, ones * self.native["nu"],
                                       ones * self.native["gamma"], ones * self.native["kappa"], device)
            solver.prepare_mu_lam(model, state, device)
            solver.time = 0.0
            solver.time_profile.clear()
            substeps = self.native["substeps"]
            dt = 1.0 / (self.native["frame_rate"] * substeps)
            joint_v = torch.zeros((self.n_pinned, 3), dtype=self.init_pos.dtype, device=device)
            face_v = torch.zeros((0, 3), dtype=self.init_pos.dtype, device=device)
            frames = [self.sim2wld(self.init_pos[self.n_elements:])]
            for frame in range(self.config["frames"]):
                for _ in range(substeps):
                    solver.p2g2p(model, state, dt, mesh_x=None, mesh_v=None, joint_traditional_v=None,
                                 joint_verts_v=joint_v, joint_faces_v=face_v, device=device)
                positions = wp.to_torch(state.particle_x)[self.n_elements:]
                assert torch.isfinite(positions).all(), f"Nonfinite simulation at frame {frame + 1}"
                frames.append(self.sim2wld(positions).clone())
            return torch.stack(frames).cpu().numpy()[:, self.inverse]


def job_end() -> float:
    end = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%e"], check=True, text=True,
                         stdout=subprocess.PIPE).stdout.strip()
    return datetime.fromisoformat(end).timestamp()


def try_lock(path: Path) -> Any:
    """Open flock that holds while the returned file stays open; None if another process holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("w")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def tasks(output: Path, config: dict[str, Any]) -> list[tuple[Path, dict[str, float], dict[str, Any], Path | None]]:
    """(result path, parameters, metadata, line dir or None for references) in run order."""
    items: list[tuple[Path, dict[str, float], dict[str, Any], Path | None]] = [
        (output / GARMENT / f"{name}.npz", dict(config[name]), {"run_id": name}, None) for name in ("center", "target")]
    for step in config["step_sizes"]:
        line = line_dir(output, step)
        for k in offsets(config["radius"]):
            values = {**config["center"], "E": config["center"]["E"] * math.exp(k * step)}
            items.append((line / f"k{k:+03d}.npz", values, {"run_id": f"E/s{step:.0e}/k{k:+03d}", "k": k,
                                                            "step": step, "offset": k * step}, line))
    return items


def bound_to(line: Path, device: str, output: Path) -> bool:
    """Bind the line to this GPU model if unbound; True when this process may run its points."""
    marker = line / "device.txt"
    if not marker.is_file():
        lock = try_lock(output / "locks" / f"bind_{line.name}.lock")
        if lock is None:
            return False
        if not marker.is_file():
            line.mkdir(parents=True, exist_ok=True)
            marker.write_text(device + "\n")
            (line / "partition.txt").write_text(os.environ["SLURM_JOB_PARTITION"] + "\n")
        lock.close()
    return marker.read_text().strip() == device


def run(output: Path) -> None:
    import torch

    config = json.loads((output / "config.json").read_text())
    end = job_end()
    simulation = HangSimulation(config)
    assert simulation.init_pos.dtype == torch.float64
    device = torch.cuda.get_device_name()
    longest = 0.0
    for path, values, metadata, line in tasks(output, config):
        if path.is_file() or (line is not None and not bound_to(line, device, output)):
            continue
        lock = try_lock(output / "locks" / (path.relative_to(output).as_posix().replace("/", "_") + ".lock"))
        if lock is None:
            continue
        if path.is_file():
            lock.close()
            continue
        if end - time.time() < 1.5 * max(longest, 600.0) + 60.0:
            print(f"Stopping before {path.relative_to(output)}: {end - time.time():.0f}s left", flush=True)
            lock.close()
            return
        print(f"Native {path.relative_to(output)}: D/E/H = {values['D']!r}/{values['E']!r}/{values['H']!r}",
              flush=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        trajectory = simulation.rollout(values)
        seconds = time.perf_counter() - start
        longest = max(longest, seconds)
        save(path, trajectory=trajectory, parameters=np.asarray([values[key] for key in CENTER]),
             parameter_names=np.asarray(list(CENTER)), seconds=seconds, device=device,
             job_id=os.environ["SLURM_JOB_ID"], node=os.uname().nodename, **metadata)
        print(f"done {path.relative_to(output)} ({seconds:.1f}s)", flush=True)
        lock.close()
    print(f"No remaining rollout for {device}", flush=True)


def summarize(output: Path) -> None:
    config = json.loads((output / "config.json").read_text())
    with np.load(output / GARMENT / "target.npz") as archive:
        target = archive["trajectory"]
    rows, missing = [], 0
    for step in config["step_sizes"]:
        line = line_dir(output, step)
        center_path = line / "k+00.npz"
        if not center_path.is_file():
            missing += 2 * config["radius"] + 1
            continue
        with np.load(center_path) as archive:
            center = archive["trajectory"]
        for k in offsets(config["radius"]):
            path = line / f"k{k:+03d}.npz"
            if not path.is_file():
                missing += 1
                continue
            with np.load(path) as archive:
                trajectory, seconds, device = archive["trajectory"], float(archive["seconds"]), str(archive["device"])
            rows.append({"step": step, "k": k, "ln_E_offset": k * step, "E": float(np.exp(np.log(100.0) + k * step)),
                         "loss_target": float(np.mean((trajectory - target) ** 2)),
                         "loss_center": float(np.mean((trajectory - center) ** 2)),
                         "max_v2v_center_mm": float(1000 * np.linalg.norm(trajectory - center, axis=-1).max()),
                         "seconds": seconds, "device": device})
    rows.sort(key=lambda row: (row["step"], row["k"]))
    with (output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rollouts summarized, {missing} missing -> {output / 'runs.csv'}", flush=True)


def video(output: Path, baseline: Path, ffmpeg: Path, playback_fps: float, elevation: float, azimuth: float) -> None:
    """Grey meshes: MPMAvatar center | MPMAvatar target | DiffCloth center (original contact, no smoothed normals)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    assert ffmpeg.is_file() and baseline.is_file(), (ffmpeg, baseline)
    config = json.loads((output / "config.json").read_text())
    scene = load_scene(Path(config["data"]))
    columns = []
    for name, label in (("center", f"MPMAvatar center (E {config['center']['E']:g})"),
                        ("target", f"MPMAvatar target (E {config['target']['E']:g})")):
        with np.load(output / GARMENT / f"{name}.npz") as archive:
            columns.append((archive["trajectory"], label))
    with np.load(baseline) as archive:
        columns.append((archive["trajectory"], "DiffCloth center (original contact)"))
    count = min(len(trajectory) for trajectory, _ in columns)
    body = scene["body_m"][:, [0, 2, 1]]
    columns = [(trajectory[:count][..., [0, 2, 1]], label) for trajectory, label in columns]  # z-up display
    combined = np.concatenate([trajectory.reshape(-1, 3) for trajectory, _ in columns] + [body])
    middle = (combined.min(axis=0) + combined.max(axis=0)) / 2
    radius = (combined.max(axis=0) - combined.min(axis=0)).max() * 0.55
    light = np.array([0.3, -0.8, 0.5]) / np.linalg.norm([0.3, -0.8, 0.5])
    frames = output / "video" / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        figure = plt.figure(figsize=(6.4 * len(columns), 7.2), dpi=100, facecolor="white")
        for column, (trajectory, label) in enumerate(columns):
            axis = figure.add_subplot(1, len(columns), column + 1, projection="3d")
            meshes = [(trajectory[index], scene["faces"], np.array([0.8, 0.8, 0.8])),
                      (body, scene["body_faces"], np.array([0.45, 0.6, 0.9]))]
            triangles = np.concatenate([vertices[faces] for vertices, faces, _ in meshes])
            colors = np.concatenate([np.broadcast_to(rgb, (len(faces), 3)) for _, faces, rgb in meshes])
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            shade = 0.35 + 0.45 * np.abs(normals @ light)
            axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors * shade[:, None] / 0.8,
                                                   edgecolors="none", linewidths=0, antialiased=False))
            axis.scatter(*trajectory[index, scene["attachment_ids"]].T, color="red", s=30, depthshade=False)
            for setter, low, high in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), middle - radius,
                                         middle + radius):
                setter(low, high)
            axis.set_box_aspect((1, 1, 1))
            axis.set_proj_type("ortho")
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_axis_off()
            axis.set_title(label, fontsize=12)
        figure.suptitle(f"Hanging square 1 m, heavy sphere (blue), corners pinned (red) · frame {index}/{count - 1} · "
                        f"t = {index / NATIVE['frame_rate']:.2f} s", fontsize=14)
        figure.subplots_adjust(left=0, right=1, bottom=0.02, top=0.88, wspace=0)
        figure.savefig(frames / f"frame_{index:03d}.png")
        plt.close(figure)
    path = output / "video" / "center_vs_target.mp4"
    subprocess.run([str(ffmpeg), "-y", "-framerate", str(playback_fps), "-i", str(frames / "frame_%03d.png"),
                    "-frames:v", str(count), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(path)], check=True)
    print(path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize", "video"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="prepare: DiffCloth hang scene dir")
    parser.add_argument("--step-sizes", nargs="+", type=float, help="prepare: ln E steps")
    parser.add_argument("--radius", type=int, help="prepare: k = -radius..radius")
    parser.add_argument("--baseline", type=Path, help="video: DiffCloth center.npz")
    parser.add_argument("--ffmpeg", type=Path, help="video: ffmpeg binary")
    parser.add_argument("--playback-fps", type=float, default=25.0)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data, args.output, args.step_sizes, args.radius)
    elif args.stage == "run":
        run(args.output)
    elif args.stage == "summarize":
        summarize(args.output)
    else:
        video(args.output, args.baseline, args.ffmpeg, args.playback_fps, args.elevation, args.azimuth)


if __name__ == "__main__":
    main()
