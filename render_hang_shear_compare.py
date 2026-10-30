"""Render synchronized E=100 original versus controlled-shear hang trajectories.

Activate mpmavatar; from MPMAvatar:
# MPLCONFIGDIR=/tmp/robustdiffcloth-matplotlib python render_hang_shear_compare.py --config output/hang_heavy_E_control_h1e-6_r10/config.json --original output/hang_heavy_E_trace/trace/run_00_k+0/trajectory.npz --controlled output/hang_heavy_E_trace/release_control/run_00_k+0/trajectory.npz --output output/hang_heavy_E_control_h1e-6_r10/video --ffmpeg /usr/bin/ffmpeg --playback-fps 12.5 --elevation 20 --azimuth -60
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from hang_local_sweep import load_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--playback-fps", type=float, required=True)
    parser.add_argument("--elevation", type=float, required=True)
    parser.add_argument("--azimuth", type=float, required=True)
    args = parser.parse_args()
    assert args.config.is_file() and args.original.is_file() and args.controlled.is_file() and args.ffmpeg.is_file()
    config = json.loads(args.config.read_text())
    scene = load_scene(Path(config["data"]))
    trajectories = []
    for path in (args.original, args.controlled):
        with np.load(path) as archive:
            assert float(archive["E"]) == 100.0
            trajectories.append(archive["trajectory"])
    a, b = trajectories
    assert a.shape == b.shape == (config["frames"] + 1, len(scene["rest_m"]), 3)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    distance = np.linalg.norm(a - b, axis=-1)
    body = scene["body_m"][:, [0, 2, 1]]
    trajectories = [x[..., [0, 2, 1]] for x in trajectories]
    light = np.array([0.3, -0.8, 0.5]) / np.linalg.norm([0.3, -0.8, 0.5])
    frames = args.output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    labels = ("Original shearing | E = 100", "Controlled shear release | E = 100")
    for index in range(len(a)):
        bounds = np.concatenate([x[index] for x in trajectories] + [body])
        middle = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        radius = np.ptp(bounds, axis=0).max() * 0.55
        figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
        for column, (x, label) in enumerate(zip(trajectories, labels)):
            axis = figure.add_subplot(1, 2, column + 1, projection="3d")
            triangles = np.concatenate([x[index, scene["faces"]], body[scene["body_faces"]]])
            colors = np.concatenate([np.tile([0.78, 0.78, 0.78], (len(scene["faces"]), 1)),
                                     np.tile([0.45, 0.6, 0.9], (len(scene["body_faces"]), 1))])
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            assert (lengths > 0).all(), f"Degenerate triangles in frame {index}"
            normals /= lengths[:, None]
            shade = 0.35 + 0.45 * np.abs(normals @ light)
            axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors * shade[:, None] / 0.8,
                                                   edgecolors="none", linewidths=0, antialiased=False))
            axis.scatter(*x[index, scene["attachment_ids"]].T, color="red", s=28, depthshade=False)
            for setter, low, high in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), middle - radius, middle + radius):
                setter(low, high)
            axis.set_box_aspect((1, 1, 1))
            axis.set_proj_type("ortho")
            axis.view_init(elev=args.elevation, azim=args.azimuth)
            axis.set_axis_off()
            axis.set_title(label, fontsize=14)
        rms = 1000 * np.sqrt(np.mean(distance[index] ** 2))
        maximum = 1000 * distance[index].max()
        figure.suptitle(f"Hang heavy | t = {index / config['native']['frame_rate']:.2f} s | "
                        f"frame {index}/{len(a) - 1}", fontsize=15)
        figure.text(0.5, 0.035, f"Original vs control: RMS {rms:.2f} mm | max vertex {maximum:.2f} mm | "
                    f"playback {args.playback_fps / config['native']['frame_rate']:g}x", ha="center", fontsize=12)
        figure.subplots_adjust(left=0, right=1, bottom=0.075, top=0.87, wspace=0)
        figure.savefig(frames / f"frame_{index:03d}.png")
        plt.close(figure)
        if index % 10 == 0 or index == len(a) - 1:
            print(f"Rendered {index + 1}/{len(a)} frames", flush=True)
    video = args.output / "original_vs_controlled.mp4"
    subprocess.run([str(args.ffmpeg), "-y", "-framerate", str(args.playback_fps), "-i", str(frames / "frame_%03d.png"),
                    "-frames:v", str(len(a)), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(video)], check=True)
    (args.output / "comparison.json").write_text(json.dumps({**vars(args), "frames": len(a),
        "final_rms_mm": float(1000 * np.sqrt(np.mean(distance[-1] ** 2))),
        "max_vertex_mm": float(1000 * distance.max())}, default=str, indent=2) + "\n")
    print(video, flush=True)


if __name__ == "__main__":
    main()
