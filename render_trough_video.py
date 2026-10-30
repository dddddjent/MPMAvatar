"""Render the MPMAvatar trough center with an overview and a camera following the cloth.

Activate mpmavatar; from MPMAvatar:
# MPLCONFIGDIR=/tmp/robustdiffcloth-matplotlib python render_trough_video.py --output output/trough20_E_smooth_h1e-6_r10 --ffmpeg /usr/bin/ffmpeg --playback-fps 25
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray

from hang_local_sweep import load_scene


def draw(axis: Axes3D, cloth: NDArray, faces: NDArray, body: NDArray, body_faces: NDArray,
         clip: bool) -> None:
    triangles = np.concatenate([body[body_faces], cloth[faces]])
    colors = np.concatenate([np.tile([0.40, 0.61, 0.79], (len(body_faces), 1)),
                             np.tile([0.94, 0.47, 0.16], (len(faces), 1))])
    if clip:
        limits = np.array([axis.get_xlim(), axis.get_ylim(), axis.get_zlim()])
        keep = ((triangles >= limits[:, 0]) & (triangles <= limits[:, 1])).all(axis=(1, 2))
        triangles, colors = triangles[keep], colors[keep]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    assert (lengths > 0).all(), "Degenerate rendered triangle"
    normals /= lengths
    light = np.array([0.3, -0.4, 0.8])
    light /= np.linalg.norm(light)
    colors *= (0.55 + 0.45 * np.abs(normals @ light))[:, None]
    axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors="none",
                                          linewidths=0, antialiased=False))


def render(output: Path, ffmpeg: Path, playback_fps: float) -> None:
    assert ffmpeg.is_file() and playback_fps > 0
    config = json.loads((output / "config.json").read_text())
    scene = load_scene(Path(config["data"]))
    with np.load(output / "runs/k+00/trajectory.npz") as archive:
        trajectory = archive["trajectory"]
        assert float(archive["E"]) == config["center"]["E"]
    assert trajectory.shape == (config["frames"] + 1, len(scene["rest_m"]), 3)
    assert np.isfinite(trajectory).all()
    cloth, body = trajectory[..., [0, 2, 1]], scene["body_m"][:, [0, 2, 1]]
    combined = np.concatenate([cloth.reshape(-1, 3), body])
    low, high = combined.min(axis=0) - 0.08, combined.max(axis=0) + 0.08
    frames = output / "video/frames"
    frames.mkdir(parents=True, exist_ok=True)
    for index, vertices in enumerate(cloth):
        figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
        middle = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        panels = [(low, high, "Whole trough", False),
                  (middle - 0.65, middle + 0.65, "Following the cloth", True)]
        for panel, (lower, upper, title, clip) in enumerate(panels, start=1):
            axis = figure.add_subplot(1, 2, panel, projection="3d")
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[1], upper[1])
            axis.set_zlim(lower[2], upper[2])
            draw(axis, vertices, scene["faces"], body, scene["body_faces"], clip)
            axis.set_box_aspect(upper - lower, zoom=1.15)
            axis.set_proj_type("ortho")
            axis.view_init(elev=20, azim=-7)
            axis.set_axis_off()
            axis.set_title(title, fontsize=13, pad=-5)
        figure.suptitle("MPMAvatar | 20° trough | smooth shear-release model", fontsize=18, y=0.97)
        figure.text(0.5, 0.905,
                    f"E = {config['center']['E']:g} | t = {index / config['native']['frame_rate']:.2f} s | "
                    f"frame {index}/{len(cloth) - 1}", ha="center", fontsize=13)
        figure.text(0.5, 0.055,
                    f"{config['native']['grid_size']}³ grid | {config['native']['substeps']} substeps/frame | "
                    f"fp64 deterministic | C2 mass band [0.5, 2] × 1e-15", ha="center", fontsize=11)
        figure.text(0.5, 0.02,
                    f"Orange: cloth | blue: fixed trough | "
                    f"{playback_fps / config['native']['frame_rate']:g}× playback", ha="center", fontsize=10)
        figure.subplots_adjust(left=0, right=1, bottom=0.10, top=0.85, wspace=0)
        figure.savefig(frames / f"frame_{index:03d}.png")
        if index == len(cloth) - 1:
            figure.savefig(output / "trough_preview.png")
        plt.close(figure)
        if index % 10 == 0 or index == len(cloth) - 1:
            print(f"Rendered {index + 1}/{len(cloth)} frames", flush=True)
    video = output / "trough.mp4"
    subprocess.run([str(ffmpeg), "-y", "-framerate", str(playback_fps), "-i", str(frames / "frame_%03d.png"),
                    "-frames:v", str(len(cloth)), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(video)], check=True)
    (output / "video/video.json").write_text(json.dumps({"frames": len(cloth), "playback_fps": playback_fps,
        "source": "runs/k+00/trajectory.npz", "display": "x,z,y; overview and cloth-following close view"}, indent=2) + "\n")
    print(video, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--playback-fps", type=float, required=True)
    args = parser.parse_args()
    render(args.output, args.ffmpeg, args.playback_fps)


if __name__ == "__main__":
    main()
