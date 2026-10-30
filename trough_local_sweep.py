"""Run the 20-degree trough with the latest shear-release/C2-mass-band settings.

Activate mpmavatar; from MPMAvatar:
# MPLCONFIGDIR=/tmp/robustdiffcloth-matplotlib python trough_local_sweep.py --stage prepare --data ../DiffCloth/data/hang/g24_trough20_imls --settings output/hang_heavy_E_mass_band/default/sweep/config.json --output output/trough20_E_smooth_h1e-6_r10 --step-size 1e-6 --radius 10 --grid-size 100 --substeps 200
# MPLCONFIGDIR=/tmp/robustdiffcloth-matplotlib python trough_local_sweep.py --stage run --output output/trough20_E_smooth_h1e-6_r10 --stride 10 --snapshot-stride 200
# MPLCONFIGDIR=/tmp/robustdiffcloth-matplotlib python trough_local_sweep.py --stage audit --output output/trough20_E_smooth_h1e-6_r10 --snapshot-stride 200
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hang_local_sweep import load_scene, offsets
from sweep_hang_e import case_path


def prepare(data: Path, settings: Path, output: Path, step: float, radius: int,
            grid_size: int, substeps: int) -> None:
    assert settings.is_file() and not output.exists()
    assert step > 0 and radius > 0 and grid_size > 0 and substeps > 0
    config = json.loads(settings.read_text())
    assert config["mapping"] == "release_tension_shear" and config["mass_band"] == [0.5e-15, 2e-15]
    assert config["precision"] == "fp64" and config["deterministic"]
    config["native"].update(grid_size=grid_size, substeps=substeps)
    scene = load_scene(data)
    assert scene["attachment_ids"].size == 0 and len(scene["rest_m"]) == 24 * 24
    assert float(scene["vee_fillet_radius"]) == 0
    extent = np.concatenate([scene["rest_m"], scene["body_m"]])
    scale = 0.5
    shift = 1.0 - scale * (extent.min(axis=0) + extent.max(axis=0)) / 2
    bounds = extent * scale + shift
    assert bounds.min() > 0.05 and bounds.max() < 1.95
    config.pop("comparison_baseline")
    config.update(data=str(data.resolve()), settings_source=str(settings.resolve()),
                  step_sizes=[step], radius=radius, sim_scale=scale, sim_shift=shift.tolist(),
                  frames=int(scene["frames"]),
                  gravity_sim=float(scene["gravity_y"]) / float(scene["scale"]) * scale)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared {2 * radius + 1} points plus E={config['target']['E']:g} target; "
          f"grid {grid_size}³, {substeps} substeps/frame, physical cell size {2 / grid_size / scale:g} m", flush=True)


def run(output: Path, stride: int, snapshot_stride: int) -> None:
    config = json.loads((output / "config.json").read_text())
    script = Path(__file__).with_name("sweep_hang_e.py")
    stages = [("center", [0]), ("target", [0])]
    stages += [("points", [k]) for k in offsets(config["radius"]) if k != 0]
    for stage, ks in stages:
        directory = output / "target" if stage == "target" else case_path(output, ks[0])
        if (directory / "trajectory.npz").is_file():
            continue
        command = [sys.executable, "-u", str(script), "--stage", stage, "--output", str(output),
                   "--offsets", *map(str, ks), "--stride", str(stride), "--snapshot-stride", str(snapshot_stride)]
        subprocess.run(command, check=True)
        if stage == "target":
            subprocess.run([sys.executable, str(script), "--stage", "plot", "--output", str(output),
                            "--offsets", "0", "--stride", str(stride),
                            "--snapshot-stride", str(snapshot_stride)], check=True)
    audit(output, snapshot_stride)


def audit(output: Path, snapshot_stride: int) -> None:
    config = json.loads((output / "config.json").read_text())
    scene = load_scene(Path(config["data"]))
    shape = (config["frames"] + 1, len(scene["rest_m"]), 3)
    steps = config["frames"] * config["native"]["substeps"]
    rows: list[dict[str, Any]] = []
    for k in range(-config["radius"], config["radius"] + 1):
        directory = case_path(output, k)
        young = config["center"]["E"] * math.exp(k * config["step_sizes"][0])
        with np.load(directory / "trajectory.npz") as archive:
            x = archive["trajectory"]
            assert x.shape == shape and x.dtype == np.float64 and np.isfinite(x).all()
            assert float(archive["E"]) == young and int(archive["k"]) == k
            assert str(archive["mapping"]) == config["mapping"]
            assert np.array_equal(archive["mass_band"], config["mass_band"])
            assert np.allclose(x[0], scene["rest_m"], rtol=0, atol=1e-14)
        for step in range(0, steps + 1, snapshot_stride):
            with np.load(directory / f"state_{step:06d}.npz") as archive:
                assert int(archive["step"]) == step and float(archive["E"]) == young
                for name in ("particle_x", "particle_v", "particle_C", "particle_d"):
                    assert np.isfinite(archive[name]).all(), (k, step, name)
                assert archive["particle_x"].min() > 0.02 and archive["particle_x"].max() < 1.97, (k, step)
        rows.append({"k": k, "E": young, "finite_saved_states": steps // snapshot_stride + 1})
    with np.load(output / "target/trajectory.npz") as archive:
        assert archive["trajectory"].shape == shape and np.isfinite(archive["trajectory"]).all()
        assert float(archive["E"]) == config["target"]["E"]
    with (output / "adjacent.csv").open() as stream:
        adjacent = list(csv.DictReader(stream))
    assert len(adjacent) == 2 * config["radius"]
    result = {"points": rows, "target_finite": True, "grid_size": config["native"]["grid_size"],
              "cell_size_m": 2 / config["native"]["grid_size"] / config["sim_scale"],
              "adjacent_rms_um_range": [min(float(r["trajectory_rms_mm"]) for r in adjacent) * 1000,
                                        max(float(r["trajectory_rms_mm"]) for r in adjacent) * 1000]}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "points"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "audit"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--snapshot-stride", type=int, default=200)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data, args.settings, args.output, args.step_size, args.radius, args.grid_size, args.substeps)
    elif args.stage == "run":
        run(args.output, args.stride, args.snapshot_stride)
    else:
        audit(args.output, args.snapshot_stride)


if __name__ == "__main__":
    main()
