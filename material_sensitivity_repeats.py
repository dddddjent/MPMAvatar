"""Nine fresh native sweeps and nine center AD evaluations per material case.

Activate mpmavatar; commands from MPMAvatar:
# python material_sensitivity_repeats.py --output output/material_sensitivity_185_repeats --stage prepare --config configs/material_sensitivity_185.json
# python material_sensitivity_repeats.py --output output/material_sensitivity_185_repeats --stage run --repeat 1 --garment upper --parameter-set soft
# python material_sensitivity_repeats.py --output output/material_sensitivity_185_repeats --stage summarize
# python material_sensitivity_repeats.py --output output/material_sensitivity_185_repeats --stage audit

Run on an allocated GPU; summarize/audit on an owned CPU allocation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import material_sensitivity as native

REPEATS = 9


def repeat_dir(output: Path, repeat: int) -> Path:
    assert 1 <= repeat <= REPEATS
    return output / f"repeat{repeat:02d}"


def read_csv(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), path
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def prepare(config_path: Path, output: Path) -> None:
    native.prepare(config_path, output)
    config = json.loads((output / "config.json").read_text())
    config["repeats"] = REPEATS
    config["plot_baseline"] = "mean of nine native center losses in each stencil"
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = []
    for repeat in range(1, REPEATS + 1):
        directory = repeat_dir(output, repeat)
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        rows = native.plan(config)
        native.write_csv(directory / "manifest.csv", rows)
        manifest.extend({"repeat": repeat, **row, "trajectory": f"{directory.name}/{row['trajectory']}"}
                        for row in rows)
    native.write_csv(output / "manifest.csv", manifest)
    assert len(manifest) == 2430
    print(f"Prepared {len(manifest)} fresh native trajectories and 54 fresh AD evaluations: {output}", flush=True)


def run_case(config: dict[str, Any], output: Path, repeat: int, garment: str, name: str) -> None:
    import numpy as np
    import torch
    from replay_material_window import replay

    assert torch.cuda.is_available(), "Use an allocated GPU"
    directory = repeat_dir(output, repeat)
    case_dir = directory / garment / name
    case_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = case_dir / "progress.json"
    tables = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {key: [] for key in native.TABLES}
    assert set(tables) == set(native.TABLES)
    assert len(tables["ad"]) <= 1 and len(tables["samples"]) == 5 * len(tables["fd"]) <= 45
    completed = {(row["direction"], row["step_size"]) for row in tables["fd"]}
    if len(completed) == 9:
        # A timeout can interrupt CSV export after the checkpoint is committed.
        native.save_progress(case_dir, tables)
        (case_dir / "complete").touch()
        return
    trainer = native.build_trainer(config, directory, garment, name)
    metadata = {"repeat": repeat, "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
                "started_at": datetime.now(timezone.utc).isoformat()}
    if not tables["ad"]:
        tables["ad"].append({**native.center_ad(trainer, garment, name, config["parameter_sets"][name]), **metadata})
        native.save_progress(case_dir, tables)
    ad = tables["ad"][0]
    rows = [row for row in native.plan(config) if (row["garment"], row["parameter_set"]) == (garment, name)]
    frame_ids = np.asarray(trainer.scene.train_frame_index)
    faces = trainer.new_cloth_faces.cpu().numpy()
    vertex_ids = trainer.reordered_cloth_v_idx.cpu().numpy()
    for direction in config["directions"]:
        for step in config["step_sizes"]:
            if (direction, step) in completed:
                continue
            group = [row for row in rows if (row["direction"], row["step_size"]) == (direction, step)]
            positions, losses, seconds = {}, {}, {}
            for row in sorted(group, key=lambda row: (abs(row["offset"]), row["offset"])):
                native.set_parameters(trainer, {key: row[key] for key in native.PARAMETERS})
                print(f"Repeat {repeat}: native {row['trajectory']}", flush=True)
                start = time.perf_counter()
                positions[row["offset"]], losses[row["offset"]] = replay(trainer)
                seconds[row["offset"]] = time.perf_counter() - start
                assert math.isfinite(losses[row["offset"]]), row
            samples, frames, fd = native.save_group(directory, group, positions, losses, ad,
                                                    frame_ids, faces, vertex_ids, trainer.num_joint_v)
            tables["samples"].extend({**row, **metadata, "seconds": seconds[row["offset"]]} for row in samples)
            tables["trajectory_by_frame"].extend({**row, **metadata} for row in frames)
            tables["fd"].append({**fd, **metadata})
            native.save_progress(case_dir, tables)
            print(f"Saved repeat {repeat} {garment}/{name} {direction} h={step}", flush=True)
    assert len(tables["samples"]) == 45 and len(tables["fd"]) == 9 and len(tables["ad"]) == 1
    (case_dir / "complete").touch()


def statistics(values: list[float]) -> tuple[float, float, float]:
    assert values and all(math.isfinite(value) for value in values)
    low, high = min(values), max(values)
    # Preserve a constant value exactly; sum/divide can move its mean by one ulp.
    mean = low if low == high else math.fsum(values) / len(values)
    return low, mean, high


def summarize(config: dict[str, Any], output: Path) -> None:
    combined: dict[str, list[dict[str, Any]]] = {key: [] for key in native.TABLES}
    for repeat in range(1, REPEATS + 1):
        directory = repeat_dir(output, repeat)
        for garment in config["garments"]:
            for name in config["parameter_sets"]:
                assert (directory / garment / name / "complete").is_file()
        native.summarize(config, directory)
        for table in native.TABLES:
            for row in read_csv(directory / f"{table}.csv"):
                assert int(row["repeat"]) == repeat
                for key in ("trajectory", "trajectory_fd"):
                    if key in row:
                        row[key] = f"{directory.name}/{row[key]}"
                combined[table].append(row)
    for table, count in {"samples": 2430, "ad": 54, "fd": 486, "trajectory_by_frame": 29160}.items():
        assert len(combined[table]) == count, (table, len(combined[table]))
        native.write_csv(output / f"{table}.csv", combined[table])

    points = []
    for planned in native.plan(config):
        keys = ("garment", "parameter_set", "direction", "step_size", "offset")
        matches = [row for row in combined["samples"] if all(str(row[key]) == str(planned[key]) for key in keys)]
        assert len(matches) == REPEATS and {int(row["repeat"]) for row in matches} == set(range(1, REPEATS + 1))
        centers = [row for row in combined["samples"]
                   if all(str(row[key]) == str(planned[key]) for key in keys[:-1]) and int(row["offset"]) == 0]
        assert len(centers) == REPEATS
        center_mean = statistics([float(row["loss"]) for row in centers])[1]
        assert center_mean > 0
        low, mean, high = statistics([float(row["loss"]) for row in matches])
        point = {key: planned[key] for key in keys}
        point["effective_parameter"] = planned[f"effective_{planned['direction']}"]
        point.update({"count": REPEATS, "center_mean": center_mean,
                      "loss_min": low, "loss_mean": mean, "loss_max": high,
                      "change_min": low - center_mean, "change_mean": mean - center_mean,
                      "change_max": high - center_mean})
        for metric in ("min", "mean", "max"):
            point[f"percent_{metric}"] = 100.0 * point[f"change_{metric}"] / center_mean
        points.append(point)
    native.write_csv(output / "point_statistics.csv", points)
    ads = []
    for garment in config["garments"]:
        for name in config["parameter_sets"]:
            rows = [row for row in combined["ad"] if (row["garment"], row["parameter_set"]) == (garment, name)]
            assert len(rows) == REPEATS
            for direction in native.PARAMETERS:
                values = [float(row[f"ad_{direction}"]) for row in rows]
                finite = [value for value in values if math.isfinite(value)]
                low, mean, high = statistics(finite) if finite else (math.nan, math.nan, math.nan)
                ads.append({"garment": garment, "parameter_set": name, "direction": direction,
                            "count": REPEATS, "finite_count": len(finite), "nonfinite_count": REPEATS - len(finite),
                            "ad_min": low, "ad_mean": mean, "ad_max": high})
    native.write_csv(output / "ad_statistics.csv", ads)
    print("Saved 270 nine-run point distributions and 18 nine-run AD distributions.", flush=True)


def audit(config: dict[str, Any], output: Path) -> None:
    for repeat in range(1, REPEATS + 1):
        native.audit(config, repeat_dir(output, repeat))
    samples = read_csv(output / "samples.csv")
    ads = read_csv(output / "ad.csv")
    fds = read_csv(output / "fd.csv")
    frames = read_csv(output / "trajectory_by_frame.csv")
    points = read_csv(output / "point_statistics.csv")
    assert len(samples) == 2430 and len({row["trajectory"] for row in samples}) == 2430
    assert len(ads) == 54 and len(points) == 270
    assert len(fds) == 486 and len(frames) == 29160
    for point in points:
        keys = ("garment", "parameter_set", "direction", "step_size", "offset")
        rows = [row for row in samples if all(row[key] == point[key] for key in keys)]
        assert len(rows) == 9 and {int(row["repeat"]) for row in rows} == set(range(1, 10))
        assert all(float(row[f"effective_{point['direction']}"]) == float(point["effective_parameter"])
                   for row in rows)
        center_rows = [row for row in samples if all(row[key] == point[key] for key in keys[:-1])
                       and int(row["offset"]) == 0]
        mean_center = statistics([float(row["loss"]) for row in center_rows])[1]
        assert float(point["center_mean"]) == mean_center
        for metric, value in zip(("min", "mean", "max"), statistics([float(row["loss"]) for row in rows])):
            assert float(point[f"loss_{metric}"]) == value
            assert float(point[f"change_{metric}"]) == value - mean_center
            assert float(point[f"percent_{metric}"]) == 100.0 * (value - mean_center) / mean_center
        if int(point["offset"]) == 0:
            assert float(point["change_mean"]) == 0
    ad_stats = read_csv(output / "ad_statistics.csv")
    assert len(ad_stats) == 18
    for stat in ad_stats:
        rows = [row for row in ads if (row["garment"], row["parameter_set"]) ==
                (stat["garment"], stat["parameter_set"])]
        assert len(rows) == 9 and {int(row["repeat"]) for row in rows} == set(range(1, 10))
        values = [float(row[f"ad_{stat['direction']}"]) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        assert len(finite) == int(stat["finite_count"])
        assert 9 - len(finite) == int(stat["nonfinite_count"])
        expected = statistics(finite) if finite else (math.nan, math.nan, math.nan)
        for metric, value in zip(("min", "mean", "max"), expected):
            actual = float(stat[f"ad_{metric}"])
            assert actual == value or (math.isnan(actual) and math.isnan(value))
    report = {"verified": True, "repeats_per_native_point": 9, "repeats_per_ad_center": 9,
              "trajectories": len(samples), "ad_evaluations": len(ads), "native_point_distributions": len(points),
              "fd_stencils": len(fds), "trajectory_frame_rows": len(frames),
              "gpu_models": sorted({row["gpu"] for row in samples}),
              "nonfinite_ad_distributions": [row for row in ad_stats if int(row["nonfinite_count"]) > 0],
              "baseline": config["plot_baseline"]}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified all 2430 trajectories, 54 AD evaluations, and nine-run distribution statistics.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=native.REPO / "configs/material_sensitivity_185.json")
    parser.add_argument("--stage", choices=("prepare", "run", "summarize", "audit"), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--garment", choices=("upper", "lower"), default="upper")
    parser.add_argument("--parameter-set", choices=("soft", "default", "stiff"), default="soft")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        prepare(args.config, output)
        return
    assert (output / "config.json").is_file()
    config = json.loads((output / "config.json").read_text())
    assert config["repeats"] == REPEATS
    if args.stage == "run":
        run_case(config, output, args.repeat, args.garment, args.parameter_set)
    elif args.stage == "summarize":
        summarize(config, output)
    else:
        audit(config, output)


if __name__ == "__main__":
    main()
