"""Prepare/run the native five-point material sweep and center-only AD.

Template commands (activate mpmavatar; from MPMAvatar):
# python material_sensitivity.py --config configs/material_sensitivity_185.json --output output/material_sensitivity_185 --stage prepare --garment both --parameter-set all
# python material_sensitivity.py --config configs/material_sensitivity_185.json --output output/material_sensitivity_185 --stage run --garment both --parameter-set all --resume
# python material_sensitivity.py --config configs/material_sensitivity_185.json --output output/material_sensitivity_185 --stage summarize --garment both --parameter-set all
# python material_sensitivity.py --config configs/material_sensitivity_185.json --output output/material_sensitivity_185 --stage audit --garment both --parameter-set all

Run requires a GPU already available to this process. No scheduler is invoked.
All derivatives use exported D/E/H units; internal E is exported E / 100.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from train_material_params import Trainer

REPO = Path(__file__).resolve().parent
PARAMETERS = ("D", "E", "H")
TABLES = ("samples", "trajectory_by_frame", "fd", "ad")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    assert rows, path
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def effective_parameters(values: dict[str, float]) -> dict[str, float]:
    """Scalar values actually passed to the native float32 material kernels."""
    return {key: float32(float32(values[key] / 100.0) * 100.0) if key == "E"
            else float32(values[key]) for key in PARAMETERS}


def plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    assert set(config["garments"]) == {"upper", "lower"}
    assert set(config["parameter_sets"]) == {"soft", "default", "stiff"}
    assert config["directions"] == list(PARAMETERS)
    assert config["offsets"] == [-2, -1, 0, 1, 2]
    assert config["step_sizes"] == [1e-6, 1e-5, 1e-4]
    assert config["grid_size"] > 0 and config["substeps"] > 0
    rows = []
    for garment, window in config["garments"].items():
        assert window["frame_count"] >= 2
        for name, center in config["parameter_sets"].items():
            assert set(center) == set(PARAMETERS)
            for direction in config["directions"]:
                for step in config["step_sizes"]:
                    group = f"{garment}/{name}/{direction}/h{step:.0e}"
                    for offset in config["offsets"]:
                        values = {key: float(center[key]) for key in PARAMETERS}
                        values[direction] += offset * step
                        assert all(math.isfinite(value) and value > 0 for value in values.values())
                        effective = effective_parameters(values)
                        rows.append({"garment": garment, "parameter_set": name,
                                     "direction": direction, "step_size": step, "offset": offset,
                                     **window, **values,
                                     **{f"effective_{key}": effective[key] for key in PARAMETERS},
                                     "trajectory": f"{group}/offset{offset:+d}.npz"})
    assert len(rows) == 270
    return rows


def prepare(config_path: Path, output: Path) -> None:
    assert config_path.is_file(), config_path
    assert not output.exists(), f"Choose a fresh output: {output}"
    config = json.loads(config_path.read_text())
    prepared = (REPO / config["prepared"]).resolve()
    config["prepared"] = str(prepared)
    assert (prepared / "preparation.json").is_file(), prepared
    metadata = json.loads((prepared / "preparation.json").read_text())
    name = f"s{metadata['train']['subject']}_t{metadata['train']['take']}"
    assets = prepared / "data" / name
    assert (assets / "mesh_processed.obj").is_file()
    assert (assets / "optimized_weights.npy").is_file()
    assert (prepared / "model" / name / "point_cloud" / "timestep_030000" / "point_cloud.ply").is_file()
    assert (prepared / "output" / "tracking" / f"{name}_{metadata['train']['start']}_100").is_dir()
    for garment, window in config["garments"].items():
        assert (assets / f"split_idx_{garment}.npz").is_file()
        assert metadata["train"]["start"] <= window["frame_start"]
        assert window["frame_start"] + window["frame_count"] <= metadata["train"]["start"] + metadata["train"]["count"]
    rows = plan(config)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_csv(output / "manifest.csv", rows)
    collapsed = sum(row["offset"] != 0 and row[f"effective_{row['direction']}"] ==
                    effective_parameters(config["parameter_sets"][row["parameter_set"]])[row["direction"]]
                    for row in rows)
    print(f"Prepared {len(rows)} native rollouts and 6 center AD evaluations: {output}", flush=True)
    print(f"Float32 rounds {collapsed} nonzero perturbations to the center; requested steps are preserved.", flush=True)


def build_trainer(config: dict[str, Any], output: Path, garment: str, parameter_set: str) -> Trainer:
    if config.get("deterministic", False):
        from simulation_runtime import configure_simulation
        configure_simulation(True)
    from train_material_params import Trainer, parse_args

    prepared = Path(config["prepared"])
    metadata = json.loads((prepared / "preparation.json").read_text())
    train, test = metadata["train"], metadata["test"]
    name = f"s{train['subject']}_t{train['take']}"
    window = config["garments"][garment]
    assets = prepared / "data" / name
    # Match the native physics scripts' two-frame camera/body setup window.
    sys.argv = [str(REPO / "train_material_params.py"),
                "--save_name", f"{garment}_{parameter_set}", "--output_dir", str(output / "trainer"),
                "--trained_model_path", str(prepared / "output" / "tracking" / f"{name}_{train['start']}_100"),
                "--model_path", str(prepared / "model" / name),
                "--dataset_dir", str(prepared / "data"), "--dataset_type", "4ddress",
                "--smplx_gender", train["gender"], "--subject", str(train["subject"]),
                "--train_take", str(train["take"]), "--test_take", str(test["take"]),
                "--verts_start_idx", str(train["start"]),
                "--split_idx_path", str(assets / f"split_idx_{garment}.npz"),
                "--uv_path", str(assets / "mesh_processed.obj"), "--test_camera_index", "0",
                "--train_frame_start_num", str(window["frame_start"]), str(window["frame_count"]),
                "--test_frame_start_num", str(window["frame_start"]), "2",
                "--iterations", "200", "--init_params_path", "", "--material_gradient", "ad",
                "--grid_size", str(config["grid_size"]), "--substep", str(config["substeps"]),
                "--seed", str(config["seed"])]
    if config.get("deterministic", False):
        sys.argv.append("--deterministic")
    args, opt, pipe, run_eval, *_ = parse_args()
    trainer = Trainer(args, opt, pipe, run_eval)
    assert trainer.n_traditional == 0
    return trainer


def set_parameters(trainer: Trainer, values: dict[str, float]) -> None:
    import torch

    with torch.no_grad():
        for key in PARAMETERS:
            trainer.torch_param[key].fill_(values[key] / (100.0 if key == "E" else 1.0))


def center_ad(trainer: Trainer, garment: str, name: str, center: dict[str, float]) -> dict[str, Any]:
    import torch

    set_parameters(trainer, center)
    trainer.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, _ = trainer._autodiff_material(str(trainer.particle_init_position.device))
    assert torch.isfinite(loss), f"Nonfinite AD loss: {garment}/{name}"
    loss.backward()
    torch.cuda.synchronize()
    gradients = {}
    for key in PARAMETERS:
        assert trainer.torch_param[key].grad is not None
        value = float(trainer.torch_param[key].grad) / (100.0 if key == "E" else 1.0)
        if not math.isfinite(value):
            print(f"WARNING: AD backward returned {value} for {garment}/{name}/{key}; "
                  "saving the measured failure and continuing the native sensitivity sweep.", flush=True)
        gradients[f"ad_{key}"] = value
    return {"garment": garment, "parameter_set": name, **center,
            "ad_loss": float(loss.detach()), **gradients, "seconds": time.perf_counter() - start}


def central_differences(values: dict[int, Any], step: float) -> dict[str, Any]:
    """Requested-step derivatives of a scalar loss or a trajectory array."""
    return {"h": (values[1] - values[-1]) / (2.0 * step),
            "2h": (values[2] - values[-2]) / (4.0 * step),
            "five_point": (8.0 * (values[1] - values[-1]) - (values[2] - values[-2])) / (12.0 * step)}


def relative_error(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0 else 0.0


def save_group(
    output: Path, rows: list[dict[str, Any]], positions: dict[int, np.ndarray],
    losses: dict[int, float], ad: dict[str, Any], frame_ids: np.ndarray,
    faces: np.ndarray, vertex_ids: np.ndarray, num_joints: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    center = positions[0].astype(np.float64)
    first = rows[0]
    step, direction = first["step_size"], first["direction"]
    group = {key: first[key] for key in ("garment", "parameter_set", "direction", "step_size")}
    center_value = next(row[f"effective_{direction}"] for row in rows if row["offset"] == 0)
    sample_rows, frame_rows = [], []
    for row in rows:
        offset = row["offset"]
        delta = positions[offset].astype(np.float64) - center
        distances = np.linalg.norm(delta, axis=-1)
        effective_offset = row[f"effective_{direction}"] - center_value
        resolved = offset == 0 or effective_offset != 0
        path = output / row["trajectory"]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, positions=positions[offset], frame_ids=frame_ids, faces=faces,
                 vertex_ids=vertex_ids, num_joints=num_joints, loss=losses[offset],
                 requested_parameters=np.asarray([row[key] for key in PARAMETERS]),
                 effective_parameters=np.asarray([row[f"effective_{key}"] for key in PARAMETERS]),
                 parameter_names=np.asarray(PARAMETERS), v2v_center_m=distances)
        sample_rows.append({**row, "effective_offset": effective_offset, "perturbation_resolved": resolved,
                            "loss": losses[offset], "center_loss": losses[0],
                            "loss_difference": losses[offset] - losses[0],
                            "loss_secant_from_center": (losses[offset] - losses[0]) / (offset * step) if offset else "",
                            "v2v_mean_m": float(distances.mean()), "v2v_max_m": float(distances.max()),
                            "v2v_final_mean_m": float(distances[-1].mean()),
                            "ad_center": ad[f"ad_{direction}"]})
        for frame_id, frame_distances in zip(frame_ids, distances):
            frame_rows.append({**group, "offset": offset, "frame_id": int(frame_id),
                               "v2v_mean_m": float(frame_distances.mean()),
                               "v2v_max_m": float(frame_distances.max())})
    fd = central_differences(losses, step)
    dpositions = central_differences({k: v.astype(np.float64) for k, v in positions.items()}, step)
    fd_path = Path(first["trajectory"]).parent / "fd.npz"
    np.savez(output / fd_path, **{f"dpositions_dparameter_{key}": value for key, value in dpositions.items()},
             frame_ids=frame_ids, vertex_ids=vertex_ids, parameter=direction, step_size=step)
    effective = {row["offset"]: row[f"effective_{direction}"] for row in rows}
    fd_row = {**group, "center_loss": losses[0], "ad_loss": ad["ad_loss"],
              "ad_native_loss_difference": ad["ad_loss"] - losses[0],
              "ad": ad[f"ad_{direction}"], "distinct_effective_values": len(set(effective.values())),
              "stencil_resolved": len(set(effective.values())) == 5,
              "trajectory_fd": str(fd_path)}
    for key, value in fd.items():
        fd_row[f"fd_{key}"] = value
        fd_row[f"ad_fd_{key}_absolute_error"] = abs(value - ad[f"ad_{direction}"])
        fd_row[f"ad_fd_{key}_relative_error"] = relative_error(value, ad[f"ad_{direction}"])
    for radius in (1, 2):
        span = effective[radius] - effective[-radius]
        fd_row[f"effective_span_{radius}h"] = span
        fd_row[f"fd_effective_{radius}h"] = (losses[radius] - losses[-radius]) / span if span else ""
    return sample_rows, frame_rows, fd_row


def save_progress(result_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    """Commit a whole stencil before exporting its derived CSVs."""
    for row in tables["fd"]:
        for key in ("h", "2h", "five_point"):
            row[f"ad_fd_{key}_absolute_error"] = abs(row[f"fd_{key}"] - row["ad"])
            row[f"ad_fd_{key}_relative_error"] = relative_error(row[f"fd_{key}"], row["ad"])
    temporary = result_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(tables, indent=2) + "\n")
    temporary.replace(result_dir / "progress.json")
    for key, rows in tables.items():
        if rows:
            write_csv(result_dir / f"{key}.csv", rows)


def run_case(config: dict[str, Any], output: Path, garment: str, name: str, resume: bool) -> None:
    import numpy as np
    import torch
    from replay_material_window import replay

    assert torch.cuda.is_available(), "Run on a GPU; this script never requests an allocation"
    result_dir = output / garment / name
    assert resume or not result_dir.exists(), f"Use --resume for {result_dir}"
    result_dir.mkdir(parents=True, exist_ok=resume)
    checkpoint = result_dir / "progress.json"
    tables: dict[str, list[dict[str, Any]]] = {key: [] for key in TABLES}
    if resume and checkpoint.is_file():
        tables = json.loads(checkpoint.read_text())
        assert set(tables) == set(TABLES)
        assert len(tables["samples"]) == 5 * len(tables["fd"]) <= 45
        assert len(tables["ad"]) <= 1
        save_progress(result_dir, tables)
    completed = {(row["direction"], row["step_size"]) for row in tables["fd"]}
    if len(completed) == 9:
        (result_dir / "complete").touch()
        print(f"Case already complete: {garment}/{name}", flush=True)
        return
    trainer = build_trainer(config, output, garment, name)
    all_rows = [row for row in plan(config) if row["garment"] == garment and row["parameter_set"] == name]
    frame_ids = np.asarray(trainer.scene.train_frame_index)
    faces = trainer.new_cloth_faces.cpu().numpy()
    vertex_ids = trainer.reordered_cloth_v_idx.cpu().numpy()
    center = config["parameter_sets"][name]
    if tables["ad"]:
        ad = tables["ad"][0]
    else:
        ad = center_ad(trainer, garment, name, center)
        tables["ad"].append(ad)
        save_progress(result_dir, tables)
    for direction in config["directions"]:
        for step in config["step_sizes"]:
            if (direction, step) in completed:
                continue
            rows = [row for row in all_rows if row["direction"] == direction and row["step_size"] == step]
            positions, losses = {}, {}
            for row in sorted(rows, key=lambda row: (abs(row["offset"]), row["offset"])):
                set_parameters(trainer, {key: row[key] for key in PARAMETERS})
                print(f"Native {row['trajectory']}: D/E/H = {row['D']}/{row['E']}/{row['H']}", flush=True)
                positions[row["offset"]], losses[row["offset"]] = replay(trainer)
                assert math.isfinite(losses[row["offset"]]), row
            sample_rows, frame_rows, fd_row = save_group(
                output, rows, positions, losses, ad, frame_ids, faces, vertex_ids, trainer.num_joint_v)
            tables["samples"].extend(sample_rows)
            tables["trajectory_by_frame"].extend(frame_rows)
            tables["fd"].append(fd_row)
            save_progress(result_dir, tables)
            print(json.dumps(fd_row), flush=True)
            if not fd_row["stencil_resolved"]:
                print(f"WARNING: {garment}/{name}/{direction} h={step} has only "
                      f"{fd_row['distinct_effective_values']} distinct float32 values; "
                      "FD can contain CUDA repeat noise from identical inputs.", flush=True)
    assert len(tables["samples"]) == 45 and len(tables["fd"]) == 9 and len(tables["ad"]) == 1
    (result_dir / "complete").touch()


def run_garment(config: dict[str, Any], output: Path, garment: str,
                parameter_set: str = "all", resume: bool = False) -> None:
    names = list(config["parameter_sets"]) if parameter_set == "all" else [parameter_set]
    for name in names:
        run_case(config, output, garment, name, resume)


def summarize(config: dict[str, Any], output: Path) -> None:
    for garment in config["garments"]:
        for name in config["parameter_sets"]:
            result_dir = output / garment / name
            checkpoint = result_dir / "progress.json"
            assert checkpoint.is_file(), checkpoint
            save_progress(result_dir, json.loads(checkpoint.read_text()))
    counts = {"samples": 270, "trajectory_by_frame": 135 * sum(w["frame_count"] for w in config["garments"].values()),
              "fd": 54, "ad": 6}
    for table in TABLES:
        rows = []
        for garment in config["garments"]:
            garment_rows = []
            for name in config["parameter_sets"]:
                path = output / garment / name / f"{table}.csv"
                assert path.is_file(), path
                with path.open(newline="") as stream:
                    case_rows = list(csv.DictReader(stream))
                if table == "ad":
                    for row in case_rows:
                        nonfinite = [key for key in PARAMETERS if not math.isfinite(float(row[f"ad_{key}"]))]
                        row["ad_status"] = "nonfinite" if nonfinite else "finite"
                        row["nonfinite_parameters"] = ",".join(nonfinite)
                garment_rows.extend(case_rows)
            write_csv(output / garment / f"{table}.csv", garment_rows)
            rows.extend(garment_rows)
        assert len(rows) == counts[table], f"Incomplete {table}: {len(rows)} / {counts[table]}"
        write_csv(output / f"{table}.csv", rows)
    print(f"Saved combined CSVs: {output}", flush=True)


def audit(config: dict[str, Any], output: Path) -> None:
    """Check every saved trajectory and derivative against the requested matrix."""
    import numpy as np

    def read_table(name: str) -> list[dict[str, str]]:
        with (output / f"{name}.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    expected = plan(config)
    samples = read_table("samples")
    assert len(samples) == len(expected) == 270
    actual = {row["trajectory"]: row for row in samples}
    assert len(actual) == 270
    assert set(actual) == {row["trajectory"] for row in expected}
    ads = {(row["garment"], row["parameter_set"]): row for row in read_table("ad")}
    assert len(ads) == 6
    for (garment, name), row in ads.items():
        assert {key: float(row[key]) for key in PARAMETERS} == config["parameter_sets"][name]
        assert math.isfinite(float(row["ad_loss"]))
        nonfinite = [key for key in PARAMETERS if not math.isfinite(float(row[f"ad_{key}"]))]
        assert row["ad_status"] == ("nonfinite" if nonfinite else "finite")
        assert row["nonfinite_parameters"] == ",".join(nonfinite)
    fd_rows = read_table("fd")
    assert len(fd_rows) == 54
    fds = {(row["garment"], row["parameter_set"], row["direction"], float(row["step_size"])): row for row in fd_rows}
    assert len(fds) == 54
    frame_rows = read_table("trajectory_by_frame")
    frames = {(row["garment"], row["parameter_set"], row["direction"], float(row["step_size"]),
               int(row["offset"]), int(row["frame_id"])): row for row in frame_rows}
    assert len(frames) == len(frame_rows) == 135 * sum(w["frame_count"] for w in config["garments"].values())
    summary = []
    for group, fd_row in fds.items():
        garment, name, direction, step = group
        rows = [row for row in expected if (row["garment"], row["parameter_set"], row["direction"], row["step_size"]) == group]
        assert len(rows) == 5
        positions, losses, distances = {}, {}, {}
        window = config["garments"][garment]
        frame_ids = np.arange(window["frame_start"], window["frame_start"] + window["frame_count"])
        center_path = output / next(row["trajectory"] for row in rows if row["offset"] == 0)
        with np.load(center_path, allow_pickle=False) as center:
            center_positions = center["positions"].astype(np.float64)
            center_loss = float(center["loss"])
            vertex_ids = center["vertex_ids"]
            faces = center["faces"]
        for row in rows:
            saved_row = actual[row["trajectory"]]
            for key in ("garment", "parameter_set", "direction"):
                assert saved_row[key] == row[key]
            for key in ("offset", "frame_start", "frame_count", "step_size", *PARAMETERS):
                assert float(saved_row[key]) == row[key], (row["trajectory"], key)
            with np.load(output / row["trajectory"], allow_pickle=False) as saved:
                assert "target" not in saved.files
                assert saved["positions"].shape == (len(frame_ids), len(vertex_ids), 3)
                assert np.isfinite(saved["positions"]).all(), row["trajectory"]
                np.testing.assert_array_equal(saved["frame_ids"], frame_ids)
                np.testing.assert_array_equal(saved["vertex_ids"], vertex_ids)
                np.testing.assert_array_equal(saved["faces"], faces)
                np.testing.assert_array_equal(saved["requested_parameters"], [row[key] for key in PARAMETERS])
                np.testing.assert_array_equal(saved["effective_parameters"], [row[f"effective_{key}"] for key in PARAMETERS])
                offset = row["offset"]
                positions[offset] = saved["positions"].astype(np.float64)
                losses[offset] = float(saved["loss"])
                assert math.isfinite(losses[offset]) and losses[offset] == float(saved_row["loss"])
                distances[offset] = np.linalg.norm(positions[offset] - center_positions, axis=-1)
                np.testing.assert_array_equal(saved["v2v_center_m"], distances[offset])
            np.testing.assert_allclose(float(saved_row["v2v_mean_m"]), distances[offset].mean(), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(float(saved_row["v2v_max_m"]), distances[offset].max(), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(float(saved_row["v2v_final_mean_m"]), distances[offset][-1].mean(), rtol=1e-12, atol=1e-15)
            assert float(saved_row["center_loss"]) == center_loss
            assert float(saved_row["loss_difference"]) == losses[offset] - center_loss
            np.testing.assert_allclose(float(saved_row["ad_center"]), float(ads[(garment, name)][f"ad_{direction}"]),
                                       rtol=0, atol=0, equal_nan=True)
            if offset:
                assert float(saved_row["loss_secant_from_center"]) == (losses[offset] - center_loss) / (offset * step)
            for frame_id, frame_distances in zip(frame_ids, distances[offset]):
                frame_row = frames[(*group, offset, int(frame_id))]
                np.testing.assert_allclose([float(frame_row["v2v_mean_m"]), float(frame_row["v2v_max_m"])],
                                           [frame_distances.mean(), frame_distances.max()], rtol=1e-12, atol=1e-15)
        derivatives = central_differences(positions, step)
        loss_derivatives = central_differences(losses, step)
        with np.load(output / fd_row["trajectory_fd"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["frame_ids"], frame_ids)
            for key, value in derivatives.items():
                np.testing.assert_array_equal(saved[f"dpositions_dparameter_{key}"], value)
                assert float(fd_row[f"fd_{key}"]) == loss_derivatives[key]
        np.testing.assert_allclose(float(fd_row["ad"]), float(ads[(garment, name)][f"ad_{direction}"]),
                                   rtol=0, atol=0, equal_nan=True)
        for key, value in loss_derivatives.items():
            np.testing.assert_allclose(float(fd_row[f"ad_fd_{key}_absolute_error"]), abs(value - float(fd_row["ad"])),
                                       rtol=0, atol=0, equal_nan=True)
            np.testing.assert_allclose(float(fd_row[f"ad_fd_{key}_relative_error"]), relative_error(value, float(fd_row["ad"])),
                                       rtol=0, atol=0, equal_nan=True)
        summary.append({"garment": garment, "parameter_set": name, "direction": direction,
                        "step_size": step, "ad": float(fd_row["ad"]), "fd_h": float(fd_row["fd_h"]),
                        "ad_fd_h_relative_error": float(fd_row["ad_fd_h_relative_error"]),
                        "stencil_resolved": fd_row["stencil_resolved"],
                        "maximum_v2v_m": max(float(value.max()) for value in distances.values())})
    case_summary = []
    for (garment, name), ad in ads.items():
        case_samples = [row for row in samples if (row["garment"], row["parameter_set"]) == (garment, name)]
        center_losses = [float(row["loss"]) for row in case_samples if int(row["offset"]) == 0]
        assert len(case_samples) == 45 and len(center_losses) == 9
        case_summary.append({"garment": garment, "parameter_set": name,
                             "ad_status": ad["ad_status"], "nonfinite_parameters": ad["nonfinite_parameters"],
                             **{key: float(ad[key]) for key in ("D", "E", "H", "ad_loss", "ad_D", "ad_E", "ad_H")},
                             "native_center_loss_min": min(center_losses), "native_center_loss_max": max(center_losses),
                             "native_center_loss_span": max(center_losses) - min(center_losses),
                             "ad_native_max_loss_relative_difference": max(relative_error(value, float(ad["ad_loss"])) for value in center_losses),
                             "maximum_vertex_distance_m": max(float(row["v2v_max_m"]) for row in case_samples),
                             "maximum_mean_vertex_distance_m": max(float(row["v2v_mean_m"]) for row in case_samples),
                             "unresolved_nonzero_perturbations": sum(int(row["offset"]) != 0 and row["perturbation_resolved"] == "False" for row in case_samples)})
    write_csv(output / "summary.csv", case_summary)
    report = {"verified": True, "trajectories": len(samples), "center_ad_evaluations": len(ads),
              "fd_stencils": len(fds), "trajectory_frame_rows": len(frames),
              "unresolved_stencils": sum(row["stencil_resolved"] == "False" for row in summary),
              "nonfinite_ad_centers": [{"garment": row["garment"], "parameter_set": row["parameter_set"],
                                        "parameters": row["nonfinite_parameters"]}
                                       for row in case_summary if row["ad_status"] == "nonfinite"],
              "groups": summary, "case_summary": case_summary}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(samples)} trajectories, {len(ads)} AD centers, {len(fds)} FD stencils, "
          f"and {len(frames)} per-frame CSV rows.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/material_sensitivity_185.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize", "audit"), required=True)
    parser.add_argument("--garment", choices=("both", "upper", "lower"), default="both")
    parser.add_argument("--parameter-set", choices=("all", "soft", "default", "stiff"), default="all")
    parser.add_argument("--resume", action="store_true", help="Reuse completed AD evaluations and five-point stencils")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        assert args.garment == "both" and args.parameter_set == "all", "Prepare the complete 270-point matrix"
        prepare(args.config, output)
        return
    assert (output / "config.json").is_file(), "Run --stage prepare first"
    config = json.loads((output / "config.json").read_text())
    if args.stage == "summarize":
        summarize(config, output)
    elif args.stage == "audit":
        audit(config, output)
    elif args.garment == "both":
        # Separate processes release each garment's native/AD grid buffers.
        for garment in config["garments"]:
            names = config["parameter_sets"] if args.parameter_set == "all" else [args.parameter_set]
            for name in names:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                                "--stage", "run", "--garment", garment, "--parameter-set", name,
                                *(["--resume"] if args.resume else [])], cwd=REPO, check=True)
        if args.parameter_set == "all":
            summarize(config, output)
    else:
        run_garment(config, output, args.garment, args.parameter_set, args.resume)


if __name__ == "__main__":
    main()
