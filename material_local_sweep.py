"""Dense one-direction sweeps of D/E/H around the default material.

Each line (garment, direction, step) holds offsets k = -radius..radius with
ln(p) = ln(p_center) + k * step for log directions (E) and p = p_center + k * step
otherwise (D, H), its own center rollout, and one GPU model:
a line resumes only on the device name that started it, so deterministic
points within a line are comparable. Every rollout is saved as soon as it ends;
a rollout whose simulation goes nonfinite is saved as failed and skipped.

Template commands (activate mpmavatar; from MPMAvatar):
# python material_local_sweep.py --config configs/material_local_sweep_185.json --output output/material_local_sweep_185 --stage prepare
# python material_local_sweep.py --output output/material_local_sweep_185 --stage run --garment upper
# python material_local_sweep.py --output output/material_local_sweep_185 --stage summarize
# python material_local_sweep.py --config configs/material_local_sweep_dgarments.json --output output/material_local_sweep_dgarments --stage prepare
# python material_local_sweep.py --output output/material_local_sweep_dgarments --stage run --garment dress

Run requires a GPU already available to this process (inside a Slurm job). No scheduler is invoked.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from material_sensitivity import PARAMETERS, build_trainer, set_parameters, write_csv

if TYPE_CHECKING:
    from train_material_params import Trainer

REPO = Path(__file__).resolve().parent


def build_dgarments_trainer(config: dict[str, Any], output: Path, garment: str, parameter_set: str) -> "Trainer":
    """run.py's material-stage trainer with the rest shape at frame 0 and a later fitting window."""
    from simulation_runtime import configure_precision, configure_simulation

    configure_precision(config["precision"])
    configure_simulation(config["deterministic"])
    from dataset_input import common_arguments, load_manifest
    from train_material_params import Trainer, parse_args

    root, appearance = Path(config["data"]), Path(config["appearance"])
    manifest = load_manifest(root)
    window = config["garments"][garment]
    assert window["frame_start"] + window["frame_count"] <= len(manifest["train_frame_ids"])
    sys.argv = [str(REPO / "train_material_params.py"), *common_arguments(root, manifest, appearance),
                "--train_frame_start_num", str(window["frame_start"]), str(window["frame_count"]),
                "--test_frame_start_num", str(window["frame_start"]), "2",
                "--iterations", "200", "--output_dir", str(output / "trainer"),
                "--save_name", f"{garment}_{parameter_set}", "--smplx_num_betas", "10",
                "--prescribed_surface_path", str(root / manifest["prescribed_surface"]),
                "--init_params_path", "", "--material_gradient", "ad",
                "--grid_size", str(config["grid_size"]), "--substep", str(config["substeps"]),
                "--precision", config["precision"], "--seed", str(config["seed"])]
    if config["deterministic"]:
        sys.argv.append("--deterministic")
    args, opt, pipe, run_eval, *_ = parse_args()
    trainer = Trainer(args, opt, pipe, run_eval)
    assert trainer.n_traditional == 0
    return trainer


TRAINERS = {"4ddress": build_trainer, "dgarments": build_dgarments_trainer}


def line_name(garment: str, direction: str, step: float) -> str:
    return f"{garment}/{direction}/s{step:.0e}"


def offsets(radius: int) -> list[int]:
    """Center first, then outward so partial lines stay symmetric."""
    return [0] + [sign * k for k in range(1, radius + 1) for sign in (1, -1)]


def point_values(config: dict[str, Any], direction: str, step: float, k: int) -> dict[str, float]:
    values = {key: float(config["center"][key]) for key in PARAMETERS}
    if direction in config["log_directions"]:
        values[direction] *= math.exp(k * step)  # exact center at k = 0
    else:
        values[direction] += k * step
    assert values[direction] > 0, (direction, step, k)
    return values


def lines(config: dict[str, Any]) -> list[tuple[str, float]]:
    return [(direction, step) for step in config["step_sizes"] for direction in config["directions"]]


def prepare(config_path: Path, output: Path) -> None:
    assert config_path.is_file(), config_path
    assert not output.exists(), f"Choose a fresh output: {output}"
    config = json.loads(config_path.read_text())
    if config["dataset"] == "4ddress":
        config["prepared"] = str((REPO / config["prepared"]).resolve())
        assert (Path(config["prepared"]) / "preparation.json").is_file(), config["prepared"]
    else:
        assert config["dataset"] == "dgarments", config["dataset"]
        config["data"] = str((REPO / config["data"]).resolve())
        config["appearance"] = str((REPO / config["appearance"]).resolve())
        assert (Path(config["data"]) / "manifest.json").is_file(), config["data"]
        assert (Path(config["appearance"]) / "point_cloud").is_dir(), config["appearance"]
    assert config["precision"] == "fp64" and config["deterministic"]
    assert config["directions"] == list(PARAMETERS) and config["radius"] > 0
    assert set(config["log_directions"]) <= set(PARAMETERS)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    count = len(config["garments"]) * len(config["directions"]) * len(config["step_sizes"]) * (2 * config["radius"] + 1)
    print(f"Prepared {count} rollouts (one center per line): {output}", flush=True)


def job_end() -> float:
    job_id = os.environ["SLURM_JOB_ID"]
    end = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%e"], check=True, text=True,
                         stdout=subprocess.PIPE).stdout.strip()
    return datetime.fromisoformat(end).timestamp()


def run(config: dict[str, Any], output: Path, garment: str) -> None:
    import numpy as np
    import torch
    from replay_material_window import replay

    end = job_end()
    # build_trainer selects fp64/deterministic execution before CUDA starts.
    trainer = TRAINERS[config["dataset"]](config, output, garment, f"job{os.environ['SLURM_JOB_ID']}")
    assert torch.cuda.is_available(), "Run on a GPU; this script never requests an allocation"
    device = torch.cuda.get_device_name()
    assert trainer.particle_init_position.dtype == torch.float64
    reference = output / garment / "reference.npz"
    if not reference.is_file():
        idx = trainer.reordered_cloth_v_idx
        reference.parent.mkdir(parents=True, exist_ok=True)
        temporary = reference.with_suffix(f".{os.environ['SLURM_JOB_ID']}.npz")
        np.savez(temporary, target=trainer.train_frame_verts[:, idx].cpu().numpy(),
                 frame_ids=np.asarray(trainer.scene.train_frame_index), faces=trainer.new_cloth_faces.cpu().numpy(),
                 vertex_ids=idx.cpu().numpy(), num_joints=int(trainer.num_joint_v))
        temporary.replace(reference)
    (output / "locks").mkdir(exist_ok=True)
    longest = 0.0
    for direction, step in lines(config):
        name = line_name(garment, direction, step)
        line = output / name
        if (line / "complete").is_file():
            continue
        lock = (output / "locks" / f"{name.replace('/', '_')}.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        line.mkdir(parents=True, exist_ok=True)
        if (line / "device.txt").is_file():
            if (line / "device.txt").read_text().strip() != device:
                lock.close()
                continue
        else:
            (line / "device.txt").write_text(device + "\n")
            (line / "partition.txt").write_text(os.environ["SLURM_JOB_PARTITION"] + "\n")
        for k in offsets(config["radius"]):
            path = line / f"k{k:+03d}.npz"
            if path.is_file():
                continue
            if end - time.time() < 2.0 * max(longest, 120.0) + 60.0:
                print(f"Stopping before {path}: {end - time.time():.0f}s left", flush=True)
                lock.close()
                return
            values = point_values(config, direction, step, k)
            set_parameters(trainer, values)
            print(f"Native {path.relative_to(output)}: D/E/H = {values['D']!r}/{values['E']!r}/{values['H']!r}", flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            metadata = {"k": k, "step": step, "offset": k * step, "direction": direction,
                        "log_direction": direction in config["log_directions"],
                        "parameters": np.asarray([values[key] for key in PARAMETERS]),
                        "parameter_names": np.asarray(PARAMETERS), "device": device,
                        "job_id": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename}
            try:
                positions, loss = replay(trainer)
                assert math.isfinite(loss), f"Nonfinite loss {loss}"
            except AssertionError as error:
                # The user asked to skip failed rollouts; the failure is kept on disk and in the summary.
                assert k != 0, f"Center rollout failed: {error}"
                seconds = time.perf_counter() - start
                np.savez(path.with_suffix(".tmp.npz"), failed=True, error=str(error), seconds=seconds, **metadata)
                path.with_suffix(".tmp.npz").replace(path)
                print(f"FAILED {path.relative_to(output)}: {error}", flush=True)
                continue
            seconds = time.perf_counter() - start
            longest = max(longest, seconds)
            np.savez(path.with_suffix(".tmp.npz"), failed=False, positions=positions, loss_target=loss,
                     seconds=seconds, **metadata)
            path.with_suffix(".tmp.npz").replace(path)
            print(f"loss_target {loss:.15e} ({seconds:.1f}s)", flush=True)
        (line / "complete").touch()
        lock.close()
    print(f"No remaining {garment} line for {device}", flush=True)


def mse_by_frame(positions: Any, reference: Any) -> Any:
    """Per-frame torch mse_loss over vertices/xyz, matching the training geometry loss."""
    return ((positions - reference) ** 2).mean(axis=(1, 2))


def summarize(config: dict[str, Any], output: Path) -> None:
    import numpy as np

    rows, frame_rows, failures, centers = [], [], [], {}
    for garment in config["garments"]:
        with np.load(output / garment / "reference.npz", allow_pickle=False) as saved:
            target = saved["target"].astype(np.float64)
        for direction, step in lines(config):
            name = line_name(garment, direction, step)
            assert (output / name / "complete").is_file(), f"Incomplete line: {name}"
            device = (output / name / "device.txt").read_text().strip()
            with np.load(output / name / "k+00.npz", allow_pickle=False) as saved:
                center = saved["positions"].astype(np.float64)
                center_loss = float(saved["loss_target"])
            centers.setdefault((garment, device), []).append((name, center))
            for k in range(-config["radius"], config["radius"] + 1):
                with np.load(output / name / f"k{k:+03d}.npz", allow_pickle=False) as saved:
                    assert str(saved["device"]) == device and int(saved["k"]) == k
                    if bool(saved["failed"]):
                        failures.append({"garment": garment, "direction": direction, "step": step, "k": k,
                                         "offset": k * step, **dict(zip(PARAMETERS, map(float, saved["parameters"]))),
                                         "error": str(saved["error"]), "trajectory": f"{name}/k{k:+03d}.npz"})
                        continue
                    positions = saved["positions"].astype(np.float64)
                    assert np.isfinite(positions).all()
                    loss_target = float(saved["loss_target"])
                    parameters = saved["parameters"]
                    seconds = float(saved["seconds"])
                    node = str(saved["node"])
                target_frames = mse_by_frame(positions, target)[1:]
                center_frames = mse_by_frame(positions, center)[1:]
                np.testing.assert_allclose(target_frames.mean(), loss_target, rtol=1e-9, atol=0)
                distances = np.linalg.norm(positions - center, axis=-1)
                rows.append({"garment": garment, "direction": direction,
                             "log_direction": direction in config["log_directions"], "step": step, "k": k,
                             "offset": k * step, **dict(zip(PARAMETERS, map(float, parameters))),
                             "loss_target": loss_target, "loss_target_change": loss_target - center_loss,
                             "loss_center": float(center_frames.mean()),
                             "v2v_center_mean_m": float(distances.mean()), "v2v_center_max_m": float(distances.max()),
                             "device": device, "node": node, "seconds": seconds,
                             "trajectory": f"{name}/k{k:+03d}.npz"})
                for frame, (frame_target, frame_center) in enumerate(zip(target_frames, center_frames), start=1):
                    frame_rows.append({"garment": garment, "direction": direction, "step": step, "k": k,
                                       "frame": frame, "loss_target": float(frame_target),
                                       "loss_center": float(frame_center)})
    write_csv(output / "samples.csv", rows)
    write_csv(output / "loss_by_frame.csv", frame_rows)
    if failures:
        write_csv(output / "failures.csv", failures)
    # Lines on the same GPU model share the same deterministic center input.
    center_check = []
    for (garment, device), entries in centers.items():
        first_name, first = entries[0]
        for name, center in entries[1:]:
            center_check.append({"garment": garment, "device": device, "reference_line": first_name, "line": name,
                                 "max_abs_difference_m": float(np.abs(center - first).max())})
    if center_check:
        write_csv(output / "center_repeatability.csv", center_check)
    print(f"Saved {len(rows)} samples and {len(failures)} failed rollouts: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/material_local_sweep_185.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), required=True)
    parser.add_argument("--garment", choices=("upper", "lower", "dress"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        prepare(args.config, output)
        return
    assert (output / "config.json").is_file(), "Run --stage prepare first"
    config = json.loads((output / "config.json").read_text())
    if args.stage == "run":
        assert args.garment, "--garment is required for run"
        run(config, output, args.garment)
    else:
        summarize(config, output)


if __name__ == "__main__":
    main()
