"""Jointly fit subject-185 lower patches at the original MPMAvatar spacing."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from local_patch_data import LocalPatch
    from local_patch_solver import PatchRollout

# Template commands (workspace root, mpmavatar environment; GPU allocation for physics stages):
# python MPMAvatar/local_patch_fit.py --stage survey --prepared data/MPMAvatar/4DDress/examples/s185_t1 --output MPMAvatar/output/local_campaign/s185_lower_native_grid/survey --windows 11:26 27:42 43:58 59:74 75:90 91:106 --workers 4 --padding-cells 4 --max-core-vertices 0 --min-scored-vertices 16 --clearance 0.02 --collar-rings 1
# python MPMAvatar/local_patch_fit.py --stage prepare --prepared data/MPMAvatar/4DDress/examples/s185_t1 --output PATCH_BUNDLE --windows 11:26 --count 1 --max-core-vertices 0 --min-scored-vertices 16 --clearance 0.02 --collar-rings 1
# python MPMAvatar/local_patch_fit.py --stage fit --patches PATCH_BUNDLE --output FIT_DIRECTORY --padding-cells 4 --substeps 400 --iterations 200 --stop-after 0 --density 1 --initial-E 100 --initial-H 1 --E-bounds 50 2500 --H-bounds 0.5 1.2 --fd-log-E 0.05 --fd-H 0.005 --learning-rate-E 0.03 --learning-rate-H 0.003 --device cuda:0
# Add --resume to the fit command to continue the same planned optimization.
# python MPMAvatar/local_patch_fit.py --stage evaluate --patches PATCH_BUNDLE --output EVALUATION_DIRECTORY --checkpoint FIT_DIRECTORY/best_param.npz --padding-cells 4 --substeps 400 --device cuda:0
# python MPMAvatar/local_patch_fit.py --stage convergence --patches PATCH_BUNDLE --output CONVERGENCE_DIRECTORY --checkpoint FIT_DIRECTORY/best_param.npz --padding-cells 4 --cell-size-factors 1 0.5 --substep-counts 400 800 --device cuda:0


@dataclass(frozen=True)
class FitSettings:
    density: float = 1.0
    initial_E: float = 100.0
    initial_H: float = 1.0
    E_bounds: tuple[float, float] = (50.0, 2500.0)
    H_bounds: tuple[float, float] = (0.5, 1.2)
    fd_log_E: float = 0.05
    fd_H: float = 0.005
    learning_rate_E: float = 0.03
    learning_rate_H: float = 0.003
    iterations: int = 200

    def __post_init__(self) -> None:
        values = (self.density, self.initial_E, self.initial_H, *self.E_bounds,
                  *self.H_bounds, self.fd_log_E, self.fd_H, self.learning_rate_E,
                  self.learning_rate_H)
        assert np.isfinite(values).all() and min(values) > 0
        assert self.E_bounds[0] < self.E_bounds[1]
        assert self.H_bounds[0] < self.H_bounds[1]
        assert self.E_bounds[0] <= self.initial_E <= self.E_bounds[1]
        assert self.H_bounds[0] <= self.initial_H <= self.H_bounds[1]
        assert self.iterations > 0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([np.log(self.E_bounds[0]), self.H_bounds[0]]),
                np.array([np.log(self.E_bounds[1]), self.H_bounds[1]]))


def finite_difference_gradient(
    objective: Callable[[np.ndarray], float], point: np.ndarray,
    offsets: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> np.ndarray:
    """Central differences; use actual bounded probe separation near a bound."""
    assert point.shape == offsets.shape == lower.shape == upper.shape == (2,)
    assert np.isfinite(point).all() and np.all(offsets > 0)
    assert np.all(lower < upper) and np.all((lower <= point) & (point <= upper))
    gradient = np.empty(2, dtype=np.float64)
    for axis in range(2):
        plus, minus = point.copy(), point.copy()
        plus[axis] = min(upper[axis], point[axis] + offsets[axis])
        minus[axis] = max(lower[axis], point[axis] - offsets[axis])
        separation = plus[axis] - minus[axis]
        assert separation > 0
        gradient[axis] = (objective(plus) - objective(minus)) / separation
    assert np.isfinite(gradient).all(), "Nonfinite finite-difference gradient"
    return gradient


def adam_update(
    point: np.ndarray, gradient: np.ndarray, first: np.ndarray, second: np.ndarray,
    step: int, learning_rates: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert step >= 1 and np.isfinite(gradient).all()
    first = 0.9 * first + 0.1 * gradient
    second = 0.999 * second + 0.001 * gradient**2
    update = learning_rates * (first / (1 - 0.9**step)) / (
        np.sqrt(second / (1 - 0.999**step)) + 1e-8)
    point = np.clip(point - update, lower, upper)
    assert np.isfinite(point).all()
    return point, first, second


def aggregate_rollouts(
    patches: Sequence[LocalPatch], rollouts: Sequence[PatchRollout],
) -> dict[str, Any]:
    """Weight by scored vertex-frame observations, excluding initialized frames."""
    assert len(patches) == len(rollouts) > 0
    counts = np.array([len(p.scored_vertex_ids) * (len(p.frame_ids) - 1)
                       for p in patches], dtype=np.float64)
    assert np.all(counts > 0)
    mse = float(np.average([r.mean_xyz_mse_m2 for r in rollouts], weights=counts))
    mean_error = float(np.average([r.mean_vertex_error_mm for r in rollouts], weights=counts))
    assert np.isfinite([mse, mean_error]).all() and mse >= 0
    return {
        "mean_xyz_mse_m2": mse,
        "vertex_rmse_mm": float(1000 * np.sqrt(3 * mse)),
        "mean_vertex_error_mm": mean_error,
        "boundary_max_error_mm": float(max(r.boundary_max_error_mm for r in rollouts)),
        "scored_vertex_frame_count": int(counts.sum()),
        "initialization_frame_scored": False,
        "prediction_scope": "local patch interior conditioned on recorded collar motion",
        "patches": [{
            "name": p.name, "frame_ids": p.frame_ids.tolist(),
            "scored_vertex_count": len(p.scored_vertex_ids),
            "mean_xyz_mse_m2": r.mean_xyz_mse_m2,
            "mean_vertex_error_mm": r.mean_vertex_error_mm,
            "vertex_rmse_mm": r.vertex_rmse_mm,
            "boundary_max_error_mm": r.boundary_max_error_mm,
            "grid_leakage_fraction": r.grid_leakage_fraction,
            "per_frame_xyz_mse_m2": r.per_frame_xyz_mse_m2.tolist(),
            "per_frame_vertex_error_mm": r.per_frame_vertex_error_mm.tolist(),
            "per_frame_vertex_rmse_mm": r.per_frame_vertex_rmse_mm.tolist(),
            "per_frame_boundary_max_error_mm": r.per_frame_boundary_max_error_mm.tolist(),
            "per_frame_grid_leakage_fraction": r.per_frame_grid_leakage_fraction.tolist(),
        } for p, r in zip(patches, rollouts)],
    }


def describe_grids(
    patches: Sequence[LocalPatch], cell_size_factor: float, padding_cells: int,
) -> list[dict[str, Any]]:
    """Report the physical spacing and fixed domain of each independent patch."""
    from local_patch_data import local_grid_domain

    records: list[dict[str, Any]] = []
    for patch in patches:
        origin, side, n_grid = local_grid_domain(patch, cell_size_factor, padding_cells)
        records.append({
            "name": patch.name,
            "local_bbox_extent_m": float(np.ptp(patch.reference_vertices_m.reshape(-1, 3), axis=0).max()),
            "global_bbox_extent_m": 1.0 / patch.simulation_scale,
            "base_cell_size_m": patch.cell_size_m,
            "cell_size_m": patch.cell_size_m * cell_size_factor,
            "origin_m": origin.tolist(), "side_m": side, "n_grid": n_grid,
        })
    return records


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_parameters(path: Path, record: dict[str, Any]) -> None:
    """Native D/E/H keys; the loss belongs to these exact evaluated parameters."""
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, D=record["D"], E=record["E"], H=record["H"],
                 loss=record["loss"], step=record["step"])
    temporary.replace(path)


def write_history(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """CSV is a derived report; the atomic optimizer JSON owns completed steps."""
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_fit_reports(output: Path, state: dict[str, Any]) -> None:
    """Regenerate derived reports from the completed optimizer checkpoint."""
    if state["best"]:
        write_parameters(output / "best_param.npz", state["best"])
        write_json(output / "best_metrics.json", state["best"])
    if state["last"]:
        write_parameters(output / "last_param.npz", state["last"])
        write_json(output / "summary.json", {
            "completed_iterations": state["next_step"],
            "planned_iterations": state["configuration"]["settings"]["iterations"],
            "best": state["best"], "last": state["last"],
            "loss_note": "Loss and D/E/H refer to the same evaluated point, before the Adam update.",
            "density_policy": "fixed", "configuration": state["configuration"],
        })


def optimize(
    objective: Callable[[np.ndarray], dict[str, Any]], output: Path,
    settings: FitSettings, context: dict[str, Any],
    on_best: Callable[[dict[str, Any]], None],
    initialize_output: Callable[[], None], *, resume: bool = False,
    stop_after: int = 0,
) -> dict[str, Any]:
    """Shared log-E/H Adam with central differences and exact evaluated checkpoints."""
    assert 0 <= stop_after <= settings.iterations
    configuration = json.loads(json.dumps({"settings": asdict(settings), "context": context}))
    checkpoint = output / "optimizer_state.json"
    fields = ["step", "loss", "vertex_rmse_mm", "mean_vertex_error_mm",
              "boundary_max_error_mm", "D", "E", "H", "gradient_log_E", "gradient_H",
              "evaluation_seconds", "gradient_seconds", "iteration_seconds"]
    if resume:
        assert checkpoint.is_file(), checkpoint
        state = json.loads(checkpoint.read_text())
        assert state["configuration"] == configuration, "Resume settings or patch bundle changed"
        assert [row["step"] for row in state["history"]] == list(range(state["next_step"]))
        write_history(output / "history.csv", state["history"], fields)
        export_fit_reports(output, state)
    else:
        assert not output.exists(), f"Output already exists: {output}; use --resume for the same fit"
        output.mkdir(parents=True)
        initialize_output()
        state = {"configuration": configuration, "next_step": 0,
                 "point": [float(np.log(settings.initial_E)), settings.initial_H],
                 "first_moment": [0.0, 0.0], "second_moment": [0.0, 0.0],
                 "best": {}, "last": {}, "history": []}
        write_history(output / "history.csv", [], fields)
        write_json(checkpoint, state)
    lower, upper = settings.bounds()
    point = np.asarray(state["point"], dtype=np.float64)
    first = np.asarray(state["first_moment"], dtype=np.float64)
    second = np.asarray(state["second_moment"], dtype=np.float64)
    offsets = np.array([settings.fd_log_E, settings.fd_H])
    rates = np.array([settings.learning_rate_E, settings.learning_rate_H])
    end_step = stop_after or settings.iterations
    assert state["next_step"] <= end_step

    def loss(probe: np.ndarray) -> float:
        return 1e6 * float(objective(probe)["mean_xyz_mse_m2"])

    for step in range(state["next_step"], end_step):
        iteration_start = perf_counter()
        metrics = objective(point)
        evaluation_seconds = perf_counter() - iteration_start
        record = {"step": step, "loss": float(metrics["mean_xyz_mse_m2"]),
                  "D": settings.density, "E": float(np.exp(point[0])), "H": float(point[1]),
                  "metrics": metrics}
        assert np.isfinite(record["loss"]) and record["loss"] >= 0
        if step == 0:
            write_parameters(output / "initial_param.npz", record)
        if not state["best"] or record["loss"] < state["best"]["loss"]:
            state["best"] = record
            write_parameters(output / "best_param.npz", record)
            write_json(output / "best_metrics.json", record)
            on_best(record)
        gradient_start = perf_counter()
        gradient = finite_difference_gradient(loss, point, offsets, lower, upper)
        gradient_seconds = perf_counter() - gradient_start
        row = {key: record[key] for key in ("step", "loss", "D", "E", "H")}
        row.update({key: metrics[key] for key in
                    ("vertex_rmse_mm", "mean_vertex_error_mm", "boundary_max_error_mm")})
        row.update(gradient_log_E=float(gradient[0]), gradient_H=float(gradient[1]))
        point, first, second = adam_update(point, gradient, first, second, step + 1, rates, lower, upper)
        row.update(evaluation_seconds=evaluation_seconds, gradient_seconds=gradient_seconds,
                   iteration_seconds=perf_counter() - iteration_start)
        state.update(next_step=step + 1, point=point.tolist(), first_moment=first.tolist(),
                     second_moment=second.tolist(), last=record)
        state["history"].append(row)
        write_json(checkpoint, state)
        write_history(output / "history.csv", state["history"], fields)
        export_fit_reports(output, state)
        print(f"Step {step}: E={record['E']:.6g}, H={record['H']:.6g}, "
              f"vertex RMSE={metrics['vertex_rmse_mm']:.4f} mm, "
              f"iteration={row['iteration_seconds']:.2f}s", flush=True)
    return state


def save_predictions(
    output: Path, patches: Sequence[LocalPatch], rollouts: Sequence[PatchRollout],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for index, (p, r) in enumerate(zip(patches, rollouts)):
        path = output / f"patch_{index:03d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, vertices_m=r.vertices_m, frame_ids=p.frame_ids,
                     vertex_ids=p.vertex_ids, faces=p.faces,
                     scored_vertex_ids=p.scored_vertex_ids,
                     boundary_vertex_ids=p.boundary_vertex_ids)
        temporary.replace(path)


def load_parameters(path: Path) -> tuple[float, float, float]:
    assert path.is_file(), path
    with np.load(path, allow_pickle=False) as data:
        values = tuple(float(data[key]) for key in ("D", "E", "H"))
    assert np.isfinite(values).all() and min(values) > 0
    return values


def parse_window(value: str) -> tuple[int, int]:
    start, end = (int(part) for part in value.split(":"))
    assert start < end
    return start, end


def survey_window(window: tuple[int, int], *, args: argparse.Namespace,
                  output: Path) -> dict[str, Any]:
    """Run spatial-index queries in an isolated process and save its window."""
    from local_patch_data import prepare_native_patches, save_bundle

    body_root = (args.body_root.resolve() if args.body_root else args.prepared.resolve()
                 / "data/4D-DRESS/00185_Inner/Inner/Take1")
    patches, metadata = prepare_native_patches(
        args.prepared.resolve(), body_root, [window], count=0,
        max_core_vertices=args.max_core_vertices, min_scored_vertices=0,
        clearance_m=args.clearance, collar_rings=args.collar_rings,
        clearance_cache=output / f"clearance_{window[0]}_{window[1]}.npz")
    records = [{"name": p.name, "frame_ids": p.frame_ids.tolist(),
                "vertex_count": len(p.vertex_ids), "triangle_count": len(p.faces),
                "boundary_vertex_count": len(p.boundary_vertex_ids),
                "scored_vertex_count": len(p.scored_vertex_ids)} for p in patches]
    report = {"metadata": metadata, "candidates": records,
              "grids": describe_grids(patches, 1.0, args.padding_cells)}
    label = f"window_{window[0]}_{window[1]}"
    write_json(output / f"{label}.json", report)
    viable = [p for p in patches if len(p.scored_vertex_ids) >= args.min_scored_vertices]
    if viable:
        save_bundle(output / label, viable, metadata)
    print(f"Survey {window[0]}:{window[1]}: "
          f"scored counts={[len(p.scored_vertex_ids) for p in patches]}; "
          f"viable={len(viable)}", flush=True)
    return report


def survey_patches(args: argparse.Namespace, output: Path) -> None:
    """Inspect independent windows concurrently and save each completed result."""
    assert not args.resume and not args.checkpoint and args.workers > 0
    assert not output.exists(), output
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        reports = list(executor.map(partial(survey_window, args=args, output=output), args.windows))
    metadata = reports[0]["metadata"].copy()
    metadata["windows"] = [list(window) for window in args.windows]
    write_json(output / "survey.json", {
        "metadata": metadata,
        "candidates": [p for report in reports for p in report["candidates"]],
        "grids": [p for report in reports for p in report["grids"]],
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("survey", "prepare", "fit", "evaluate", "convergence"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data/MPMAvatar/4DDress/examples/s185_t1")
    parser.add_argument("--body-root", type=Path, help="Take1 directory containing SMPLX/")
    parser.add_argument("--windows", type=parse_window, nargs="+", default=[(45, 50), (51, 56)])
    parser.add_argument("--count", type=int, default=1, help="Patches per training window")
    parser.add_argument("--workers", type=int, default=4, help="Independent CPU windows in survey")
    parser.add_argument("--max-core-vertices", type=int, default=0, help="0 selects largest eligible component")
    parser.add_argument("--min-scored-vertices", type=int, default=16)
    parser.add_argument("--clearance", type=float, default=0.02, help="Body/non-neighbor cloth clearance (meters)")
    parser.add_argument("--collar-rings", type=int, default=1)
    parser.add_argument("--core-vertex-ids", type=int, nargs="+", help="Explicit original tracking vertex IDs")
    parser.add_argument("--patches", type=Path, help="Prepared local patch bundle")
    parser.add_argument("--substeps", type=int, default=400)
    parser.add_argument("--padding-cells", type=int, default=4,
                        help="Domain padding in original-spacing cell units (minimum 4)")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--initial-E", type=float, default=100.0)
    parser.add_argument("--initial-H", type=float, default=1.0)
    parser.add_argument("--E-bounds", type=float, nargs=2, default=(50.0, 2500.0))
    parser.add_argument("--H-bounds", type=float, nargs=2, default=(0.5, 1.2))
    parser.add_argument("--fd-log-E", type=float, default=0.05)
    parser.add_argument("--fd-H", type=float, default=0.005)
    parser.add_argument("--learning-rate-E", type=float, default=0.03)
    parser.add_argument("--learning-rate-H", type=float, default=0.003)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cell-size-factors", type=float, nargs="+", default=(1.0, 0.5),
                        help="Convergence spacing multipliers relative to the original spacing (0 < factor <= 1)")
    parser.add_argument("--substep-counts", type=int, nargs="+", default=(400, 800))
    parser.add_argument("--convergence-tolerance-mm", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from local_patch_data import load_bundle, prepare_native_patches, save_bundle

    output = args.output.resolve()
    if args.stage == "survey":
        survey_patches(args, output)
        return
    if args.stage == "prepare":
        assert not args.resume and not args.checkpoint
        body_root = (args.body_root.resolve() if args.body_root else args.prepared.resolve()
                     / "data/4D-DRESS/00185_Inner/Inner/Take1")
        patches, metadata = prepare_native_patches(
            args.prepared.resolve(), body_root, args.windows, count=args.count,
            max_core_vertices=args.max_core_vertices, min_scored_vertices=args.min_scored_vertices,
            clearance_m=args.clearance,
            collar_rings=args.collar_rings, core_vertex_ids=args.core_vertex_ids)
        save_bundle(output, patches, metadata)
        print(f"Prepared {len(patches)} subject-185 lower patches: {output}")
        return
    assert args.patches, "Choose the subject-185 lower patch bundle with --patches"
    source_bundle = output / "patches" if args.stage == "fit" and args.resume else args.patches.resolve()
    patches, metadata = load_bundle(source_bundle)
    assert args.stage == "fit" or not args.resume
    assert args.stage == "fit" or args.stop_after == 0
    assert args.stage != "fit" or not args.checkpoint, "Fit uses --initial-E/--initial-H and fixed --density"
    assert args.substeps > 0
    assert args.padding_cells >= 4
    assert args.device.startswith("cuda:"), "Physics stages require a CUDA allocation"
    from local_patch_solver import PatchSimulator

    def simulators(cell_size_factor: float, substeps: int) -> list[Any]:
        return [PatchSimulator(p, substeps, args.device, padding_cells=args.padding_cells,
                               cell_size_factor=cell_size_factor)
                for p in patches]

    if args.stage == "fit":
        settings = FitSettings(
            density=args.density, initial_E=args.initial_E, initial_H=args.initial_H,
            E_bounds=tuple(args.E_bounds), H_bounds=tuple(args.H_bounds),
            fd_log_E=args.fd_log_E, fd_H=args.fd_H, learning_rate_E=args.learning_rate_E,
            learning_rate_H=args.learning_rate_H, iterations=args.iterations)
        runners = simulators(1.0, args.substeps)
        latest: list[Any] = []

        def evaluate_point(point: np.ndarray) -> dict[str, Any]:
            nonlocal latest
            latest = [runner.rollout(settings.density, float(np.exp(point[0])), float(point[1]))
                      for runner in runners]
            return aggregate_rollouts(patches, latest)

        def save_best(record: dict[str, Any]) -> None:
            save_predictions(output / "best_predictions", patches, latest)

        def snapshot_inputs() -> None:
            save_bundle(output / "patches", patches, metadata)

        optimize(evaluate_point, output, settings,
                 {"patch_bundle": str(args.patches.resolve()), "metadata": metadata,
                  "grids": describe_grids(patches, 1.0, args.padding_cells),
                  "substeps": args.substeps, "padding_cells": args.padding_cells,
                  "device": args.device, "native_simulation_constants": {
                      "nu": 0.3, "gamma": 500.0, "kappa": 500.0, "friction_angle": 40.0,
                      "gravity_sim": [0.0, -9.8, 0.0], "damping": 1.1, "thickness_sim": 1e-5,
                  }}, save_best, snapshot_inputs, resume=args.resume, stop_after=args.stop_after)
        return
    assert args.checkpoint, "Select frozen material explicitly with --checkpoint"
    density, E, H = load_parameters(args.checkpoint.resolve())
    assert not output.exists(), f"Output already exists: {output}"
    output.mkdir(parents=True)
    if args.stage == "evaluate":
        runners = simulators(1.0, args.substeps)
        rollouts = [runner.rollout(density, E, H) for runner in runners]
        report = aggregate_rollouts(patches, rollouts)
        report.update(D=density, E=E, H=H, grids=describe_grids(patches, 1.0, args.padding_cells),
                      substeps=args.substeps,
                      padding_cells=args.padding_cells,
                      checkpoint=str(args.checkpoint.resolve()), patch_bundle=str(args.patches.resolve()))
        write_json(output / "metrics.json", report)
        save_predictions(output / "predictions", patches, rollouts)
        print(f"Patch vertex RMSE: {report['vertex_rmse_mm']:.4f} mm; {output / 'metrics.json'}")
        return
    assert np.isfinite(args.cell_size_factors).all() and 0 < min(args.cell_size_factors)
    assert max(args.cell_size_factors) <= 1.0
    assert min(args.substep_counts) > 0
    assert np.isfinite(args.convergence_tolerance_mm) and args.convergence_tolerance_mm > 0
    records = []
    trajectories = []
    for cell_size_factor in args.cell_size_factors:
        for substeps in args.substep_counts:
            runners = simulators(cell_size_factor, substeps)
            rollouts = [runner.rollout(density, E, H) for runner in runners]
            report = aggregate_rollouts(patches, rollouts)
            report.update(cell_size_factor=cell_size_factor, substeps=substeps,
                          grids=describe_grids(patches, cell_size_factor, args.padding_cells))
            records.append(report)
            trajectories.append([r.vertices_m for r in rollouts])
            del runners
    reference_index = next(index for index, record in enumerate(records)
                           if record["cell_size_factor"] == min(args.cell_size_factors)
                           and record["substeps"] == max(args.substep_counts))
    for record, run in zip(records, trajectories):
        squared_distance_sum, count = 0.0, 0
        per_patch = []
        for patch, prediction, reference in zip(patches, run, trajectories[reference_index]):
            difference = prediction[1:, patch.scored_vertex_ids] - reference[1:, patch.scored_vertex_ids]
            total = float(np.sum(difference.astype(np.float64)**2))
            observations = difference.shape[0] * difference.shape[1]
            per_patch.append({"name": patch.name,
                              "trajectory_rmse_mm": float(1000 * np.sqrt(total / observations))})
            squared_distance_sum += total
            count += observations
        difference_mm = float(1000 * np.sqrt(squared_distance_sum / count))
        record.update(trajectory_rmse_to_finest_mm=difference_mm,
                      within_tolerance=difference_mm <= args.convergence_tolerance_mm,
                      per_patch_trajectory_differences=per_patch)
    write_json(output / "convergence.json", {
        "D": density, "E": E, "H": H, "checkpoint": str(args.checkpoint.resolve()),
        "patch_bundle": str(args.patches.resolve()), "runs": records,
        "reference": {"cell_size_factor": min(args.cell_size_factors), "substeps": max(args.substep_counts)},
        "tolerance_mm": args.convergence_tolerance_mm,
        "padding_cells": args.padding_cells,
        "policy": "Frozen identical material and reference collars across all grid/timestep combinations",
    })


if __name__ == "__main__":
    main()
