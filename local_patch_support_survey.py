"""Diagnose native grid-support separation using cached contact-free geometry.

This does not change production scoring or remove runtime leakage checks.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
from typing import Any

import numpy as np

from local_patch_data import (
    LocalPatch, build_local_patches, local_grid_domain,
)

# Template command (workspace root, mpmavatar environment, allocated CPU node):
# python MPMAvatar/local_patch_support_survey.py --cache MPMAvatar/output/local_campaign/s185_lower_native_grid/survey8 --output MPMAvatar/output/local_campaign/s185_lower_native_grid/support_survey.json --minimum-frames 7 --maximum-frames 100 --substeps 400 --padding-cells 4 --workers 4


def support_separated_vertices(patch: LocalPatch, substeps: int = 400,
                               padding_cells: int = 4) -> np.ndarray:
    """Keep free vertices with disjoint 3x3x3 footprints at every reference substep.

    Coordinates and interpolation use the simulator's float32 normalization.
    Treating all 27 stencil nodes as active is conservative at zero-weight nodes.
    Reference separation does not certify separation of the simulated trajectory.
    """
    assert substeps > 0
    origin, _, _ = local_grid_domain(patch, padding_cells=padding_cells)
    reference = np.asarray((patch.reference_vertices_m - origin) * patch.simulation_scale,
                           dtype=np.float32)
    inverse_dx = np.float32(1.0 / (patch.cell_size_m * patch.simulation_scale))
    active = np.setdiff1d(np.arange(len(patch.vertex_ids)), patch.boundary_vertex_ids)
    boundary_faces = patch.faces[np.isin(patch.faces, patch.boundary_vertex_ids).all(axis=1)]

    def remove_overlapping(positions: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        movers = np.concatenate((positions[patch.boundary_vertex_ids],
                                 positions[boundary_faces].mean(axis=1)))
        mover_base = np.trunc(movers * inverse_dx - np.float32(.5)).astype(np.int32)
        candidate_base = np.trunc(positions[candidates] * inverse_dx - np.float32(.5)).astype(np.int32)
        assert np.all(mover_base >= 0) and np.all(candidate_base >= 0)
        overlapping = (np.abs(candidate_base[:, None] - mover_base[None]) <= 2).all(axis=2).any(axis=1)
        return candidates[~overlapping]

    # Reject on saved poses first, before checking the finer interpolated motion.
    for positions in reference:
        active = remove_overlapping(positions, active)
        if len(active) == 0:
            return active
    for start, end in zip(reference[:-1], reference[1:]):
        for step in range(1, substeps):
            positions = start + np.float32(step / substeps) * (end - start)
            active = remove_overlapping(positions, active)
            if len(active) == 0:
                return active
    return active


def read_clearance_cache(path: Path) -> dict[str, np.ndarray]:
    files = sorted(path.glob("clearance_*.npz"))
    assert len(files) > 0
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    first: dict[str, np.ndarray] = {}
    static_keys = ("vertex_ids", "faces", "rest_vertices_m", "attachment_vertex_count",
                   "simulation_scale", "clearance_m", "collar_rings")
    for index, filename in enumerate(files):
        with np.load(filename, allow_pickle=False) as data:
            if index == 0:
                first = {key: data[key].copy() for key in static_keys}
            else:
                for key in static_keys:
                    assert np.array_equal(first[key], data[key]), f"Inconsistent cache {filename}: {key}"
            for frame_id, positions, mask in zip(data["frame_ids"], data["reference_vertices_m"], data["frame_masks"]):
                frame = int(frame_id)
                if frame in frames:
                    assert np.array_equal(frames[frame][0], positions)
                    assert np.array_equal(frames[frame][1], mask)
                frames[frame] = (positions.copy(), mask.copy())
    frame_ids = np.asarray(sorted(frames), dtype=np.int64)
    assert np.all(np.diff(frame_ids) == 1)
    first["frame_ids"] = frame_ids
    first["reference_vertices_m"] = np.stack([frames[int(frame)][0] for frame in frame_ids])
    first["frame_masks"] = np.stack([frames[int(frame)][1] for frame in frame_ids])
    assert float(first["clearance_m"]) >= .02 and int(first["collar_rings"]) == 1
    return first


def survey_length(length: int, *, cache: Path, substeps: int,
                  padding_cells: int) -> dict[str, Any]:
    data = read_clearance_cache(cache)
    results: list[dict[str, Any]] = []
    for offset in range(len(data["frame_ids"]) - length + 1):
        frame_ids = data["frame_ids"][offset:offset + length]
        patches = build_local_patches(
            f"s185_lower_{frame_ids[0]}_{frame_ids[-1]}", frame_ids, data["vertex_ids"],
            data["reference_vertices_m"][offset:offset + length], data["rest_vertices_m"],
            data["faces"], data["frame_masks"][offset:offset + length].all(axis=0),
            int(data["attachment_vertex_count"]), float(data["simulation_scale"]),
            count=0, min_scored_vertices=0, collar_rings=1,
        )
        records: list[dict[str, Any]] = []
        for patch in patches:
            ids = support_separated_vertices(patch, substeps, padding_cells)
            _, side, n_grid = local_grid_domain(patch, padding_cells=padding_cells)
            records.append({"name": patch.name, "vertices": len(patch.vertex_ids),
                            "free_vertices": len(patch.vertex_ids) - len(patch.boundary_vertex_ids),
                            "distance_rule_scored": len(patch.scored_vertex_ids),
                            "support_rule_scored": len(ids),
                            "support_rule_tracking_vertex_ids": patch.vertex_ids[ids].tolist(),
                            "grid_nodes_per_axis": n_grid, "grid_side_m": side,
                            "cell_size_m": patch.cell_size_m})
        results.append({"window": [int(frame_ids[0]), int(frame_ids[-1])], "patches": records})
    count = sum(p["support_rule_scored"] for result in results for p in result["patches"])
    print(f"Support survey {length} frames: {len(results)} windows, {count} separated vertex-window observations", flush=True)
    return {"frame_count": length, "windows": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=7)
    parser.add_argument("--maximum-frames", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=400)
    parser.add_argument("--padding-cells", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.cache.is_dir() and not args.output.exists()
    assert 7 <= args.minimum_frames <= args.maximum_frames <= 100
    assert args.workers > 0 and args.substeps > 0 and args.padding_cells >= 4
    task = partial(survey_length, cache=args.cache, substeps=args.substeps,
                   padding_cells=args.padding_cells)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(task, range(args.minimum_frames, args.maximum_frames + 1)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"contact_free_only": True, "clearance_m": .02,
                                      "substeps": args.substeps, "padding_cells": args.padding_cells,
                                      "lengths": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
