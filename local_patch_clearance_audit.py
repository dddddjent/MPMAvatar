"""Separate body clearance from the lower-cloth vertex-pair rejection heuristic."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from local_patch_data import (
    build_local_patches, close_smplx_mouth, expand_vertex_rings, mesh_adjacency,
    tracking_surface_complement,
)
from local_patch_support_survey import read_clearance_cache

# Template command (workspace root, mpmavatar environment, allocated CPU node):
# python MPMAvatar/local_patch_clearance_audit.py --prepared data/MPMAvatar/4DDress/examples/s185_t1 --cache MPMAvatar/output/local_campaign/s185_lower_native_grid/survey8 --output MPMAvatar/output/local_campaign/s185_lower_native_grid/clearance_audit.json --frames 11 45 60 100 --path-batch-size 128


def audit_frame(data: dict[str, np.ndarray], prepared: Path, frame: int,
                path_batch_size: int) -> dict[str, Any]:
    import trimesh
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    offset = int(np.flatnonzero(data["frame_ids"] == frame)[0])
    positions = data["reference_vertices_m"][offset]
    faces = data["faces"]
    clearance = float(data["clearance_m"])
    adjacency = mesh_adjacency(faces, len(positions))
    neighbors = [set(map(int, expand_vertex_rings(np.asarray([vertex]), adjacency, 2)))
                 for vertex in range(len(positions))]
    pairs = cKDTree(positions).query_pairs(clearance, output_type="ndarray")
    rejected = np.asarray([pair for pair in pairs if int(pair[1]) not in neighbors[int(pair[0])]],
                          dtype=np.int64).reshape(-1, 2)
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    assert np.all(lengths > 0)
    graph = csr_matrix((np.concatenate((lengths, lengths)),
                        (np.concatenate((edges[:, 0], edges[:, 1])),
                         np.concatenate((edges[:, 1], edges[:, 0])))),
                       shape=(len(positions), len(positions)))
    local_path = np.zeros(len(rejected), dtype=bool)
    sources = np.unique(rejected[:, 0])
    for start in range(0, len(sources), path_batch_size):
        batch = sources[start:start + path_batch_size]
        distances = dijkstra(graph, directed=False, indices=batch, limit=clearance)
        selected = np.flatnonzero(np.isin(rejected[:, 0], batch))
        rows = np.searchsorted(batch, rejected[selected, 0])
        local_path[selected] = np.isfinite(distances[rows, rejected[selected, 1]])

    tracking = prepared / "output/tracking/s185_t1_11_100" / f"params_{frame}.npz"
    offsets_path = prepared / "model/s185_t1/point_cloud/timestep_030000/verts_offset.npy"
    split_path = prepared / "data/s185_t1/split_idx_lower.npz"
    body_path = prepared / "data/4D-DRESS/00185_Inner/Inner/Take1/SMPLX" / f"mesh-f{frame:05d}_smplx.ply"
    assert tracking.is_file() and offsets_path.is_file() and split_path.is_file() and body_path.is_file()
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    with np.load(tracking, allow_pickle=False) as tracked, np.load(split_path, allow_pickle=False) as split:
        vertices = np.asarray(tracked["vertices"] + offsets[frame - 11], dtype=np.float64)
        np.testing.assert_array_equal(vertices[data["vertex_ids"]], positions)
        external_faces = tracking_surface_complement(tracked["faces"], data["vertex_ids"], faces,
                                                     split["reordered_cloth_f_idx"])
    body = close_smplx_mouth(trimesh.load(body_path, force="mesh", process=False))
    external = trimesh.Trimesh(vertices=vertices, faces=external_faces, process=False)
    samples = np.concatenate((positions, positions[faces].mean(axis=1)))
    body_safe = trimesh.proximity.signed_distance(body, samples) <= -clearance
    _, distances, _ = trimesh.proximity.closest_point(external, samples)
    external_safe = distances >= clearance

    def vertex_mask(sample_mask: np.ndarray) -> np.ndarray:
        mask = sample_mask[:len(positions)].copy()
        mask[np.unique(faces[~sample_mask[len(positions):]])] = False
        return mask

    geometric_safe = vertex_mask(body_safe & external_safe)
    current_safe = geometric_safe.copy()
    current_safe[np.unique(rejected)] = False
    np.testing.assert_array_equal(current_safe, data["frame_masks"][offset])
    diagnostic_safe = geometric_safe.copy()
    diagnostic_safe[np.unique(rejected[~local_path])] = False

    def pose_patch_counts(mask: np.ndarray) -> dict[str, int]:
        # A single-pose geometry diagnostic, without trajectory validation or fitting.
        patches = build_local_patches(
            f"audit_{frame}", np.asarray([frame], dtype=np.int64), data["vertex_ids"],
            positions[None], data["rest_vertices_m"], faces, mask,
            int(data["attachment_vertex_count"]), float(data["simulation_scale"]),
            count=0, min_scored_vertices=0, collar_rings=1,
        )
        return {"candidate_regions": len(patches),
                "scored_vertices": sum(len(p.scored_vertex_ids) for p in patches),
                "maximum_scored_vertices_per_region": max((len(p.scored_vertex_ids) for p in patches), default=0)}

    return {
        "frame": frame, "lower_vertices": len(positions),
        "body_clear_vertices": int(vertex_mask(body_safe).sum()),
        "body_and_external_clear_vertices": int(geometric_safe.sum()),
        "current_eligible_vertices": int(current_safe.sum()),
        "two_ring_rejected_pairs": len(rejected),
        "rejected_pairs_with_surface_edge_path_under_clearance": int(local_path.sum()),
        "diagnostic_eligible_after_excluding_only_longer_path_pairs": int(diagnostic_safe.sum()),
        "clearance_m": clearance,
        "median_edge_length_m": float(np.median(lengths)),
        "single_pose_interiors": {
            "current_filter": pose_patch_counts(current_safe),
            "body_external_only_diagnostic": pose_patch_counts(geometric_safe),
            "short_surface_path_exemption_diagnostic": pose_patch_counts(diagnostic_safe),
        },
        "production_filter_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[11, 45, 60, 100])
    parser.add_argument("--path-batch-size", type=int, default=128)
    args = parser.parse_args()
    assert args.prepared.is_dir() and args.cache.is_dir() and not args.output.exists()
    assert args.path_batch_size > 0
    data = read_clearance_cache(args.cache)
    results = []
    for frame in args.frames:
        assert frame in data["frame_ids"]
        result = audit_frame(data, args.prepared, frame, args.path_batch_size)
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps({"frames": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
