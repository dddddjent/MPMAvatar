"""Compare independently prepared templates against author assets, without hashes.

Geometry and UVs are compared at corresponding triangle corners. Connectivity
is compared up to vertex renumbering, and garment partitions by triangle ID.
This comparison applies to templates retaining the source scan's face order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Template command (mpmavatar environment, workspace root):
# python MPMAvatar/preprocess/compare_4ddress_templates.py --derived data/MPMAvatar/4DDress/00190_Inner/data/s190_t2 --reference MPMAvatar/data/s190_t2 --output data/MPMAvatar/4DDress/00190_Inner/author_comparison.json


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert path.is_file(), path
    vertices, faces, uv, uv_faces = [], [], [], []
    with path.open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == "vt":
                uv.append([float(x) for x in fields[1:3]])
            elif fields[0] == "f":
                assert len(fields) == 4, "Template must contain triangles"
                corners = [field.split("/") for field in fields[1:]]
                faces.append([int(corner[0]) - 1 for corner in corners])
                uv_faces.append([int(corner[1]) - 1 for corner in corners])
    v, f, t, tf = (np.asarray(value) for value in (vertices, faces, uv, uv_faces))
    assert v.ndim == t.ndim == 2 and f.shape == tf.shape
    assert f.min() >= 0 and f.max() < len(v) and tf.min() >= 0 and tf.max() < len(t)
    assert np.isfinite(v).all() and np.isfinite(t).all()
    return v, f, t, tf


def garments(path: Path, faces: np.ndarray, vertex_count: int) -> dict[str, np.ndarray]:
    with np.load(path / "cloth_vertices.npz", allow_pickle=False) as archive:
        result = {}
        for label in archive.files:
            ids = archive[label]
            assert ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
            assert ids.min() >= 0 and ids.max() < vertex_count
            result[label] = np.isin(faces, ids).all(axis=1)
        return result


def split_report(directory: Path, faces: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    with np.load(directory / "split_idx.npz", allow_pickle=False) as split:
        cloth_ids = split["reordered_cloth_f_idx"]
        human_ids = split["reordered_human_f_idx"]
        cloth_vertices = split["reordered_cloth_v_idx"]
        human_vertices = split["reordered_human_v_idx"]
        joint = np.intersect1d(cloth_vertices, human_vertices)
        return {"cloth_partition_matches_labels": bool(np.array_equal(np.sort(cloth_ids), np.flatnonzero(mask))),
                "cloth_local_faces_valid": bool(np.array_equal(cloth_vertices[split["new_cloth_faces"]], faces[cloth_ids])),
                "human_local_faces_valid": bool(np.array_equal(human_vertices[split["new_human_faces"]], faces[human_ids])),
                "joint_vertices": int(split["num_joint_v"]), "joint_faces": int(split["num_joint_f"]),
                "joint_vertex_count_valid": bool(len(joint) == int(split["num_joint_v"]))}


def compare(derived: Path, reference: Path) -> dict[str, Any]:
    dv, df, duv, duf = read_obj(derived / "mesh_processed.obj")
    rv, rf, ruv, ruf = read_obj(reference / "mesh_processed.obj")
    assert df.shape == rf.shape, "Different triangle counts: this comparator requires retained source face order"
    corner_dist = np.linalg.norm(dv[df] - rv[rf], axis=2)
    uv_dist = np.linalg.norm(duv[duf] - ruv[ruf], axis=2)
    pairs = np.unique(np.stack((df.ravel(), rf.ravel()), axis=1), axis=0)
    bijective = len(pairs) == len(np.unique(df)) == len(np.unique(rf))
    dg, rg = garments(derived, df, len(dv)), garments(reference, rf, len(rv))
    masks = {}
    for label in sorted(set(dg) | set(rg)):
        d = dg.get(label, np.zeros(len(df), dtype=bool))
        r = rg.get(label, np.zeros(len(rf), dtype=bool))
        masks[label] = {"derived_faces": int(d.sum()), "reference_faces": int(r.sum()),
                        "disagreeing_faces": int(np.count_nonzero(d != r)),
                        "face_iou": float(np.count_nonzero(d & r) / max(1, np.count_nonzero(d | r)))}
    dsplit = split_report(derived, df, np.logical_or.reduce(list(dg.values())))
    rsplit = split_report(reference, rf, np.logical_or.reduce(list(rg.values())))
    equivalent = bool(np.all(corner_dist == 0) and np.all(uv_dist == 0) and bijective
                      and set(dg) == set(rg) and all(v["disagreeing_faces"] == 0 for v in masks.values())
                      and dsplit == rsplit and all(dsplit[k] for k in dsplit if k.endswith("valid") or k.endswith("labels")))
    return {"derived": str(derived.resolve()), "reference": str(reference.resolve()),
            "equivalent_geometry_uv_connectivity_and_partitions": equivalent,
            "comparison_scope": "Initial template, UV layout, garment indices and split only; excludes tracking, skinning weights, AO, appearance and material fitting.",
            "vertices": {"derived": len(dv), "reference": len(rv)}, "faces": len(df),
            "max_triangle_corner_distance_m": float(corner_dist.max()),
            "p95_triangle_corner_distance_m": float(np.percentile(corner_dist, 95)),
            "max_triangle_corner_uv_distance": float(uv_dist.max()),
            "connectivity_identical_up_to_vertex_numbering": bool(bijective),
            "vertex_numbering_identical": bool(np.array_equal(df, rf) and np.array_equal(dv, rv)),
            "garments": masks, "derived_split": dsplit, "reference_split": rsplit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.derived, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
