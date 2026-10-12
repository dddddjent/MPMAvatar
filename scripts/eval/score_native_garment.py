"""Score one native MPMAvatar garment rollout (100 test uvmesh frames) against the labeled raw test scans.

Garment-only: the prediction is the split's cloth (reordered_cloth_v_idx, new_cloth_faces); the target is the
scan faces whose three vertices carry the garment label. Metric and seeds match WarpXPBD/rescore_test_scans.py.

# Template command, from the workspace root on an allocated CPU (mpmavatar env):
# python MPMAvatar/scripts/eval/score_native_garment.py --prediction-dir data/MPMAvatar/4DDress/examples/s170_t1/output/phys/s170_t1_upper/seed0 --prepared data/MPMAvatar/4DDress/examples/s170_t1 --split split_idx_upper.npz --label 3 --test-take 5 --test-start 21 --output MPMAvatar/output/differentiable_gpu/initial/scores/s170_t1_upper_final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import trimesh

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "WarpXPBD"))
from rescore_test_scans import metric_row, read_obj  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--test-take", type=int, required=True)
    parser.add_argument("--test-start", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = args.prepared.name
    subject = int(name[1:].split("_")[0])
    assets = args.prepared / "data" / name
    take = args.prepared / f"data/4D-DRESS/{subject:05d}_Inner/Inner/Take{args.test_take}"
    assert (args.prediction_dir / "uvmesh/099.obj").is_file() and take.is_dir()
    split = dict(np.load(assets / args.split))
    _, reference_faces = read_obj(assets / "mesh_processed.obj")
    rows = []
    for index in range(100):
        frame = args.test_start + index
        vertices, faces = read_obj(args.prediction_dir / "uvmesh" / f"{index:03d}.obj")
        assert np.array_equal(faces, reference_faces), index
        with (take / "Meshes_pkl" / f"mesh-f{frame:05d}.pkl").open("rb") as stream:
            scan = pickle.load(stream)
        with (take / "Semantic/labels" / f"label-f{frame:05d}.pkl").open("rb") as stream:
            labels = np.asarray(pickle.load(stream)["scan_labels"])
        scan_vertices, scan_faces = np.asarray(scan["vertices"]), np.asarray(scan["faces"])
        assert len(labels) == len(scan_vertices)
        garment_faces = scan_faces[(labels[scan_faces] == args.label).all(axis=1)]
        assert len(garment_faces) > 0, frame
        target = trimesh.Trimesh(scan_vertices, garment_faces, process=False)
        prediction = trimesh.Trimesh(vertices[split["reordered_cloth_v_idx"]], split["new_cloth_faces"],
                                     process=False, maintain_order=True)
        rows.append({"sequence_index": index, "frame_id": frame, "garment": metric_row(target, prediction, frame)})
        if index % 25 == 0 or index == 99:
            print(json.dumps(rows[-1]), flush=True)
    mean = {metric: float(np.mean([row["garment"][metric] for row in rows])) for metric in ("CD", "F-Score")}
    report = {"prediction_dir": str(args.prediction_dir.resolve()), "scan_take": str(take.resolve()),
              "split": args.split, "garment_label": args.label,
              "metric_function": "MPMAvatar/metric.py:all_mesh_metrics, unchanged",
              "sampling": "trimesh.sample.sample_surface; scan seed=2*frame; prediction seed=2*frame+1",
              "mean": mean, "frames": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(mean), flush=True)


if __name__ == "__main__":
    main()
