"""Per-frame geometry of the tracked training meshes against the 4D-DRESS scans.

Uses metric.all_mesh_metrics exactly as eval.py does for the held-out rollout
(CD = 1000 * (mean d^2 + mean d^2) in m^2 x 1000, F-Score at d^2 <= 1e-3, 100k samples):
- full: whole tracked mesh (mesh_cloth_<frame>.obj) vs the whole scan, as eval.py scores the test take;
- garment: tracked faces whose vertices are all in cloth_vertices.npz for the simulated labels vs the scan
  faces whose three vertices carry one of those labels.
Writes <tracking>/tracking_geo_metric.npz and tracking_geo_metric.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from tqdm import tqdm

from metric import all_mesh_metrics

# Template command (mpmavatar environment, from MPMAvatar on a CPU allocation):
# python eval_tracking.py --prepared ../data/MPMAvatar/4DDress/examples/s148_t9 --seed 0


def read_pickle(path: Path) -> dict[str, Any]:
    assert path.is_file(), path
    with path.open("rb") as stream:
        return pickle.load(stream)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Raw v/f records in file order."""
    assert path.is_file(), path
    vertices, faces = [], []
    with path.open() as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.append([item.split("/")[0] for item in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64) - 1


def compact(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Mesh of the given faces with only their referenced vertices."""
    used = np.unique(faces)
    mapping = np.full(len(vertices), -1, dtype=np.int64)
    mapping[used] = np.arange(len(used))
    return trimesh.Trimesh(vertices[used], mapping[faces], process=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Seeds trimesh surface sampling")
    args = parser.parse_args()
    config = json.loads((args.prepared / "preparation.json").read_text())
    train, labels = config["train"], config["labels"]
    name = f"s{train['subject']}_t{train['take']}"
    tracking = args.prepared / "output" / "tracking" / f"{name}_{train['start']}_{train['count']}"
    take = Path(train["source"])
    frames = list(range(train["start"], train["start"] + train["count"]))
    with np.load(args.prepared / "data" / name / "cloth_vertices.npz") as cloth:
        cloth_ids = np.concatenate([cloth[str(label)] for label in labels])
    np.random.seed(args.seed)
    metrics: dict[str, list[float]] = {"full_CD": [], "full_F-Score": [], "garment_CD": [], "garment_F-Score": []}
    for frame in tqdm(frames, desc=name):
        points, faces = read_obj(tracking / f"mesh_cloth_{frame}.obj")
        scan = read_pickle(take / "Meshes_pkl" / f"mesh-f{frame:05d}.pkl")
        scan_points, scan_faces = np.asarray(scan["vertices"], dtype=np.float64), np.asarray(scan["faces"], dtype=np.int64)
        scan_labels = np.asarray(read_pickle(take / "Semantic" / "labels" / f"label-f{frame:05d}.pkl")["scan_labels"])
        assert scan_labels.shape == (len(scan_points),)
        tracked_cloth = np.zeros(len(points), dtype=bool)
        tracked_cloth[cloth_ids] = True
        pairs = {"full": (trimesh.Trimesh(scan_points, scan_faces, process=False), trimesh.Trimesh(points, faces, process=False)),
                 "garment": (compact(scan_points, scan_faces[np.isin(scan_labels, labels)[scan_faces].all(axis=1)]),
                             compact(points, faces[tracked_cloth[faces].all(axis=1)]))}
        for key, (gt, pred) in pairs.items():
            fscore_value, chamfer = all_mesh_metrics(gt, pred)
            metrics[f"{key}_CD"].append(float(chamfer))
            metrics[f"{key}_F-Score"].append(float(fscore_value))
    np.savez(tracking / "tracking_geo_metric.npz", frames=np.asarray(frames), **{k: np.asarray(v) for k, v in metrics.items()})
    summary = {"subject": train["subject"], "take": train["take"], "frames": [frames[0], frames[-1]], "labels": labels,
               "seed": args.seed, "mean": {k: float(np.mean(v)) for k, v in metrics.items()},
               "max": {k: float(np.max(v)) for k, v in metrics.items() if k.endswith("CD")},
               "worst_frame": {k: frames[int(np.argmax(v))] for k, v in metrics.items() if k.endswith("CD")},
               "per_frame": {k: v for k, v in metrics.items()}}
    (tracking / "tracking_geo_metric.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("mean", "max", "worst_frame")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
