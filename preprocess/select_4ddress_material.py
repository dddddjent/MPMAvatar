"""Material-fit setup for new native 4D-DRESS cases (no author splits or windows).

Window: for each simulated garment, the contiguous --frames window inside the
tracked training frames with the largest summed mean vertex speed of that garment,
among windows whose every frame is well tracked: garment CD vs the scan
(eval_tracking.py, tracking_geo_metric.json) <= --max-garment-cd and largest
principal stretch of the garment's triangles vs the rest (first) frame
<= --max-stretch. The thresholds are the worst frames inside the shipped
170/185/190/191 fit windows (logged in the case README). No such window is an error.
Lower attachment (two-garment cases): fix_v is the lower garment's top (waist)
boundary loop, as in the shipped 170/185 assets, and the split_garments.py
--iteration is the one whose joint-vertex fraction is closest to --joint-fraction
(170: 17.0%, 185: 18.7% of lower vertices).
Writes <prepared>/material_setup.json and, for a lower garment, <assets>/fix_v.npy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Template command (mpmavatar environment, workspace root, CPU node):
# python MPMAvatar/preprocess/select_4ddress_material.py --prepared data/MPMAvatar/4DDress/examples/s187_t7 --frames 10 --joint-fraction 0.18 --max-garment-cd 0.222 --max-stretch 35.9


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Raw v/f records in file order (no loader re-indexing)."""
    assert path.is_file(), path
    vertices, faces = [], []
    with path.open() as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.append([item.split("/")[0] for item in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64) - 1


def max_stretch(tracking: Path, frames: list[int], faces: np.ndarray) -> np.ndarray:
    """Per frame, the largest principal stretch over `faces` relative to frames[0] (the rest shape)."""
    def edges(vertices: np.ndarray) -> np.ndarray:
        triangles = vertices[faces]
        return np.stack([triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]], axis=2)
    rest = edges(read_obj(tracking / f"mesh_cloth_{frames[0]}.obj")[0])
    rest_metric = np.einsum("fij,fik->fjk", rest, rest)
    stretches = []
    for frame in frames:
        current = edges(read_obj(tracking / f"mesh_cloth_{frame}.obj")[0])
        # Eigenvalues of rest_metric^-1 current_metric are the squared principal stretches.
        squared = np.linalg.eigvals(np.linalg.solve(rest_metric, np.einsum("fij,fik->fjk", current, current))).real
        stretches.append(float(np.sqrt(squared.max())))
    return np.asarray(stretches)


def speed_window(tracking: Path, frames: list[int], vertex_ids: np.ndarray, count: int,
                 valid: np.ndarray) -> dict[str, object]:
    positions = np.stack([read_obj(tracking / f"mesh_cloth_{frame}.obj")[0][vertex_ids] for frame in frames])
    speeds = np.linalg.norm(positions[1:] - positions[:-1], axis=2).mean(axis=1)
    scores = np.convolve(speeds, np.ones(count - 1), mode="valid")
    allowed = np.convolve(valid, np.ones(count, dtype=int), mode="valid") == count
    assert allowed.any(), f"No {count}-frame window with every frame well tracked (valid frames: {np.asarray(frames)[valid].tolist()})"
    best = int(np.argmax(np.where(allowed, scores, -np.inf)))
    unrestricted = int(np.argmax(scores))
    return {"start": frames[best], "frames": count, "score_m": float(scores[best]),
            "mean_score_m": float(scores.mean()), "unrestricted_start": frames[unrestricted],
            "allowed_windows": int(allowed.sum()), "speeds_m_per_frame": speeds.tolist()}


def top_loop(faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    # Connected components of the boundary edges by label propagation.
    nodes = np.unique(boundary)
    label = {int(node): int(node) for node in nodes}
    changed = True
    while changed:
        changed = False
        for a, b in boundary:
            low = min(label[int(a)], label[int(b)])
            if label[int(a)] != low or label[int(b)] != low:
                label[int(a)] = label[int(b)] = low
                changed = True
    loops = {}
    for node, root in label.items():
        loops.setdefault(root, []).append(node)
    loops = [np.asarray(sorted(loop), dtype=np.int64) for loop in loops.values()]
    return max(loops, key=lambda loop: vertices[loop, 1].mean())


def joint_fraction(faces: np.ndarray, cloth_faces: np.ndarray, fix_v: np.ndarray, iterations: int) -> float:
    """split_garments.py fix_v branch, counting joint vertices."""
    expanded = np.flatnonzero(np.isin(faces, fix_v).any(axis=1))
    for _ in range(iterations):
        expanded = np.flatnonzero(np.isin(faces, faces[expanded]).sum(axis=1) > 1)
    human_faces = ~cloth_faces
    human_faces[expanded] = True
    cloth_v = np.unique(faces[cloth_faces])
    human_v = np.unique(faces[human_faces])
    return len(np.intersect1d(cloth_v, human_v)) / len(cloth_v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--joint-fraction", type=float, required=True)
    parser.add_argument("--max-garment-cd", type=float, required=True,
                        help="Largest per-frame garment CD (eval_tracking.py units) allowed inside the window")
    parser.add_argument("--max-stretch", type=float, required=True,
                        help="Largest triangle principal stretch vs the rest frame allowed inside the window")
    args = parser.parse_args()
    config = json.loads((args.prepared / "preparation.json").read_text())
    train = config["train"]
    name = f"s{train['subject']}_t{train['take']}"
    assets = args.prepared / "data" / name
    tracking = args.prepared / "output" / "tracking" / f"{name}_{train['start']}_{train['count']}"
    assert (args.prepared / "tracking_report.json").is_file(), "Run tracking first"
    output = args.prepared / "material_setup.json"
    assert not output.exists(), output
    frames = list(range(train["start"], train["start"] + train["count"]))
    cloth = np.load(assets / "cloth_vertices.npz")
    geometry = json.loads((tracking / "tracking_geo_metric.json").read_text())
    garment_cd = np.asarray(geometry["per_frame"]["garment_CD"])
    assert len(garment_cd) == len(frames), "tracking_geo_metric.json must cover every tracked frame"
    faces = read_obj(tracking / f"mesh_cloth_{frames[0]}.obj")[1]
    windows = {}
    for label in config["labels"]:
        stretch = max_stretch(tracking, frames, faces[np.isin(faces, cloth[str(label)]).all(axis=1)])
        valid = (garment_cd <= args.max_garment_cd) & (stretch <= args.max_stretch)
        windows[str(label)] = speed_window(tracking, frames, cloth[str(label)], args.frames, valid)
        windows[str(label)]["max_stretch"] = stretch.tolist()
    setup: dict[str, object] = {"labels": config["labels"], "windows": windows,
                                "max_garment_cd": args.max_garment_cd, "max_stretch": args.max_stretch,
                                "garment_cd": garment_cd.tolist()}
    if 4 in config["labels"]:
        assert config["labels"] == [3, 4], config["labels"]
        vertices, faces = read_obj(assets / "mesh_processed.obj")
        lower = cloth["4"]
        cloth_faces = np.isin(faces, lower).all(axis=1)
        fix_v = top_loop(faces[cloth_faces], vertices)
        assert np.isin(fix_v, lower).all()
        fractions = [joint_fraction(faces, cloth_faces, fix_v, iterations) for iterations in range(41)]
        iteration = int(np.argmin(np.abs(np.asarray(fractions) - args.joint_fraction)))
        np.save(assets / "fix_v.npy", fix_v)
        setup["lower_split"] = {"iteration": iteration, "joint_fraction": fractions[iteration],
                                "target_joint_fraction": args.joint_fraction, "fix_v": len(fix_v),
                                "fix_v_y": [float(vertices[fix_v, 1].min()), float(vertices[fix_v, 1].max())],
                                "lower_y": [float(vertices[lower, 1].min()), float(vertices[lower, 1].max())],
                                "fractions_by_iteration": fractions}
    output.write_text(json.dumps(setup, indent=2) + "\n")
    print(json.dumps(setup, indent=2), flush=True)


if __name__ == "__main__":
    main()
