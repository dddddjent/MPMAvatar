"""Pick a native MPMAvatar train/test case for a 4D-DRESS Inner subject.

Two distinct takes are drawn at random (seeded) among takes with at least 101
contiguous scan frames: the first is the training take, the second the test take.
Both use their first scan frame and 100 frames, as in the shipped examples
(the training start is the tracking template and rest shape).
Garments follow the shipped examples: the upper garment (label 3) is always
simulated; the lower garment (label 4) only when it ends above the shoes
(skirts/shorts in 170/185), not when it reaches them (trousers in 190/191).
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

# Template command (mpmavatar environment, workspace root, CPU node):
# python MPMAvatar/preprocess/select_4ddress_case.py --inner ../datasets/4DDress/00187_Inner_1 ../datasets/4DDress/00187_Inner_2 --seed 187 --output MPMAvatar/data/4ddress_cases/s187.json


def read_pickle(path: Path) -> dict[str, Any]:
    assert path.is_file(), path
    with path.open("rb") as stream:
        return pickle.load(stream)


def take_frames(take: Path) -> list[int]:
    frames = sorted(int(frame) for frame in read_pickle(take / "basic_info.pkl")["scan_frames"])
    # Native scripts address frames as start..start+count-1, so only the leading contiguous run counts.
    run = 1
    while run < len(frames) and frames[run] == frames[0] + run:
        run += 1
    return frames[:run]


def garment_extent(take: Path, frame: int) -> dict[int, list[float]]:
    vertices = np.asarray(read_pickle(take / "Meshes_pkl" / f"mesh-f{frame:05d}.pkl")["vertices"])
    labels = np.asarray(read_pickle(take / "Semantic" / "labels" / f"label-f{frame:05d}.pkl")["scan_labels"])
    assert labels.shape == (len(vertices),)
    return {int(label): [int(np.count_nonzero(labels == label)), float(vertices[labels == label, 1].min()),
                         float(vertices[labels == label, 1].max())] for label in np.unique(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", type=Path, nargs="+", required=True,
                        help="Extracted <subject>_Inner[_n] folders of one subject")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    takes = {}
    for inner in args.inner:
        assert (inner / "Inner").is_dir(), inner
        for take in sorted((inner / "Inner").glob("Take*")):
            assert take.name not in takes, f"Duplicate {take.name}"
            takes[take.name] = take.resolve()
    subjects = {int(path.parent.parent.name.split("_")[0]) for path in takes.values()}
    assert len(subjects) == 1, subjects
    frames = {name: take_frames(path) for name, path in takes.items()}
    genders = {read_pickle(path / "basic_info.pkl")["gender"] for path in takes.values()}
    assert len(genders) == 1, genders
    # Tracking reads the pose after the last training frame, hence 101.
    eligible = sorted((name for name in takes if len(frames[name]) >= 101), key=lambda name: int(name[4:]))
    assert len(eligible) >= 2, frames
    train, test = random.Random(args.seed).sample(eligible, 2)
    train_start, test_start = frames[train][0], frames[test][0]
    extent = garment_extent(takes[train], train_start)
    assert 3 in extent, f"No upper garment: {extent}"
    labels = [3]
    if 4 in extent and extent[4][1] > extent[2][2]:
        labels.append(4)
    case = {"subject": subjects.pop(), "gender": genders.pop(), "seed": args.seed,
            "takes": {name: {"source": str(takes[name]), "first": frames[name][0], "frames": len(frames[name])}
                      for name in sorted(takes, key=lambda name: int(name[4:]))},
            "eligible": eligible, "train_take": int(train[4:]), "test_take": int(test[4:]),
            "train_source": str(takes[train]), "test_source": str(takes[test]),
            "train_start": train_start, "test_start": test_start, "labels": labels,
            "train_start_label_extent": {str(label): {"vertices": count, "y_min": low, "y_max": high}
                                         for label, (count, low, high) in extent.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps(case, indent=2), flush=True)


if __name__ == "__main__":
    main()
