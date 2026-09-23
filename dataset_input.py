"""Read completed avatar dataset exports in the ActorsHQ layout."""

import json
from pathlib import Path
from typing import Any


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    assert path.is_file(), f"A complete avatar dataset export is required: {path}"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["component"] in ("cape_mpmavatar_export", "clothtransformer_mpmavatar_export", "dgarments_mpmavatar_export")
    assert manifest["fps"] == 25, "The original solver uses 25 Hz observations"
    train = manifest["train_frame_ids"]
    future = manifest["evaluation_frame_ids"]
    assert len(train) >= 2 and len(future) >= 2
    assert train + future == manifest["frame_ids"] == list(range(len(train) + len(future)))
    assert manifest["sequence_key"] == f"a{manifest['actor']}_s{manifest['sequence']}"
    assert manifest["camera_ids"]
    actor_dir = root / manifest["sequence_key"]
    cameras = json.loads((actor_dir / "cam_info.json").read_text())
    assert list(cameras) == manifest["camera_ids"]
    body_model = manifest.get("body_model", "smplx")
    assert body_model in ("raw", "smplx"), f"Unsupported body model: {body_model}"
    required = [manifest["split_path"], manifest["prescribed_surface"]]
    if body_model == "raw":
        body_path = root / manifest["body_motion_manifest"]
        assert body_path.is_file(), body_path
        body = json.loads(body_path.read_text())
        assert body["status"] == "complete" and body["body_type"] == "raw_mesh"
        assert body["fixed_topology"] and body["units"] == "metres" and body["world_axes"] == "Y-up"
        assert body["frame_ids"] == manifest["frame_ids"] and body["fps"] == manifest["fps"]
        assert body_path.parent / body["sequence_path"] == root / manifest["collider_sequence"]
        required.append(manifest["collider_sequence"])
    else:
        required += ["body_models/TR00_E096.pt",
                     f"body_models/smplx/SMPLX_{manifest['gender'].upper()}.npz"]
    for relative in required:
        assert (root / relative).is_file(), root / relative
    tracking = root / manifest["tracking_directory"]
    actual = sorted(int(path.stem.split("_")[-1]) for path in tracking.glob("params_*.npz"))
    assert actual == train, "Tracking must contain exactly the training prefix"
    return manifest


def common_arguments(root: Path, manifest: dict[str, Any], model: Path) -> list[str]:
    actor, sequence = manifest["actor"], manifest["sequence"]
    arguments = [
        "--dataset_dir", str(root), "--dataset_type", "actorshq",
        "--actor", str(actor), "--sequence", str(sequence),
        "--trained_model_path", str(root / manifest["tracking_directory"]),
        "--uv_path", str(root / manifest["sequence_key"] / f"a{actor}s{sequence}_uv.obj"),
        "--split_idx_path", str(root / manifest["split_path"]),
        "--verts_start_idx", "0", "--model_path", str(model),
        "--test_camera_index", *[str(i) for i in range(len(manifest["camera_ids"]))],
    ]
    if manifest.get("body_model", "smplx") == "smplx":
        arguments += ["--smplx_gender", manifest["gender"]]
    return arguments
