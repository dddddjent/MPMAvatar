"""Prepare native MPMAvatar inputs from 4D-DRESS scans without moving raw data.

Template geometry and UVs come from the first training scan. Coincident vertices
are welded, with the first source label retained, then separated by garment.
Camera labels follow eth-ait/4d-dress/dataset/extract_garment.py: rasterize the
original scan and use barycentric votes for the original vertex labels.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Template command (mpmavatar environment, workspace root, allocated GPU):
# python MPMAvatar/preprocess/prepare_4ddress.py --train-sequence ../datasets/4DDress/00190_Inner/Inner/Take2 --test-sequence ../datasets/4DDress/00190_Inner/Inner/Take5 --output data/MPMAvatar/4DDress/00190_Inner --smplx ../datasets/smplx --vposer ../datasets/vposer_v1_0/snapshots/TR00_E096.pt --train-start 11 --train-count 100 --test-start 11 --test-count 100 --labels 3 --stage all

COLORS = np.array([[128, 128, 128], [255, 128, 0], [128, 0, 255],
                   [180, 50, 50], [50, 180, 50], [0, 128, 255]], dtype=np.uint8)
NAMES = ("skin", "hair", "shoe", "upper", "lower", "outer")
CAMERAS = ("0004", "0028", "0052", "0076")


def read_pickle(path: Path) -> dict[str, Any]:
    assert path.is_file(), path
    with path.open("rb") as stream:
        return pickle.load(stream)


def scan_frame(sequence: Path, frame: int) -> tuple[dict[str, Any], np.ndarray]:
    scan = read_pickle(sequence / "Meshes_pkl" / f"mesh-f{frame:05d}.pkl")
    labels = np.asarray(read_pickle(sequence / "Semantic" / "labels" /
                                   f"label-f{frame:05d}.pkl")["scan_labels"])
    assert labels.shape == (len(scan["vertices"]),)
    assert np.isin(labels, np.arange(6)).all(), "Unexpected 4D-DRESS labels"
    return scan, labels.astype(np.int64)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray,
              uv: np.ndarray, uv_faces: np.ndarray) -> None:
    """Keep UV seams separate from physical vertex connectivity."""
    assert faces.shape == uv_faces.shape
    with path.open("w") as stream:
        for vertex in vertices:
            stream.write("v " + " ".join(format(float(x), ".17g") for x in vertex) + "\n")
        for coord in uv:
            stream.write("vt " + " ".join(format(float(x), ".17g") for x in coord) + "\n")
        for face, uv_face in zip(faces + 1, uv_faces + 1):
            stream.write("f " + " ".join(f"{v}/{t}" for v, t in zip(face, uv_face)) + "\n")


def split_indices(faces: np.ndarray, cloth_vertices: np.ndarray,
                  vertex_count: int) -> dict[str, np.ndarray]:
    """Native split_garments.py semantics for --iteration 0 --fix_v None."""
    cloth_face = np.isin(faces, cloth_vertices).all(axis=1)
    cloth_v = np.zeros(vertex_count, dtype=bool)
    human_v = np.zeros(vertex_count, dtype=bool)
    cloth_v[faces[cloth_face].ravel()] = True
    human_v[faces[~cloth_face].ravel()] = True
    human_v |= ~cloth_v
    joint = np.flatnonzero(cloth_v & human_v)
    cv = np.concatenate((joint, np.flatnonzero(~human_v)))
    hv = np.concatenate((joint, np.flatnonzero(~cloth_v)))
    cf, hf = np.flatnonzero(cloth_face), np.flatnonzero(~cloth_face)
    cm = np.full(vertex_count, -1, dtype=np.int32)
    hm = cm.copy()
    cm[cv], hm[hv] = np.arange(len(cv)), np.arange(len(hv))
    return {"num_joint_v": np.asarray(len(joint)), "num_joint_f": np.asarray(0),
            "reordered_cloth_v_idx": cv.astype(np.int32),
            "reordered_human_v_idx": hv.astype(np.int32),
            "reordered_cloth_f_idx": cf.astype(np.int32),
            "reordered_human_f_idx": hf.astype(np.int32),
            "new_cloth_faces": cm[faces[cf]], "new_human_faces": hm[faces[hf]]}


def build_template(sequence: Path, frame: int, garments: list[int], output: Path) -> dict[str, Any]:
    scan, source_labels = scan_frame(sequence, frame)
    source_v = np.asarray(scan["vertices"], dtype=np.float64)
    source_f = np.asarray(scan["faces"], dtype=np.int64)
    uv = np.asarray(scan["uvs"], dtype=np.float64)
    assert np.isfinite(source_v).all() and np.isfinite(uv).all()
    assert source_f.ndim == 2 and source_f.shape[1] == 3
    assert source_f.min() >= 0 and source_f.max() < len(source_v)
    assert uv.shape == (len(source_v), 2)
    welded, first, inverse = np.unique(source_v, axis=0, return_index=True, return_inverse=True)
    welded_labels = source_labels[first]
    welded_faces = inverse[source_f]
    # Each selected garment is a separate simulation surface; the rest is the
    # visible avatar surface. This duplicates shared garment/body boundaries.
    face_part = np.zeros(len(source_f), dtype=np.int64)
    for label in garments:
        selected = (welded_labels[welded_faces] == label).all(axis=1)
        assert selected.any(), f"No faces for garment label {label}"
        face_part[selected] = label
    faces = np.empty_like(source_f)
    vertices = []
    cloth = {}
    offset = 0
    for part in [0, *garments]:
        selected = face_part == part
        used = np.unique(welded_faces[selected])
        assert len(used), f"Empty template part {part}"
        mapping = np.full(len(welded), -1, dtype=np.int64)
        mapping[used] = np.arange(len(used)) + offset
        faces[selected] = mapping[welded_faces[selected]]
        vertices.append(welded[used])
        if part:
            cloth[str(part)] = np.arange(offset, offset + len(used), dtype=np.int32)
        offset += len(used)
    vertex_array = np.concatenate(vertices)
    # Exact geometric equality is intentional, not an approximate surface test.
    assert np.array_equal(vertex_array[faces], source_v[source_f])
    triangles = vertex_array[faces]
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    assert (double_area > 0).all(), "Source contains degenerate faces; repair explicitly before tracking"
    output.mkdir(parents=True)
    write_obj(output / "mesh_processed.obj", vertex_array, faces, uv, source_f)
    np.savez(output / "cloth_vertices.npz", **cloth)
    np.savez(output / "split_idx.npz", **split_indices(faces, np.concatenate(list(cloth.values())), len(vertex_array)))
    for label in garments:
        selected = face_part == label
        ids = cloth[str(label)]
        write_obj(output / f"{NAMES[label]}.obj", vertex_array[ids], faces[selected] - ids[0],
                  uv, source_f[selected])
    report = {"source_vertices": len(source_v), "vertices": len(vertex_array),
              "faces": len(faces), "uv_vertices": len(uv),
              "coincident_vertex_label_conflicts": int(np.count_nonzero(source_labels != welded_labels[inverse])),
              "garment_faces": {str(k): int(np.count_nonzero(face_part == k)) for k in garments},
              "garment_vertices": {k: len(v) for k, v in cloth.items()},
              "source_triangle_corners_exact": True, "source_uv_exact": True,
              "label_rule": "first source vertex at each exactly coincident position; all three face labels match",
              "geometry_rule": "weld exact positions separately within each selected garment and remaining surface",
              "dense_tracker_laplacian_gib": len(vertex_array) ** 2 * 4 / 1024 ** 3}
    (output / "template_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def link(source: Path, target: Path) -> None:
    assert source.exists(), source
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def sequence_info(path: Path, start: int, count: int, tracking: bool) -> dict[str, Any]:
    assert path.is_dir(), path
    assert path.parent.name == "Inner", "Native MPMAvatar scripts currently support Inner outfits"
    subject = int(path.parent.parent.name.split("_")[0])
    take = int(path.name.removeprefix("Take"))
    info = read_pickle(path / "basic_info.pkl")
    assert count > 0
    frames = list(range(start, start + count))
    assert set(frames).issubset({int(f) for f in info["scan_frames"]})
    cameras = read_pickle(path / "Capture" / "cameras.pkl")
    assert set(cameras) == set(CAMERAS)
    for frame in frames:
        required = [path / "Meshes_pkl" / f"mesh-f{frame:05d}.pkl",
                    path / "Semantic" / "labels" / f"label-f{frame:05d}.pkl",
                    path / "SMPLX" / f"mesh-f{frame:05d}_smplx.pkl",
                    path / "SMPLX" / f"mesh-f{frame:05d}_smplx.ply"]
        for camera in CAMERAS:
            required.extend([path / "Capture" / camera / "images" / f"capture-f{frame:05d}.png",
                             path / "Capture" / camera / "masks" / f"mask-f{frame:05d}.png"])
        for item in required:
            assert item.is_file(), item
    if tracking:
        next_pose = path / "SMPLX" / f"mesh-f{frames[-1] + 1:05d}_smplx.pkl"
        assert next_pose.is_file(), f"Tracking also reads the next pose: {next_pose}"
    return {"source": str(path), "subject": subject, "take": take, "start": start,
            "count": count, "gender": info["gender"]}


def sequence_links(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in ("basic_info.pkl", "Meshes_pkl", "SMPLX"):
        link(source / name, destination / name)
    (destination / "Semantic").mkdir()
    link(source / "Semantic" / "labels", destination / "Semantic" / "labels")
    (destination / "Semantic" / "clothes").mkdir()
    (destination / "Capture").mkdir()
    link(source / "Capture" / "cameras.pkl", destination / "Capture" / "cameras.pkl")
    for camera in CAMERAS:
        target = destination / "Capture" / camera
        target.mkdir()
        for name in ("images", "masks"):
            link(source / "Capture" / camera / name, target / name)
        (target / "labels").mkdir()


def extract_clothes(scan: dict[str, Any], labels: np.ndarray, garments: list[int]) -> dict[str, Any]:
    """Reference scan garment surfaces use original labels, as in official tools."""
    faces = np.asarray(scan["faces"])
    result = {}
    for label in garments:
        ids = np.flatnonzero(labels == label)
        selected = (labels[faces] == label).all(axis=1)
        mapping = np.full(len(labels), -1, dtype=np.int64)
        mapping[ids] = np.arange(len(ids))
        result[NAMES[label]] = {"vertices": scan["vertices"][ids],
                                "faces": mapping[faces[selected]],
                                "colors": scan["colors"][ids], "uvs": scan["uvs"][ids]}
    return result


def render_observations(sequence: Path, start: int, count: int, garments: list[int]) -> dict[str, Any]:
    import torch
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes

    assert torch.cuda.is_available(), "Label rendering requires an allocated CUDA GPU"
    camera_data = read_pickle(sequence / "Capture" / "cameras.pkl")
    renderers = {}
    for camera in CAMERAS:
        with Image.open(sequence / "Capture" / camera / "images" / f"capture-f{start:05d}.png") as im:
            width, height = im.size
        data = camera_data[camera]
        intrinsic = torch.as_tensor(data["intrinsics"], dtype=torch.float32, device="cuda")
        extrinsic = torch.tensor(data["extrinsics"], dtype=torch.float32, device="cuda")
        extrinsic[:2] *= -1
        model = PerspectiveCameras(focal_length=intrinsic.diag()[:2][None],
                                   principal_point=intrinsic[:2, 2][None],
                                   R=extrinsic[:, :3].T[None], T=extrinsic[:, 3][None],
                                   in_ndc=False, image_size=((height, width),), device="cuda")
        settings = RasterizationSettings(image_size=(height, width), blur_radius=0,
                                         faces_per_pixel=1, max_faces_per_bin=80000)
        renderers[camera] = MeshRasterizer(cameras=model, raster_settings=settings)
    mask_ious = {camera: [] for camera in CAMERAS}
    with torch.no_grad():
        for frame in range(start, start + count):
            scan, labels = scan_frame(sequence, frame)
            vertices = torch.as_tensor(scan["vertices"], dtype=torch.float32, device="cuda")
            faces = torch.as_tensor(scan["faces"], dtype=torch.int64, device="cuda")
            vertex_labels = torch.as_tensor(labels, dtype=torch.int64, device="cuda")
            mesh = Meshes(verts=[vertices], faces=[faces])
            for camera, rasterizer in renderers.items():
                fragments = rasterizer(mesh)
                face_ids = fragments.pix_to_face[0, :, :, 0]
                visible = face_ids >= 0
                face_labels = vertex_labels[faces[face_ids[visible]]]
                barycentric = fragments.bary_coords[0, :, :, 0][visible]
                votes = torch.zeros((len(face_labels), 6), dtype=torch.float32, device="cuda")
                for corner in range(3):
                    votes.scatter_add_(1, face_labels[:, corner:corner + 1], barycentric[:, corner:corner + 1])
                pixel_labels = votes.argmax(dim=1).cpu().numpy()
                foreground = visible.cpu().numpy()
                colored = np.full((*foreground.shape, 3), 255, dtype=np.uint8)
                colored[foreground] = COLORS[pixel_labels]
                directory = sequence / "Capture" / camera
                Image.fromarray(colored).save(directory / "labels" / f"label-f{frame:05d}.png")
                with Image.open(directory / "masks" / f"mask-f{frame:05d}.png") as image:
                    source_mask = np.asarray(image.convert("L")) > 127
                assert source_mask.shape == foreground.shape
                union = np.count_nonzero(source_mask | foreground)
                assert union > 0, f"Empty projection for {camera}, frame {frame}"
                mask_ious[camera].append(float(np.count_nonzero(source_mask & foreground) / union))
                if frame == start:
                    with Image.open(directory / "images" / f"capture-f{frame:05d}.png") as image:
                        rgb = np.asarray(image.convert("RGB"))
                    overlay = np.rint(0.5 * rgb + 0.5 * colored).astype(np.uint8)
                    Image.fromarray(overlay).save(directory / "labels" / f"overlay-f{frame:05d}.png")
            clothes = extract_clothes(scan, labels, garments)
            with (sequence / "Semantic" / "clothes" / f"cloth-f{frame:05d}.pkl").open("wb") as stream:
                pickle.dump(clothes, stream)
            print(f"Rendered labels/extracted garments: {sequence.name} frame {frame}", flush=True)
    report = {"frames": count, "camera_label_images": count * len(CAMERAS),
              "silhouette_iou_with_dataset_masks": {camera: {"mean": float(np.mean(values)),
                                                               "min": min(values)}
                                                      for camera, values in mask_ious.items()}}
    (sequence / "observation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sequence", type=Path, required=True)
    parser.add_argument("--test-sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smplx", type=Path, required=True)
    parser.add_argument("--vposer", type=Path, required=True)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--test-start", type=int, required=True)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--labels", type=int, nargs="+", choices=(3, 4), required=True)
    parser.add_argument("--stage", choices=("assets", "observations", "all"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    garments = sorted(set(args.labels))
    assert not output.is_relative_to(args.train_sequence.resolve())
    assert not output.is_relative_to(args.test_sequence.resolve())
    train = sequence_info(args.train_sequence.resolve(), args.train_start, args.train_count, True)
    test = sequence_info(args.test_sequence.resolve(), args.test_start, args.test_count, False)
    assert train["subject"] == test["subject"] and train["gender"] == test["gender"]
    assert train["take"] != test["take"], "Choose separate training and testing takes"
    config = {"train": train, "test": test, "labels": garments,
              "smplx": str(args.smplx.resolve()), "vposer": str(args.vposer.resolve())}
    name = f"s{train['subject']}_t{train['take']}"
    relative = Path("data/4D-DRESS") / f"{train['subject']:05d}_Inner" / "Inner"
    if args.stage in ("assets", "all"):
        assert not output.exists(), f"Choose a new output directory: {output}"
        for gender in ("FEMALE", "MALE", "NEUTRAL"):
            assert (args.smplx / f"SMPLX_{gender}.npz").is_file()
        assert args.vposer.is_file(), args.vposer
        output.mkdir(parents=True)
        (output / "preprocess").mkdir()
        (output / "data/body_models").mkdir(parents=True)
        link(args.smplx, output / "data/body_models/smplx")
        link(args.vposer, output / "data/body_models/TR00_E096.pt")
        for item in (train, test):
            sequence_links(Path(item["source"]), output / relative / f"Take{item['take']}")
        report = build_template(Path(train["source"]), train["start"], garments, output / "data" / name)
        (output / "preparation.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    if args.stage in ("observations", "all"):
        assert (output / "preparation.json").is_file()
        assert json.loads((output / "preparation.json").read_text()) == config, "Preparation arguments changed"
        for item in (train, test):
            report = render_observations(output / relative / f"Take{item['take']}",
                                         item["start"], item["count"], garments)
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
