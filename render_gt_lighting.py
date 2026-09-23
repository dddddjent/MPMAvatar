"""Render predicted surfaces with the prepared dataset's original capture settings."""

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import bpy
from mathutils import Matrix
import numpy as np

from cape_avatar import render_utils as render
from MPMAvatar.evaluation_video import encode_comparison

# Template command (dataset's render_python environment, from clothes-reconstruction):
# python -m MPMAvatar.render_gt_lighting --data /absolute/dataset --evaluation /absolute/output/evaluation/seed0 --skip-video
# Omit --skip-video to encode prediction/reference/comparison videos.


def surface_regions(root: Path, component: str, settings: dict[str, Any]
                    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Recover the exact source face regions and fixed material colors."""
    with np.load(root / "preparation/sequence.npz", allow_pickle=False) as data:
        faces = data["faces"]
        if component == "clothtransformer_mpmavatar_export":
            body_count = len(data["body_faces"])
            regions = [(name, region_faces, np.asarray(settings["materials"][name]["color_linear_rgb"]))
                       for name, region_faces in (("body", faces[:body_count]), ("cloth", faces[body_count:]))]
        else:
            assert component in ("cape_mpmavatar_export", "dgarments_mpmavatar_export")
            if component == "cape_mpmavatar_export":
                with np.load(root / "preparation/garment.npz", allow_pickle=False) as garment:
                    labels, colors = garment["face_labels"], garment["face_colors"]
                names = ("skin", "shirt", "lower")
            else:
                labels, colors = data["face_labels"], data["face_colors"]
                names = ("skin", "dress")
            regions = []
            for index, name in enumerate(names):
                selected = labels == index
                assert selected.any(), name
                assert np.allclose(colors[selected], colors[selected][0]), name
                regions.append((name, faces[selected], colors[selected][0]))
    return faces, regions


def capture_cameras(scene: Any, settings: dict[str, Any], info: dict[str, Any],
                    component: str) -> dict[str, Any]:
    """Restore saved camera matrices without refitting cameras to predicted cloth."""
    cameras = {}
    rig = settings["camera_rig"]
    for name, calibration in info.items():
        width, height = calibration["W"], calibration["H"]
        assert [width, height] == settings["capture"]["resolution"]
        data = bpy.data.cameras.new(name)
        data.type = "PERSP"
        data.lens, data.sensor_width = rig["lens_mm"], rig["sensor_width_mm"]
        data.sensor_fit = "HORIZONTAL"
        data.dof.use_dof = False
        # CAPE stores rig clipping; the other exporters store per-camera clipping.
        if component == "cape_mpmavatar_export":
            data.clip_start, data.clip_end = rig["clip_start_m"], rig["clip_end_m"]
        else:
            data.clip_start, data.clip_end = calibration["clip_planes"]
        assert np.allclose(render.camera_intrinsics(width, height, data.lens, data.sensor_width),
                           calibration["K"])
        camera = bpy.data.objects.new(name, data)
        scene.collection.objects.link(camera)
        camera.matrix_world = Matrix((render.SCIENTIFIC_TO_BLENDER @ np.asarray(calibration["RT"])
                                      @ render.POSITIVE_DEPTH_TO_BLENDER_CAMERA).tolist())
        cameras[name] = camera
    bpy.context.view_layer.update()
    for name, camera in cameras.items():
        assert np.allclose(render.positive_depth_camera_to_world(np.asarray(camera.matrix_world)),
                           info[name]["RT"], atol=1e-6)
    return cameras


def render_predictions(root: Path, evaluation: Path, skip_video: bool) -> None:
    """Render simulated cloth and the selected prediction body with source lighting."""
    manifest = json.loads((root / "manifest.json").read_text())
    metric_path = evaluation / "geometry_metrics.json"
    assert metric_path.is_file(), "Run material evaluation first to compute held-out error."
    metrics = json.loads(metric_path.read_text())
    assert metrics["protocol"] == "from_first_evaluation_frame", (
        "These metrics use the previous evaluation protocol; rerun run.py --stage evaluate first.")
    assert metrics["evaluation_frame_ids"] == manifest["evaluation_frame_ids"]
    print(f"Held-out error: {metrics['mean_v2v_mm']:.6f} mm V2V, "
          f"{metrics['mean_xyz_mse_m2']:.9g} m^2 XYZ MSE ({metric_path})", flush=True)
    capture = json.loads((root / "capture/manifest.json").read_text())
    assert manifest["status"] == capture["status"] == "complete"
    settings = capture["settings"]
    info = json.loads((root / "capture/cam_info.json").read_text())
    assert list(info) == manifest["camera_ids"] == capture["camera_ids"]
    with np.load(evaluation / "predictions.npz", allow_pickle=False) as data:
        frames, vertices = data["frame_ids"].tolist(), data["vertices"]
    assert frames in (manifest["evaluation_frame_ids"], manifest["frame_ids"])
    assert len(vertices) == len(frames) and np.isfinite(vertices).all()
    faces, regions = surface_regions(root, manifest["component"], settings)
    assert vertices.ndim == 3 and vertices.shape[2] == 3
    assert int(faces.max()) + 1 == vertices.shape[1]
    surfaces = {name: vertices for name, _, _ in regions}
    fitted_body = manifest.get("body_shape_experiment", {}).get("prediction_body") == "fitted_smplx"
    if fitted_body:
        assert manifest["component"] == "dgarments_mpmavatar_export"
        body_path = evaluation / "predicted_body.npz"
        assert body_path.is_file(), f"Missing exact simulated body geometry: {body_path}"
        with np.load(body_path, allow_pickle=False) as data:
            assert data["frame_ids"].tolist() == frames
            body_vertices, body_faces = data["vertices"], data["faces"]
        assert body_vertices.ndim == 3 and body_vertices.shape[0] == len(frames)
        assert body_vertices.shape[2] == 3 and np.isfinite(body_vertices).all()
        assert body_faces.ndim == 2 and body_faces.shape[1] == 3
        assert body_faces.min() >= 0 and body_faces.max() < body_vertices.shape[1]
        surfaces["skin"] = body_vertices
        regions = [(name, body_faces if name == "skin" else region_faces, color)
                   for name, region_faces, color in regions]
    output = evaluation / "gt_lighting"
    assert not output.exists(), output
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    devices = render.configure_scene(scene, settings)
    scene.render.fps = manifest["fps"]
    objects = {}
    for index, (name, region_faces, color) in enumerate(regions):
        material = render.make_material(name, color, settings["materials"][name])
        first = render.scientific_points_to_blender(surfaces[name][0])
        objects[name] = render.create_mesh_object(name, first, region_faces, material)
        objects[name].pass_index = index + 1
    cameras = capture_cameras(scene, settings, info, manifest["component"])
    render.create_lights(scene, settings["lights"])
    # Match the capture compositor's unmodified RGB output.
    scene.use_nodes = True
    tree = scene.node_tree
    assert tree is not None
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    for name in cameras:
        for kind in ("pred", "gt"):
            (output / name / kind).mkdir(parents=True)
    for frame_index, frame in enumerate(frames):
        for name, obj in objects.items():
            render.update_vertices({name: obj}, render.scientific_points_to_blender(surfaces[name][frame_index]))
        for offset, (name, camera) in enumerate(cameras.items()):
            print(f"GT lighting: frame {frame}, {name}", flush=True)
            scene.camera = camera
            scene.cycles.seed = settings["capture"]["seed"] + frame * 8 + offset
            png = output / name / "pred" / f"{frame:04d}.png"
            scene.render.filepath = str(png)
            bpy.ops.render.render(write_still=True)
            render.assert_rgb_png(png, info[name]["W"], info[name]["H"])
            reference = root / "capture/rgbs" / name / f"{name}_rgb{frame:06d}.png"
            assert reference.is_file(), reference
            shutil.copyfile(reference, output / name / "gt" / f"{frame:04d}.png")
    if not skip_video:
        for name in cameras:
            encode_comparison(output / name, frames, manifest["fps"])
    (output / "manifest.json").write_text(json.dumps({
        "status": "complete", "geometry": str(evaluation / "predictions.npz"),
        "capture": str(root / "capture/manifest.json"), "frame_ids": frames,
        "camera_ids": list(cameras), "devices": devices,
        "appearance": "Source materials, lights, cameras and color management; predicted surface geometry",
        "prediction_body": "Fitted SMPL-X simulation collider" if fitted_body else "Original recorded body surface",
        "body_geometry": str(evaluation / "predicted_body.npz") if fitted_body else str(evaluation / "predictions.npz"),
        "evaluation_metrics": str(metric_path),
        "reference": "Original capture PNGs", "video_layout": "prediction left, reference right",
        "videos": not skip_video,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    render_predictions(args.data.resolve(), args.evaluation.resolve(), args.skip_video)


if __name__ == "__main__":
    main()
