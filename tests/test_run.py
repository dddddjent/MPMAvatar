"""Check shared stage dispatch and dataset split boundaries without training."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import run

# Template command (activate mpmavatar; run from MPMAvatar):
# python -m unittest discover -s tests -p test_run.py -v


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "data with spaces"
        self.root.mkdir()
        self.output = self.root.parent / "output with spaces"
        self.manifest: dict[str, Any] = {
            "status": "complete", "component": "cape_mpmavatar_export", "fps": 25,
            "actor": 1, "sequence": 1, "sequence_key": "a1_s1", "gender": "female",
            "frame_ids": list(range(6)), "train_frame_ids": [0, 1, 2, 3],
            "evaluation_frame_ids": [4, 5], "camera_ids": ["Cam001", "Cam002"],
            "split_path": "a1_s1/split_idx.npz",
            "prescribed_surface": "a1_s1/prescribed_surface.npz",
            "tracking_directory": "tracking/a1_s1_0_4", "attachment_count": 3,
        }
        for name in (self.manifest["split_path"], self.manifest["prescribed_surface"],
                     "body_models/TR00_E096.pt", "body_models/smplx/SMPLX_FEMALE.npz"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (self.root / "a1_s1/cam_info.json").write_text(json.dumps({"Cam001": {}, "Cam002": {}}))
        tracking = self.root / self.manifest["tracking_directory"]
        tracking.mkdir(parents=True)
        for frame in self.manifest["train_frame_ids"]:
            (tracking / f"params_{frame}.npz").touch()
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def checkpoint(self, relative: str) -> Path:
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def launch(self, stage: str, *options: str) -> list[str]:
        argv = ["run.py", "--data", str(self.root), "--output", str(self.output),
                "--stage", stage, *options]
        with patch("sys.argv", argv), patch("run.subprocess.run") as process:
            run.main()
        process.assert_called_once()
        self.assertEqual(process.call_args.kwargs, {"cwd": Path(run.__file__).resolve().parent, "check": True})
        return process.call_args.args[0]

    def test_all_sources_and_stages(self) -> None:
        for component, attachments in (("cape_mpmavatar_export", 3),
                                       ("clothtransformer_mpmavatar_export", 0),
                                       ("dgarments_mpmavatar_export", 0)):
            self.manifest.update(component=component, attachment_count=attachments)
            self.write_manifest()
            self.output = self.root.parent / component
            with self.subTest(component=component, stage="appearance"):
                command = self.launch("appearance", "--appearance-iterations", "12")
                self.assertEqual(command[1], "train_appearance.py")
                start = command.index("--train_frame_start_num")
                self.assertEqual(command[start + 1:start + 3], ["0", "4"])
                self.assertEqual(command[command.index("--save_iterations") + 1], "12")
                self.assertEqual(command[command.index("--smplx_gender") + 1], "female")
                self.assertEqual(command[command.index("--dataset_dir") + 1], str(self.root))
            self.checkpoint("appearance/point_cloud/timestep_000012/point_cloud.ply")
            with self.subTest(component=component, stage="material"):
                command = self.launch("material", "--appearance-iterations", "12", "--material-iterations", "7")
                self.assertEqual(command[1], "train_material_params.py")
                start = command.index("--train_frame_start_num")
                self.assertEqual(command[start + 1:start + 3], ["0", "4"])
                self.assertNotIn("--run_eval", command)
            material = self.checkpoint("material/seed0/last_param_00006.npz")
            with self.subTest(component=component, stage="evaluate"):
                command = self.launch("evaluate", "--appearance-iterations", "12", "--material-iterations", "7",
                                      "--material-frames", "2", "--skip-render", "--skip-video")
                start = command.index("--train_frame_start_num")
                self.assertEqual(command[start + 1:start + 3], ["0", "2"])
                start = command.index("--test_frame_start_num")
                self.assertEqual(command[start + 1:start + 3], ["4", "2"])
                self.assertEqual(command[command.index("--init_params_path") + 1], str(material))
                self.assertIn("--run_eval", command)
                self.assertIn("--skip_render", command)
                self.assertIn("--skip_video", command)

    def test_evaluation_from_start_keeps_fitting_window(self) -> None:
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        self.checkpoint("material/seed0/last_param_00003.npz")
        command = self.launch("evaluate", "--evaluate-from-start", "--material-frames", "2", "--skip-render")
        start = command.index("--test_frame_start_num")
        self.assertEqual(command[start + 1:start + 3], ["0", "6"])
        start = command.index("--train_frame_start_num")
        self.assertEqual(command[start + 1:start + 3], ["0", "2"])
        for stage in ("appearance", "material"):
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "only supported for --stage evaluate"):
                self.launch(stage, "--evaluate-from-start")

    def test_evaluation_uses_latest_saved_iteration(self) -> None:
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        latest = self.checkpoint("material/seed0/last_param_100000.npz")
        self.checkpoint("material/seed0/last_param_99999.npz")
        self.checkpoint("material/seed0/best_param_100001.npz")
        self.checkpoint("material/seed0/last_param_invalid.npz")
        (latest.parent / "last_param_100002.npz").mkdir()
        command = self.launch("evaluate", "--skip-render")
        self.assertEqual(command[command.index("--init_params_path") + 1], str(latest))

    def test_evaluation_accepts_explicit_checkpoint(self) -> None:
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        for name in ("last_param_00003.npz", "best_param_00003.npz"):
            checkpoint = self.checkpoint(f"selected checkpoints/{name}")
            with self.subTest(checkpoint=name):
                command = self.launch("evaluate", "--checkpoint", os.path.relpath(checkpoint),
                                      "--skip-render")
                self.assertEqual(command[command.index("--init_params_path") + 1], str(checkpoint))

    def test_shared_appearance_for_material_and_evaluation(self) -> None:
        shared = self.root.parent / "baseline appearance"
        checkpoint = shared / "point_cloud/timestep_030000/point_cloud.ply"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        material = self.launch("material", "--appearance-model", str(shared))
        self.checkpoint("material/seed0/last_param_00003.npz")
        evaluation = self.launch("evaluate", "--appearance-model", str(shared), "--skip-render")
        for command in (material, evaluation):
            self.assertEqual(command[command.index("--model_path") + 1], str(shared))
            self.assertEqual(command[command.index("--dataset_dir") + 1], str(self.root))
            self.assertEqual(command[command.index("--output_dir") + 1], str(self.output))
        self.assertFalse((self.output / "appearance").exists())
        with self.assertRaisesRegex(AssertionError, "only supported for material/evaluate"):
            self.launch("appearance", "--appearance-model", str(shared))

    def test_checkpoint_option_errors_do_not_launch_evaluation(self) -> None:
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        self.checkpoint("material/seed0/last_param_00199.npz")
        with self.assertRaisesRegex(AssertionError, "Material checkpoint does not exist"):
            self.launch("evaluate", "--checkpoint", str(self.output / "missing.npz"), "--skip-render")
        for stage in ("appearance", "material"):
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "only supported for --stage evaluate"):
                self.launch(stage, "--checkpoint", "checkpoint.npz")

    def test_stage_prerequisites_and_split_guards(self) -> None:
        for stage in ("material", "evaluate"):
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "Run appearance first"):
                self.launch(stage)
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        with self.assertRaisesRegex(AssertionError, "Run material fitting first"):
            self.launch("evaluate")
        argv = ["run.py", "--data", str(self.root), "--output", str(self.output), "--stage", "appearance"]
        with patch("sys.argv", argv), patch("run.subprocess.run") as process:
            run.main()
        process.assert_not_called()
        for count in ("1", "5"):
            with self.subTest(count=count), self.assertRaisesRegex(AssertionError, "training prefix"):
                self.launch("material", "--material-frames", count)
        with self.assertRaises(AssertionError):
            self.launch("material", "--skip-render")
        (self.output / "material").mkdir()
        with self.assertRaisesRegex(AssertionError, "No optimizer checkpoint"):
            self.launch("material")
        (self.root / self.manifest["tracking_directory"] / "params_4.npz").touch()
        with self.assertRaisesRegex(AssertionError, "Tracking must contain exactly the training prefix"):
            self.launch("evaluate")

    def test_resume_dispatch(self) -> None:
        self.checkpoint("appearance/training_state.pt")
        command = self.launch("appearance")
        self.assertEqual(command[command.index("--start_checkpoint") + 1],
                         str(self.output / "appearance/training_state.pt"))
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        self.checkpoint("material/seed0/last_param_00045.npz")
        command = self.launch("material", "--resume-parameters")
        self.assertIn("--resume", command)
        self.assertIn("--resume_parameters", command)
        self.checkpoint("material/seed0/training_state.pt")
        command = self.launch("material")
        self.assertIn("--resume", command)
        self.assertNotIn("--resume_parameters", command)

    def test_partial_old_appearance_requires_training_state(self) -> None:
        self.checkpoint("appearance/point_cloud/timestep_005000/point_cloud.ply")
        with self.assertRaisesRegex(AssertionError, "No appearance training checkpoint"):
            self.launch("appearance")

    def test_raw_body_stages_without_parametric_assets(self) -> None:
        self.manifest.pop("gender")
        self.manifest.update(body_model="raw", consumer_status="raw_mesh_loader_required",
                             body_motion_manifest="body_motion/manifest.json",
                             collider_sequence="body_motion/sequence.npz")
        body = self.root / "body_motion"
        body.mkdir()
        (body / "sequence.npz").touch()
        (body / "manifest.json").write_text(json.dumps({
            "status": "complete", "body_type": "raw_mesh", "fixed_topology": True,
            "units": "metres", "world_axes": "Y-up", "fps": 25,
            "frame_ids": self.manifest["frame_ids"], "sequence_path": "sequence.npz",
        }))
        for path in (self.root / "body_models").rglob("*"):
            if path.is_file():
                path.unlink()
        self.write_manifest()
        appearance = self.launch("appearance")
        self.assertNotIn("--smplx_gender", appearance)
        self.assertNotIn("--raw_collider_path", appearance)
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        material = self.launch("material")
        self.checkpoint("material/seed0/last_param_00199.npz")
        evaluation = self.launch("evaluate", "--skip-render")
        for command in (material, evaluation):
            self.assertNotIn("--smplx_gender", command)
            self.assertEqual(command[command.index("--raw_collider_path") + 1], str(body / "sequence.npz"))
        (body / "sequence.npz").unlink()
        with self.assertRaises(AssertionError):
            self.launch("material")

    def test_evaluation_renders_both_appearances(self) -> None:
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        self.checkpoint("material/seed0/last_param_00199.npz")
        (self.root / "config.json").write_text(json.dumps({"render_python": run.sys.executable}))
        for relative in ("capture/manifest.json", "capture/cam_info.json", "preparation/sequence.npz"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        argv = ["run.py", "--data", str(self.root), "--output", str(self.output),
                "--stage", "evaluate", "--skip-video"]
        with patch("sys.argv", argv), patch("run.subprocess.run") as process:
            run.main()
        self.assertEqual(process.call_count, 2)
        fitted, gt = process.call_args_list
        self.assertIn("--run_eval", fitted.args[0])
        self.assertNotIn("--skip_render", fitted.args[0])
        self.assertIn("--skip_video", fitted.args[0])
        self.assertEqual(gt.args[0], [run.sys.executable, "-B", "-m", "MPMAvatar.render_gt_lighting",
                                     "--data", str(self.root), "--evaluation",
                                     str(self.output / "evaluation/seed0"), "--skip-video"])
        self.assertEqual(gt.kwargs, {"cwd": Path(run.__file__).resolve().parent.parent, "check": True})
        with patch("sys.argv", argv), patch("run.subprocess.run", side_effect=run.subprocess.CalledProcessError(1, "evaluate")) as process:
            with self.assertRaises(run.subprocess.CalledProcessError):
                run.main()
        process.assert_called_once()

    def test_fixed_beta_requires_explicit_mesh_rendering(self) -> None:
        self.manifest["body_shape_experiment"] = {"prediction_body": "fitted_smplx"}
        self.write_manifest()
        self.checkpoint("appearance/point_cloud/timestep_030000/point_cloud.ply")
        self.checkpoint("material/seed0/last_param_00003.npz")
        with self.assertRaisesRegex(AssertionError, "--render-appearance gt_lighting"):
            self.launch("evaluate")
        command = self.launch("evaluate", "--skip-render")
        self.assertIn("--skip_render", command)
        (self.root / "config.json").write_text(json.dumps({"render_python": run.sys.executable}))
        for relative in ("capture/manifest.json", "capture/cam_info.json", "preparation/sequence.npz"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        argv = ["run.py", "--data", str(self.root), "--output", str(self.output),
                "--stage", "evaluate", "--render-appearance", "gt_lighting", "--skip-video"]
        with patch("sys.argv", argv), patch("run.subprocess.run") as process:
            run.main()
        self.assertEqual(process.call_count, 2)
        simulation, rendering = process.call_args_list
        self.assertIn("--skip_render", simulation.args[0])
        self.assertIn("MPMAvatar.render_gt_lighting", rendering.args[0])
        self.assertIn("--skip-video", rendering.args[0])


if __name__ == "__main__":
    unittest.main()
