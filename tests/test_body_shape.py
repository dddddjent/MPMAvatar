"""Verify one-beta fitting, frozen inputs, and exact optimizer continuation on CPU."""

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from body_shape import SingleBetaBody, fit_body_shape, load_body_shape

# Template command (activate mpmavatar; run from MPMAvatar):
# python -m unittest discover -s tests -p test_body_shape.py -v


class BodyShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "tweaked body"
        self.source.mkdir()
        self.material = self.root / "another body's material.npz"
        np.savez(self.material, D=0.7, E=230., H=0.9,
                 dataset_dir="/different/body", fitting_frame_ids=[0, 1, 2])
        self.betas = torch.arange(10, dtype=torch.float32).reshape(1, 10) / 10
        self.betas[0, 4] = 2.
        self.body = SimpleNamespace(beta_index=4, initial_beta=2., source_betas=self.betas,
                                    apply=self.apply_beta)

    @staticmethod
    def apply_beta(trainer: Any, beta: float) -> None:
        trainer.active_beta = beta

    @staticmethod
    def objective(trainer: Any) -> float:
        return (trainer.active_beta - 0.35) ** 2

    def trainer(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            output_path=str(self.root / name), args=SimpleNamespace(init_params_path=str(self.material)),
            accelerator=SimpleNamespace(num_processes=1),
            scene=SimpleNamespace(dataset_dir=str(self.source), train_frame_index=[0, 1]),
            train_frame_collider=torch.zeros(2, 3, 3),
            torch_param={key: torch.tensor(-1.) for key in ("D", "E", "H")},
            initial_cloth_velocity=torch.zeros(3, 3),
            resume_context={"iterations": 100, "fitting_frame_ids": [0, 1], "substep": 2},
        )

    def fit(self, trainer: SimpleNamespace, iterations: int, resume: bool = False,
            finite_difference: float = 0.01) -> tuple[dict[str, Any], str]:
        output = io.StringIO()
        with patch("body_shape.SingleBetaBody", return_value=self.body), patch(
            "body_shape.geometry_loss", side_effect=self.objective
        ), redirect_stdout(output):
            fit_body_shape(trainer, self.source, self.material, iterations=iterations,
                           learning_rate=0.1, finite_difference=finite_difference, resume=resume)
        state = torch.load(Path(trainer.output_path) / "body_shape_state.pt", weights_only=True)
        return state, output.getvalue()

    def test_finite_difference_improves_beta_with_independent_frozen_material(self) -> None:
        trainer = self.trainer("fit")
        before = self.betas.clone()
        state, terminal = self.fit(trainer, 4)
        self.assertLess(state["best"]["loss"], self.objective(SimpleNamespace(active_beta=2.)))
        self.assertEqual(trainer.active_beta, state["best"]["beta"])
        torch.testing.assert_close(self.betas, before, rtol=0, atol=0)
        for key, expected in (("D", .7), ("E", 2.3), ("H", .9)):
            self.assertAlmostEqual(trainer.torch_param[key].item(), expected, places=6)
            self.assertFalse(trainer.torch_param[key].requires_grad)
        for row in state["history"][1:]:
            self.assertAlmostEqual(row["gradient"], 2 * (row["beta_before"] - .35), places=6)
            self.assertAlmostEqual(row["loss"], (row["beta"] - .35) ** 2, places=12)
        root = Path(trainer.output_path)
        with np.load(root / "best_body_shape.npz", allow_pickle=False) as saved:
            self.assertEqual(float(saved["E"]), 230.)
            self.assertEqual(str(saved["material_checkpoint"]), str(self.material))
            self.assertEqual(str(saved["source_dataset"]), str(self.source))
            self.assertEqual(float(saved["beta_value"]), state["best"]["beta"])
            self.assertAlmostEqual(float(saved["loss"]), self.objective(trainer))
            mask = np.arange(10) != 4
            np.testing.assert_array_equal(saved["betas"][:, mask], before.numpy()[:, mask])
        summary = json.loads((root / "beta_summary.json").read_text())
        self.assertEqual(summary["completed_iterations"], 4)
        self.assertEqual(summary["last"], state["last"])
        self.assertIn("Best beta:", (root / "beta_summary.txt").read_text())
        self.assertIn("Body step 4/4: beta[4]=", terminal)
        with (root / "beta_history.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 5)

    def test_resumed_optimizer_matches_uninterrupted_fit_exactly(self) -> None:
        complete, _ = self.fit(self.trainer("complete"), 9)
        resumed = self.trainer("resumed")
        self.fit(resumed, 3)
        resumed.resume_context["iterations"] = 999
        continuation, _ = self.fit(resumed, 9, resume=True)
        self.assertEqual(continuation["next_step"], 9)
        for field in ("last", "best", "history", "context"):
            self.assertEqual(continuation[field], complete[field])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(continuation["optimizer"]["state"][0][key],
                                       complete["optimizer"]["state"][0][key], rtol=0, atol=0)
        self.assertEqual(continuation["optimizer"]["param_groups"], complete["optimizer"]["param_groups"])

    def test_resume_rejects_changed_simulation_or_material(self) -> None:
        trainer = self.trainer("resume guards")
        self.fit(trainer, 2)
        trainer.resume_context["substep"] = 3
        with self.assertRaisesRegex(AssertionError, "settings or input selection changed"):
            self.fit(trainer, 3, resume=True)
        trainer.resume_context["substep"] = 2
        np.savez(self.material, D=.8, E=230., H=.9)
        with self.assertRaisesRegex(AssertionError, "settings or input selection changed"):
            self.fit(trainer, 3, resume=True)

    def test_evaluation_loads_selected_beta_and_rejects_different_material(self) -> None:
        trainer = self.trainer("evaluation")
        state, _ = self.fit(trainer, 3)
        checkpoint = Path(trainer.output_path) / "best_body_shape.npz"
        trainer.active_beta = 100.
        with patch("body_shape.SingleBetaBody", return_value=self.body), redirect_stdout(io.StringIO()):
            load_body_shape(trainer, checkpoint, self.source)
            self.assertEqual(trainer.active_beta, state["best"]["beta"])
            np.savez(self.material, D=.7, E=231., H=.9)
            with self.assertRaisesRegex(AssertionError, "material E differs"):
                load_body_shape(trainer, checkpoint, self.source)

    def test_regeneration_changes_one_beta_and_refreshes_interval_velocities(self) -> None:
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        pose = np.zeros((3, 165), dtype=np.float32)
        pose[:, 0] = [0., 1., 3.]
        translation = np.array([[0., 0., 0.], [.1, .2, .3], [.3, .4, .5]], dtype=np.float32)
        template = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        captured: list[torch.Tensor] = []

        def fake_lbs(betas: torch.Tensor, full_pose: torch.Tensor, *args: Any) -> tuple[torch.Tensor, None]:
            captured.append(betas.clone())
            offset = betas[:, 4] * (1 + full_pose[:, 0])
            return template[None] + offset[:, None, None], None

        expected = template.numpy()[None] + (2 * (1 + pose[:, 0]))[:, None, None] + translation[:, None]
        model_path = self.source / "model.npz"
        model_path.touch()
        np.savez(self.source / "collider.npz", betas=self.betas.numpy(), full_pose=pose,
                 transl=translation, vertices=expected, faces=faces, frame_ids=[0, 1, 2],
                 timestamps=np.arange(3) / 25)
        (self.source / "manifest.json").write_text(json.dumps({
            "status": "complete", "component": "dgarments_mpmavatar_export", "frame_ids": [0, 1, 2],
            "body_shape_experiment": {"body_motion": "collider.npz", "fixed_betas": self.betas[0].tolist(),
                                      "perturbation": {"indexing": "zero_based", "beta_index": 4}},
        }))
        (self.source / "config.json").write_text(json.dumps({"smplx_model": str(model_path), "gender": "female"}))
        model = MagicMock()
        model.to.return_value = model
        model.faces = faces
        with patch("smplx.SMPLX", return_value=model), patch("smplx.lbs.lbs", side_effect=fake_lbs):
            body = SingleBetaBody(self.source, torch.device("cpu"), batch_size=2)
            np.testing.assert_allclose(body.vertices(2.).numpy(), expected, rtol=0, atol=1e-6)
            trainer = SimpleNamespace(collider_faces=torch.as_tensor(faces), motion_frame_ids=[0, 1, 2],
                                      scene=SimpleNamespace(train_frame_index=[0, 1], test_frame_index=[1, 2]))
            body.apply(trainer, 1.)
        changed = template.numpy()[None] + (1 + pose[:, 0])[:, None, None] + translation[:, None]
        np.testing.assert_allclose(trainer.body_motion.numpy(), changed, rtol=0, atol=1e-6)
        np.testing.assert_allclose(trainer.body_motion_velocity.numpy(), np.diff(changed, axis=0) * 25,
                                   rtol=0, atol=1e-5)
        torch.testing.assert_close(trainer.train_frame_collider_velo, trainer.body_motion_velocity[:1])
        torch.testing.assert_close(trainer.test_frame_collider_velo, trainer.body_motion_velocity[1:])
        torch.testing.assert_close(body.source_betas, self.betas, rtol=0, atol=0)
        np.testing.assert_array_equal(body.pose.numpy(), pose)
        np.testing.assert_array_equal(body.translation.numpy(), translation)
        unchanged = torch.arange(10) != 4
        for batch in captured:
            torch.testing.assert_close(batch[:, unchanged], self.betas[:, unchanged].expand(len(batch), -1),
                                       rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
