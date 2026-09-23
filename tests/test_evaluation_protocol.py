"""Held-out metrics remain independent of the requested video frame range."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from future_evaluation import simulate_future

# Template command (activate mpmavatar; run from MPMAvatar):
# python -m unittest discover -s tests -p test_evaluation_protocol.py -v


class EvaluationProtocolTests(unittest.TestCase):
    def test_direct_rerender_scores_even_when_display_simulation_is_skipped(self) -> None:
        from train_material_params import Trainer

        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(prescribed_surface_path="surface.npz")
        trainer.scene = SimpleNamespace(dataset_dir="recorded_dataset")
        manifest = {"evaluation_frame_ids": [3, 4, 5]}
        with patch("train_material_params.load_manifest", return_value=manifest), patch(
            "train_material_params.score_evaluation"
        ) as score, patch("train_material_params.simulate_future") as display:
            trainer.eval(skip_sim=True, skip_render=True, skip_video=True)
        score.assert_called_once_with(trainer, manifest)
        display.assert_not_called()

    def test_scoring_uses_held_out_rollout_for_all_render_ranges_and_body_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = root / "uv.obj"
            uv.write_text("vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
            surface = torch.zeros(6, 4, 3)
            np.savez(root / "surface.npz", vertices=surface.numpy())
            evaluation = surface[3:].clone()
            evaluation[:, 1:3, 0] = torch.tensor([0., .06, .12])[:, None]
            # Neither attached cloth nor original body vertices enter the metric.
            evaluation[:, 0] = 500.
            evaluation[:, 3] = 1000.
            rendered_from_start = surface + 50.
            metric_values = []
            for tweaked in (False, True):
                for from_start in (False, True):
                    with self.subTest(tweaked=tweaked, from_start=from_start):
                        frames = list(range(6)) if from_start else [3, 4, 5]
                        output = root / f"tweaked_{tweaked}_from_start_{from_start}"
                        output.mkdir()
                        manifest = dict(frame_ids=list(range(6)), train_frame_ids=[0, 1, 2],
                                        evaluation_frame_ids=[3, 4, 5], simulation_domain={})
                        if tweaked:
                            manifest["body_shape_experiment"] = {"prediction_body": "fitted_smplx"}
                        trainer = SimpleNamespace(
                            scene=SimpleNamespace(dataset_dir=str(root), test_frame_index=frames,
                                                  train_frame_index=[0, 1], train_frame_num=2,
                                                  uv_path=str(uv)),
                            output_path=str(output), body_motion=surface[:, :3] + 100.,
                            collider_faces=torch.tensor([[0, 1, 2]]),
                            args=SimpleNamespace(prescribed_surface_path=str(root / "surface.npz"),
                                                 init_params_path=str(root / "material.npz")),
                            reordered_cloth_v_idx=torch.tensor([0, 1, 2]), num_joint_v=1,
                            torch_param={"D": torch.tensor(1.), "E": torch.tensor(2.), "H": torch.tensor(1.)},
                        )
                        with patch("future_evaluation.load_manifest", return_value=manifest), patch(
                            "future_evaluation.evaluation_predictions", return_value=list(evaluation)
                        ) as held_out, patch(
                            "future_evaluation.continuous_predictions", return_value=list(rendered_from_start)
                        ) as full:
                            result = torch.stack(simulate_future(trainer))
                        held_out.assert_called_once_with(trainer, [3, 4, 5])
                        if from_start:
                            full.assert_called_once_with(trainer)
                        else:
                            full.assert_not_called()
                        expected_render = rendered_from_start if from_start else evaluation
                        torch.testing.assert_close(result, expected_render)
                        report = json.loads((output / "geometry_metrics.json").read_text())
                        self.assertEqual(report["evaluation_scope"], "held_out")
                        self.assertEqual(report["protocol"], "from_first_evaluation_frame")
                        self.assertEqual(report["evaluation_frame_ids"], [3, 4, 5])
                        self.assertEqual(report["render_frame_ids"], frames)
                        self.assertEqual(report["initial_velocity_frame_ids"], [3, 4])
                        self.assertEqual(report["free_cloth_vertex_count"], 2)
                        self.assertAlmostEqual(report["mean_v2v_m"], .06, places=7)
                        self.assertAlmostEqual(report["mean_xyz_mse_m2"], .002, places=7)
                        metric_values.append(report["per_frame_v2v_m"])
                        with np.load(output / "evaluation_predictions.npz") as saved:
                            np.testing.assert_array_equal(saved["frame_ids"], [3, 4, 5])
                            np.testing.assert_array_equal(saved["vertices"], evaluation.numpy())
                        with np.load(output / "predictions.npz") as saved:
                            np.testing.assert_array_equal(saved["frame_ids"], frames)
                            np.testing.assert_array_equal(saved["vertices"], expected_render.numpy())
            self.assertTrue(all(values == metric_values[0] for values in metric_values))


if __name__ == "__main__":
    unittest.main()
