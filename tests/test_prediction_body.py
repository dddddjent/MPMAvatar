"""Ensure displayed experiment bodies are the actual simulation colliders."""

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
# python -m unittest discover -s tests -p test_prediction_body.py -v


class PredictionBodyTests(unittest.TestCase):
    def test_archive_uses_collider_without_changing_scored_cloth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = root / "uv.obj"
            uv.write_text("vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
            surface = torch.arange(48, dtype=torch.float32).reshape(4, 4, 3)
            collider = surface[:, :3] + 100.
            collider_faces = torch.tensor([[0, 1, 2]])
            np.savez(root / "surface.npz", vertices=surface.numpy())
            manifest = dict(
                frame_ids=[0, 1, 2, 3], train_frame_ids=[0, 1], evaluation_frame_ids=[2, 3],
                simulation_domain={}, body_shape_experiment={"prediction_body": "fitted_smplx"},
            )
            trainer = SimpleNamespace(
                scene=SimpleNamespace(dataset_dir=str(root), test_frame_index=[2, 3],
                                      train_frame_index=[0, 1], train_frame_num=2, uv_path=str(uv)),
                output_path=str(root), body_motion=collider, collider_faces=collider_faces,
                args=SimpleNamespace(prescribed_surface_path=str(root / "surface.npz"),
                                     init_params_path=str(root / "material.npz")),
                reordered_cloth_v_idx=torch.tensor([0, 1, 2]), num_joint_v=0,
                torch_param={"D": torch.tensor(1.), "E": torch.tensor(2.), "H": torch.tensor(1.)},
            )
            with patch("future_evaluation.load_manifest", return_value=manifest), patch(
                "future_evaluation.evaluation_predictions", return_value=list(surface[2:])
            ):
                simulate_future(trainer)
            with np.load(root / "predicted_body.npz") as saved:
                np.testing.assert_array_equal(saved["frame_ids"], [2, 3])
                np.testing.assert_array_equal(saved["vertices"], collider[2:].numpy())
                np.testing.assert_array_equal(saved["faces"], collider_faces.numpy())
            with np.load(root / "predictions.npz") as saved:
                np.testing.assert_array_equal(saved["vertices"], surface[2:].numpy())
            report = json.loads((root / "geometry_metrics.json").read_text())
            self.assertEqual(report["mean_v2v_m"], 0.)
            self.assertIn("exact simulation collider", report["prediction_body"])


if __name__ == "__main__":
    unittest.main()
