"""Numerical and orchestration checks; no MPM simulation or GPU allocation.

# Template command (mpmavatar environment, from MPMAvatar):
# python -m unittest discover -s tests -p test_material_sensitivity.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import torch

from material_sensitivity import (
    REPO, PARAMETERS, audit, center_ad, central_differences, effective_parameters,
    plan, relative_error, run_case, run_garment, save_group, save_progress, summarize,
)


class AlgebraTrainer:
    """CPU algebra fixture using the Trainer's internal parameter convention."""

    def __init__(self, frame_start: int) -> None:
        self.torch_param = {key: torch.tensor(1.0, requires_grad=True) for key in PARAMETERS}
        self.optimizer = Mock()
        self.optimizer.zero_grad.side_effect = self.zero_grad
        self.particle_init_position = torch.zeros((3, 3))
        self.scene = SimpleNamespace(train_frame_index=list(range(frame_start, frame_start + 12)))
        self.new_cloth_faces = torch.tensor([[0, 1, 2]])
        self.reordered_cloth_v_idx = torch.tensor([17, 8, 5])
        self.num_joint_v = 1
        self.ad_evaluations = 0
        self.native_evaluations = 0

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none
        for value in self.torch_param.values():
            value.grad = None

    def _autodiff_material(self, device: str) -> tuple[torch.Tensor, None]:
        assert device == "cpu"
        self.ad_evaluations += 1
        d, e, h = (self.torch_param[key] for key in PARAMETERS)
        return d.square() + 100.0 * e.square() + h.square(), None

    def replay(self) -> tuple[np.ndarray, float]:
        self.native_evaluations += 1
        d = float(self.torch_param["D"].detach())
        e = float((self.torch_param["E"].detach() * 100.0))
        h = float(self.torch_param["H"].detach())
        positions = np.zeros((12, 3, 3), dtype=np.float32)
        positions[1:, :, 0] = d + e / 100.0 + h
        return positions, d * d + e * e / 100.0 + h * h


class MaterialSensitivityTest(unittest.TestCase):
    def test_fp64_resolves_all_requested_perturbations(self) -> None:
        self.config["precision"] = "fp64"
        rows = plan(self.config)
        for row in rows:
            center = effective_parameters(self.config["parameter_sets"][row["parameter_set"]], "fp64")
            if row["offset"]:
                self.assertNotEqual(row[f"effective_{row['direction']}"], center[row["direction"]])

    def setUp(self) -> None:
        self.config = json.loads((REPO / "configs/material_sensitivity_185.json").read_text())

    def test_full_matrix_and_float32_resolution(self) -> None:
        rows = plan(self.config)
        self.assertEqual(len(rows), 270)
        self.assertEqual(len({row["trajectory"] for row in rows}), 270)
        for garment in ("upper", "lower"):
            subset = [row for row in rows if row["garment"] == garment]
            self.assertEqual(len(subset), 135)
            for row in subset:
                center = self.config["parameter_sets"][row["parameter_set"]]
                for key in PARAMETERS:
                    expected = center[key] + (row["offset"] * row["step_size"] if key == row["direction"] else 0)
                    self.assertEqual(row[key], expected)
        stiff = [row for row in rows if row["parameter_set"] == "stiff"
                 and row["direction"] == "E" and row["step_size"] == 1e-6]
        self.assertEqual({row["effective_E"] for row in stiff}, {1000.0})
        for row in rows:
            tensors = {key: torch.tensor(row[key] / (100.0 if key == "E" else 1.0)) for key in PARAMETERS}
            actual = {key: float(tensors[key] * (100.0 if key == "E" else 1.0)) for key in PARAMETERS}
            self.assertEqual(effective_parameters({key: row[key] for key in PARAMETERS}), actual)

    def test_three_and_five_point_derivatives(self) -> None:
        step = 0.1
        fd = central_differences({offset: (1.0 + offset * step) ** 3 for offset in range(-2, 3)}, step)
        self.assertAlmostEqual(fd["h"], 3.01)
        self.assertAlmostEqual(fd["2h"], 3.04)
        self.assertAlmostEqual(fd["five_point"], 3.0)
        constant = {offset: np.full((2, 3, 3), 1.234567) for offset in range(-2, 3)}
        for value in central_differences(constant, 1e-6).values():
            np.testing.assert_array_equal(value, np.zeros((2, 3, 3)))

    @patch("torch.cuda.synchronize")
    def test_ad_exported_E_units_and_no_updates(self, synchronize: Mock) -> None:
        trainer = AlgebraTrainer(19)
        ad = center_ad(trainer, "upper", "stiff", self.config["parameter_sets"]["stiff"])
        self.assertEqual(ad["ad_D"], 2.0)
        self.assertEqual(ad["ad_E"], 20.0)
        self.assertEqual(ad["ad_H"], 2.0)
        self.assertEqual(float(trainer.torch_param["E"].detach()), 10.0)
        trainer.optimizer.step.assert_not_called()

    @patch("torch.cuda.synchronize")
    def test_nonfinite_ad_is_saved_and_warned(self, synchronize: Mock) -> None:
        trainer = AlgebraTrainer(19)

        def singular_loss(device: str) -> tuple[torch.Tensor, None]:
            d = trainer.torch_param["D"]
            e = trainer.torch_param["E"]
            h = trainer.torch_param["H"]
            return (d - d.detach()).sqrt() + e.square() + h.square(), None

        trainer._autodiff_material = singular_loss
        message = io.StringIO()
        with contextlib.redirect_stdout(message):
            ad = center_ad(trainer, "lower", "soft", self.config["parameter_sets"]["soft"])
        self.assertTrue(np.isinf(ad["ad_D"]))
        self.assertIn("WARNING", message.getvalue())
        self.assertTrue(np.isfinite(ad["ad_loss"]))
        self.assertTrue(np.isnan(relative_error(0.0, float("nan"))))
        trainer.optimizer.step.assert_not_called()

    def test_euclidean_distances_and_saved_trajectory_FD(self) -> None:
        rows = [row for row in plan(self.config) if row["garment"] == "upper"
                and row["parameter_set"] == "default" and row["direction"] == "D"
                and row["step_size"] == 1e-4]
        step = rows[0]["step_size"]
        direction = np.asarray([3.0, 4.0, 0.0])
        positions = {offset: np.ones((12, 3, 3)) + offset * step * direction for offset in range(-2, 3)}
        losses = {offset: 7.0 + 2.0 * offset * step for offset in range(-2, 3)}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            samples, frames, fd = save_group(output, rows, positions, losses,
                                             {"ad_D": 2.0, "ad_loss": 7.0}, np.arange(19, 31),
                                             np.asarray([[0, 1, 2]]), np.asarray([17, 8, 5]), 1)
            for sample in samples:
                self.assertAlmostEqual(sample["v2v_mean_m"], abs(sample["offset"]) * step * 5.0)
                with np.load(output / sample["trajectory"], allow_pickle=False) as saved:
                    self.assertNotIn("target", saved.files)
                    np.testing.assert_array_equal(saved["vertex_ids"], [17, 8, 5])
            self.assertEqual(len(frames), 60)
            with np.load(output / fd["trajectory_fd"], allow_pickle=False) as saved:
                for key in ("h", "2h", "five_point"):
                    np.testing.assert_allclose(saved[f"dpositions_dparameter_{key}"],
                                               np.broadcast_to(direction, (12, 3, 3)), atol=1e-10)

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.is_available", return_value=True)
    def test_complete_orchestration_with_algebra_only(self, available: Mock, synchronize: Mock) -> None:
        trainers = {key: AlgebraTrainer(window["frame_start"]) for key, window in self.config["garments"].items()}

        def lower_ad_loss(device: str) -> tuple[torch.Tensor, None]:
            trainer = trainers["lower"]
            loss, _ = AlgebraTrainer._autodiff_material(trainer, device)
            if abs(float(trainer.torch_param["E"].detach()) - 0.1) < 1e-6:
                d = trainer.torch_param["D"]
                loss = loss + (d - d.detach()).sqrt()
            return loss, None

        trainers["lower"]._autodiff_material = lower_ad_loss
        replay_module = SimpleNamespace(replay=lambda trainer: trainer.replay())
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp)
            with patch.dict("sys.modules", {"replay_material_window": replay_module}):
                for garment, trainer in trainers.items():
                    with patch("material_sensitivity.build_trainer", return_value=trainer):
                        run_garment(self.config, output, garment)
                        run_garment(self.config, output, garment, resume=True)
                    self.assertEqual(trainer.ad_evaluations, 3)
                    self.assertEqual(trainer.native_evaluations, 135)
                    trainer.optimizer.step.assert_not_called()
            checkpoint = output / "lower/soft/progress.json"
            cached = json.loads(checkpoint.read_text())
            cached["fd"][0]["ad_fd_h_relative_error"] = 0.0
            cached["fd"][0]["ad_fd_h_absolute_error"] = 0.0
            checkpoint.write_text(json.dumps(cached))
            summarize(self.config, output)
            repaired = json.loads(checkpoint.read_text())["fd"][0]
            self.assertTrue(np.isnan(repaired["ad_fd_h_relative_error"]))
            self.assertTrue(np.isinf(repaired["ad_fd_h_absolute_error"]))
            for table, count in (("samples", 270), ("fd", 54), ("ad", 6), ("trajectory_by_frame", 3240)):
                with (output / f"{table}.csv").open(newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            self.assertEqual(len(list(output.rglob("offset*.npz"))), 270)
            self.assertEqual(len(list(output.rglob("fd.npz"))), 54)
            audit(self.config, output)
            report = json.loads((output / "audit.json").read_text())
            self.assertTrue(report["verified"])
            self.assertEqual(report["trajectories"], 270)
            self.assertEqual(report["nonfinite_ad_centers"],
                             [{"garment": "lower", "parameter_set": "soft", "parameters": "D"}])

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.is_available", return_value=True)
    def test_resume_after_one_committed_stencil(self, available: Mock, synchronize: Mock) -> None:
        trainer = AlgebraTrainer(19)
        replay_module = SimpleNamespace(replay=lambda trainer: trainer.replay())

        def interrupted_save(result_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
            save_progress(result_dir, tables)
            if len(tables["fd"]) == 1:
                raise InterruptedError("Simulated allocation end")

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temp)
            with patch.dict("sys.modules", {"replay_material_window": replay_module}), \
                    patch("material_sensitivity.build_trainer", return_value=trainer):
                with patch("material_sensitivity.save_progress", side_effect=interrupted_save):
                    with self.assertRaises(InterruptedError):
                        run_case(self.config, output, "upper", "soft", False)
                run_case(self.config, output, "upper", "soft", True)
            self.assertEqual(trainer.ad_evaluations, 1)
            self.assertEqual(trainer.native_evaluations, 45)
            with (output / "upper/soft/samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 45)
            self.assertEqual(len({row["trajectory"] for row in rows}), 45)


if __name__ == "__main__":
    unittest.main()
