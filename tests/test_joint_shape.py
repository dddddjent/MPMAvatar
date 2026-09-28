"""Check simultaneous derivatives, paired snapshots and exact joint continuation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import torch

from body_shape import load_body_shape
from joint_shape import fit_joint_shape

# Template command (activate mpmavatar; run from MPMAvatar):
# python -m unittest discover -s tests -p test_joint_shape.py -v


class JointShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "body"
        self.source.mkdir()
        self.material = self.root / "initial_material.npz"
        np.savez(self.material, D=.7, E=230., H=.9)
        self.betas = torch.zeros(1, 10)
        self.betas[0, 4] = 2.
        self.body = SimpleNamespace(beta_index=4, initial_beta=2., source_betas=self.betas,
                                    apply=self.apply_beta)

    @staticmethod
    def apply_beta(trainer: Any, beta: float) -> None:
        trainer.active_beta = beta

    @staticmethod
    def objective(trainer: Any) -> float:
        beta = trainer.active_beta - .35
        density = float(trainer.torch_param["D"]) - .8
        youngs = float(trainer.torch_param["E"]) - 2.
        height = float(trainer.torch_param["H"]) - .95
        return beta ** 2 + density ** 2 + youngs ** 2 + height ** 2 + .2 * beta * density

    def trainer(self, name: str) -> SimpleNamespace:
        parameters = {key: torch.tensor(-1.) for key in ("D", "E", "H")}
        optimizer = torch.optim.Adam([{"params": [value], "lr": .03} for value in parameters.values()])
        return SimpleNamespace(
            output_path=str(self.root / name), args=SimpleNamespace(init_params_path=str(self.material)),
            accelerator=SimpleNamespace(num_processes=1),
            scene=SimpleNamespace(dataset_dir=str(self.source), train_frame_index=[0, 1]),
            train_frame_collider=torch.zeros(2, 3, 3), torch_param=parameters,
            initial_cloth_velocity=torch.zeros(3, 3),
            resume_context={"iterations": 9, "fitting_frame_ids": [0, 1], "substep": 2},
            optimizer=optimizer, scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12),
            param_ranges={"D": [.1, 3.], "E": [.5, 20.], "H": [.8, 1.2]},
        )

    def fit(self, trainer: SimpleNamespace, iterations: int, resume: bool = False,
            stop_after: int = 0) -> dict[str, Any]:
        with patch("joint_shape.SingleBetaBody", return_value=self.body), patch(
            "joint_shape.geometry_loss", side_effect=self.objective
        ), patch("joint_shape.render_material"), patch("joint_shape.render_body"), redirect_stdout(io.StringIO()):
            fit_joint_shape(trainer, self.source, self.material, iterations=iterations,
                            learning_rate=.1, finite_difference=.01, resume=resume, stop_after=stop_after)
        return torch.load(Path(trainer.output_path) / "joint_shape_state.pt", weights_only=True)

    def test_derivatives_use_common_beta_material_and_checkpoint_loss_matches_pair(self) -> None:
        trainer = self.trainer("joint")
        initial_betas = self.betas.clone()
        state = self.fit(trainer, 4)
        first = state["history"][1]
        beta = 2. - .35
        density = float(torch.tensor(.7)) - .8
        youngs = float(torch.tensor(2.3)) - 2.
        height = float(torch.tensor(.9)) - .95
        self.assertAlmostEqual(first["gradient_beta"], 2 * beta + .2 * density, places=6)
        self.assertAlmostEqual(first["gradient_D"], 2 * density + .2 * beta + .05, places=6)
        self.assertAlmostEqual(first["gradient_E"], 2 * youngs + .05, places=6)
        self.assertAlmostEqual(first["gradient_H"], 2 * height + .005, places=6)
        self.assertLess(state["best"]["loss"], state["history"][0]["loss"])
        self.assertEqual(trainer.active_beta, state["best"]["beta"])
        torch.testing.assert_close(self.betas, initial_betas, rtol=0, atol=0)
        root = Path(trainer.output_path)
        for name in ("best", "last"):
            with np.load(root / f"{name}_joint_shape.npz", allow_pickle=False) as saved:
                values = {key: torch.tensor(float(saved[key]) / (100. if key == "E" else 1.))
                          for key in ("D", "E", "H")}
                actual_loss = self.objective(SimpleNamespace(active_beta=float(saved["beta_value"]),
                                                             torch_param=values))
                self.assertAlmostEqual(float(saved["loss"]), actual_loss, places=12)
                self.assertEqual(float(saved["beta_value"]), state[name]["beta"])
                self.assertEqual(str(saved["optimization_kind"]), "joint_beta_material")
        trainer.args.init_params_path = str(root / "best_joint_shape.npz")
        with patch("body_shape.SingleBetaBody", return_value=self.body), redirect_stdout(io.StringIO()):
            load_body_shape(trainer, root / "best_joint_shape.npz", self.source)
        self.assertEqual(trainer.active_beta, state["best"]["beta"])

    def test_stop_cap_resumes_both_optimizers_and_original_scheduler_exactly(self) -> None:
        complete = self.fit(self.trainer("complete"), 9)
        resumed = self.trainer("resumed")
        partial = self.fit(resumed, 9, stop_after=3)
        self.assertEqual(partial["next_step"], 3)
        self.assertEqual(partial["context"]["simulation"]["iterations"], 9)
        self.assertNotIn("stop_after_iterations", partial["context"])
        continuation = self.fit(resumed, 9, resume=True, stop_after=9)
        for field in ("last", "best", "history", "context", "material_scheduler"):
            self.assertEqual(continuation[field], complete[field])
        for optimizer in ("beta_optimizer", "material_optimizer"):
            self.assertEqual(continuation[optimizer]["param_groups"], complete[optimizer]["param_groups"])
            for index, values in complete[optimizer]["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(continuation[optimizer]["state"][index][key], value, rtol=0, atol=0)
        summary = json.loads((Path(resumed.output_path) / "joint_summary.json").read_text())
        self.assertEqual((summary["planned_iterations"], summary["stop_after_iterations"], summary["completed_iterations"]), (9, 9, 9))

    def test_resume_rejects_changed_material_or_simulation(self) -> None:
        trainer = self.trainer("guard")
        self.fit(trainer, 2)
        trainer.resume_context["substep"] = 3
        with self.assertRaisesRegex(AssertionError, "settings or input selection changed"):
            self.fit(trainer, 3, resume=True)
        trainer.resume_context["substep"] = 2
        np.savez(self.material, D=.8, E=230., H=.9)
        with self.assertRaisesRegex(AssertionError, "settings or input selection changed"):
            self.fit(trainer, 3, resume=True)


if __name__ == "__main__":
    unittest.main()
