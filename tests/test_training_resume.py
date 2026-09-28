"""Exercise the real stage checkpoint code without loading an experiment."""

from argparse import ArgumentParser
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from accelerate import Accelerator

from arguments import OptimizationParams
from material_progress import restore_training_state
from scene import MeshGaussianModel
from train_material_params import Trainer

# Template command (activate mpmavatar; run from MPMAvatar, requires CUDA):
# python -m unittest discover -s tests -p test_training_resume.py -v


class TrainingResumeTests(unittest.TestCase):
    def test_material_stop_cap_resumes_original_optimizer_schedule_and_progress(self) -> None:
        accelerator = Accelerator(cpu=True)

        def make_trainer(directory: Path, cap: int) -> Trainer:
            directory.mkdir(parents=True, exist_ok=True)
            trainer = Trainer.__new__(Trainer)
            trainer.output_path = str(directory)
            trainer.step = 0
            trainer.iterations = 3
            trainer.args = SimpleNamespace(stop_after=cap)
            trainer.accelerator = accelerator
            trainer.scene = SimpleNamespace(train_frame_index=[0, 1], dataset_dir="/dataset")
            trainer.initial_cloth_velocity = torch.ones(3, 3)
            trainer.torch_param = {key: torch.tensor(value) for key, value in
                                   zip(("D", "E", "H"), (0.6, 7.3, 0.9))}
            optimizer = torch.optim.Adam(list(trainer.torch_param.values()), lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3.)
            trainer.optimizer, trainer.scheduler = accelerator.prepare(optimizer, scheduler)
            trainer.best_params = trainer.last_params = {"loss": 1e9}
            trainer.resume_context = {"dataset_dir": "/dataset", "fitting_frame_ids": [0, 1], "iterations": 3}
            trainer.optimizer_reset_step = None
            return trainer

        def update(trainer: Trainer) -> None:
            evaluated = {f"evaluated_{key}": value.item() * (100. if key == "E" else 1.)
                         for key, value in trainer.torch_param.items()}
            loss = sum(float(value.square()) for value in trainer.torch_param.values())
            trainer.optimizer.zero_grad()
            for value in trainer.torch_param.values():
                value.grad = value.square()
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.last_params = {"step": trainer.step, "loss": loss, **evaluated,
                                   **{key: value.item() * (100. if key == "E" else 1.)
                                      for key, value in trainer.torch_param.items()}}
            if loss < trainer.best_params["loss"]:
                trainer.best_params = trainer.last_params.copy()

        def train(trainer: Trainer) -> None:
            with patch.object(trainer, "train_one_step", side_effect=lambda: update(trainer)), patch(
                "train_material_params.render_material"
            ):
                trainer.train()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = make_trainer(root / "complete", 0)
            train(complete)
            partial = make_trainer(root / "resumed", 1)
            train(partial)
            summary = json.loads((root / "resumed/summary.json").read_text())
            self.assertEqual((summary["completed_iterations"], summary["planned_iterations"], summary["stop_after_iterations"]), (1, 3, 1))
            resumed = make_trainer(root / "resumed", 3)
            state = restore_training_state(root / "resumed", resumed.torch_param, resumed.optimizer,
                                           resumed.scheduler, resumed.resume_context, False)
            resumed.step = state["next_step"]
            resumed.best_params, resumed.last_params = state["best"], state["last"]
            train(resumed)
            summary = json.loads((root / "resumed/summary.json").read_text())
            self.assertEqual((summary["completed_iterations"], summary["planned_iterations"], summary["stop_after_iterations"]), (3, 3, 3))
            self.assertEqual(resumed.step, 3)
            self.assertEqual(resumed.resume_context, complete.resume_context)
            self.assertEqual(resumed.scheduler.state_dict(), complete.scheduler.state_dict())
            self.assertEqual(resumed.best_params, complete.best_params)
            self.assertEqual(resumed.last_params, complete.last_params)
            self.assertEqual((root / "resumed/history.csv").read_text(), (root / "complete/history.csv").read_text())
            for key in complete.torch_param:
                torch.testing.assert_close(resumed.torch_param[key], complete.torch_param[key], rtol=0, atol=0)
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(resumed.optimizer.state[resumed.torch_param[key]][field],
                                               complete.optimizer.state[complete.torch_param[key]][field], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Gaussian optimizer groups")
    def test_appearance_optimizer_and_auxiliary_parameters(self) -> None:
        parser = ArgumentParser()
        optimization = OptimizationParams(parser)
        opt = optimization.extract(parser.parse_args([]))

        def make_model() -> MeshGaussianModel:
            model = MeshGaussianModel(3, device="cuda")
            for name, shape in (("_xyz", (3, 3)), ("_features_dc", (3, 1, 3)),
                                ("_features_rest", (3, 15, 3)), ("_scaling", (3, 3)),
                                ("_rotation", (3, 4)), ("_opacity", (3, 1)),
                                ("verts_offset", (2, 3, 3)), ("cam_m", (2, 3)), ("cam_c", (2, 3))):
                setattr(model, name, torch.nn.Parameter(torch.rand(shape, device="cuda")))
            model.binding = torch.arange(3, device="cuda")
            model.binding_counter = torch.ones(3, dtype=torch.int32, device="cuda")
            model.face_center = torch.zeros(3, 3, device="cuda")
            model.face_orien_mat = torch.eye(3, device="cuda").repeat(3, 1, 1)
            model.face_scaling = torch.ones(3, 1, device="cuda")
            model.max_radii2D = torch.ones(3, device="cuda")
            model.spatial_lr_scale = np.float64(1.2)
            model.shadow_net = torch.nn.Linear(3, 3).cuda()
            model.training_setup(opt)
            return model

        def update(model: MeshGaussianModel, step: int) -> None:
            model.update_learning_rate(step)
            model.optimizer.zero_grad()
            loss = sum(value.square().sum() for group in model.optimizer.param_groups
                       for value in group["params"])
            loss.backward()
            model.optimizer.step()

        original = make_model()
        update(original, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(original.capture_training(), path)
            resumed = make_model()
            resumed.restore_training(torch.load(path, weights_only=False), opt)
        update(original, 2)
        update(resumed, 2)
        for a, b in zip(original.optimizer.param_groups, resumed.optimizer.param_groups):
            self.assertEqual(a["name"], b["name"])
            self.assertEqual(a["lr"], b["lr"])
            for x, y in zip(a["params"], b["params"]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(original.optimizer.state[x][key],
                                               resumed.optimizer.state[y][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
