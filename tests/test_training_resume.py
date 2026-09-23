"""Exercise the real stage checkpoint code without loading an experiment."""

from argparse import ArgumentParser
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
    def test_material_training_writes_progress_with_accelerate(self) -> None:
        accelerator = Accelerator(cpu=True)
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer.__new__(Trainer)
            trainer.output_path = directory
            trainer.step = 2
            trainer.iterations = 3
            trainer.accelerator = accelerator
            trainer.scene = SimpleNamespace(train_frame_index=[0, 1], dataset_dir="/dataset")
            trainer.initial_cloth_velocity = torch.ones(3, 3)
            trainer.torch_param = {key: torch.tensor(value) for key, value in
                                   zip(("D", "E", "H"), (0.6, 7.3, 0.9))}
            optimizer = torch.optim.Adam(list(trainer.torch_param.values()), lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50.)
            trainer.optimizer, trainer.scheduler = accelerator.prepare(optimizer, scheduler)
            for value in trainer.torch_param.values():
                value.grad = value.square()
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.best_params = trainer.last_params = {
                "step": 2, "loss": 0.01, "D": 0.6, "E": 730., "H": 0.9,
                "evaluated_D": 0.61, "evaluated_E": 740., "evaluated_H": 0.91,
            }
            trainer.resume_context = {"dataset_dir": "/dataset", "fitting_frame_ids": [0, 1]}
            trainer.optimizer_reset_step = None
            with patch.object(trainer, "train_one_step") as simulation:
                trainer.train()
            simulation.assert_called_once()
            state = restore_training_state(Path(directory), trainer.torch_param, trainer.optimizer,
                                           trainer.scheduler, trainer.resume_context, False)
            self.assertEqual(state["next_step"], 3)
            self.assertEqual(state["last"]["evaluated_E"], 740.)
            self.assertTrue((Path(directory) / "history.csv").is_file())
            self.assertTrue((Path(directory) / "summary.json").is_file())
            with np.load(Path(directory) / "last_param_00002.npz") as saved:
                self.assertEqual(saved["evaluated_E"].item(), 740.)

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
