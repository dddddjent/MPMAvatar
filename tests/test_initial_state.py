"""Small contract and real-solver checks; no external training data required."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from dataset_input import load_manifest
from future_evaluation import load_driving_surface, continuous_predictions, simulate_future
import run
from warp_mpm.initial_state import estimate_velocity, particle_velocities
from warp_mpm.mpm_data_structure import MPMModelStruct, MPMStateStruct
from warp_mpm.mpm_solver import MPMWARP

# Template command (activate mpmavatar; run from MPMAvatar):
# MPM_TEST_DEVICE=cuda:0 python -m unittest discover -s tests -p test_initial_state.py -v


class InputTests(unittest.TestCase):
    def test_velocity_scaling_and_particle_order(self) -> None:
        x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        speed = torch.tensor([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
        frames = torch.stack((x, x + speed / 25))
        v = estimate_velocity(frames, [0, 1])
        torch.testing.assert_close(v, speed)
        packed = particle_velocities(v, torch.tensor([[0, 1, 2]]), torch.tensor(2.))
        torch.testing.assert_close(packed[0], torch.tensor([4., 6., 8.]))
        torch.testing.assert_close(packed[1:], speed * 2)
        frames.zero_()
        torch.testing.assert_close(v, speed)
        with self.assertRaises(AssertionError):
            estimate_velocity(frames, [0, 2])

    def test_manifest_rejects_future_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(status="complete", component="cape_mpmavatar_export", fps=25,
                            actor=127, sequence=1, sequence_key="a127_s1", gender="male",
                            frame_ids=[0, 1, 2, 3], train_frame_ids=[0, 1],
                            evaluation_frame_ids=[2, 3], camera_ids=["Cam001", "Cam002"],
                            split_path="a127_s1/split_idx.npz",
                            prescribed_surface="a127_s1/prescribed_surface.npz",
                            tracking_directory="tracking/a127_s1_0_2")
            (root / "manifest.json").write_text(json.dumps(manifest))
            for name in (manifest["split_path"], manifest["prescribed_surface"],
                         "body_models/TR00_E096.pt", "body_models/smplx/SMPLX_MALE.npz"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "a127_s1/cam_info.json").write_text(json.dumps({"Cam001": {}, "Cam002": {}}))
            tracking = root / manifest["tracking_directory"]
            tracking.mkdir(parents=True)
            for frame in (0, 1):
                (tracking / f"params_{frame}.npz").touch()
            self.assertEqual(load_manifest(root), manifest)
            output = root / "result"
            appearance = output / "appearance/point_cloud/timestep_030000/point_cloud.ply"
            material = output / "material/seed0/last_param_00199.npz"
            for path in (appearance, material):
                path.parent.mkdir(parents=True)
                path.touch()
            argv = ["run.py", "--data", str(root), "--output", str(output),
                    "--stage", "evaluate", "--material-frames", "2", "--skip-render"]
            with patch("sys.argv", argv), patch("run.subprocess.run") as launch:
                run.main()
            command = launch.call_args.args[0]
            start = command.index("--test_frame_start_num")
            self.assertEqual(command[start + 1:start + 3], ["2", "2"])
            self.assertEqual(command[command.index("--init_params_path") + 1], str(material))
            self.assertIn("--run_eval", command)
            self.assertIn("--skip_render", command)
            (tracking / "params_2.npz").touch()
            with self.assertRaises(AssertionError):
                load_manifest(root)

    def test_driving_inputs_exclude_free_cloth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.npz"
            vertices = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
            np.savez(path, vertices=vertices)
            ids = list(range(5))
            joint_ids, human_ids = torch.tensor([0]), torch.tensor([0, 3])
            original = load_driving_surface(str(path), ids, joint_ids, human_ids)
            vertices[2:, 1:3] = np.nan
            np.savez(path, vertices=vertices)
            changed = load_driving_surface(str(path), ids, joint_ids, human_ids)
            for before, after in zip(original, changed):
                torch.testing.assert_close(before, after)


class SolverTests(unittest.TestCase):
    @torch.no_grad()
    def test_reset_motion_and_rest_height_force(self) -> None:
        device = os.environ.get("MPM_TEST_DEVICE", "cpu")
        wp.config.kernel_cache_dir = "/tmp/clothes_mpm_warp_cache"
        wp.init()
        wp.get_device(device)
        vertices = torch.tensor([[.9, .9, 1.], [1.1, .9, 1.], [.9, 1.1, 1.]], device=device)
        faces = torch.tensor([[0, 1, 2]], device=device)
        positions = torch.cat((vertices[faces].mean(1), vertices))
        direction = torch.diag(torch.tensor([.2, .2, 1.], device=device)).unsqueeze(0)
        rest = torch.tensor([[5., 0., 5.]], device=device)
        velocity = particle_velocities(torch.tensor([[.1, 0., 0.]] * 3, device=device),
                                       faces, torch.tensor(1., device=device))
        state = MPMStateStruct()
        state.init(4, 1, 3, device=device)
        state.from_torch(positions, torch.full((4,), 5e-8, device=device),
                         torch.linalg.inv(direction), rest, faces,
                         np.zeros(4, dtype=np.int32), np.array([0, 1, 1, 1], dtype=np.int32),
                         np.array([1, 0, 0, 0], dtype=np.int32),
                         tensor_velocity=velocity, n_grid=20, grid_lim=2.,
                         device=device, requires_grad=False)
        model = MPMModelStruct()
        model.init(4, device=device)
        model.init_other_params(n_grid=20, grid_lim=2., device=device)
        solver = MPMWARP(4, 1, 3, n_grid=20, grid_lim=2.,
                         mesh_vertices=(vertices + .5).cpu().numpy(),
                         mesh_faces=faces.cpu().numpy(), device=device)
        solver.set_parameters_dict(model, state, {"material": "cloth", "density": 1.,
                                                 "g": [0., 0., 0.]}, device=device)
        solver.set_E_nu(model, E=100., nu=.3, gamma=500., kappa=500., device=device)
        solver.prepare_mu_lam(model, state, device=device)

        def rollout(v: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
            modified_rest = rest.clone()
            modified_rest[:, 2] /= h
            state.reset_state(3, positions.clone(), direction.clone(), tensor_velocity=v.clone(),
                              tensor_R_inv=modified_rest, device=device, requires_grad=False)
            solver.time = 0.
            solver.time_profile.clear()
            for _ in range(4):
                solver.p2g2p(model, state, 1e-4, device=device)
            return wp.to_torch(state.particle_x).clone(), wp.to_torch(state.vertex_force).clone()

        moving, force = rollout(velocity, 1.)
        stationary, _ = rollout(torch.zeros_like(velocity), 1.)
        repeated, _ = rollout(velocity, 1.)
        torch.testing.assert_close(moving, repeated, atol=1e-7, rtol=0)
        torch.testing.assert_close(stationary, positions, atol=1e-6, rtol=0)
        self.assertGreater(float((moving - stationary)[:, 0].mean()), 1e-5)
        self.assertLess(float(force.abs().max()), 1e-9)
        _, tension_force = rollout(torch.zeros_like(velocity), .9)
        assert torch.isfinite(tension_force).all()
        self.assertGreater(float(tension_force.abs().max()), 1e-8)
        torch.testing.assert_close(tension_force.sum(0), torch.zeros(3, device=device), atol=1e-9, rtol=0)

        # Exercise the production continuous-prediction function across an unscored gap.
        from train_material_params import Trainer
        trainer = Trainer.__new__(Trainer)
        trainer.mpm_state, trainer.mpm_model, trainer.mpm_solver = state, model, solver
        trainer.particle_init_position = positions
        trainer.particle_init_dir = direction
        trainer.particle_init_velo = velocity
        trainer.vertices_init_position = vertices
        trainer.first_frame_verts = vertices
        trainer.new_cloth_faces = faces
        trainer.torch_param = {"D": torch.tensor(1.), "E": torch.tensor(1.), "H": torch.tensor(1.)}
        trainer.poisson_ratio = torch.full((4,), .3, device=device)
        trainer.gamma = trainer.kappa = torch.full((4,), 500., device=device)
        trainer.n_vertices, trainer.n_elements, trainer.n_traditional = 3, 1, 0
        trainer.num_joint_f = 0
        trainer.scale = torch.tensor(1., device=device)
        trainer.wld2sim = trainer.sim2wld = lambda x: x
        trainer.args = SimpleNamespace(substep=400)
        trainer.scene = SimpleNamespace(train_frame_index=[0, 1], test_frame_index=[3, 4])
        trainer.motion_frame_ids = list(range(5))
        trainer.body_motion = (vertices + .5).repeat(5, 1, 1)
        trainer.body_motion_velocity = torch.zeros_like(trainer.body_motion[:-1])
        trainer.prescribed_joint_positions = trainer.prescribed_human_positions = torch.empty(5, 0, 3, device=device)
        trainer.reordered_cloth_v_idx = torch.arange(3, device=device)
        trainer.reordered_human_v_idx = torch.empty(0, dtype=torch.long, device=device)
        predicted = torch.stack(continuous_predictions(trainer))
        displacement = (predicted - vertices)[..., 0].mean(1)
        # 1600 float32 position updates accumulate roundoff around coordinates of 1 m.
        torch.testing.assert_close(displacement, torch.tensor([.012, .016], device=device), atol=5e-5, rtol=0)
        self.assertAlmostEqual(solver.time, 4 / 25)
        torch.testing.assert_close(torch.stack(continuous_predictions(trainer)), predicted, atol=1e-7, rtol=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(status="complete", component="cape_mpmavatar_export", fps=25,
                            actor=127, sequence=1, sequence_key="a127_s1", gender="male",
                            frame_ids=list(range(5)), train_frame_ids=[0, 1, 2],
                            evaluation_frame_ids=[3, 4], camera_ids=["Cam001"],
                            split_path="a127_s1/split_idx.npz",
                            prescribed_surface="a127_s1/prescribed_surface.npz",
                            tracking_directory="tracking/a127_s1_0_3",
                            simulation_domain={"scale_sim_per_world_m": 1.})
            (root / "manifest.json").write_text(json.dumps(manifest))
            for name in (manifest["split_path"], "body_models/TR00_E096.pt",
                         "body_models/smplx/SMPLX_MALE.npz"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "a127_s1/cam_info.json").write_text(json.dumps({"Cam001": {}}))
            tracking = root / manifest["tracking_directory"]
            tracking.mkdir(parents=True)
            for frame in (0, 1, 2):
                (tracking / f"params_{frame}.npz").touch()
            uv_path = root / "uv.obj"
            uv_path.write_text("vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
            surface = root / manifest["prescribed_surface"]
            target = vertices.repeat(5, 1, 1).cpu().numpy()
            target[3:] = predicted.cpu().numpy()
            np.savez(surface, vertices=target)
            trainer.scene.dataset_dir = str(root)
            trainer.scene.train_frame_num = 2
            trainer.scene.uv_path = str(uv_path)
            trainer.args.prescribed_surface_path = str(surface)
            trainer.args.init_params_path = str(root / "material.npz")
            trainer.num_joint_v = 0
            for suffix, shift in (("first", 0.), ("changed_target", .1)):
                target[3:, :, 0] += shift
                np.savez(surface, vertices=target)
                output = root / suffix
                output.mkdir()
                trainer.output_path = str(output)
                result = torch.stack(simulate_future(trainer))
                torch.testing.assert_close(result, predicted, atol=1e-7, rtol=0)
                report = json.loads((output / "geometry_metrics.json").read_text())
                self.assertEqual(report["evaluation_frame_ids"], [3, 4])
                self.assertEqual(report["evaluation_scope"], "held_out")
                self.assertAlmostEqual(report["mean_v2v_m"], shift, places=6)
                with np.load(output / "predictions.npz") as saved:
                    np.testing.assert_array_equal(saved["frame_ids"], [3, 4])
                self.assertEqual(len(list((output / "uvmesh").glob("*.obj"))), 2)

            trainer.scene.test_frame_index = list(range(5))
            output = root / "full_sequence"
            output.mkdir()
            trainer.output_path = str(output)
            result = torch.stack(simulate_future(trainer))
            torch.testing.assert_close(result[0], vertices, atol=0, rtol=0)
            torch.testing.assert_close(result[3:], predicted, atol=1e-7, rtol=0)
            displacement = (result - vertices)[..., 0].mean(1)
            torch.testing.assert_close(displacement, torch.arange(5, device=device) * .004, atol=5e-5, rtol=0)
            self.assertAlmostEqual(solver.time, 4 / 25)
            report = json.loads((output / "geometry_metrics.json").read_text())
            self.assertEqual(report["evaluation_scope"], "full_sequence")
            self.assertEqual(report["evaluation_frame_ids"], list(range(5)))
            self.assertEqual(report["fitting_frame_ids"], [0, 1])
            self.assertAlmostEqual(report["per_frame_v2v_m"][0], 0.)
            with np.load(output / "predictions.npz") as saved:
                np.testing.assert_array_equal(saved["frame_ids"], np.arange(5))
                np.testing.assert_allclose(saved["vertices"], result.cpu().numpy())
            self.assertEqual(len(list((output / "uvmesh").glob("*.obj"))), 5)


if __name__ == "__main__":
    unittest.main()
