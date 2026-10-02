"""Small CPU checks of scoring, material coordinates, and resumable optimization."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from local_patch_data import LocalPatch, save_bundle
from local_patch_fit import FitSettings, aggregate_rollouts, finite_difference_gradient, main, optimize
from local_patch_solver import cloth_directions_and_volume, rest_direction_inverse, score_rollout

# Template command (activate mpmavatar; from MPMAvatar, no CUDA or simulation):
# OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_local_patch_fit.py -v


class LocalPatchFitTests(unittest.TestCase):
    def test_volume_quadrature_and_rest_translation(self) -> None:
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        directions, volumes = cloth_directions_and_volume(vertices, faces)
        np.testing.assert_allclose(directions[0], np.diag([2., 3., 1.]))
        self.assertAlmostEqual(float(volumes.sum()), 3e-5)
        np.testing.assert_allclose(volumes, np.full(4, 0.75e-5))
        inverse = rest_direction_inverse(vertices, faces, 0.5)
        np.testing.assert_allclose(inverse, [[0.5, 0., 2 / 3]])
        np.testing.assert_allclose(rest_direction_inverse(vertices + [8., -5., 3.], faces, 0.5), inverse)

    def test_scoring_excludes_initialized_frame_and_collar(self) -> None:
        reference = np.zeros((3, 2, 3))
        patch = SimpleNamespace(reference_vertices_m=reference, scored_vertex_ids=np.array([0]),
                                boundary_vertex_ids=np.array([1]))
        prediction = reference.copy()
        prediction[0, 0, 0] = 1.0
        prediction[1:, 0, 0] = [0.003, 0.004]
        prediction[:, 1, 0] = 0.02
        result = score_rollout(patch, prediction)
        self.assertAlmostEqual(result.mean_xyz_mse_m2, 12.5e-6 / 3)
        self.assertAlmostEqual(result.vertex_rmse_mm, np.sqrt(12.5))
        self.assertAlmostEqual(result.mean_vertex_error_mm, 3.5)
        self.assertAlmostEqual(result.boundary_max_error_mm, 20.0)

    def test_aggregate_weights_vertex_frame_observations(self) -> None:
        patches = [SimpleNamespace(name="a", frame_ids=np.array([1, 2]),
                                   scored_vertex_ids=np.array([0]), boundary_vertex_ids=np.array([1]),
                                   reference_vertices_m=np.zeros((2, 2, 3))),
                   SimpleNamespace(name="b", frame_ids=np.array([3, 4, 5]),
                                   scored_vertex_ids=np.array([0, 1]), boundary_vertex_ids=np.array([2]),
                                   reference_vertices_m=np.zeros((3, 3, 3)))]
        predictions = [p.reference_vertices_m.copy() for p in patches]
        predictions[0][1:, 0, 0] = 0.001
        predictions[1][1:, :2, 0] = 0.003
        rollouts = [score_rollout(p, x) for p, x in zip(patches, predictions)]
        report = aggregate_rollouts(patches, rollouts)
        self.assertEqual(report["scored_vertex_frame_count"], 5)
        self.assertAlmostEqual(report["mean_xyz_mse_m2"], 37e-6 / 15)
        self.assertAlmostEqual(report["mean_vertex_error_mm"], 2.6)
        self.assertAlmostEqual(report["vertex_rmse_mm"], np.sqrt(7.4))

    def test_finite_difference_respects_bounds(self) -> None:
        probes: list[np.ndarray] = []

        def objective(point: np.ndarray) -> float:
            self.assertTrue(np.all((point >= [0., 0.]) & (point <= [1., 1.])))
            probes.append(point.copy())
            return float(3 * point[0] + 5 * point[1])

        gradient = finite_difference_gradient(objective, np.array([0., 1.]), np.array([0.1, 0.1]),
                                              np.zeros(2), np.ones(2))
        np.testing.assert_allclose(gradient, [3., 5.])
        self.assertEqual(len(probes), 4)

    def test_resume_matches_uninterrupted_and_repairs_derived_reports(self) -> None:
        settings = FitSettings(iterations=4)

        def objective(point: np.ndarray) -> dict[str, float]:
            mse = float(((point[0] - np.log(120))**2 + (point[1] - 0.9)**2) / 1e6)
            return {"mean_xyz_mse_m2": mse, "vertex_rmse_mm": float(1000 * np.sqrt(3 * mse)),
                    "mean_vertex_error_mm": float(1000 * np.sqrt(3 * mse)), "boundary_max_error_mm": 0.0}

        def callback(record: dict[str, object]) -> None:
            self.assertAlmostEqual(float(record["loss"]), objective(
                np.array([np.log(float(record["E"])), float(record["H"])]))["mean_xyz_mse_m2"])

        def initialize() -> None:
            return None

        with tempfile.TemporaryDirectory() as directory:
            full_path, resumed_path = Path(directory) / "full", Path(directory) / "resumed"
            full = optimize(objective, full_path, settings, {"fixture": "analytic"}, callback, initialize)
            optimize(objective, resumed_path, settings, {"fixture": "analytic"}, callback, initialize, stop_after=2)
            resumed = optimize(objective, resumed_path, settings, {"fixture": "analytic"}, callback, initialize, resume=True)
            timing_fields = {"evaluation_seconds", "gradient_seconds", "iteration_seconds"}
            for state in (resumed, full):
                for row in state["history"]:
                    for key in timing_fields:
                        self.assertGreaterEqual(row.pop(key), 0)
            self.assertEqual(resumed, full)
            (resumed_path / "summary.json").write_text("{}\n")
            (resumed_path / "last_param.npz").unlink()
            with (resumed_path / "history.csv").open("a") as stream:
                stream.write("uncommitted-row\n")
            optimize(objective, resumed_path, settings, {"fixture": "analytic"}, callback, initialize, resume=True)
            self.assertEqual(json.loads((resumed_path / "summary.json").read_text())["completed_iterations"], 4)
            with np.load(resumed_path / "last_param.npz", allow_pickle=False) as data:
                self.assertEqual(int(data["step"]), 3)
                self.assertAlmostEqual(float(data["loss"]), full["last"]["loss"])
            self.assertEqual(len((resumed_path / "history.csv").read_text().splitlines()), 5)
            with self.assertRaisesRegex(AssertionError, "Resume settings"):
                optimize(objective, resumed_path, settings, {"fixture": "changed"}, callback, initialize, resume=True)

    def test_cli_snapshot_evaluation_and_convergence_without_physics(self) -> None:
        vertices = np.array([[x * 0.05, y * 0.05, 0.] for y in range(3) for x in range(3)])
        faces = np.array([[a, a + 1, a + 3] for a in (0, 1, 3, 4)]
                         + [[a + 1, a + 4, a + 3] for a in (0, 1, 3, 4)], dtype=np.int64)
        local_patch = LocalPatch("fixture", np.array([45, 46]), np.arange(9),
                                 np.stack((vertices, vertices + [0., 0.001, 0.])), vertices,
                                 faces, np.array([0, 1, 2, 3, 5, 6, 7, 8]), np.array([4]), 2.0)

        class MockSimulator:
            def __init__(self, data: LocalPatch, substeps: int, device: str,
                         padding_cells: int = 4, cell_size_factor: float = 1.0) -> None:
                self.data, self.substeps = data, substeps
                self.cell_size = data.cell_size_m * cell_size_factor

            def rollout(self, D: float, E: float, H: float) -> object:
                predicted = self.data.reference_vertices_m.copy()
                predicted[1:, self.data.scored_vertex_ids, 0] += self.cell_size + 0.001 / self.substeps
                return score_rollout(self.data, predicted)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fit, evaluation, convergence = [root / name for name in ("source", "fit", "evaluation", "convergence")]
            save_bundle(source, [local_patch], {"subject": 185, "garment": "lower"})
            base = ["local_patch_fit.py", "--patches", str(source)]
            fake_module = SimpleNamespace(PatchSimulator=MockSimulator)
            with patch.dict("sys.modules", {"local_patch_solver": fake_module}):
                arguments = base + ["--stage", "fit", "--output", str(fit), "--iterations", "1"]
                with patch("sys.argv", arguments):
                    main()
                self.assertTrue((fit / "patches/manifest.json").is_file())
                # Resume must use frozen fit inputs even if the original source changes.
                (source / "manifest.json").write_text("{}\n")
                with patch("sys.argv", arguments + ["--resume"]):
                    main()
                parameters = ["--checkpoint", str(fit / "best_param.npz")]
                with patch("sys.argv", ["local_patch_fit.py", "--patches", str(fit / "patches"),
                                        "--stage", "evaluate", "--output", str(evaluation), *parameters]):
                    main()
                report = json.loads((evaluation / "metrics.json").read_text())
                self.assertFalse(report["initialization_frame_scored"])
                self.assertEqual(report["scored_vertex_frame_count"], 1)
                self.assertEqual(report["grids"][0]["cell_size_m"], 2 * .5 / 200)
                self.assertEqual(report["grids"][0]["global_bbox_extent_m"], .5)
                with patch("sys.argv", ["local_patch_fit.py", "--patches", str(fit / "patches"),
                                        "--stage", "convergence", "--output", str(convergence), *parameters,
                                        "--cell-size-factors", "1", "0.5", "--substep-counts", "400", "800"]):
                    main()
                comparison = json.loads((convergence / "convergence.json").read_text())
                self.assertEqual(len(comparison["runs"]), 4)
                self.assertEqual(comparison["reference"], {"cell_size_factor": 0.5, "substeps": 800})
                self.assertEqual(comparison["runs"][-1]["trajectory_rmse_to_finest_mm"], 0.0)
                self.assertAlmostEqual(comparison["runs"][0]["trajectory_rmse_to_finest_mm"], 2.50125)
                coarse, fine = comparison["runs"][0]["grids"][0], comparison["runs"][-1]["grids"][0]
                self.assertEqual(coarse["cell_size_m"], 2 * fine["cell_size_m"])
                self.assertEqual(coarse["origin_m"], fine["origin_m"])


if __name__ == "__main__":
    unittest.main()
