"""Bounded GPU integration check on a real raw-body export; no full training."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import warp as wp

from collider_motion import load_raw_collider
from dataset_input import common_arguments, load_manifest
from future_evaluation import continuous_predictions
from train_material_params import Trainer, parse_args

# Template command (activate mpmavatar; run from MPMAvatar):
# python validate_raw_collider.py --data ../data/MPMAvatar/ClothTransformer/sim_00000_raw --output ../data/outputs/mpmavatar_raw_validation --grid-size 40 --substeps 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--substeps", type=int, default=400)
    args = parser.parse_args()
    root, output = args.data.resolve(), args.output.resolve()
    assert not output.exists(), output
    assert args.grid_size > 0 and args.substeps > 0
    manifest = load_manifest(root)
    assert manifest["body_model"] == "raw"
    assert not (root / "body_models").exists()
    vertices, faces, velocities = load_raw_collider(root / manifest["collider_sequence"], manifest["frame_ids"])
    output.mkdir(parents=True)
    wp.config.kernel_cache_dir = "/tmp/clothes_mpm_warp_cache"
    with tempfile.TemporaryDirectory(prefix="mpm_raw_validation_") as directory:
        work = Path(directory)
        common = common_arguments(root, manifest, work / "appearance")
        command = [sys.executable, "train_appearance.py", *common,
                   "--train_frame_start_num", "0", "2", "--test_frame_start_num", "0", "2",
                   "--test_camera_index", "0", "--iterations", "2", "--test_iterations", "2",
                   "--save_iterations", "2"]
        with (output / "appearance.log").open("w") as log:
            subprocess.run(command, cwd=Path(__file__).resolve().parent, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        sys.argv = ["train_material_params.py", *common,
                    "--train_frame_start_num", "0", "2", "--test_frame_start_num", "0", "2",
                    "--test_camera_index", "0", "--iterations", "1",
                    "--output_dir", str(work), "--save_name", "material",
                    "--prescribed_surface_path", str(root / manifest["prescribed_surface"]),
                    "--raw_collider_path", str(root / manifest["collider_sequence"]),
                    "--grid_size", str(args.grid_size), "--substep", str(args.substeps)]
        model, opt, pipe, *_ = parse_args()
        trainer = Trainer(model, opt, pipe, run_eval=False)
        assert not hasattr(trainer, "lbs_deformer")
        np.testing.assert_array_equal(trainer.collider_faces.numpy(), faces)
        np.testing.assert_array_equal(trainer.body_motion.cpu().numpy(), vertices[:2])
        np.testing.assert_array_equal(trainer.body_motion_velocity.cpu().numpy(), velocities[:1])
        trainer.train()
        checkpoint = work / "material/seed0/last_param_00000.npz"
        assert checkpoint.is_file(), checkpoint
        with np.load(checkpoint, allow_pickle=False) as saved:
            fitted = {key: float(saved[key]) for key in ("D", "E", "H", "loss")}
        assert all(np.isfinite(value) for value in fitted.values()), fitted
        del trainer
        torch.cuda.empty_cache()
        model.test_frame_start_num = [2, 2]
        model.init_params_path = str(checkpoint)
        opt.save_name = "prediction"
        trainer = Trainer(model, opt, pipe, run_eval=True)
        predictions = torch.stack(continuous_predictions(trainer))
        assert torch.isfinite(predictions).all()
        torch.testing.assert_close(trainer.prescribed_human_positions,
                                   trainer.body_motion, atol=1e-6, rtol=0)
        torch.testing.assert_close(predictions[:, trainer.reordered_human_v_idx.long()],
                                   trainer.body_motion[2:4], atol=1e-6, rtol=0)
        last_collider = trainer.wld2sim(trainer.body_motion[2]) + (
            trainer.body_motion_velocity[2] * trainer.scale * (args.substeps - 1) / (25. * args.substeps))
        torch.testing.assert_close(wp.to_torch(trainer.mpm_solver.mesh.points), last_collider, atol=1e-6, rtol=0)
        report = {
            "status": "passed", "dataset": str(root), "device": torch.cuda.get_device_name(),
            "body_vertices": vertices.shape[1], "body_faces": len(faces),
            "archive_frames_checked": len(vertices), "parametric_model_created": False,
            "appearance_iterations": 2, "material_iterations": 1, "material_frames": [0, 1],
            "prediction_frames": [2, 3], "grid_size": args.grid_size, "substeps": args.substeps,
            "fitted_smoke_parameters": fitted,
            "scope": "Real appearance optimization, material finite differences and continuous prediction; early frames only, not the full held-out suffix or convergence",
            "temporary_training_outputs": "removed",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
