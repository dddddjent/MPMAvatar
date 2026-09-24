"""Train or evaluate MPMAvatar on a completed CAPE, ClothTransformer or D-Garment export."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from dataset_input import common_arguments, load_manifest

# Template command (activate mpmavatar; run from MPMAvatar):
# python run.py --data ../data/MPMAvatar/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage appearance --appearance-iterations 30000 --material-iterations 200 --material-frames all --grid-size 200 --substeps 400
# Run --stage material after appearance; optionally --stage body before --stage evaluate.
# Body: --stage body --checkpoint /path/to/material.npz --body-iterations 100 --beta-lr 0.01 --beta-epsilon 0.01 --body-batch-size 8
# Optimized-body evaluation: --stage evaluate --checkpoint /path/to/material.npz --body-checkpoint /path/to/best_body_shape.npz --render-appearance gt_lighting
# Evaluation-only optional flags: --checkpoint /path/to/material.npz --evaluate-from-start --skip-render --skip-video.
# Reuse appearance: --appearance-model /path/to/baseline/appearance (material/evaluate only).
# Fixed-beta evaluation: --render-appearance gt_lighting.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("appearance", "material", "body", "evaluate"), required=True)
    parser.add_argument("--appearance-iterations", type=int, default=30000)
    parser.add_argument("--appearance-model", type=Path,
                        help="Existing appearance directory to reuse for material/body/evaluate; default: <output>/appearance")
    parser.add_argument("--material-iterations", type=int, default=200)
    parser.add_argument("--material-frames", default="all", help="all (the manifest's complete training prefix), or a count from 2 to its length")
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=400)
    parser.add_argument("--checkpoint", type=Path,
                        help="Frozen material checkpoint for body/evaluate; required for body, otherwise defaults to latest local last_param")
    parser.add_argument("--body-checkpoint", type=Path,
                        help="Optimized body NPZ for evaluation; also supply its frozen material with --checkpoint")
    parser.add_argument("--body-iterations", type=int, default=100, help="Total body fitting steps, including resumed steps")
    parser.add_argument("--beta-lr", type=float, default=0.01)
    parser.add_argument("--beta-epsilon", type=float, default=0.01, help="Central finite-difference offset in beta units")
    parser.add_argument("--body-batch-size", type=int, default=8, help="Frames per SMPL-X regeneration batch")
    parser.add_argument("--evaluate-from-start", action="store_true",
                        help="Render a rollout from frame 0; error always uses a separate rollout from the first evaluation frame")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--render-appearance", choices=("both", "gt_lighting"), default="both",
                        help="Evaluation videos: both branches, or source materials/lighting only")
    parser.add_argument("--resume-parameters", action="store_true",
                        help="One-time material resume from old NPZ files, resetting Adam and the LR schedule")
    args = parser.parse_args()
    assert args.stage in ("body", "evaluate") or args.checkpoint is None, "--checkpoint is only supported for --stage evaluate or body"
    assert args.stage == "evaluate" or args.body_checkpoint is None, "--body-checkpoint is only supported for --stage evaluate"
    if args.stage == "body" or args.body_checkpoint:
        assert args.checkpoint, "Choose the frozen material explicitly with --checkpoint PATH"
    assert args.stage == "evaluate" or not args.evaluate_from_start, "--evaluate-from-start is only supported for --stage evaluate"
    assert args.stage == "material" or not args.resume_parameters, "--resume-parameters is only supported for --stage material"
    assert args.stage != "appearance" or args.appearance_model is None, "--appearance-model is only supported for material/evaluate or body"
    assert args.stage == "evaluate" or args.render_appearance == "both", "--render-appearance is only supported for --stage evaluate"
    root, output = args.data.resolve(), args.output.resolve()
    manifest = load_manifest(root)
    if args.stage == "body" or args.body_checkpoint:
        assert "body_shape_experiment" in manifest, "Choose a tweaked-beta export with --data"
        assert manifest.get("body_model", "smplx") == "smplx", "Body optimization requires a parametric SMPL-X collider"
        assert (root / manifest["body_shape_experiment"]["body_motion"]).is_file()
        assert args.body_iterations > 0 and args.beta_lr > 0 and args.beta_epsilon > 0 and args.body_batch_size > 0
    if args.body_checkpoint:
        assert args.body_checkpoint.is_file(), f"Body checkpoint does not exist: {args.body_checkpoint}"
    train_count = len(manifest["train_frame_ids"])
    material_count = train_count if args.material_frames == "all" else int(args.material_frames)
    assert 2 <= material_count <= train_count, "Choose --material-frames within the exported training prefix"
    assert args.appearance_iterations > 0 and args.material_iterations > 0
    assert args.grid_size > 0 and args.substeps > 0
    assert args.stage == "evaluate" or not (args.skip_render or args.skip_video)
    model = args.appearance_model.resolve() if args.appearance_model else output / "appearance"
    if args.stage == "evaluate" and not args.skip_render and "body_shape_experiment" in manifest:
        assert args.render_appearance == "gt_lighting", (
            "Fixed-beta SMPL-X bodies require --render-appearance gt_lighting: "
            "the existing learned body appearance is bound to the original SMPL-H topology")
    common = common_arguments(root, manifest, model)
    if args.stage == "appearance":
        final = model / "point_cloud" / f"timestep_{args.appearance_iterations:06d}" / "point_cloud.ply"
        if final.is_file():
            print(f"Appearance already completed {args.appearance_iterations} iterations: {final}", flush=True)
            return
        resume = model / "training_state.pt"
        assert not model.exists() or resume.is_file(), f"No appearance training checkpoint in {model}; cannot resume this old partial run"
        command = [sys.executable, "train_appearance.py", *common,
                   "--train_frame_start_num", "0", str(train_count),
                   "--test_frame_start_num", "0", str(train_count),
                   "--iterations", str(args.appearance_iterations),
                   "--save_iterations", str(args.appearance_iterations)]
        if resume.is_file():
            command += ["--start_checkpoint", str(resume)]
    else:
        checkpoint = model / "point_cloud" / f"timestep_{args.appearance_iterations:06d}" / "point_cloud.ply"
        assert checkpoint.is_file(), f"Run appearance first or supply --appearance-model: {checkpoint}"
        stage_name = {"evaluate": "evaluation", "body": "body", "material": "material"}[args.stage]
        if args.stage == "evaluate":
            assert not (output / stage_name).exists(), f"Stage output already exists: {output / stage_name}"
        test_start, test_count = 0, 2
        if args.stage == "evaluate":
            evaluation_frames = manifest["frame_ids"] if args.evaluate_from_start else manifest["evaluation_frame_ids"]
            test_start, test_count = evaluation_frames[0], len(evaluation_frames)
        command = [sys.executable, "train_material_params.py", *common,
                   "--train_frame_start_num", "0", str(material_count),
                   "--test_frame_start_num", str(test_start), str(test_count),
                   "--iterations", str(args.material_iterations),
                   "--output_dir", str(output), "--save_name", stage_name,
                   "--smplx_num_betas", "10",
                   "--prescribed_surface_path", str(root / manifest["prescribed_surface"]),
                   "--grid_size", str(args.grid_size), "--substep", str(args.substeps)]
        if manifest.get("body_model", "smplx") == "raw":
            command += ["--raw_collider_path", str(root / manifest["collider_sequence"])]
        if args.stage == "material" and (output / stage_name).exists():
            resume = output / stage_name / "seed0/training_state.pt"
            assert resume.is_file() or args.resume_parameters, (
                "No optimizer checkpoint; use --resume-parameters once to resume this old material run "
                "with a fresh optimizer and LR schedule")
            command.append("--resume")
        if args.resume_parameters:
            assert (output / stage_name).is_dir(), "--resume-parameters requires an existing material run"
            command.append("--resume_parameters")
        if args.stage in ("body", "evaluate"):
            if args.checkpoint:
                material = args.checkpoint.resolve()
            else:
                directory = output / "material/seed0"
                checkpoints = [path for path in directory.glob("last_param_*.npz")
                               if path.is_file() and path.stem.removeprefix("last_param_").isdigit()]
                assert checkpoints, f"Run material fitting first: no last_param checkpoints in {directory}; or pass --checkpoint PATH"
                material = max(checkpoints, key=lambda path: int(path.stem.removeprefix("last_param_")))
            assert material.is_file(), f"Material checkpoint does not exist: {material}"
            command += ["--init_params_path", str(material)]
        if args.stage == "body":
            command += ["--body_shape", "--body_iterations", str(args.body_iterations),
                        "--beta_lr", str(args.beta_lr), "--beta_epsilon", str(args.beta_epsilon),
                        "--body_batch_size", str(args.body_batch_size)]
        if args.stage == "evaluate":
            command.append("--run_eval")
            if args.body_checkpoint:
                command += ["--body_checkpoint", str(args.body_checkpoint.resolve()),
                            "--body_batch_size", str(args.body_batch_size)]
            if args.skip_render or args.render_appearance == "gt_lighting":
                command.append("--skip_render")
            if args.skip_video:
                command.append("--skip_video")
    gt_command: list[str] = []
    if args.stage == "evaluate" and not args.skip_render:
        if args.render_appearance == "gt_lighting":
            print("Rendering source materials/lighting only (--render-appearance gt_lighting).", flush=True)
        config = json.loads((root / "config.json").read_text())
        render_python = Path(config["render_python"])
        assert render_python.is_file(), render_python
        for relative in ("capture/manifest.json", "capture/cam_info.json", "preparation/sequence.npz"):
            assert (root / relative).is_file(), root / relative
        gt_command = [str(render_python), "-B", "-m", "MPMAvatar.render_gt_lighting",
                      "--data", str(root), "--evaluation", str(output / "evaluation/seed0")]
        if args.skip_video:
            gt_command.append("--skip-video")
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
    if gt_command:
        print(shlex.join(gt_command), flush=True)
        subprocess.run(gt_command, cwd=Path(__file__).resolve().parent.parent, check=True)


if __name__ == "__main__":
    main()
