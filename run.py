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
# Run --stage material after appearance; run --stage evaluate after material fitting.
# Evaluation-only optional flags: --checkpoint /path/to/material.npz --evaluate-from-start --skip-render --skip-video.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("appearance", "material", "evaluate"), required=True)
    parser.add_argument("--appearance-iterations", type=int, default=30000)
    parser.add_argument("--material-iterations", type=int, default=200)
    parser.add_argument("--material-frames", default="all", help="all (the manifest's complete training prefix), or a count from 2 to its length")
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--substeps", type=int, default=400)
    parser.add_argument("--checkpoint", type=Path,
                        help="Evaluation material checkpoint; default: highest last_param iteration in <output>/material/seed0")
    parser.add_argument("--evaluate-from-start", action="store_true",
                        help="Save, score and render every frame from frame 0, including the training prefix")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--resume-parameters", action="store_true",
                        help="One-time material resume from old NPZ files, resetting Adam and the LR schedule")
    args = parser.parse_args()
    assert args.stage == "evaluate" or args.checkpoint is None, "--checkpoint is only supported for --stage evaluate"
    assert args.stage == "evaluate" or not args.evaluate_from_start, "--evaluate-from-start is only supported for --stage evaluate"
    assert args.stage == "material" or not args.resume_parameters, "--resume-parameters is only supported for --stage material"
    root, output = args.data.resolve(), args.output.resolve()
    manifest = load_manifest(root)
    train_count = len(manifest["train_frame_ids"])
    material_count = train_count if args.material_frames == "all" else int(args.material_frames)
    assert 2 <= material_count <= train_count, "Choose --material-frames within the exported training prefix"
    assert args.appearance_iterations > 0 and args.material_iterations > 0
    assert args.grid_size > 0 and args.substeps > 0
    assert args.stage == "evaluate" or not (args.skip_render or args.skip_video)
    model = output / "appearance"
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
        assert checkpoint.is_file(), f"Run appearance first: {checkpoint}"
        stage_name = "evaluation" if args.stage == "evaluate" else "material"
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
        if args.stage == "evaluate":
            if args.checkpoint:
                material = args.checkpoint.resolve()
            else:
                directory = output / "material/seed0"
                checkpoints = [path for path in directory.glob("last_param_*.npz")
                               if path.is_file() and path.stem.removeprefix("last_param_").isdigit()]
                assert checkpoints, f"Run material fitting first: no last_param checkpoints in {directory}; or pass --checkpoint PATH"
                material = max(checkpoints, key=lambda path: int(path.stem.removeprefix("last_param_")))
            assert material.is_file(), f"Material checkpoint does not exist: {material}"
            command += ["--run_eval", "--init_params_path", str(material)]
            if args.skip_render:
                command.append("--skip_render")
            if args.skip_video:
                command.append("--skip_video")
    gt_command: list[str] = []
    if args.stage == "evaluate" and not args.skip_render:
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
