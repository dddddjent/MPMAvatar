"""Run native tracking, UV transfer, AO baking and skin-weight inpainting.

The working directory keeps upstream relative paths inside the prepared output.
The tracking algorithm and its default optimization settings are unchanged.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

# Template command (mpmavatar environment, workspace root, allocated GPU):
# python MPMAvatar/preprocess/track_4ddress.py --prepared data/MPMAvatar/4DDress/00190_Inner --render-python /work/nvme/bivb/junlinl6/conda/envs/synthetic_avatar/bin/python --stage all --wandb-entity ''


def run(command: list[str], cwd: Path) -> None:
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--render-python", type=Path, required=True,
                        help="Python executable with Blender bpy installed (synthetic_avatar)")
    parser.add_argument("--stage", choices=("tracking", "postprocess", "all"), default="all")
    parser.add_argument("--wandb-entity", default="", help="Empty uses ordinary console logging")
    args = parser.parse_args()
    prepared = args.prepared.resolve()
    config_path = prepared / "preparation.json"
    assert config_path.is_file(), config_path
    config = json.loads(config_path.read_text())
    train = config["train"]
    name = f"s{train['subject']}_t{train['take']}"
    save_name = f"{name}_{train['start']}_{train['count']}"
    source = prepared / "data/4D-DRESS" / f"{train['subject']:05d}_Inner" / "Inner" / f"Take{train['take']}"
    assets = prepared / "data" / name
    output = prepared / "output/tracking" / save_name
    cwd = prepared / "preprocess"
    repo = Path(__file__).resolve().parents[1]
    assert cwd.is_dir() and (assets / "mesh_processed.obj").is_file()
    assert args.render_python.is_file(), args.render_python
    for frame in range(train["start"], train["start"] + train["count"]):
        for camera in ("0004", "0028", "0052", "0076"):
            label = source / "Capture" / camera / "labels" / f"label-f{frame:05d}.png"
            assert label.is_file(), f"Generate observations first: {label}"
    if args.stage in ("tracking", "all"):
        assert not output.exists(), f"Tracking output already exists: {output}"
        command = [sys.executable, str(repo / "preprocess/train_mesh_lbs_4ddress.py"),
                   "--save_name", save_name, "--seq", name, "--start_idx", str(train["start"]),
                   "--num_frames", str(train["count"]), "--labels", *map(str, config["labels"]),
                   "--data_path", str(source)]
        if args.wandb_entity:
            command += ["--wandb", "--wandb_entity", args.wandb_entity, "--wandb_name", f"track_{save_name}"]
        run(command, cwd)
    if args.stage in ("postprocess", "all"):
        for frame in range(train["start"], train["start"] + train["count"]):
            assert (output / f"mesh_cloth_{frame}.obj").is_file()
            assert (output / f"params_{frame}.npz").is_file()
        run([sys.executable, str(repo / "blender/add_uv_4ddress.py"),
             "--uv_path", str(assets / "mesh_processed.obj"), "--output_path", str(output)], cwd)
        run([str(args.render_python.resolve()), str(repo / "blender/bake.py"), "--",
             "--output_path", str(output), "--ao_res", "256"], cwd)
        first = f"mesh-f{train['start']:05d}_smplx"
        run([sys.executable, str(repo / "preprocess/lbs_weights_inpainting_4ddress.py"),
             "--smplx_gender", train["gender"], "--src_mesh_path", str(source / "SMPLX" / f"{first}.ply"),
             "--src_param_path", str(source / "SMPLX" / f"{first}.pkl"),
             "--target_mesh_path", str(output / f"mesh_cloth_{train['start']}.obj"),
             "--output_path", str(assets)], cwd)
        assert (assets / "optimized_weights.npy").is_file()
        for frame in range(train["start"], train["start"] + train["count"]):
            assert (output / "aomap" / f"mesh_cloth_{frame}.png").is_file()
        (prepared / "tracking_report.json").write_text(json.dumps({
            "status": "complete", "tracking": str(output), "frames": train["count"],
            "template": str(assets), "appearance_training_ready": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
