"""Material training checkpoints and readable progress exports."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Export existing material results without training (activate mpmavatar):
# python material_progress.py --directory output/ClothTransformer/sim_00000_raw/material/seed0


def parameter_files(directory: Path) -> list[Path]:
    return sorted((path for path in directory.glob("last_param_*.npz")
                   if path.is_file() and path.stem.removeprefix("last_param_").isdigit()),
                  key=lambda path: int(path.stem.removeprefix("last_param_")))


def read_parameters(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].item() for key in
                ("step", "loss", "D", "E", "H", "evaluated_D", "evaluated_E", "evaluated_H")
                if key in data}


def export_progress(directory: Path, last_step: int | None = None) -> None:
    paths = parameter_files(directory)
    if last_step is not None:
        paths = [path for path in paths if int(path.stem.removeprefix("last_param_")) <= last_step]
    assert paths, f"No material checkpoints in {directory}"
    fields = ["step", "loss", "D", "E", "H", "evaluated_D", "evaluated_E", "evaluated_H"]
    temporary = directory / "history.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            writer.writerow(read_parameters(path))
    temporary.replace(directory / "history.csv")
    last = paths[-1]
    best = last.with_name(last.name.replace("last_param_", "best_param_"))
    assert best.is_file(), best
    summary = {
        "last": read_parameters(last), "best": read_parameters(best),
        "last_checkpoint": last.name, "best_checkpoint": best.name,
        "loss_note": "Loss is measured before the update; D/E/H are after it. "
                     "evaluated_D/E/H identify the loss inputs when recorded. "
                     "Best retains the original trainer's pre-update loss/post-update parameter convention.",
        "E_note": "E uses the exported/terminal units (100 times the internal optimizer variable).",
    }
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(directory / "summary.json")


def save_training_state(directory: Path, next_step: int, params: dict[str, torch.Tensor],
                        optimizer: Any, scheduler: Any, best: dict[str, Any],
                        last: dict[str, Any], context: dict[str, Any],
                        optimizer_reset_step: int | None) -> None:
    state = {
        "next_step": next_step,
        "params": {key: value.detach().cpu().clone() for key, value in params.items()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "best": best, "last": last, "context": context,
        "optimizer_reset_step": optimizer_reset_step,
    }
    temporary = directory / "training_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "training_state.pt")


def restore_training_state(directory: Path, params: dict[str, torch.Tensor],
                           optimizer: Any, scheduler: Any, context: dict[str, Any],
                           resume_parameters: bool) -> dict[str, Any]:
    checkpoint = directory / "training_state.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["context"] == context, "Resume requires the same material fitting settings"
        for key, value in params.items():
            value.copy_(state["params"][key])
        scheduler.load_state_dict(state["scheduler"])
        optimizer.load_state_dict(state["optimizer"])
        print(f"Resuming material at step {state['next_step']} with optimizer and scheduler state", flush=True)
        return state

    assert resume_parameters, "No optimizer checkpoint; explicitly pass --resume-parameters for this old run"
    paths = parameter_files(directory)
    assert paths, f"No material checkpoints in {directory}"
    latest = paths[-1]
    with np.load(latest, allow_pickle=False) as saved:
        assert str(saved["dataset_dir"].item()) == context["dataset_dir"], "Resume dataset differs"
        assert saved["fitting_frame_ids"].tolist() == context["fitting_frame_ids"], "Resume fitting frames differ"
    last = read_parameters(latest)
    assert last["step"] == int(latest.stem.removeprefix("last_param_")), "Stale last_param step"
    best = read_parameters(latest.with_name(latest.name.replace("last_param_", "best_param_")))
    for key, value in params.items():
        value.fill_(last[key] / (100.0 if key == "E" else 1.0))
    next_step = last["step"] + 1
    print(f"Resuming material at step {next_step} from {latest}; resetting Adam and the LR schedule", flush=True)
    return {"next_step": next_step, "best": best, "last": last,
            "optimizer_reset_step": next_step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    export_progress(args.directory.resolve())
    print(f"Wrote {args.directory / 'history.csv'} and {args.directory / 'summary.json'}")


if __name__ == "__main__":
    main()
