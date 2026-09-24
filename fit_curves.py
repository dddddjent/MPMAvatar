"""Render material and one-beta fitting histories as PNG curves."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Template commands (activate mpmavatar, run from MPMAvatar):
# python fit_curves.py --kind material --history output/DGarments/C24_beta0_offset0.1/material/seed0/history.csv --output output/DGarments/C24_beta0_offset0.1/material/seed0/fit_curves.png
# python fit_curves.py --kind body --history output/DGarments/C24_beta_recovery_offset0.1_material0.1/body/seed0/beta_history.csv --output output/DGarments/C24_beta_recovery_offset0.1_material0.1/body/seed0/fit_curves.png --beta-index 0
# python fit_curves.py --kind material --history output/DGarments/C24_beta0_offset0.1/material/seed0/history.csv --output output/DGarments/C24_beta0_offset0.1/evaluation/seed0/material_fit_curves.png --selected-step 63


def read_history(path: Path, fields: tuple[str, ...]) -> dict[str, list[float]]:
    assert path.is_file(), path
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None and set(fields) <= set(reader.fieldnames), path
        values = {field: [] for field in fields}
        for row in reader:
            for field in fields:
                values[field].append(float(row[field]))
    assert values["step"], f"Empty fitting history: {path}"
    return values


def render_material(history: Path, output: Path, selected_step: int | None = None) -> None:
    values = read_history(history, ("step", "loss", "D", "E", "H"))
    assert all(density > 0 for density in values["D"]), f"Nonpositive D in {history}"
    values["E / D"] = [youngs / density for youngs, density in zip(values["E"], values["D"])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for axis, name, color in zip(axes.flat, ("loss", "D", "E", "H", "E / D"),
                                 ("#284b8c", "#159895", "#c05c28", "#8654a8", "#a85050")):
        axis.plot(values["step"], values[name], color=color, linewidth=1.8)
        if selected_step is not None:
            axis.axvline(selected_step, color="#555555", linestyle="--", linewidth=1,
                         label=f"selected step {selected_step}")
            axis.legend(loc="best", fontsize=8)
        axis.set_title("Training cloth MSE" if name == "loss" else name)
        axis.set_xlabel("Material update step")
        axis.set_ylabel("MSE (m²)" if name == "loss" else
                        f"{name} (export units)" if name in ("E", "E / D") else name)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("MPMAvatar material fit · loss before update, D/E/H and E/D after update")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    fig.savefig(temporary, format="png", dpi=150)
    plt.close(fig)
    temporary.replace(output)


def render_body(history: Path, output: Path, beta_index: int,
                selected_step: int | None = None) -> None:
    values = read_history(history, ("step", "loss", "beta"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, name, color in zip(axes, ("loss", "beta"), ("#284b8c", "#c05c28")):
        axis.plot(values["step"], values[name], marker="o", markersize=2.5,
                  color=color, linewidth=1.8)
        if selected_step is not None:
            axis.axvline(selected_step, color="#555555", linestyle="--", linewidth=1,
                         label=f"selected step {selected_step}")
            axis.legend(loc="best", fontsize=8)
        axis.set_title("Training cloth MSE" if name == "loss" else f"Fitted beta[{beta_index}]")
        axis.set_xlabel("Body update step")
        axis.set_ylabel("MSE (m²)" if name == "loss" else "SMPL-X beta value")
        axis.grid(alpha=0.25)
    fig.suptitle(f"MPMAvatar body fit · beta[{beta_index}] · loss at listed beta")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    fig.savefig(temporary, format="png", dpi=150)
    plt.close(fig)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("material", "body"), required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta-index", type=int)
    parser.add_argument("--selected-step", type=int)
    args = parser.parse_args()
    if args.kind == "material":
        render_material(args.history, args.output, args.selected_step)
    else:
        assert args.beta_index is not None, "Body curves require --beta-index"
        render_body(args.history, args.output, args.beta_index, args.selected_step)
    print(args.output)


if __name__ == "__main__":
    main()
