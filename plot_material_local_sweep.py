"""Plot dense D/E/H sweeps: GT loss and center loss, one 3x3 page each per garment.

Activate mpmavatar; run from MPMAvatar on an owned CPU allocation after summarize:
# python plot_material_local_sweep.py --output output/material_local_sweep_185
# python plot_material_local_sweep.py --output output/material_local_sweep_dgarments
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BLUE = "#0072B2"
LOSSES = {"loss_target": "GT loss (geometry MSE vs tracked target)",
          "loss_center": "Center loss (geometry MSE vs center rollout)"}


def plot(output: Path) -> None:
    config = json.loads((output / "config.json").read_text())
    with (output / "samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = 2 * config["radius"] + 1
    failures = []
    if (output / "failures.csv").is_file():
        with (output / "failures.csv").open(newline="") as stream:
            failures = list(csv.DictReader(stream))
    failed = len(failures)
    assert len(rows) == len(config["garments"]) * len(config["directions"]) * len(config["step_sizes"]) * points - failed
    pdf_path = output / "loss_curves.pdf"
    with PdfPages(pdf_path) as pdf:
        for garment, window in config["garments"].items():
            for metric, title in LOSSES.items():
                figure, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
                for row_index, direction in enumerate(config["directions"]):
                    for column, step in enumerate(config["step_sizes"]):
                        line = sorted((row for row in rows if row["garment"] == garment and row["direction"] == direction
                                       and float(row["step"]) == step), key=lambda row: int(row["k"]))
                        missing = [row for row in failures if row["garment"] == garment and row["direction"] == direction
                                   and float(row["step"]) == step]
                        assert len(line) + len(missing) == points
                        x = [float(row["offset"]) for row in line]
                        y = [float(row[metric]) for row in line]
                        axis = axes[row_index, column]
                        axis.plot(x, y, color=BLUE, linewidth=1.5, marker="o", markersize=3)
                        axis.axvline(0.0, color="0.6", linewidth=0.8, zorder=0)
                        log = direction in config["log_directions"]
                        axis.set_title(f"{'ln ' if log else ''}{direction}, step {step:.0e} ({line[0]['device']}"
                                       f"{f', {len(missing)} failed, skipped' if missing else ''})", fontsize=10)
                        axis.set_xlabel(f"Δ ln {direction}" if log else f"Δ {direction}")
                        axis.set_ylabel(metric)
                        axis.grid(color="0.9", linewidth=0.6)
                        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=True)
                figure.suptitle(f"{config['title']} {garment}, frames {window['frame_start']}–"
                                f"{window['frame_start'] + window['frame_count'] - 1}: {title}\n"
                                f"center D/E/H = {config['center']['D']}/{config['center']['E']}/{config['center']['H']}, "
                                f"fp64 deterministic, k = ±{config['radius']}; ln steps for {', '.join(config['log_directions'])}, linear otherwise")
                pdf.savefig(figure)
                figure.savefig(output / f"{garment}_{metric}.png", dpi=120)
                plt.close(figure)
    print(f"Saved {pdf_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
