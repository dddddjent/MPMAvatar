"""Overlay one sweep line (direction, step, loss) for every garment of a dense local sweep on one axis.

Activate mpmavatar; run from MPMAvatar after summarize:
# python plot_material_local_sweep_overlay.py --output output/material_local_sweep_185 --direction E --step 1e-4 --metric loss_center
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#0072B2", "#E69F00", "#009E73"]
LOSSES = {"loss_target": "GT loss", "loss_center": "Center loss"}


def plot(output: Path, direction: str, step: float, metric: str) -> None:
    config = json.loads((output / "config.json").read_text())
    with (output / "samples.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["direction"] == direction and float(row["step"]) == step]
    log = direction in config["log_directions"]
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for color, garment in zip(COLORS, config["garments"]):
        line = sorted((row for row in rows if row["garment"] == garment), key=lambda row: int(row["k"]))
        assert len(line) == 2 * config["radius"] + 1, f"{garment}: {len(line)} points"
        axis.plot([float(row["offset"]) for row in line], [float(row[metric]) for row in line], color=color,
                  linewidth=1.5, marker="o", markersize=3, label=f"{garment} ({line[0]['device']})")
    axis.axvline(0.0, color="0.6", linewidth=0.8, zorder=0)
    axis.set_xlabel(f"Δ ln {direction}" if log else f"Δ {direction}")
    axis.set_ylabel(metric)
    axis.grid(color="0.9", linewidth=0.6)
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=True)
    axis.legend(frameon=False)
    axis.set_title(f"{config['title']}: {LOSSES[metric]}, {'ln ' if log else ''}{direction} step {step:.0e}\n"
                   f"center D/E/H = {config['center']['D']}/{config['center']['E']}/{config['center']['H']}, "
                   f"fp64 deterministic, k = ±{config['radius']}", fontsize=10)
    path = output / f"overlay_{direction}_s{step:.0e}_{metric}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", choices=["D", "E", "H"], required=True)
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--metric", choices=list(LOSSES), required=True)
    args = parser.parse_args()
    plot(args.output.resolve(), args.direction, args.step, args.metric)


if __name__ == "__main__":
    main()
