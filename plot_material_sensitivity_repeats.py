"""Plot nine-run loss distributions and AD prediction ranges on shared axes.

Activate mpmavatar; run from MPMAvatar on an owned CPU allocation:
# python plot_material_sensitivity_repeats.py --output output/material_sensitivity_185_repeats
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PointKey = tuple[str, str, str, float]
AdKey = tuple[str, str, str]
BLUE = "#0072B2"
ORANGE = "#D55E00"
OFFSETS = (-2, -1, 0, 1, 2)
STEPS = (1e-6, 1e-5, 1e-4)
PERCENT_LINEAR_THRESHOLD = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), path
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def ad_predictions(
    stat: dict[str, str], offsets: list[float], step: float, factor: float,
) -> tuple[list[float], list[float], list[float]]:
    gradients = [float(stat[f"ad_{metric}"]) for metric in ("min", "mean", "max")]
    assert all(math.isfinite(value) for value in gradients)
    low_gradient, mean_gradient, high_gradient = gradients
    assert low_gradient <= mean_gradient <= high_gradient
    low, mean, high = [], [], []
    for offset in offsets:
        change = offset * step * factor
        bounds = (low_gradient * change, high_gradient * change)
        low.append(min(bounds))
        mean.append(mean_gradient * change)
        high.append(max(bounds))
    assert all(lower <= middle <= upper for lower, middle, upper in zip(low, mean, high))
    assert all(lower == middle == upper == 0 for offset, lower, middle, upper
               in zip(offsets, low, mean, high) if offset == 0)
    return low, mean, high


def analyze(output: Path) -> None:
    assert (output / "audit.json").is_file(), "Audit the completed experiment before plotting"
    audit = json.loads((output / "audit.json").read_text())
    assert audit["verified"] and audit["trajectories"] == 2430 and audit["ad_evaluations"] == 54
    points = read_csv(output / "point_statistics.csv")
    stats = read_csv(output / "ad_statistics.csv")
    assert len(points) == 270 and len(stats) == 18
    cases = list(dict.fromkeys((row["garment"], row["parameter_set"]) for row in points))
    assert len(cases) == 6
    groups: dict[PointKey, dict[int, dict[str, str]]] = {}
    for row in points:
        assert int(row["count"]) == 9
        key = row["garment"], row["parameter_set"], row["direction"], float(row["step_size"])
        group = groups.setdefault(key, {})
        offset = int(row["offset"])
        assert offset not in group
        group[offset] = row
    ads = {(row["garment"], row["parameter_set"], row["direction"]): row for row in stats}
    assert len(groups) == 54 and len(ads) == 18
    assert all(set(group) == set(OFFSETS) for group in groups.values())
    extents = {"absolute": 0.0, "percent": 0.0}
    for key, group in groups.items():
        center_mean = float(group[0]["center_mean"])
        assert center_mean > 0 and float(group[0]["change_mean"]) == 0
        assert all(float(row["center_mean"]) == center_mean for row in group.values())
        stat = ads[key[:3]]
        assert int(stat["count"]) == 9
        for view, prefix, factor in (("absolute", "change", 1.0), ("percent", "percent", 100.0 / center_mean)):
            candidates = [abs(float(row[f"{prefix}_{metric}"])) for row in group.values()
                          for metric in ("min", "mean", "max")]
            if int(stat["finite_count"]) > 0:
                low, mean, high = ad_predictions(stat, list(OFFSETS), key[3], factor)
                candidates.extend(abs(value) for values in (low, mean, high) for value in values)
            extents[view] = max(extents[view], *candidates)
    assert all(math.isfinite(extent) and extent > 0 for extent in extents.values())
    for view in ("absolute", "percent"):
        filename = "loss_distribution" if view == "absolute" else "loss_distribution_percent"
        with PdfPages(output / f"{filename}.pdf") as pdf:
            for step in STEPS:
                render_page(output, pdf, filename, view, step, cases, groups, ads, extents[view])
    plot_report = {"verified_input_audit": True, "native_ranges_per_view": 270,
                   "native_repeats_per_range": 9, "ad_repeats_per_center": 9,
                   "baseline": audit["baseline"], "gray_band": False,
                   "absolute_ylim": [-1.15 * extents["absolute"], 1.15 * extents["absolute"]],
                   "percentage_ylim": [-1.5 * extents["percent"], 1.5 * extents["percent"]],
                   "percentage_linear_threshold": PERCENT_LINEAR_THRESHOLD}
    (output / "plot_summary.json").write_text(json.dumps(plot_report, indent=2) + "\n")
    print(f"Saved nine-run native bars and AD ranges in {output}", flush=True)


def render_page(
    output: Path, pdf: PdfPages, filename: str, view: str, step: float,
    cases: list[tuple[str, str]], groups: dict[PointKey, dict[int, dict[str, str]]],
    ads: dict[AdKey, dict[str, str]], extent: float,
) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 16), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.91, bottom=0.075, hspace=0.8, wspace=0.3)
    prefix = "change" if view == "absolute" else "percent"
    metric = "Loss changes, shared linear axes" if view == "absolute" else "Percentage loss changes, shared signed-log axes"
    fig.suptitle(f"Subject 185: {metric}\nNine runs per point and AD center; absolute h = {step:.0e}", fontsize=14, y=0.993)
    offsets = list(OFFSETS)
    prediction_offsets = [-2.0 + index / 100.0 for index in range(401)]
    for index, case in enumerate(cases):
        for column, direction in enumerate(("D", "E", "H")):
            axis = axes[index, column]
            group = groups[(*case, direction, step)]
            stat = ads[(*case, direction)]
            center_mean = float(group[0]["center_mean"])
            factor = 1.0 if view == "absolute" else 100.0 / center_mean
            means = [float(group[offset][f"{prefix}_mean"]) for offset in offsets]
            lows = [float(group[offset][f"{prefix}_min"]) for offset in offsets]
            highs = [float(group[offset][f"{prefix}_max"]) for offset in offsets]
            assert all(low <= mean <= high for low, mean, high in zip(lows, means, highs))
            finite_count = int(stat["finite_count"])
            if finite_count > 0:
                low_ad, mean_ad, high_ad = ad_predictions(stat, prediction_offsets, step, factor)
                axis.fill_between(prediction_offsets, low_ad, high_ad, color=ORANGE, alpha=0.24, zorder=1)
                axis.plot(prediction_offsets, mean_ad, color=ORANGE, linewidth=1.5, zorder=2)
            if finite_count < 9:
                axis.text(0.04, 0.87, f"AD: {finite_count}/9 finite", color="#B00020",
                          transform=axis.transAxes, fontsize=9)
            distinct_inputs = len({float(row["effective_parameter"]) for row in group.values()})
            if distinct_inputs < 5:
                axis.text(0.04, 0.75, f"{distinct_inputs}/5 distinct native inputs", color="#B00020",
                          transform=axis.transAxes, fontsize=9)
            errors = [[mean - low for mean, low in zip(means, lows)],
                      [high - mean for mean, high in zip(means, highs)]]
            axis.errorbar(offsets, means, yerr=errors, fmt="_", color=BLUE, ecolor=BLUE,
                          markersize=13, markeredgewidth=2.0, elinewidth=1.4,
                          capsize=4, capthick=1.2, zorder=3)
            axis.axhline(0, color="black", linewidth=0.5, alpha=0.4)
            axis.set_title(f"{case[0]}/{case[1]}: {direction}", fontsize=10)
            axis.set_xticks(offsets)
            if view == "absolute":
                axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useOffset=False)
            axis.tick_params(labelsize=9)
            axis.grid(alpha=0.15)
            if column == 0:
                axis.set_ylabel("Loss - mean center loss" if view == "absolute" else "Loss change (%)", fontsize=9)
            if index == len(cases) - 1:
                axis.set_xlabel("Offset n (parameter change = n h)", fontsize=9)
    if view == "absolute":
        axes[0, 0].set_ylim(-1.15 * extent, 1.15 * extent)
        note = "Same linear y range on every page; large jumps can flatten small changes."
    else:
        axes[0, 0].set_yscale("symlog", linthresh=PERCENT_LINEAR_THRESHOLD)
        axes[0, 0].set_yticks([-10, -1e-3, -1e-6, 0, 1e-6, 1e-3, 10])
        axes[0, 0].set_ylim(-1.5 * extent, 1.5 * extent)
        note = f"Same y scale on every page: linear within +/-{PERCENT_LINEAR_THRESHOLD:g}%, logarithmic outside."
    handles = [Line2D([], [], color=BLUE, marker="_", markersize=13, markeredgewidth=2,
                      linestyle="none", label="Native mean dash + min-max bar (9 runs)"),
               Patch(facecolor=ORANGE, alpha=0.24, label="AD min-max prediction range"),
               Line2D([], [], color=ORANGE, linewidth=1.5, label="Mean AD prediction")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.953),
               ncol=3, frameon=False, fontsize=9)
    fig.text(0.5, 0.017, note + "\n"
             "Change = loss - mean of nine center losses in this stencil; percentage = 100 x change / that mean.\n"
             "Ranges are observed min-max, not confidence intervals. AD predictions share the mean center anchor at zero.",
             ha="center", fontsize=8.5)
    pdf.savefig(fig)
    fig.savefig(output / f"{filename}_h{step:.0e}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.output.resolve())


if __name__ == "__main__":
    main()
