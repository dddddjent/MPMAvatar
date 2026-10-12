"""Compare saved loss stencils with common absolute and percentage axes.

Activate mpmavatar; run from MPMAvatar on a CPU allocation:
# python plot_material_sensitivity.py --output output/material_sensitivity_185
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read_csv(path: Path) -> list[dict[str, str]]:
    assert path.is_file(), path
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def analyze(output: Path) -> None:
    samples = read_csv(output / "samples.csv")
    fds = read_csv(output / "fd.csv")
    assert len(samples) == 270 and len(fds) == 54
    cases = list(dict.fromkeys((row["garment"], row["parameter_set"]) for row in samples))
    centers = {case: [float(row["loss"]) for row in samples
                      if (row["garment"], row["parameter_set"]) == case and int(row["offset"]) == 0]
               for case in cases}
    assert all(len(values) == 9 for values in centers.values())
    references = {case: math.fsum(values) / len(values) for case, values in centers.items()}
    groups = {(row["garment"], row["parameter_set"], row["direction"], float(row["step_size"])):
              {int(sample["offset"]): float(sample["loss"]) for sample in samples
               if all(sample[key] == row[key] for key in ("garment", "parameter_set", "direction", "step_size"))}
              for row in fds}
    diagnostics = []
    for row in fds:
        case = row["garment"], row["parameter_set"]
        step = float(row["step_size"])
        losses = groups[(*case, row["direction"], step)]
        assert set(losses) == {-2, -1, 0, 1, 2}
        span = max(centers[case]) - min(centers[case])
        ad = float(row["ad"])
        diagnostics.append({
            **{key: row[key] for key in ("garment", "parameter_set", "direction", "step_size", "ad",
                                         "fd_h", "fd_2h", "fd_five_point", "ad_fd_h_relative_error",
                                         "ad_fd_2h_relative_error", "ad_fd_five_point_relative_error",
                                         "distinct_effective_values")},
            "native_center_repeat_span": span,
            "ad_linear_loss_change_at_h": abs(ad * step),
            "ad_change_over_center_span": abs(ad * step) / span if span else "",
            "left_slope_h": (losses[0] - losses[-1]) / step,
            "right_slope_h": (losses[1] - losses[0]) / step,
            "left_slope_2h": (losses[0] - losses[-2]) / (2 * step),
            "right_slope_2h": (losses[2] - losses[0]) / (2 * step),
            "maximum_loss_change_from_center": max(abs(value - losses[0]) for value in losses.values()),
        })
    with (output / "local_loss_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    # Curves start at zero; fixed mean-relative bands and scales are shared across steps.
    extents = {"absolute": 0.0, "percent": 0.0}
    for row in fds:
        case = row["garment"], row["parameter_set"]
        step = float(row["step_size"])
        losses = groups[(*case, row["direction"], step)]
        reference = references[case]
        assert reference > 0
        candidates = [abs(value - losses[0]) for value in losses.values()]
        candidates.extend(abs(value - reference) for value in centers[case])
        ad = float(row["ad"])
        if math.isfinite(ad):
            candidates.extend(abs(ad * offset * step) for offset in (-2, 2))
        extents["absolute"] = max(extents["absolute"], *candidates)
        extents["percent"] = max(extents["percent"], *(value * 100.0 / reference for value in candidates))
    assert all(extent > 0 for extent in extents.values())

    for view in ("absolute", "percent"):
        filename = "loss_curves" if view == "absolute" else "loss_curves_percent"
        with PdfPages(output / f"{filename}.pdf") as pdf:
            for step in (1e-6, 1e-5, 1e-4):
                render_page(output, pdf, filename, view, step, cases, centers, references, groups, fds, extents[view])

    for case in cases:
        values = [float(row["loss"]) for row in samples if (row["garment"], row["parameter_set"]) == case]
        print(f"{case[0]}/{case[1]}: native loss [{min(values):.12g}, {max(values):.12g}], "
              f"center repeat span {max(centers[case]) - min(centers[case]):.12g}")
    print(f"Saved 54 diagnostic rows and shared-axis absolute/percentage loss curves in {output}")


def render_page(
    output: Path, pdf: PdfPages, filename: str, view: str, step: float,
    cases: list[tuple[str, str]], centers: dict[tuple[str, str], list[float]],
    references: dict[tuple[str, str], float],
    groups: dict[tuple[str, str, str, float], dict[int, float]],
    fds: list[dict[str, str]], extent: float,
) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 16), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.92, bottom=0.065, hspace=0.8, wspace=0.3)
    metric = "loss change, shared linear axes" if view == "absolute" else "percentage loss change, shared signed-log axes"
    fig.suptitle(f"Subject 185: {metric}\nabsolute h = {step:.0e}", fontsize=14, y=0.995)
    handles = []
    for index, case in enumerate(cases):
        for column, direction in enumerate(("D", "E", "H")):
            axis = axes[index, column]
            losses = groups[(*case, direction, step)]
            row = next(row for row in fds if (row["garment"], row["parameter_set"], row["direction"],
                                             float(row["step_size"])) == (*case, direction, step))
            offsets = [-2, -1, 0, 1, 2]
            reference = references[case]
            factor = 1.0 if view == "absolute" else 100.0 / reference
            low = (min(centers[case]) - reference) * factor
            high = (max(centers[case]) - reference) * factor
            changes = [(losses[offset] - losses[0]) * factor for offset in offsets]
            band = axis.axhspan(low, high,
                               color="gray", alpha=0.2, label="Center repeats relative to their mean")
            measured, = axis.plot(offsets, changes,
                                  "o-", color="#0072B2", linewidth=1.5, label="Native measurements")
            ad = float(row["ad"])
            if math.isfinite(ad):
                prediction = [ad * offset * step * factor for offset in offsets]
                tangent, = axis.plot(offsets, prediction,
                                     "--", color="#D55E00", linewidth=1.5, label="AD linear prediction")
                if not handles:
                    handles = [measured, tangent, band]
            else:
                axis.text(0.04, 0.88, "AD = NaN", transform=axis.transAxes, color="#B00020")
            if int(row["distinct_effective_values"]) < 5:
                axis.text(0.04, 0.88, f"{row['distinct_effective_values']}/5 distinct native inputs",
                          transform=axis.transAxes, color="#B00020", fontsize=9)
            axis.axhline(0, color="black", linewidth=0.5, alpha=0.4)
            axis.set_title(f"{case[0]}/{case[1]}: {direction}", fontsize=10)
            axis.set_xticks(offsets)
            if view == "absolute":
                axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useOffset=False)
            axis.tick_params(labelsize=9)
            axis.grid(alpha=0.15)
            if column == 0:
                axis.set_ylabel("Loss - stencil center" if view == "absolute" else "Loss change (%)", fontsize=9)
            if index == len(cases) - 1:
                axis.set_xlabel("Offset n (parameter change = n h)", fontsize=9)
    if view == "absolute":
        axes[0, 0].set_ylim(-1.15 * extent, 1.15 * extent)
        note = "Same linear y range on every page; large jumps can flatten small changes."
    else:
        axes[0, 0].set_yscale("symlog", linthresh=0.001)
        axes[0, 0].set_ylim(-1.5 * extent, 1.5 * extent)
        note = "Same y scale on every page: linear within +/-0.001%, logarithmic outside. Use the linear PDF for curve shape."
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.018, note + "\n"
             "Curves = loss - stencil center; gray = nine center losses - their mean. Percentage = 100 x change / mean center loss.",
             ha="center", fontsize=9)
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
