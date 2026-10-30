"""Measure the C2 mass band on saved crossings and force-free transfer fixtures.

Activate mpmavatar; from MPMAvatar:
# MPLCONFIGDIR=/tmp/robustdiffcloth-matplotlib python probe_hang_mass_band.py --baseline output/hang_heavy_E_control_h1e-6_r10 --output output/hang_heavy_E_mass_band
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hang_local_sweep import HangSimulation

BANDS = {"narrow": (0.75e-15, 1.5e-15), "default": (0.5e-15, 2e-15), "wide": (0.25e-15, 4e-15)}


def saved_crossings(baseline: Path, output: Path) -> list[dict[str, Any]]:
    diagnosis = baseline / "diagnosis"
    cases = [
        ("coarse", diagnosis / "bracket_coarse/input_a.npz", diagnosis / "bracket_coarse/input_b.npz",
         diagnosis / "bracket_coarse/replay"),
        ("fine", diagnosis / "bracket_fine/input_a.npz", diagnosis / "bracket_fine/input_b.npz",
         diagnosis / "bracket_fine/replay"),
        ("step4019", diagnosis / "pair_m4_m3_to4018/result_0.npz", diagnosis / "pair_m4_m3_to4018/result_1.npz",
         diagnosis / "pair_m4_m3_step4019"),
        ("step4199", diagnosis / "pair_m4_m3_to4198/result_0.npz", diagnosis / "pair_m4_m3_to4198/result_1.npz",
         diagnosis / "pair_m4_m3_step4199"),
    ]
    rows = []
    for band, bounds in BANDS.items():
        for case, a, b, reference in cases:
            assert a.is_file() and b.is_file() and reference.is_dir()
            path = output / band / case
            path.mkdir(parents=True, exist_ok=True)
            with np.load(a) as state_a, np.load(b) as state_b:
                young_a, young_b = float(state_a["E"]), float(state_b["E"])
            command = [sys.executable, "-u", "replay_hang_e.py", "--config", str(baseline / "config.json"),
                       "--state-a", str(a), "--state-b", str(b), "--E-a", repr(young_a), "--E-b", repr(young_b),
                       "--substeps", "400", "--steps", "1", "--output", str(path), "--release-tension-shear",
                       "--mass-band", str(bounds[0]), str(bounds[1]), "--capture-stages", "4", "13", "14"]
            with (path / "run.log").open("w") as log:
                subprocess.run(command, cwd=Path(__file__).parent, stdout=log, stderr=subprocess.STDOUT, check=True)
            row: dict[str, Any] = {"band": band, "case": case, "bounds": bounds, "delta_E": young_b - young_a}
            for stage, field, name in ((4, "grid_v_out", "grid_velocity"), (14, "particle_C", "APIC_C"),
                                       (14, "particle_x", "position"), (14, "particle_d", "d")):
                with np.load(path / f"stage_{stage:02d}_run0.npz") as result_a, \
                        np.load(path / f"stage_{stage:02d}_run1.npz") as result_b, \
                        np.load(reference / f"stage_{stage:02d}_run0.npz") as native_a, \
                        np.load(reference / f"stage_{stage:02d}_run1.npz") as native_b:
                    assert np.isfinite(result_a[field]).all() and np.isfinite(result_b[field]).all()
                    row[f"band_{name}_max"] = float(np.abs(result_b[field] - result_a[field]).max())
                    row[f"hard_{name}_max"] = float(np.abs(native_b[field] - native_a[field]).max())
                    if stage == 4:
                        for result, native in ((result_a, native_a), (result_b, native_b)):
                            assert np.array_equal(result["nodes"], native["nodes"])
                            assert np.array_equal(result["grid_m"].view(np.uint64), native["grid_m"].view(np.uint64))
                            high = result["grid_m"] >= bounds[1]
                            assert np.array_equal(result[field][high].view(np.uint64), native[field][high].view(np.uint64))
                        row["mass_identical_and_high_mass_velocity_bitwise_equal"] = True
            rows.append(row)
            (output / "crossings.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(f"{band} {case}: grid {row['hard_grid_velocity_max']:.6g} -> {row['band_grid_velocity_max']:.6g}; "
                  f"C {row['hard_APIC_C_max']:.6g} -> {row['band_APIC_C_max']:.6g}", flush=True)
    return rows


def uniform_transfers(baseline: Path, output: Path) -> list[dict[str, Any]]:
    config = json.loads((baseline / "config.json").read_text())
    config["frames"] = 0
    simulation = HangSimulation(config)
    import torch
    import warp as wp
    from hang_mass_band import grid_normalization_mass_band
    from warp_mpm.mpm_solver import g2p_e, g2p_v, grid_normalization_and_gravity, p2g_apic_with_stress, zero_grid
    from warp_mpm.precision import vec3

    simulation.rollout(config["center"])
    state, model = simulation.state, simulation.model
    dt = 1 / (config["native"]["frame_rate"] * config["native"]["substeps"])
    grid_shape = (model.n_grid, model.n_grid, model.n_grid)
    translation = np.array([0.2, -0.1, 0.3])
    rows = []
    for fixture, path in (("rest", baseline / "runs/k+00/state_000000.npz"),
                          ("deformed4018", baseline / "diagnosis/pair_m4_m3_to4018/result_0.npz")):
        with np.load(path) as archive:
            incoming_x, incoming_d = archive["particle_x"], archive["particle_d"]
        positions = torch.tensor(incoming_x, device=simulation.device)
        directions = torch.tensor(incoming_d, device=simulation.device)
        for gravity in (0.0, config["gravity_sim"]):
            model.gravitational_accelaration = vec3(0.0, gravity, 0.0)
            expected = translation + dt * np.array([0.0, gravity, 0.0])
            for band, bounds in (("hard", (0.0, 0.0)), *BANDS.items()):
                with torch.no_grad():
                    wp.to_torch(state.particle_x).copy_(positions)
                    wp.to_torch(state.particle_d).copy_(directions)
                    wp.to_torch(state.particle_v).copy_(torch.tensor(translation, device=simulation.device))
                    wp.to_torch(state.particle_C).zero_()
                    wp.to_torch(state.particle_stress).zero_()
                    wp.to_torch(state.vertex_force).zero_()
                wp.launch(zero_grid, dim=grid_shape, inputs=[state, model], device=simulation.device)
                wp.launch(p2g_apic_with_stress, dim=simulation.n_elements + simulation.n_vertices,
                          inputs=[state, model, dt, simulation.n_elements], device=simulation.device)
                kernel = grid_normalization_and_gravity if band == "hard" else grid_normalization_mass_band
                inputs = [state, model, dt] if band == "hard" else [state, model, dt, *bounds]
                wp.launch(kernel, dim=grid_shape, inputs=inputs, device=simulation.device)
                wp.launch(g2p_v, dim=simulation.n_vertices, inputs=[state, model, dt, simulation.n_elements],
                          device=simulation.device)
                wp.launch(g2p_e, dim=simulation.n_elements, inputs=[state, model, dt, simulation.n_elements],
                          device=simulation.device)
                velocities = state.particle_v.numpy()
                affine = state.particle_C.numpy()
                directions_out = state.particle_d.numpy()
                assert np.isfinite(velocities).all() and np.isfinite(affine).all() and np.isfinite(directions_out).all()
                error = velocities - expected
                momentum_error = np.sum(state.particle_mass.numpy()[:, None] * error, axis=0)
                row = {"fixture": fixture, "gravity_sim": gravity, "band": band,
                       "vertex_velocity_error_max_sim": float(np.abs(error[simulation.n_elements:]).max()),
                       "all_particle_velocity_error_max_sim": float(np.abs(error).max()),
                       "APIC_C_error_max": float(np.abs(affine).max()),
                       "d3_step_error_max": float(np.abs(directions_out[:simulation.n_elements, :, 2]
                                                       - incoming_d[:simulation.n_elements, :, 2]).max()),
                       "momentum_error_sim": momentum_error.tolist()}
                rows.append(row)
                print(f"Uniform {fixture} g={gravity:g} {band}: vertex v error "
                      f"{row['vertex_velocity_error_max_sim']:.6g}, d3 error {row['d3_step_error_max']:.6g}", flush=True)
    (output / "uniform.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def center_figure(output: Path, baseline: Path) -> Any:
    import matplotlib.pyplot as plt

    with np.load(baseline / "runs/k+00/trajectory.npz") as archive:
        reference = archive["trajectory"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    rows, trajectories = [], {}
    for band, bounds in BANDS.items():
        directory = output / band / ("sweep" if band == "default" else "rollout") / "runs/k+00"
        for path in directory.glob("trajectory.npz"):
            with np.load(path) as archive:
                x = archive["trajectory"]
                assert np.array_equal(archive["mass_band"], bounds)
            assert x.shape == reference.shape and np.isfinite(x).all()
            trajectories[band] = x
            distance = np.linalg.norm(x - reference, axis=-1)
            rms = np.sqrt(np.mean(distance * distance, axis=1))
            axes[0].plot(np.arange(len(rms)) / 25, 1000 * rms, label=band)
            axes[1].plot(np.arange(len(rms)) / 25, 1000 * distance.max(axis=1), label=band)
            rows.append({"band": band, "trajectory_rms_mm": float(1000 * np.sqrt(np.mean(distance * distance))),
                         "peak_frame_rms_mm": float(1000 * rms.max()), "final_rms_mm": float(1000 * rms[-1]),
                         "max_vertex_mm": float(1000 * distance.max())})
    pairs = []
    for i, a in enumerate(BANDS):
        for b in list(BANDS)[i + 1:]:
            if a in trajectories and b in trajectories:
                delta = trajectories[a] - trajectories[b]
                pairs.append({"a": a, "b": b, "trajectory_rms_mm": float(1000 * np.sqrt(np.mean(np.sum(delta * delta, axis=-1)))),
                              "max_vertex_mm": float(1000 * np.linalg.norm(delta, axis=-1).max())})
    (output / "centers.json").write_text(json.dumps({"centers": rows, "pairs": pairs}, indent=2) + "\n")
    axes[0].set(ylabel="Frame vertex RMS separation (mm)", title="Center motion change from hard cutoff")
    axes[1].set(ylabel="Maximum vertex separation (mm)", title="Largest local motion change")
    for axis in axes:
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=.25)
        axis.legend()
    figure.suptitle(f"Mass-band sensitivity: E=100; {len(rows)}/3 full center trajectories")
    figure.savefig(output / "centers.png", dpi=170)
    return figure


def equation_figure() -> Any:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 8))
    figure.text(.07, .92, "Smooth grid-mass activation", fontsize=22, weight="bold")
    figure.text(.07, .81, r"$t=\mathrm{clamp}((m-m_0)/(m_1-m_0),0,1),\quad s(t)=6t^5-15t^4+10t^3$", fontsize=19)
    figure.text(.07, .71, r"$\mathbf{v}^{+}=s(t)(\mathbf{q}/m+\Delta t\,\mathbf{g})\quad(m>m_0);\qquad\mathbf{v}^{+}=0\quad(m\leq m_0)$", fontsize=19)
    figure.text(.07, .61, "Default band: [0.5, 2] x 1e-15; narrow: [0.75, 1.5]; wide: [0.25, 4].\n"
                "The zero branch skips division. The activation has zero first/second derivatives at both ends.\n"
                "For masses above the band, saved replay grid velocities match the hard baseline bitwise.",
                fontsize=13, linespacing=1.7, va="top")
    figure.text(.07, .40, "Scope and limits", fontsize=17, weight="bold")
    figure.text(.07, .33, "Crossings: one-step replays from the SAME incoming states and E values.\n"
                "The coarse/fine inputs straddle the old cutoff; their spans differ by 1000x.\n"
                "Uniform fixtures use native P2G/G2P, zero internal forces, and no pins or collider.\n"
                "Damping and momentum error are measured; C2 activation does not guarantee conservation\n"
                "or smoothness of other material/contact operations. No full-rollout conclusion from this test alone.",
                fontsize=13, linespacing=1.7, va="top")
    return figure


def plot(output: Path, crossings: list[dict[str, Any]], uniform: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    masses = np.geomspace(0.1e-15, 10e-15, 500)
    axes[0, 0].semilogx(masses / 1e-15, (masses > 1e-15).astype(float), "k--", label="Hard")
    for band, (lower, upper) in BANDS.items():
        t = np.clip((masses - lower) / (upper - lower), 0, 1)
        axes[0, 0].semilogx(masses / 1e-15, t**3 * (10 + t * (-15 + 6*t)), label=band)
    axes[0, 0].set(xlabel="Grid mass / old cutoff", ylabel="Activation", title="C2 mass bands")
    axes[0, 0].legend()
    for axis, metric, title in ((axes[0, 1], "grid_velocity", "Grid velocity difference"),
                                (axes[1, 0], "APIC_C", "Particle APIC C difference")):
        hard = [row for row in crossings if row["band"] == "default" and row["case"] in ("coarse", "fine")]
        axis.loglog([row["delta_E"] for row in hard], [row[f"hard_{metric}_max"] for row in hard], "ko--", label="Hard")
        for band in BANDS:
            rows = [row for row in crossings if row["band"] == band and row["case"] in ("coarse", "fine")]
            axis.loglog([row["delta_E"] for row in rows], [row[f"band_{metric}_max"] for row in rows], "o-", label=band)
        axis.set(xlabel="Local incoming-state bracket E span", ylabel="Maximum component difference", title=title)
        axis.legend()
    labels = ["hard", *BANDS]
    maxima = [max(row["d3_step_error_max"] for row in uniform if row["band"] == band) for band in labels]
    axes[1, 1].bar(labels, maxima)
    axes[1, 1].set(ylabel="Max thickness-direction error per step", title="Uniform-motion bias (worst fixture)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.suptitle("Controlled shear + C2 grid-mass band: local tests, fp64 deterministic")
    figure.savefig(output / "local_tests.png", dpi=170)
    with PdfPages(output / "local_tests.pdf") as pdf:
        pdf.savefig(figure)
        equation = equation_figure()
        pdf.savefig(equation)
        plt.close(equation)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert (args.baseline / "completion.json").is_file()
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    crossings = saved_crossings(args.baseline, args.output)
    uniform = uniform_transfers(args.baseline, args.output)
    plot(args.output, crossings, uniform)
    print("Local mass-band tests complete", flush=True)


if __name__ == "__main__":
    main()
