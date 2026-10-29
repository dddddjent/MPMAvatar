"""Inspect native QR return-map decisions on two saved fp64 hang states.

Activate mpmavatar; from MPMAvatar:
# python inspect_hang_mapping.py --config output/hang_heavy_E_trace/config.json --state-a output/hang_heavy_E_trace/early_replay_278/result_0.npz --state-b output/hang_heavy_E_trace/early_replay_278/result_1.npz --output output/hang_heavy_E_trace/mapping_279
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation_runtime import configure_precision, configure_simulation

configure_precision("fp64")
configure_simulation(True)

import warp as wp
from warp_mpm.mpm_solver import MPMModelStruct, anisotropy_return_mapping
from warp_mpm.precision import scalar, mat33
from hang_local_sweep import HangSimulation, save


@wp.kernel
def inspect(d: wp.array(dtype=mat33), model: MPMModelStruct,
            factors: wp.array2d(dtype=scalar), mapped: wp.array(dtype=mat33)) -> None:
    p = wp.tid()
    q, r = wp.qr3(d[p])
    sign0 = scalar(1.0)
    sign1 = scalar(1.0)
    if r[0, 0] < scalar(0.0):
        sign0 = scalar(-1.0)
    if r[1, 1] < scalar(0.0):
        sign1 = scalar(-1.0)
    r33 = sign0 * sign1 * r[2, 2]
    r13 = sign0 * r[0, 2]
    r23 = sign1 * r[1, 2]
    fn = model.kappa[p] * (scalar(1.0) - r33) * (scalar(1.0) - r33)
    ff = model.gamma[p] * wp.sqrt(r13 * r13 + r23 * r23)
    branch = scalar(2.0)  # elastic compression
    if r33 > scalar(1.0):
        branch = scalar(0.0)  # extension: retain shear
    elif ff > model.friction_coeff * fn:
        branch = scalar(1.0)  # compression: project shear
    factors[p, 0] = r33
    factors[p, 1] = r13
    factors[p, 2] = r23
    factors[p, 3] = fn
    factors[p, 4] = ff
    factors[p, 5] = model.friction_coeff * fn
    factors[p, 6] = branch
    mapped[p] = anisotropy_return_mapping(d[p], model, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-a", type=Path, required=True)
    parser.add_argument("--state-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.config.is_file() and args.state_a.is_file() and args.state_b.is_file()
    config = json.loads(args.config.read_text())
    simulation = HangSimulation(config)
    args.output.mkdir(parents=True, exist_ok=True)
    inputs, factors, mapped = [], [], []
    for run, path in enumerate((args.state_a, args.state_b)):
        with np.load(path) as archive:
            inputs.append(archive["particle_d"])
        d = wp.array(inputs[-1], dtype=mat33, device=simulation.device)
        result = wp.empty_like(d)
        qr = wp.empty((len(inputs[-1]), 7), dtype=scalar, device=simulation.device)
        wp.launch(inspect, dim=len(inputs[-1]), inputs=[d, simulation.model, qr, result], device=simulation.device)
        factors.append(qr.numpy())
        mapped.append(result.numpy())
        save(args.output / f"mapping_{run}.npz", input=inputs[-1], factors=factors[-1], mapped=mapped[-1])
    distance_in = np.linalg.norm(inputs[0] - inputs[1], axis=(1, 2))
    distance_out = np.linalg.norm(mapped[0] - mapped[1], axis=(1, 2))
    worst = int(distance_out.argmax())
    changed = np.flatnonzero(factors[0][:, 6] != factors[1][:, 6])
    report = {"factor_columns": ["r33", "r13", "r23", "fn", "ff", "friction_fn", "branch"],
              "changed_branches": len(changed), "worst_element": worst,
              "input_distance": float(distance_in[worst]), "output_distance": float(distance_out[worst]),
              "factor_a": factors[0][worst].tolist(), "factor_b": factors[1][worst].tolist(),
              "largest_changed_elements": sorted(changed.tolist(), key=lambda p: distance_out[p], reverse=True)[:10]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
