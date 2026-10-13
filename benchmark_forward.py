"""Time the native forward simulation after warm-up, including reset and geometry loss.

Activate mpmavatar; run on an allocated GPU from MPMAvatar:
# python benchmark_forward.py --config configs/material_sensitivity_185.json --output output/forward_precision --precision fp64 --garment both --parameter-set default --warmup 1 --repeats 3
# python benchmark_forward.py --config configs/material_sensitivity_185.json --output output/forward_precision/deterministic --precision fp64 --deterministic --garment both --parameter-set default --warmup 1 --repeats 3
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def check_arrays(struct: Any, dtype: Any) -> dict[str, str]:
    """Check every floating buffer, including collider geometry and contact grids."""
    import torch
    import warp as wp

    result = {}
    for key, value in vars(struct).items():
        if isinstance(value, wp.array):
            tensor = wp.to_torch(value, requires_grad=False)
            if tensor.is_floating_point():
                assert tensor.dtype == dtype, (key, tensor.dtype, dtype)
                assert torch.isfinite(tensor).all(), key
                result[key] = str(value.dtype.__name__)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable Warp GPU_TO_GPU ordering and deterministic tensor preparation")
    parser.add_argument("--garment", choices=("upper", "lower", "both"), default="both")
    parser.add_argument("--parameter-set", choices=("soft", "default", "stiff"), default="default")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.config.is_file(), args.config
    assert args.warmup >= 1 and args.repeats >= 1
    report_path = args.output / f"{args.precision}.json"
    assert not report_path.exists(), report_path

    config = json.loads(args.config.read_text())
    config["deterministic"] = args.deterministic or config.get("deterministic", False)
    from simulation_runtime import configure_precision, configure_simulation
    configure_precision(args.precision)
    configure_simulation(config["deterministic"])
    import numpy as np
    import torch
    import warp as wp
    from material_sensitivity import REPO, build_trainer, set_parameters
    from warp_mpm.precision import torch_dtype

    assert torch.cuda.is_available(), "Run on an allocated GPU"
    config["prepared"] = str((REPO / config["prepared"]).resolve())
    config["precision"] = args.precision
    config["material_gradient"] = "fd"
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "precision": args.precision, "gpu": torch.cuda.get_device_name(),
        "deterministic": config["deterministic"],
        "warp_deterministic_mode": wp.config.deterministic.name,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warp_version": wp.__version__, "torch_version": torch.__version__,
        "grid_size": config["grid_size"], "substeps": config["substeps"],
        "parameters": config["parameter_sets"][args.parameter_set],
        "warmup": args.warmup, "repeats": args.repeats,
        "timing_scope": "native forward, state reset, material setup and geometry loss; synchronized wall time; excludes loading, JIT compilation, rendering and disk output",
        "garments": {},
    }
    garments = ("upper", "lower") if args.garment == "both" else (args.garment,)
    for garment in garments:
        trainer = build_trainer(config, args.output / args.precision, garment, args.parameter_set)
        set_parameters(trainer, config["parameter_sets"][args.parameter_set])
        device = str(trainer.particle_init_position.device)
        assert trainer.particle_init_position.dtype == torch_dtype
        assert all(value.dtype == torch_dtype for value in trainer.torch_param.values())
        for _ in range(args.warmup):
            trainer._native_material_loss(device, (0.0, 0.0, 0.0), False)
        torch.cuda.synchronize()
        times, losses, states = [], [], []
        for repeat in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss, _ = trainer._native_material_loss(device, (0.0, 0.0, 0.0), False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            assert np.isfinite(loss), loss
            times.append(elapsed)
            losses.append(loss)
            states.append({key: wp.to_torch(getattr(trainer.mpm_state, key)).detach().clone()
                           for key in ("particle_x", "particle_v", "particle_C", "particle_d")})
            print(f"{args.precision} {garment} repeat {repeat + 1}: {elapsed:.6f} s, loss {loss:.9e}", flush=True)
        buffers = {"state": check_arrays(trainer.mpm_state, torch_dtype),
                   "model": check_arrays(trainer.mpm_model, torch_dtype),
                   "mesh": check_arrays(trainer.mpm_solver.mesh, torch_dtype),
                   "collider": check_arrays(trainer.mpm_solver.mesh_collider_params[0], torch_dtype),
                   "attachment": check_arrays(trainer.mpm_solver.particle_mover_params[0], torch_dtype)}
        transitions = trainer.scene.train_frame_num - 1
        median = statistics.median(times)
        differences = {key: max(float((state[key] - states[0][key]).abs().max()) for state in states)
                       for key in states[0]}
        identical = all(torch.equal(state[key], states[0][key]) for state in states for key in states[0])
        identical_loss = all(loss == losses[0] for loss in losses)
        report["garments"][garment] = {
            "frame_ids": trainer.scene.train_frame_index, "transitions": transitions,
            "n_particles": trainer.n_particles, "n_vertices": trainer.n_vertices,
            "n_elements": trainer.n_elements, "seconds": times, "losses": losses,
            "median_seconds": median, "mean_seconds": statistics.mean(times),
            "seconds_per_frame": median / transitions,
            "milliseconds_per_substep": 1000.0 * median / (transitions * config["substeps"]),
            "repeat_state_identical": identical, "repeat_loss_identical": identical_loss,
            "repeat_state_max_abs_difference": differences,
            "checked_buffers": buffers,
        }
        np.savez(args.output / f"{args.precision}_{garment}_final.npz",
                 positions=trainer.sim2wld(wp.to_torch(trainer.mpm_state.particle_x)[trainer.n_elements:]).detach().cpu().numpy())
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{args.precision} {garment} identical repeated state: {identical}; "
              f"identical loss: {identical_loss}; max differences: {differences}", flush=True)
        assert not config["deterministic"] or (identical and identical_loss), (
            f"Deterministic repeats differ; see {report_path}")
        print(f"{args.precision} {garment} median: {median:.6f} s ({median / transitions:.6f} s/frame)", flush=True)
        del trainer
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
