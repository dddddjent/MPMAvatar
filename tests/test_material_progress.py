"""Verify optimizer continuation and readable material records on CPU."""

import csv
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import numpy as np
import torch

from material_progress import export_progress, restore_training_state, save_training_state

# Template command (activate mpmavatar; run from MPMAvatar):
# python -m unittest discover -s tests -p test_material_progress.py -v


def optimizer_bundle() -> tuple[dict[str, torch.Tensor], Any, Any]:
    params = {key: torch.tensor(value) for key, value in zip(("D", "E", "H"), (1., 2., 3.))}
    optimizer = torch.optim.Adam([{"params": [value], "lr": 0.01 * (i + 1)}
                                  for i, value in enumerate(params.values())])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    return params, optimizer, scheduler


def update(params: dict[str, torch.Tensor], optimizer: Any, scheduler: Any) -> None:
    optimizer.zero_grad()
    for value in params.values():
        value.grad = value.square() / 7
    optimizer.step()
    scheduler.step()


class MaterialProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.context = {"dataset_dir": "/dataset", "fitting_frame_ids": [0, 1], "iterations": 200}

    def write_pair(self, step: int) -> None:
        for kind in ("last", "best"):
            np.savez(self.directory / f"{kind}_param_{step:05d}.npz", step=step,
                     D=0.6, E=733., H=0.93, loss=0.001,
                     dataset_dir="/dataset", fitting_frame_ids=[0, 1])

    def test_optimizer_and_scheduler_continue_identically(self) -> None:
        params, optimizer, scheduler = optimizer_bundle()
        for _ in range(4):
            update(params, optimizer, scheduler)
        record = {"step": 3, "loss": 0.1, "D": 1., "E": 200., "H": 3.}
        save_training_state(self.directory, 4, params, optimizer, scheduler,
                            record, record, self.context, None)
        other, other_optimizer, other_scheduler = optimizer_bundle()
        state = restore_training_state(self.directory, other, other_optimizer, other_scheduler,
                                       self.context, False)
        self.assertEqual(state["next_step"], 4)
        self.assertEqual(state["best"], record)
        for _ in range(3):
            update(params, optimizer, scheduler)
            update(other, other_optimizer, other_scheduler)
            for key in params:
                torch.testing.assert_close(params[key], other[key], rtol=0, atol=0)
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[params[key]][field],
                                               other_optimizer.state[other[key]][field], rtol=0, atol=0)
            self.assertEqual(scheduler.state_dict(), other_scheduler.state_dict())
        with self.assertRaisesRegex(AssertionError, "same material fitting settings"):
            restore_training_state(self.directory, other, other_optimizer, other_scheduler,
                                   {**self.context, "fitting_frame_ids": [0, 1, 2]}, False)

    def test_old_checkpoint_reset_is_explicit_and_preserves_units_and_step(self) -> None:
        self.write_pair(9)
        self.write_pair(45)
        params, optimizer, scheduler = optimizer_bundle()
        with self.assertRaisesRegex(AssertionError, "explicitly pass --resume-parameters"):
            restore_training_state(self.directory, params, optimizer, scheduler, self.context, False)
        state = restore_training_state(self.directory, params, optimizer, scheduler, self.context, True)
        self.assertEqual(state["next_step"], 46)
        self.assertEqual(state["optimizer_reset_step"], 46)
        self.assertAlmostEqual(params["E"].item(), 7.33, places=5)
        self.assertAlmostEqual(params["D"].item(), 0.6)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(scheduler.last_epoch, 0)
        save_training_state(self.directory, 46, params, optimizer, scheduler, state["best"],
                            state["last"], self.context, 46)
        restore_training_state(self.directory, params, optimizer, scheduler, self.context, False)

    def test_history_exports_numeric_order_and_unknown_old_loss_inputs(self) -> None:
        self.write_pair(100000)
        self.write_pair(99999)
        export_progress(self.directory)
        with (self.directory / "history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["step"] for row in rows], ["99999", "100000"])
        self.assertEqual(rows[0]["evaluated_D"], "")
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["last"]["step"], 100000)
        self.assertEqual(summary["best"]["E"], 733.)
        export_progress(self.directory)
        self.assertEqual(len((self.directory / "history.csv").read_text().splitlines()), 3)
        export_progress(self.directory, last_step=99999)
        summary = json.loads((self.directory / "summary.json").read_text())
        self.assertEqual(summary["last"]["step"], 99999)


if __name__ == "__main__":
    unittest.main()
