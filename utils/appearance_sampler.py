"""Preserve the native appearance loader's shuffled epoch across interruptions."""
from collections.abc import Iterator, Sized
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler

# Template usage (mpmavatar environment, workspace root):
# bash MPMAvatar/scripts/native_4ddress_pipeline.sh 190 appearance


class StatefulRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source
        assert len(data_source) > 0
        self.order: list[int] = []
        self.cursor = 0
        self.resume_mid_epoch = False

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        if not self.order or self.cursor == len(self.order):
            # RandomSampler keeps the original permutation and global RNG draws.
            self.order = list(RandomSampler(self.data_source))
            self.cursor = 0
        while self.cursor < len(self.order):
            index = self.order[self.cursor]
            self.cursor += 1
            yield index

    def state_dict(self) -> dict[str, Any]:
        return {"size": len(self), "order": self.order.copy(), "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        assert state["size"] == len(self)
        order = state["order"]
        assert not order or sorted(order) == list(range(len(self)))
        assert 0 <= state["cursor"] <= len(order)
        self.order = order.copy()
        self.cursor = state["cursor"]
        self.resume_mid_epoch = 0 < self.cursor < len(self.order)


def cycle_native(loader: DataLoader[Any], sampler: StatefulRandomSampler) -> Iterator[Any]:
    assert loader.num_workers == 0 and loader.batch_size == 1
    assert loader.sampler is sampler and loader.generator is None
    while True:
        if sampler.resume_mid_epoch:
            # Recreating a mid-epoch iterator must not consume a second base_seed.
            rng = torch.get_rng_state()
            epoch = iter(loader)
            torch.set_rng_state(rng)
            sampler.resume_mid_epoch = False
        else:
            epoch = iter(loader)
        yield from epoch
