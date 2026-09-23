"""Tensor conversion through the public API of the installed Warp runtime."""

from typing import Any

import torch
import warp as wp


def from_torch_safe(
    t: torch.Tensor,
    dtype: Any = None,
    requires_grad: bool | None = None,
    grad: torch.Tensor | wp.array | None = None,
) -> wp.array:
    return wp.from_torch(t, dtype=dtype, requires_grad=requires_grad, grad=grad)
