"""Observed cloth velocity for the original MPMAvatar particle ordering."""

import torch


def estimate_velocity(vertices: torch.Tensor, frame_ids: list[int]) -> torch.Tensor:
    """Estimate once from two consecutive 25 Hz cloth observations."""
    assert vertices.ndim == 3 and vertices.shape[-1] == 3
    assert len(frame_ids) == len(vertices) and len(frame_ids) >= 2
    assert frame_ids[1] == frame_ids[0] + 1
    assert torch.isfinite(vertices[:2]).all()
    return ((vertices[1] - vertices[0]) * 25.0).detach().clone()


def particle_velocities(
    vertex_velocity: torch.Tensor, faces: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Pack triangle centres followed by vertices; velocity has no translation."""
    scaled = vertex_velocity * scale
    return torch.cat((scaled[faces.long()].mean(dim=1), scaled), dim=0).contiguous()
