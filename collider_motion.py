"""Read prescribed, fixed-topology body motion in world metres at 25 Hz."""

from pathlib import Path

import numpy as np


def load_raw_collider(path: Path, frame_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vertices, triangles and interval velocities without a body model."""
    assert path.is_file(), path
    with np.load(path, allow_pickle=False) as data:
        vertices = data["vertices"]
        faces = data["faces"]
        velocities = data["interval_velocities"]
        timestamps = data["timestamps"]
        assert np.array_equal(data["frame_ids"], frame_ids), "Collider frames must match the export"
    assert vertices.ndim == 3 and vertices.shape[0] == len(frame_ids) and vertices.shape[2] == 3
    assert vertices.shape[1] > 0 and len(frame_ids) >= 2
    assert faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
    assert np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0 and faces.max() < vertices.shape[1]
    assert np.isfinite(vertices).all() and np.isfinite(velocities).all()
    assert timestamps.shape == (len(frame_ids),) and np.isfinite(timestamps).all()
    delta = np.diff(timestamps)
    assert np.allclose(delta, 1.0 / 25.0), "Collider motion must be sampled at 25 Hz"
    assert velocities.shape == vertices[:-1].shape
    expected = np.diff(vertices.astype(np.float64), axis=0) / delta[:, None, None]
    assert np.allclose(velocities, expected, atol=1e-6, rtol=1e-5), "Inconsistent collider velocities"
    return vertices, faces, velocities
