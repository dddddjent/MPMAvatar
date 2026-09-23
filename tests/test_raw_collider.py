"""Check collider coordinates, timing and invalid export rejection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from collider_motion import load_raw_collider

# Template command (activate mpmavatar; run from MPMAvatar):
# python -m unittest discover -s tests -p test_raw_collider.py -v


class RawColliderTests(unittest.TestCase):
    def test_motion_and_rejected_inputs(self) -> None:
        vertices = np.asarray([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
                               for _ in range(4)], dtype=np.float32)
        vertices[:, :, 0] += np.arange(4)[:, None] * .04
        timestamps = np.arange(4) / 25.
        values = dict(vertices=vertices, faces=np.array([[0, 1, 2]]), frame_ids=np.arange(4),
                      timestamps=timestamps,
                      interval_velocities=np.diff(vertices.astype(np.float64), axis=0) * 25.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.npz"
            np.savez(path, **values)
            actual, faces, velocities = load_raw_collider(path, list(range(4)))
            np.testing.assert_array_equal(actual, vertices)
            np.testing.assert_array_equal(faces, values["faces"])
            np.testing.assert_allclose(velocities[:, :, 0], 1., atol=3e-6)
            np.testing.assert_allclose(actual[:-1] + velocities / 25., actual[1:])
            invalid = [dict(frame_ids=np.array([0, 1, 3, 4])),
                       dict(timestamps=np.arange(4) / 30.),
                       dict(faces=np.array([[0, 1, 3]])),
                       dict(interval_velocities=np.zeros_like(velocities)),
                       dict(vertices=np.full_like(vertices, np.nan))]
            for update in invalid:
                with self.subTest(field=next(iter(update))):
                    np.savez(path, **{**values, **update})
                    with self.assertRaises(AssertionError):
                        load_raw_collider(path, list(range(4)))


if __name__ == "__main__":
    unittest.main()
