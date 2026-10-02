"""Small synthetic CPU checks for native local-patch geometry contracts."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch as mock_patch

import numpy as np

from local_patch_data import (
    LocalPatch, build_local_patches, close_smplx_mouth, contact_free_mask, expand_vertex_rings,
    extract_patch_topology, load_bundle, local_grid_domain, mesh_adjacency,
    save_bundle, scored_interior, select_patch_cores, tracking_surface_complement,
    validate_patch,
)
from local_patch_support_survey import support_separated_vertices

# Template command (activate mpmavatar; run from MPMAvatar):
# python -m unittest discover -s tests -p test_local_patch_data.py -v


def plane_mesh(side: int = 9, spacing: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([[x * spacing, y * spacing, 0.0]
                           for y in range(side) for x in range(side)])
    faces: list[list[int]] = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            faces.extend(([a, a + 1, a + side], [a + 1, a + side + 1, a + side]))
    return vertices, np.asarray(faces, dtype=np.int64)


def synthetic_patch() -> LocalPatch:
    vertices, faces = plane_mesh()
    reference = np.stack((vertices, vertices + [0.003, 0.004, 0.005]))
    eligible = np.ones(len(vertices), dtype=bool)
    core = np.asarray([y * 9 + x for y in range(2, 7) for x in range(2, 7)])
    return build_local_patches("synthetic", np.asarray([45, 46]), np.arange(len(vertices)) + 200,
                               reference, vertices, faces, eligible, 0, 2.5,
                               min_scored_vertices=4,
                               core_vertex_ids=core)[0]


class LocalPatchDataTests(unittest.TestCase):
    def test_smplx_mouth_cap_closes_only_the_expected_loop(self) -> None:
        import trimesh

        body = trimesh.creation.cylinder(radius=.1, height=1., sections=32)
        body.update_faces(body.face_normals[:, 2] < .5)
        vertices, faces = body.vertices.copy(), body.faces.copy()
        self.assertFalse(body.is_watertight)
        closed = close_smplx_mouth(body)
        self.assertTrue(closed.is_watertight and closed.is_winding_consistent)
        np.testing.assert_array_equal(closed.vertices[:-1], vertices)
        np.testing.assert_array_equal(closed.faces[:-32], faces)
        np.testing.assert_array_equal(body.vertices, vertices)
        np.testing.assert_array_equal(body.faces, faces)
        with self.assertRaises(AssertionError):
            close_smplx_mouth(trimesh.creation.box())

    def test_original_triangles_collar_and_attachment_exclusion(self) -> None:
        patch = synthetic_patch()
        validate_patch(patch)
        full_vertices, full_faces = plane_mesh()
        ids = patch.vertex_ids - 200
        actual_global = ids[patch.faces]
        expected = full_faces[np.isin(full_faces, ids).all(axis=1)]
        np.testing.assert_array_equal(actual_global, expected)
        np.testing.assert_array_equal(patch.rest_vertices_m, full_vertices[ids])
        self.assertEqual(patch.simulation_scale, 2.5)
        core = np.asarray([y * 9 + x for y in range(2, 7) for x in range(2, 7)])
        collar = np.flatnonzero(~np.isin(ids, core))
        self.assertTrue(np.isin(collar, patch.boundary_vertex_ids).all())
        attached = build_local_patches("attached", patch.frame_ids, np.arange(81) + 200,
                                      np.stack((full_vertices, full_vertices)), full_vertices,
                                      full_faces, np.ones(81, dtype=bool), 30, 2.5,
                                      min_scored_vertices=1, core_vertex_ids=core)[0]
        self.assertTrue(np.all(attached.vertex_ids[attached.scored_vertex_ids] - 200 >= 30))
        self.assertTrue(np.isin(np.flatnonzero(attached.vertex_ids - 200 < 30),
                              attached.boundary_vertex_ids).all())

    def test_boundary_buffer_uses_all_frames(self) -> None:
        positions = np.asarray([[[0., 0., 0.], [.02, 0., 0.], [.04, 0., 0.]],
                                [[0., 0., 0.], [.01, 0., 0.], [.04, 0., 0.]]])
        scored = scored_interior(positions, np.asarray([0]), np.asarray([1, 2]), .005)
        np.testing.assert_array_equal(scored, [2])
        diagonal = np.asarray([[[0., 0., 0.], [.012, .012, .012]]])
        self.assertEqual(len(scored_interior(diagonal, np.asarray([0]), np.asarray([1]), .005)), 0)
        around_center = np.asarray([[[-.1, -.1, 0.], [.1, -.1, 0.],
                                     [0., .2, 0.], [0., 0., .005]]])
        scored = scored_interior(around_center, np.asarray([0, 1, 2]), np.asarray([3]),
                                  .005, np.asarray([[0, 1, 2]]))
        self.assertEqual(len(scored), 0)

    def test_connected_selection_and_unsafe_collar_rejection(self) -> None:
        vertices, faces = plane_mesh()
        adjacency = mesh_adjacency(faces, len(vertices))
        ring = expand_vertex_rings(np.asarray([40]), adjacency, 1)
        self.assertGreater(len(ring), 1)
        eligible = np.ones(len(vertices), dtype=bool)
        selected = select_patch_cores(eligible, faces, vertices[None], max_core_vertices=10)[0]
        self.assertEqual(len(selected), 10)
        with self.assertRaises(AssertionError):
            select_patch_cores(eligible, faces, vertices[None], count=2)
        eligible[39] = False
        with self.assertRaises(AssertionError):
            select_patch_cores(eligible, faces, vertices[None], core_vertex_ids=np.asarray([40]))
        with self.assertRaises(AssertionError):
            extract_patch_topology(faces, np.asarray([0, 80]), len(vertices))

    def test_grid_support_checks_interpolated_motion(self) -> None:
        start = np.asarray([[0., 0., 0.], [.002, 0., 0.], [0., .002, 0.],
                            [-.05, 0., 0.]])
        end = start.copy()
        end[3, 0] = .05
        patch = LocalPatch("crossing", np.asarray([11, 12]), np.arange(4),
                           np.stack((start, end)), start, np.asarray([[0, 1, 2]]),
                           np.asarray([0, 1, 2]), np.asarray([3]), 1.)
        np.testing.assert_array_equal(support_separated_vertices(patch, substeps=1), [3])
        self.assertEqual(len(support_separated_vertices(patch, substeps=400)), 0)

    def test_grid_support_includes_prescribed_face_centers(self) -> None:
        positions = np.asarray([[-.1, -.1, 0.], [.1, -.1, 0.], [0., .2, 0.],
                                [0., 0., .001]])
        patch = LocalPatch("face_center", np.asarray([11, 12]), np.arange(4),
                           np.stack((positions, positions)), positions,
                           np.asarray([[0, 1, 2]]), np.asarray([0, 1, 2]), np.asarray([3]), 1.)
        self.assertEqual(len(support_separated_vertices(patch, substeps=1)), 0)

    def test_grid_support_matches_explicit_stencil_intersections(self) -> None:
        patch = synthetic_patch()
        origin, _, _ = local_grid_domain(patch)
        positions = np.asarray((patch.reference_vertices_m - origin) * patch.simulation_scale,
                               dtype=np.float32)
        candidates = np.setdiff1d(np.arange(len(patch.vertex_ids)), patch.boundary_vertex_ids)
        boundary_faces = patch.faces[np.isin(patch.faces, patch.boundary_vertex_ids).all(axis=1)]
        expected = set(map(int, candidates))
        offsets = np.asarray([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
        for frame in positions:
            movers = np.concatenate((frame[patch.boundary_vertex_ids], frame[boundary_faces].mean(axis=1)))
            mover_nodes = np.trunc(movers * np.float32(100.) - np.float32(.5)).astype(np.int32)[:, None] + offsets
            occupied = set(map(tuple, mover_nodes.reshape(-1, 3)))
            for vertex in list(expected):
                base = np.trunc(frame[vertex] * np.float32(100.) - np.float32(.5)).astype(np.int32)
                if any(tuple(node) in occupied for node in base + offsets):
                    expected.remove(vertex)
        np.testing.assert_array_equal(support_separated_vertices(patch, substeps=1), sorted(expected))

    def test_grid_origin_scale_and_roundtrip(self) -> None:
        patch = synthetic_patch()
        spacing = patch.cell_size_m
        origin_coarse, side_coarse, count_coarse = local_grid_domain(patch)
        origin_fine, side_fine, count_fine = local_grid_domain(patch, .5)
        np.testing.assert_array_equal(origin_coarse, origin_fine)
        self.assertEqual(side_coarse, count_coarse * spacing)
        self.assertEqual(side_fine, count_fine * spacing * .5)
        self.assertLessEqual(count_coarse, 209)
        points = patch.reference_vertices_m.reshape(-1, 3)
        self.assertTrue(np.all(points >= origin_coarse + 4 * spacing - 1e-12))
        self.assertTrue(np.all(points <= origin_coarse + side_coarse - 4 * spacing + 1e-12))
        with self.assertRaises(AssertionError):
            local_grid_domain(patch, 1.1)
        with self.assertRaises(AssertionError):
            local_grid_domain(patch, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            metadata = {"subject": 185, "garment": "lower", "simulation_scale": 2.5}
            save_bundle(destination, [patch], metadata)
            recovered, actual_metadata = load_bundle(destination)
            self.assertEqual(actual_metadata, metadata)
            np.testing.assert_array_equal(recovered[0].reference_vertices_m, patch.reference_vertices_m)
            np.testing.assert_array_equal(recovered[0].vertex_ids, patch.vertex_ids)
            self.assertEqual(recovered[0].simulation_scale, patch.simulation_scale)
            self.assertEqual(recovered[0].cell_size_m, spacing)
            with self.assertRaises(AssertionError):
                save_bundle(destination, [patch], metadata)

    def test_original_spacing_is_independent_of_patch_extent(self) -> None:
        patch = synthetic_patch()
        # Patch extent is 0.124 m across both frames; original garment is 0.4 m.
        self.assertAlmostEqual(patch.cell_size_m, 2 * .4 / 200)
        global_limited = replace(patch, simulation_scale=.25)
        self.assertAlmostEqual(global_limited.cell_size_m, 2 * 4.0 / 200)
        moving = patch.reference_vertices_m.copy()
        moving[1, :, 0] += 1.0
        swept = replace(patch, reference_vertices_m=moving)
        self.assertAlmostEqual(swept.cell_size_m, patch.cell_size_m)
        translated = replace(patch, reference_vertices_m=patch.reference_vertices_m + [5., -3., 2.])
        self.assertAlmostEqual(translated.cell_size_m, patch.cell_size_m)

    def test_validation_rejects_bad_frames_topology_and_margin(self) -> None:
        patch = synthetic_patch()
        invalid = [replace(patch, frame_ids=np.asarray([45, 47])),
                   replace(patch, simulation_scale=-1.),
                   replace(patch, faces=np.asarray([[0, 1, len(patch.vertex_ids)]])),
                   replace(patch, scored_vertex_ids=patch.boundary_vertex_ids[:1]),
                   replace(patch, rest_vertices_m=np.full_like(patch.rest_vertices_m, np.nan))]
        for broken in invalid:
            with self.subTest(patch=broken.name), self.assertRaises(AssertionError):
                validate_patch(broken)

    def test_body_sign_and_nonneighbor_cloth_contacts(self) -> None:
        import trimesh

        vertices, faces = plane_mesh(side=3, spacing=.05)
        disconnected = np.concatenate((vertices, vertices + [0., 0., .005]))
        disconnected_faces = np.concatenate((faces, faces + len(vertices)))
        body = trimesh.creation.box(extents=[1., 1., 1.])
        reference = np.stack((disconnected, disconnected))
        with mock_patch("trimesh.proximity.signed_distance", return_value=np.full(34, -.1)):
            mask = contact_free_mask(reference, disconnected_faces, [body, body])
        self.assertFalse(mask.any())
        distances = np.full(17, -.1)
        distances[4] = .1
        with mock_patch("trimesh.proximity.signed_distance", return_value=distances):
            mask = contact_free_mask(np.stack((vertices, vertices)), faces, [body, body])
        self.assertFalse(mask[4])
        self.assertTrue(mask[0])
        centroid_distances = np.full(17, -.1)
        centroid_distances[9] = .01
        frame_masks: list[np.ndarray] = []
        with mock_patch("trimesh.proximity.signed_distance", side_effect=[distances, centroid_distances]):
            mask = contact_free_mask(np.stack((vertices, vertices)), faces, [body, body],
                                      frame_masks=frame_masks)
        self.assertFalse(mask[faces[0]].any())
        np.testing.assert_array_equal(mask, np.logical_and.reduce(frame_masks))
        self.assertFalse(frame_masks[0][4])
        self.assertTrue(frame_masks[1][4])
        body.faces = body.faces[:-1]
        with self.assertRaises(AssertionError):
            contact_free_mask(np.stack((vertices, vertices)), faces, [body, body])

    def test_upper_triangle_clearance_uses_surface_not_vertices(self) -> None:
        import trimesh

        vertices, faces = plane_mesh(side=3, spacing=.05)
        vertices += [0., 0., .005]
        upper = trimesh.Trimesh(vertices=[[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]],
                                faces=[[0, 1, 2]], process=False)
        self.assertGreater(np.linalg.norm(vertices[:, None] - upper.vertices[None], axis=2).min(), .5)
        body = trimesh.creation.box()
        reference = np.stack((vertices, vertices))
        # Single-face broad-phase fixture; closest_point still computes actual
        # triangle projections, while the unit test does not require Rtree.
        with mock_patch("trimesh.proximity.signed_distance", return_value=np.full(17, -.1)), \
                mock_patch("trimesh.proximity.nearby_faces", return_value=[[0]] * 17):
            mask = contact_free_mask(reference, faces, [body, body],
                                      external_surface_meshes=[upper, upper])
        self.assertFalse(mask.any())

    def test_tracking_complement_keeps_upper_cloth_and_skin(self) -> None:
        vertices, lower_faces = plane_mesh(side=3)
        other_faces = np.asarray([[9, 10, 11], [12, 13, 14]])
        full_faces = np.concatenate((lower_faces, other_faces))
        lower_face_ids = np.arange(len(lower_faces))
        external = tracking_surface_complement(full_faces, np.arange(len(vertices)),
                                                lower_faces, lower_face_ids)
        np.testing.assert_array_equal(external, other_faces)
        with self.assertRaises(AssertionError):
            tracking_surface_complement(full_faces, np.arange(len(vertices)),
                                        lower_faces, lower_face_ids + 1)


if __name__ == "__main__":
    unittest.main()
