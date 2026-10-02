"""Native, constant-topology lower-garment patches in world meters.

Contact checks sample garment vertices and triangle centroids in every reference frame. They do not
certify continuous triangle/triangle separation between samples or frames.
"""

from collections import deque
from dataclasses import dataclass
from importlib.util import find_spec
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class LocalPatch:
    name: str
    frame_ids: np.ndarray
    vertex_ids: np.ndarray
    reference_vertices_m: np.ndarray
    rest_vertices_m: np.ndarray
    faces: np.ndarray
    boundary_vertex_ids: np.ndarray
    scored_vertex_ids: np.ndarray
    simulation_scale: float

    @property
    def cell_size_m(self) -> float:
        """Original 200-node MPMAvatar spacing in its twice-garment-size cube."""
        assert self.simulation_scale > 0 and np.isfinite(self.simulation_scale)
        return 2.0 / (200.0 * self.simulation_scale)


def mesh_adjacency(faces: np.ndarray, vertex_count: int) -> list[set[int]]:
    """Construct sparse vertex connectivity without a dense distance matrix."""
    assert faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
    assert np.issubdtype(faces.dtype, np.integer)
    assert faces.min() >= 0 and faces.max() < vertex_count
    adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        adjacency[int(a)].update((int(b), int(c)))
        adjacency[int(b)].update((int(a), int(c)))
        adjacency[int(c)].update((int(a), int(b)))
    return adjacency


def expand_vertex_rings(vertex_ids: np.ndarray, adjacency: list[set[int]],
                        rings: int) -> np.ndarray:
    assert rings >= 0
    selected = set(map(int, vertex_ids))
    frontier = selected.copy()
    for _ in range(rings):
        frontier = {neighbor for vertex in frontier for neighbor in adjacency[vertex]} - selected
        selected.update(frontier)
    return np.asarray(sorted(selected), dtype=np.int64)


def connected_components(mask: np.ndarray, adjacency: list[set[int]]) -> list[np.ndarray]:
    assert mask.shape == (len(adjacency),) and mask.dtype == bool
    remaining = set(map(int, np.flatnonzero(mask)))
    components: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = deque([seed])
        component = [seed]
        while queue:
            vertex = queue.popleft()
            neighbors = sorted(adjacency[vertex] & remaining)
            remaining.difference_update(neighbors)
            queue.extend(neighbors)
            component.extend(neighbors)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return sorted(components, key=lambda ids: (-len(ids), int(ids[0])))


def extract_patch_topology(faces: np.ndarray, vertex_ids: np.ndarray,
                           vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep original triangles whose three vertices are selected; return local boundary."""
    assert len(vertex_ids) > 0 and np.unique(vertex_ids).size == len(vertex_ids)
    assert vertex_ids.min() >= 0 and vertex_ids.max() < vertex_count
    mapping = np.full(vertex_count, -1, dtype=np.int64)
    mapping[vertex_ids] = np.arange(len(vertex_ids))
    local_faces = mapping[faces[np.isin(faces, vertex_ids).all(axis=1)]]
    assert len(local_faces) > 0, "Selected patch contains no complete native triangles"
    assert np.unique(local_faces).size == len(vertex_ids), "Selected patch has isolated vertices"
    edges = np.sort(np.concatenate((local_faces[:, [0, 1]], local_faces[:, [1, 2]],
                                    local_faces[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts <= 2), "Patch topology must be manifold"
    boundary = np.unique(unique_edges[counts == 1])
    assert len(boundary) > 0, "A local patch requires a prescribed open boundary"
    return local_faces, boundary


def close_smplx_mouth(body: Any) -> Any:
    """Cap the native SMPL-X 32-edge mouth opening for signed-clearance queries."""
    import trimesh

    assert body.is_winding_consistent and body.volume > 0
    directed = np.concatenate((body.faces[:, [0, 1]], body.faces[:, [1, 2]],
                               body.faces[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    assert np.all(counts <= 2), "SMPL-X clearance body has nonmanifold edges"
    boundary = directed[counts[inverse] == 1]
    assert len(boundary) == 32, "Expected the native SMPL-X 32-edge mouth opening"
    mouth_ids = np.unique(boundary)
    assert len(mouth_ids) == 32
    assert np.unique(boundary[:, 0]).size == np.unique(boundary[:, 1]).size == 32
    following = {int(a): int(b) for a, b in boundary}
    ordered = [int(mouth_ids[0])]
    for _ in range(31):
        ordered.append(following[ordered[-1]])
    assert len(set(ordered)) == 32 and following[ordered[-1]] == ordered[0], "Mouth must be a single boundary loop"
    center_id = len(body.vertices)
    vertices = np.concatenate((body.vertices, body.vertices[mouth_ids].mean(axis=0, keepdims=True)))
    cap = np.column_stack((boundary[:, 1], boundary[:, 0], np.full(32, center_id)))
    closed = trimesh.Trimesh(vertices=vertices, faces=np.concatenate((body.faces, cap)), process=False)
    assert closed.is_watertight and closed.is_winding_consistent and closed.volume > 0
    return closed


def contact_free_mask(vertices_m: np.ndarray, faces: np.ndarray,
                      body_meshes: Sequence[Any], clearance_m: float = 0.02,
                      topology_exclusion_rings: int = 2,
                      external_surface_meshes: Sequence[Any] | None = None,
                      frame_masks: list[np.ndarray] | None = None) -> np.ndarray:
    """Require body/external-surface sample clearance and lower nonneighbor vertex clearance."""
    import trimesh
    from scipy.spatial import cKDTree

    assert clearance_m >= 0.02 and topology_exclusion_rings >= 1
    assert vertices_m.ndim == 3 and vertices_m.shape[2] == 3
    assert len(body_meshes) == len(vertices_m) and np.isfinite(vertices_m).all()
    if external_surface_meshes is not None:
        assert len(external_surface_meshes) == len(vertices_m)
    adjacency = mesh_adjacency(faces, vertices_m.shape[1])
    eligible = np.ones(vertices_m.shape[1], dtype=bool)
    neighbors = [set(map(int, expand_vertex_rings(np.asarray([i]), adjacency,
                                                topology_exclusion_rings)))
                 for i in range(len(adjacency))]
    for frame_index, (positions, body) in enumerate(zip(vertices_m, body_meshes)):
        assert body.is_watertight and body.is_winding_consistent, "Body must be watertight and consistently oriented"
        assert body.volume > 0, "Body must have outward orientation for signed clearance"
        samples = np.concatenate((positions, positions[faces].mean(axis=1)))
        signed_distance = trimesh.proximity.signed_distance(body, samples)
        assert np.isfinite(signed_distance).all()
        safe_samples = signed_distance <= -clearance_m
        if external_surface_meshes is not None:
            external_surface = external_surface_meshes[frame_index]
            assert len(external_surface.faces) > 0 and np.isfinite(external_surface.vertices).all()
            _, distances, _ = trimesh.proximity.closest_point(external_surface, samples)
            assert np.isfinite(distances).all()
            safe_samples &= distances >= clearance_m
        frame_eligible = safe_samples[:len(positions)].copy()
        frame_eligible[np.unique(faces[~safe_samples[len(positions):]])] = False
        pairs = cKDTree(positions).query_pairs(clearance_m, output_type="ndarray")
        for a, b in pairs:
            if int(b) not in neighbors[int(a)]:
                frame_eligible[int(a)] = False
                frame_eligible[int(b)] = False
        eligible &= frame_eligible
        if frame_masks is not None:
            frame_masks.append(frame_eligible)
    return eligible


def tracking_surface_complement(tracking_faces: np.ndarray, cloth_vertex_ids: np.ndarray,
                                 cloth_faces: np.ndarray, cloth_face_ids: np.ndarray) -> np.ndarray:
    """Extract every non-lower tracking face, including upper garment and tracked skin."""
    assert tracking_faces.ndim == 2 and tracking_faces.shape[1] == 3
    assert np.issubdtype(tracking_faces.dtype, np.integer) and tracking_faces.min() >= 0
    assert cloth_face_ids.shape == (len(cloth_faces),) and len(cloth_face_ids) > 0
    assert np.issubdtype(cloth_face_ids.dtype, np.integer)
    assert np.unique(cloth_face_ids).size == len(cloth_face_ids)
    assert cloth_face_ids.min() >= 0 and cloth_face_ids.max() < len(tracking_faces)
    assert np.array_equal(tracking_faces[cloth_face_ids], cloth_vertex_ids[cloth_faces]), "Native lower face IDs must match lower triangles"
    external_mask = np.ones(len(tracking_faces), dtype=bool)
    external_mask[cloth_face_ids] = False
    assert external_mask.any(), "Full tracked surface must contain non-lower triangles"
    return tracking_faces[external_mask]


def select_patch_cores(eligible: np.ndarray, faces: np.ndarray, vertices_m: np.ndarray,
                       count: int = 1, max_core_vertices: int = 0, collar_rings: int = 1,
                       core_vertex_ids: np.ndarray | None = None) -> list[np.ndarray]:
    """Select connected cores whose entire prescribed collar is contact free."""
    assert count >= 0 and max_core_vertices >= 0 and collar_rings >= 1
    adjacency = mesh_adjacency(faces, len(eligible))
    core_safe = eligible.copy()
    for vertex in np.flatnonzero(eligible):
        ring = expand_vertex_rings(np.asarray([vertex]), adjacency, collar_rings)
        core_safe[vertex] = eligible[ring].all()
    if core_vertex_ids is not None:
        assert count == 1, "Explicit core describes exactly one patch per window"
        assert core_vertex_ids.ndim == 1 and len(core_vertex_ids) > 0
        assert np.unique(core_vertex_ids).size == len(core_vertex_ids)
        assert core_vertex_ids.min() >= 0 and core_vertex_ids.max() < len(eligible)
        assert core_safe[core_vertex_ids].all(), "Explicit core/collar fails clearance"
        mask = np.zeros_like(eligible)
        mask[core_vertex_ids] = True
        assert len(connected_components(mask, adjacency)) == 1, "Explicit core must be connected"
        return [np.sort(core_vertex_ids)]
    components = connected_components(core_safe, adjacency)
    assert len(components) >= count, "Requested contact-free connected patches are absent"
    cores: list[np.ndarray] = []
    for component in (components[:count] if count else components):
        if max_core_vertices > 0 and len(component) > max_core_vertices:
            allowed = set(map(int, component))
            center = vertices_m[0, component].mean(axis=0)
            seed = int(component[np.argmin(np.linalg.norm(vertices_m[0, component] - center, axis=1))])
            queue = deque([seed])
            visited = {seed}
            chosen: list[int] = []
            while queue and len(chosen) < max_core_vertices:
                vertex = queue.popleft()
                chosen.append(vertex)
                neighbors = sorted((adjacency[vertex] & allowed) - visited)
                visited.update(neighbors)
                queue.extend(neighbors)
            component = np.asarray(sorted(chosen), dtype=np.int64)
        cores.append(component)
    return cores


def scored_interior(vertices_m: np.ndarray, boundary_ids: np.ndarray,
                     candidates: np.ndarray, cell_size_m: float,
                     faces: np.ndarray | None = None) -> np.ndarray:
    """Exclude candidates approaching prescribed vertices or face-center particles."""
    from scipy.spatial import cKDTree

    assert cell_size_m > 0 and len(boundary_ids) > 0
    selected = np.ones(len(candidates), dtype=bool)
    for positions in vertices_m:
        prescribed = positions[boundary_ids]
        if faces is not None:
            boundary_faces = faces[np.isin(faces, boundary_ids).all(axis=1)]
            prescribed = np.concatenate((prescribed, positions[boundary_faces].mean(axis=1)))
        distances, _ = cKDTree(prescribed).query(positions[candidates], p=np.inf)
        selected &= distances >= 3.0 * cell_size_m
    return candidates[selected]


def build_local_patches(name_prefix: str, frame_ids: np.ndarray, vertex_ids: np.ndarray,
                        reference_vertices_m: np.ndarray, rest_vertices_m: np.ndarray,
                        faces: np.ndarray, eligible: np.ndarray, attachment_vertex_count: int,
                        simulation_scale: float, *, count: int = 1, max_core_vertices: int = 0,
                        min_scored_vertices: int = 16,
                        collar_rings: int = 1,
                        core_vertex_ids: np.ndarray | None = None) -> list[LocalPatch]:
    """Build patches from lower-local arrays; vertex_ids maps them to tracking IDs."""
    assert 0 <= attachment_vertex_count <= len(vertex_ids) and min_scored_vertices >= 0
    adjacency = mesh_adjacency(faces, len(vertex_ids))
    cores = select_patch_cores(eligible, faces, reference_vertices_m, count,
                               max_core_vertices, collar_rings, core_vertex_ids)
    patches: list[LocalPatch] = []
    for index, core in enumerate(cores):
        patch_ids = expand_vertex_rings(core, adjacency, collar_rings)
        assert eligible[patch_ids].all(), "Core and collar must both satisfy clearance"
        patch_faces, perimeter = extract_patch_topology(faces, patch_ids, len(vertex_ids))
        collar = np.flatnonzero(~np.isin(patch_ids, core) | (patch_ids < attachment_vertex_count))
        boundary = np.union1d(perimeter, collar)
        local_core = np.flatnonzero(np.isin(patch_ids, core) & (patch_ids >= attachment_vertex_count))
        patch = LocalPatch(f"{name_prefix}_{index:03d}", frame_ids.copy(), vertex_ids[patch_ids],
                           reference_vertices_m[:, patch_ids], rest_vertices_m[patch_ids],
                           patch_faces, boundary, local_core, simulation_scale)
        patch.scored_vertex_ids = scored_interior(patch.reference_vertices_m, boundary,
                                                  local_core, patch.cell_size_m, patch_faces)
        if min_scored_vertices > 0:
            assert len(patch.scored_vertex_ids) >= min_scored_vertices, "Patch lacks requested scored interior after grid-boundary exclusion"
            validate_patch(patch)
        patches.append(patch)
    return patches


def validate_patch(patch: LocalPatch) -> None:
    assert patch.name and patch.simulation_scale > 0 and np.isfinite(patch.simulation_scale)
    assert patch.frame_ids.ndim == 1 and len(patch.frame_ids) >= 2
    assert np.issubdtype(patch.frame_ids.dtype, np.integer)
    assert patch.frame_ids[0] >= 0 and np.all(np.diff(patch.frame_ids) == 1)
    vertex_count = len(patch.vertex_ids)
    assert patch.vertex_ids.shape == (vertex_count,) and vertex_count >= 3
    assert np.issubdtype(patch.vertex_ids.dtype, np.integer)
    assert patch.vertex_ids.min() >= 0 and np.unique(patch.vertex_ids).size == vertex_count
    assert patch.reference_vertices_m.shape == (len(patch.frame_ids), vertex_count, 3)
    assert patch.rest_vertices_m.shape == (vertex_count, 3)
    assert np.isfinite(patch.reference_vertices_m).all() and np.isfinite(patch.rest_vertices_m).all()
    for ids in (patch.boundary_vertex_ids, patch.scored_vertex_ids):
        assert ids.ndim == 1 and len(ids) > 0 and np.issubdtype(ids.dtype, np.integer)
        assert ids.min() >= 0 and ids.max() < vertex_count and np.unique(ids).size == len(ids)
    adjacency = mesh_adjacency(patch.faces, vertex_count)
    assert len(connected_components(np.ones(vertex_count, dtype=bool), adjacency)) == 1
    _, boundary = extract_patch_topology(patch.faces, np.arange(vertex_count), vertex_count)
    assert np.isin(boundary, patch.boundary_vertex_ids).all(), "Every cut edge must be prescribed"
    assert not np.isin(patch.scored_vertex_ids, patch.boundary_vertex_ids).any()
    assert np.array_equal(scored_interior(patch.reference_vertices_m, patch.boundary_vertex_ids,
                                          patch.scored_vertex_ids, patch.cell_size_m, patch.faces),
                          patch.scored_vertex_ids), "Scored vertices violate 3-cell boundary margin"
    for positions in (patch.rest_vertices_m, *patch.reference_vertices_m):
        triangles = positions[patch.faces]
        assert np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                              triangles[:, 2] - triangles[:, 0]), axis=1) > 0), "Degenerate native triangle"


def local_grid_domain(patch: LocalPatch, cell_size_factor: float = 1.0,
                       padding_cells: int = 4) -> tuple[np.ndarray, float, int]:
    """Return fixed world origin, cube side, and cell count, with no crop rescaling.

    Padding uses the original spacing, keeping the origin fixed during
    convergence refinements and H updates. Rounding adds less than one cell.
    """
    assert 0 < cell_size_factor <= 1.0 and padding_cells >= 4
    positions = patch.reference_vertices_m.reshape(-1, 3)
    assert len(positions) > 0 and np.isfinite(positions).all()
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    base_cell_size_m = patch.cell_size_m
    cell_size_m = base_cell_size_m * cell_size_factor
    base_side = float((upper - lower).max() + 2 * padding_cells * base_cell_size_m)
    origin = (upper + lower) / 2 - base_side / 2
    n_grid = int(np.ceil(base_side / cell_size_m))
    return origin, float(n_grid * cell_size_m), n_grid


def save_bundle(path: Path, patches: Sequence[LocalPatch], metadata: dict[str, Any]) -> None:
    assert not path.exists(), "Bundle output must be fresh"
    assert len(patches) > 0 and len({patch.name for patch in patches}) == len(patches)
    for patch in patches:
        validate_patch(patch)
    path.mkdir(parents=True)
    files: list[str] = []
    for index, patch in enumerate(patches):
        filename = f"patch_{index:03d}.npz"
        np.savez_compressed(path / filename, name=np.asarray(patch.name), frame_ids=patch.frame_ids,
                            vertex_ids=patch.vertex_ids, reference_vertices_m=patch.reference_vertices_m,
                            rest_vertices_m=patch.rest_vertices_m, faces=patch.faces,
                            boundary_vertex_ids=patch.boundary_vertex_ids,
                            scored_vertex_ids=patch.scored_vertex_ids,
                            simulation_scale=np.asarray(patch.simulation_scale))
        files.append(filename)
    manifest = {"metadata": metadata, "patch_files": files}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_bundle(path: Path) -> tuple[list[LocalPatch], dict[str, Any]]:
    assert (path / "manifest.json").is_file()
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(manifest["patch_files"]) > 0
    patches: list[LocalPatch] = []
    for filename in manifest["patch_files"]:
        assert Path(filename).name == filename and (path / filename).is_file()
        with np.load(path / filename, allow_pickle=False) as data:
            patch = LocalPatch(str(data["name"].item()), data["frame_ids"], data["vertex_ids"],
                               data["reference_vertices_m"], data["rest_vertices_m"], data["faces"],
                               data["boundary_vertex_ids"], data["scored_vertex_ids"],
                               float(data["simulation_scale"]))
        validate_patch(patch)
        patches.append(patch)
    assert len({patch.name for patch in patches}) == len(patches)
    return patches, manifest["metadata"]


def prepare_native_patches(prepared_dir: Path, body_root: Path,
                           windows: Sequence[tuple[int, int]] = ((45, 50), (51, 56)), *,
                           count: int = 1, max_core_vertices: int = 0,
                           min_scored_vertices: int = 16,
                           clearance_m: float = 0.02, collar_rings: int = 1,
                           topology_exclusion_rings: int = 2,
                           core_vertex_ids: Sequence[int] | None = None,
                           clearance_cache: Path | None = None) -> tuple[list[LocalPatch], dict[str, Any]]:
    """Read native assets without producing preparation outputs or changing meshes."""
    assert find_spec("rtree") is not None, "Native patch preparation requires rtree; install MPMAvatar/requirements-local-patches.txt in the mpmavatar environment"
    import trimesh

    assert prepared_dir.is_dir() and (body_root / "SMPLX").is_dir()
    assert len(windows) > 0 and all(11 <= start < end <= 110 for start, end in windows)
    assert clearance_cache is None or len(windows) == 1
    tracking = prepared_dir / "output/tracking/s185_t1_11_100"
    offsets_path = prepared_dir / "model/s185_t1/point_cloud/timestep_030000/verts_offset.npy"
    split_path = prepared_dir / "data/s185_t1/split_idx_lower.npz"
    assert tracking.is_dir() and offsets_path.is_file() and split_path.is_file()
    offsets = np.load(offsets_path, allow_pickle=False, mmap_mode="r")
    assert offsets.ndim == 3 and offsets.shape[0] == 100 and offsets.shape[2] == 3
    with np.load(split_path, allow_pickle=False) as split:
        vertex_ids = split["reordered_cloth_v_idx"].astype(np.int64)
        faces = split["new_cloth_faces"].astype(np.int64)
        cloth_face_ids = split["reordered_cloth_f_idx"].astype(np.int64)
        attachment_count = int(split["num_joint_v"])
        joint_face_count = int(split["num_joint_f"])
    mesh_adjacency(faces, len(vertex_ids))
    assert 0 <= attachment_count <= len(vertex_ids) and 0 <= joint_face_count <= len(faces)
    assert np.unique(vertex_ids).size == len(vertex_ids)
    assert vertex_ids.min() >= 0 and vertex_ids.max() < offsets.shape[1]
    requested_frames = sorted({11, 45} | {frame for start, end in windows for frame in range(start, end + 1)})
    positions: dict[int, np.ndarray] = {}
    tracking_positions: dict[int, np.ndarray] = {}
    tracking_faces: np.ndarray | None = None
    for frame in requested_frames:
        param_path = tracking / f"params_{frame}.npz"
        assert param_path.is_file(), f"Missing native tracking frame {frame}"
        with np.load(param_path, allow_pickle=False) as data:
            vertices = data["vertices"]
            current_faces = data["faces"]
            assert vertices.shape == offsets.shape[1:]
            assert np.isfinite(vertices).all() and np.isfinite(offsets[frame - 11]).all()
            if frame == requested_frames[0]:
                tracking_faces = current_faces.copy()
            else:
                assert np.array_equal(current_faces, tracking_faces), "Native tracking topology changed"
            full_positions = np.asarray(vertices + offsets[frame - 11], dtype=np.float64)
            positions[frame] = full_positions[vertex_ids]
            tracking_positions[frame] = full_positions
    assert tracking_faces is not None
    external_faces = tracking_surface_complement(tracking_faces, vertex_ids, faces, cloth_face_ids)
    extent = float(np.ptp(positions[45], axis=0).max())
    assert extent > 0
    scale = 1.0 / extent
    explicit_core: np.ndarray | None = None
    if core_vertex_ids is not None:
        supplied = np.asarray(core_vertex_ids, dtype=np.int64)
        assert len(supplied) > 0 and np.isin(supplied, vertex_ids).all()
        assert np.unique(supplied).size == len(supplied)
        explicit_core = np.asarray([int(np.flatnonzero(vertex_ids == vertex)[0]) for vertex in supplied])
    patches: list[LocalPatch] = []
    for start, end in windows:
        frame_ids = np.arange(start, end + 1, dtype=np.int64)
        reference = np.stack([positions[int(frame)] for frame in frame_ids])
        bodies: list[Any] = []
        for frame in frame_ids:
            body_path = body_root / "SMPLX" / f"mesh-f{frame:05d}_smplx.ply"
            assert body_path.is_file()
            body = trimesh.load(body_path, force="mesh", process=False)
            assert isinstance(body, trimesh.Trimesh)
            mouth_ids = np.unique(body.edges[np.bincount(body.edges_unique_inverse)[body.edges_unique_inverse] == 1])
            assert body.vertices[mouth_ids, 1].min() > reference[:, :, 1].max() + clearance_m, "Mouth closure must lie above the complete lower garment"
            bodies.append(close_smplx_mouth(body))
        external_meshes = [trimesh.Trimesh(vertices=tracking_positions[int(frame)], faces=external_faces,
                                         process=False) for frame in frame_ids]
        frame_masks: list[np.ndarray] = []
        eligible = contact_free_mask(reference, faces, bodies, clearance_m,
                                      topology_exclusion_rings, external_meshes, frame_masks)
        if clearance_cache is not None:
            assert not clearance_cache.exists(), clearance_cache
            np.savez_compressed(clearance_cache, frame_ids=frame_ids,
                                reference_vertices_m=reference, rest_vertices_m=positions[11],
                                vertex_ids=vertex_ids, faces=faces, frame_masks=np.asarray(frame_masks),
                                attachment_vertex_count=attachment_count, simulation_scale=scale,
                                clearance_m=clearance_m, collar_rings=collar_rings)
        patches.extend(build_local_patches(f"s185_lower_{start}_{end}", frame_ids, vertex_ids,
                                           reference, positions[11], faces, eligible, attachment_count,
                                           scale, count=count, max_core_vertices=max_core_vertices,
                                           min_scored_vertices=min_scored_vertices,
                                           collar_rings=collar_rings,
                                           core_vertex_ids=explicit_core))
    metadata = {"subject": 185, "garment": "lower", "take": 1, "rest_frame": 11,
                "scale_frame": 45, "windows": [list(window) for window in windows],
                "simulation_scale": scale, "global_bbox_extent_m": extent,
                "grid_spacing_rule": "2 * original_global_bbox_extent_m / 200 (native MPMAvatar)",
                "clearance_m": clearance_m, "collar_rings": collar_rings,
                "body_clearance_geometry": "native SMPL-X with its single 32-edge mouth opening capped in memory; cap above the full lower garment",
                "topology_exclusion_rings": topology_exclusion_rings,
                "contact_sampling": "lower vertices and face centroids against body SDF and every non-lower tracking triangle (upper cloth and tracked skin) in every window frame; collar included",
                "self_contact_sampling": "vertex pairs only; graph-neighbor rings excluded",
                "score_exclusion": "Chebyshev distance >=3*patch.cell_size_m from prescribed vertices and fully prescribed face centroids in every window frame",
                "prepared_dir": str(prepared_dir.resolve()), "body_root": str(body_root.resolve()),
                "tracking_dir": str(tracking.resolve()), "offsets_path": str(offsets_path.resolve()),
                "split_path": str(split_path.resolve())}
    return patches, metadata
