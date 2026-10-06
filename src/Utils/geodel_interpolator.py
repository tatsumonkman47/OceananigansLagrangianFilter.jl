"""Optional GeoDel-backed 2D and 3D linear interpolation for the Julia package.

GeoDel supplies simplices and facet adjacency. This module supplies the rest
of the LinearNDInterpolator-shaped operation: target location, reusable
barycentric weights, vector-valued evaluation, and outside-hull fill values.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


NO_CELL = -1


class _GeoDelInterpolator:
    """Construct and apply a reusable linear interpolation map."""

    dimension = None

    def __init__(
        self,
        points,
        targets,
        *,
        parallel=True,
        nb_threads=0,
        tolerance=1e-11,
        max_steps=512,
    ):
        try:
            import geodel
        except ImportError as exc:
            raise ImportError(
                "The GeoDel backend requires the optional geodel package. "
                "Install it for PythonCall's interpreter or set "
                "LF_GEODEL_PYTHONPATH to its installation directory."
            ) from exc

        points = np.ascontiguousarray(points, dtype=np.float64)
        targets = np.ascontiguousarray(targets, dtype=np.float64)
        dimension = self.dimension
        if points.ndim != 2 or points.shape[1] != dimension:
            raise ValueError(f"points must have shape (N, {dimension})")
        if targets.ndim != 2 or targets.shape[1] != dimension:
            raise ValueError(f"targets must have shape (M, {dimension})")
        if not np.isfinite(points).all() or not np.isfinite(targets).all():
            raise ValueError("points and targets must be finite")

        if dimension == 2:
            triangulation = geodel.Triangulation2D(points)
        else:
            triangulation = geodel.Triangulation(
                points, parallel=parallel, nb_threads=nb_threads
            )
        cells = np.asarray(triangulation.cells).astype(np.int64, copy=False)
        neighbor_indices = np.asarray(triangulation.neighbors)
        neighbors = neighbor_indices.astype(np.int64)
        neighbors[neighbor_indices == geodel.NO_INDEX] = NO_CELL

        simplices = points[cells]
        origins = simplices[:, dimension]
        bases = np.transpose(
            simplices[:, :dimension] - origins[:, None, :], (0, 2, 1)
        )
        inverse_bases = np.linalg.inv(bases)

        # Duplicate input points can be absent from the tessellation.  Seed from
        # the nearest vertex that GeoDel actually retained.
        used_vertices = np.unique(cells)
        vertex_seed = np.full(points.shape[0], NO_CELL, dtype=np.int64)
        repeated_cells = np.repeat(
            np.arange(cells.shape[0], dtype=np.int64), dimension + 1
        )
        vertex_seed[cells.ravel()] = repeated_cells
        seed_tree = cKDTree(points[used_vertices])
        query_workers = nb_threads if nb_threads > 0 else -1
        nearest_used = used_vertices[seed_tree.query(targets, workers=query_workers)[1]]
        initial_cells = vertex_seed[nearest_used]

        vertices, weights, outside, unresolved = self._walk_targets(
            targets,
            cells,
            neighbors,
            origins,
            inverse_bases,
            initial_cells,
            tolerance=tolerance,
            max_steps=max_steps,
        )
        if unresolved.any():
            raise RuntimeError(
                f"GeoDel point location did not converge for {unresolved.sum()} "
                f"of {targets.shape[0]} targets within {max_steps} steps"
            )

        self.points = points
        self.targets = targets
        self.triangulation = triangulation
        self.cells = cells
        self.neighbors = neighbors
        self.vertices = vertices
        self.weights = weights
        self.outside = outside

    @staticmethod
    def _walk_targets(
        targets,
        cells,
        neighbors,
        origins,
        inverse_bases,
        initial_cells,
        *,
        tolerance,
        max_steps,
    ):
        count = targets.shape[0]
        simplex_size = cells.shape[1]
        vertices = np.zeros((count, simplex_size), dtype=np.int64)
        weights = np.full((count, simplex_size), np.nan, dtype=np.float64)
        outside = np.zeros(count, dtype=bool)
        unresolved = np.zeros(count, dtype=bool)
        current = initial_cells.copy()
        active = np.arange(count, dtype=np.int64)

        for _ in range(max_steps):
            if active.size == 0:
                break
            active_cells = current[active]
            delta = targets[active] - origins[active_cells]
            first_weights = np.einsum(
                "mij,mj->mi", inverse_bases[active_cells], delta
            )
            active_weights = np.column_stack(
                (first_weights, 1.0 - first_weights.sum(axis=1))
            )
            inside = np.min(active_weights, axis=1) >= -tolerance

            found = active[inside]
            if found.size:
                vertices[found] = cells[active_cells[inside]]
                weights[found] = active_weights[inside]

            remaining = ~inside
            if not remaining.any():
                active = active[:0]
                break

            remaining_targets = active[remaining]
            remaining_cells = active_cells[remaining]
            exit_facets = np.argmin(active_weights[remaining], axis=1)
            next_cells = neighbors[remaining_cells, exit_facets]
            hit_hull = next_cells == NO_CELL
            outside[remaining_targets[hit_hull]] = True

            continuing = ~hit_hull
            active = remaining_targets[continuing]
            current[active] = next_cells[continuing]
        else:
            unresolved[active] = True

        return vertices, weights, outside, unresolved

    def __call__(self, values, fill_value=np.nan):
        values = np.asarray(values)
        scalar = values.ndim == 1
        if scalar:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != self.points.shape[0]:
            raise ValueError("values must have shape (N,) or (N, K)")

        result = np.full(
            (self.targets.shape[0], values.shape[1]),
            fill_value,
            dtype=np.result_type(values.dtype, self.weights.dtype),
        )
        valid = ~self.outside
        gathered = values[self.vertices[valid]]
        result[valid] = np.einsum("mi,mik->mk", self.weights[valid], gathered)
        return result[:, 0] if scalar else result


class GeoDelInterpolator2D(_GeoDelInterpolator):
    """Construct and apply a reusable 2D linear interpolation map."""

    dimension = 2


class GeoDelInterpolator3D(_GeoDelInterpolator):
    """Construct and apply a reusable 3D linear interpolation map."""

    dimension = 3
