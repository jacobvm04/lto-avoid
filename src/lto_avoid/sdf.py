"""Signed distance field computation from occupancy grids."""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped
from scipy.ndimage import distance_transform_edt, map_coordinates

from lto_avoid.grid import Grid, world_to_grid
from lto_avoid.types import Points2D, SDFArray


@jaxtyped(typechecker=beartype)
def compute_sdf(grid: Grid) -> SDFArray:
    """Compute a signed distance field from an occupancy grid.

    Uses two passes of scipy's Euclidean distance transform.

    Returns:
        (H, W) float64 array. Positive = free space (distance to nearest
        obstacle), negative = inside obstacle (distance to nearest free
        cell, negated). Zero ≈ obstacle boundary. Values are in meters.
    """
    occupied = grid.obstacles
    dist_free = distance_transform_edt(~occupied, sampling=grid.resolution)
    dist_occ = distance_transform_edt(occupied, sampling=grid.resolution)
    return dist_free - dist_occ


@jaxtyped(typechecker=beartype)
def sample_sdf(
    sdf: SDFArray,
    grid: Grid,
    points: Points2D,
) -> Float[np.ndarray, " n"]:
    """Sample SDF values at world-coordinate points via bilinear interpolation.

    This is a numpy/scipy utility for tests and visualization.
    The CasADi optimizer uses its own interpolant instead.

    Args:
        sdf: (H, W) signed distance field.
        grid: Grid metadata for coordinate conversion.
        points: (N, 2) world coordinates.

    Returns:
        (N,) interpolated SDF values.
    """
    points = np.asarray(points, dtype=np.float64)
    rc = world_to_grid(grid, points)  # (N, 2) as (row, col)
    # map_coordinates expects coordinates as (row_coords, col_coords)
    coords = [rc[:, 0], rc[:, 1]]
    return map_coordinates(sdf, coords, order=1, mode="nearest")
