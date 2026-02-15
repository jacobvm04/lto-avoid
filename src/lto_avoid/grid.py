"""Occupancy grid representation and coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Immutable 2D occupancy grid.

    Attributes:
        obstacles: (H, W) boolean array. True = occupied cell.
        resolution: Meters per cell edge.
        origin: (2,) array — world (x, y) of the grid[0, 0] corner.

    Coordinate convention:
        obstacles[row, col] where row ↔ y, col ↔ x.
        Row 0 is at origin[1] (minimum y).
    """

    obstacles: np.ndarray
    resolution: float
    origin: np.ndarray

    @property
    def height(self) -> int:
        return self.obstacles.shape[0]

    @property
    def width(self) -> int:
        return self.obstacles.shape[1]

    @property
    def world_extent(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) in world coordinates."""
        x_min = float(self.origin[0])
        y_min = float(self.origin[1])
        x_max = x_min + self.width * self.resolution
        y_max = y_min + self.height * self.resolution
        return (x_min, x_max, y_min, y_max)


def make_empty_grid(
    width: int,
    height: int,
    resolution: float,
    origin: np.ndarray | tuple[float, float] = (0.0, 0.0),
) -> Grid:
    """Create an all-free occupancy grid.

    Args:
        width: Number of columns.
        height: Number of rows.
        resolution: Meters per cell.
        origin: World (x, y) of grid[0, 0].
    """
    return Grid(
        obstacles=np.zeros((height, width), dtype=bool),
        resolution=resolution,
        origin=np.asarray(origin, dtype=np.float64),
    )


def world_to_grid(grid: Grid, xy: np.ndarray) -> np.ndarray:
    """Convert world (x, y) to continuous grid (row, col).

    Args:
        grid: The occupancy grid.
        xy: (..., 2) world coordinates.

    Returns:
        (..., 2) array of (row, col) as floats.
    """
    xy = np.asarray(xy, dtype=np.float64)
    col = (xy[..., 0] - grid.origin[0]) / grid.resolution
    row = (xy[..., 1] - grid.origin[1]) / grid.resolution
    return np.stack([row, col], axis=-1)


def grid_to_world(grid: Grid, rc: np.ndarray) -> np.ndarray:
    """Convert grid (row, col) to world (x, y).

    Args:
        grid: The occupancy grid.
        rc: (..., 2) grid indices (row, col).

    Returns:
        (..., 2) world coordinates (x, y).
    """
    rc = np.asarray(rc, dtype=np.float64)
    x = rc[..., 1] * grid.resolution + grid.origin[0]
    y = rc[..., 0] * grid.resolution + grid.origin[1]
    return np.stack([x, y], axis=-1)


def add_rectangular_obstacle(
    grid: Grid,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> Grid:
    """Return a new Grid with a filled rectangle marked as occupied.

    Args:
        grid: Source grid.
        x_min, y_min, x_max, y_max: Rectangle bounds in world coordinates.
    """
    new_obs = grid.obstacles.copy()

    col_min = max(0, int(np.floor((x_min - grid.origin[0]) / grid.resolution)))
    col_max = min(grid.width, int(np.ceil((x_max - grid.origin[0]) / grid.resolution)))
    row_min = max(0, int(np.floor((y_min - grid.origin[1]) / grid.resolution)))
    row_max = min(grid.height, int(np.ceil((y_max - grid.origin[1]) / grid.resolution)))

    new_obs[row_min:row_max, col_min:col_max] = True
    return Grid(obstacles=new_obs, resolution=grid.resolution, origin=grid.origin)


def add_circular_obstacle(
    grid: Grid,
    cx: float,
    cy: float,
    radius: float,
) -> Grid:
    """Return a new Grid with a filled circle marked as occupied.

    Args:
        grid: Source grid.
        cx, cy: Circle center in world coordinates.
        radius: Circle radius in meters.
    """
    new_obs = grid.obstacles.copy()

    # World coordinates of each cell center
    rows = np.arange(grid.height)
    cols = np.arange(grid.width)
    col_grid, row_grid = np.meshgrid(cols, rows)

    cell_x = grid.origin[0] + (col_grid + 0.5) * grid.resolution
    cell_y = grid.origin[1] + (row_grid + 0.5) * grid.resolution

    dist_sq = (cell_x - cx) ** 2 + (cell_y - cy) ** 2
    new_obs[dist_sq <= radius**2] = True

    return Grid(obstacles=new_obs, resolution=grid.resolution, origin=grid.origin)
