"""Trajectory representation and utilities."""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from lto_avoid.types import Trajectory


@jaxtyped(typechecker=beartype)
def straight_line_trajectory(
    start: tuple[float, float] | np.ndarray,
    end: tuple[float, float] | np.ndarray,
    n_points: int,
) -> Trajectory:
    """Create a straight-line trajectory between two points.

    Returns:
        (n_points, 2) array of uniformly spaced waypoints.
    """
    return np.linspace(start, end, n_points, dtype=np.float64)


@jaxtyped(typechecker=beartype)
def trajectory_length(traj: Trajectory) -> float:
    """Total arc length of a trajectory."""
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


@jaxtyped(typechecker=beartype)
def resample_trajectory(traj: Trajectory, n_points: int) -> Float[np.ndarray, "m 2"]:
    """Resample a trajectory to n_points uniformly spaced by arc length.

    First and last points are preserved exactly.
    """
    diffs = np.diff(traj, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_length[-1]

    if total < 1e-12:
        return np.tile(traj[0], (n_points, 1))

    t_norm = cum_length / total
    t_new = np.linspace(0.0, 1.0, n_points)

    x_new = np.interp(t_new, t_norm, traj[:, 0])
    y_new = np.interp(t_new, t_norm, traj[:, 1])
    return np.column_stack([x_new, y_new])


@jaxtyped(typechecker=beartype)
def trajectory_smoothness(traj: Trajectory) -> float:
    """Sum of squared second differences — measures trajectory curvature."""
    if len(traj) < 3:
        return 0.0
    accel = traj[:-2] - 2 * traj[1:-1] + traj[2:]
    return float(np.sum(accel**2))
