"""Trajectory representation and utilities."""

from __future__ import annotations

import numpy as np


def straight_line_trajectory(
    start: tuple[float, float] | np.ndarray,
    end: tuple[float, float] | np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Create a straight-line trajectory between two points.

    Returns:
        (n_points, 2) array of uniformly spaced waypoints.
    """
    return np.linspace(start, end, n_points, dtype=np.float64)


def trajectory_length(traj: np.ndarray) -> float:
    """Total arc length of a trajectory."""
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def resample_trajectory(traj: np.ndarray, n_points: int) -> np.ndarray:
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


def trajectory_smoothness(traj: np.ndarray) -> float:
    """Sum of squared second differences — measures trajectory curvature."""
    if len(traj) < 3:
        return 0.0
    accel = traj[:-2] - 2 * traj[1:-1] + traj[2:]
    return float(np.sum(accel**2))
