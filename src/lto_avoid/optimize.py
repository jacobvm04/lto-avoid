"""Trajectory optimizer using CasADi Opti + IPOPT."""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi
import numpy as np

from lto_avoid.grid import Grid
from lto_avoid.sdf import compute_sdf


@dataclass(frozen=True)
class OptimizeResult:
    """Result of trajectory optimization.

    Attributes:
        trajectory: (N, 2) optimized waypoints.
        original: (N, 2) original waypoints.
        cost: Final objective value.
        success: True if IPOPT converged.
        n_iterations: IPOPT iteration count.
        solve_time_s: Wall-clock solve time in seconds.
    """

    trajectory: np.ndarray
    original: np.ndarray
    cost: float
    success: bool
    n_iterations: int
    solve_time_s: float


def build_sdf_interpolant(sdf: np.ndarray, grid: Grid) -> casadi.Function:
    """Create a CasADi B-spline interpolant from a numpy SDF array.

    The interpolant maps world (x, y) → SDF value and is automatically
    differentiable through CasADi's AD.

    Data layout: CasADi interpolant for grids [x_axis, y_axis] expects
    data where the first axis (x) varies fastest. Our sdf is (rows, cols)
    = (y, x), so we transpose to (x, y) then flatten in Fortran order.
    """
    n_rows, n_cols = sdf.shape

    x_axis = grid.origin[0] + np.arange(n_cols) * grid.resolution
    y_axis = grid.origin[1] + np.arange(n_rows) * grid.resolution

    # sdf[row, col] = sdf[y_idx, x_idx] → transpose to (x_idx, y_idx)
    sdf_xy = sdf.T  # (n_cols, n_rows)
    data_flat = sdf_xy.ravel(order="F")

    return casadi.interpolant(
        "sdf",
        "bspline",
        [x_axis.tolist(), y_axis.tolist()],
        data_flat.tolist(),
    )


def optimize_trajectory(
    trajectory: np.ndarray,
    grid: Grid,
    sdf: np.ndarray | None = None,
    w_smooth: float = 1.0,
    w_deviation: float = 0.5,
    safety_margin: float = 0.3,
    max_iter: int = 200,
    fix_endpoints: bool = True,
    verbose: bool = False,
) -> OptimizeResult:
    """Optimize a trajectory to avoid obstacles using CasADi Opti + IPOPT.

    Obstacle avoidance is formulated as hard constraints (sdf >= margin),
    not soft penalties. IPOPT handles this natively.

    Args:
        trajectory: (N, 2) initial waypoints in world coordinates.
        grid: Occupancy grid defining obstacles.
        sdf: Pre-computed SDF. Computed from grid if None.
        w_smooth: Weight for smoothness cost (squared second differences).
        w_deviation: Weight for deviation from original trajectory.
        safety_margin: Minimum clearance from obstacles in meters.
        max_iter: Maximum IPOPT iterations.
        fix_endpoints: If True, first and last waypoints are fixed.
        verbose: If True, print IPOPT output.

    Returns:
        OptimizeResult with the optimized trajectory and metadata.
    """
    original = trajectory.copy()
    n = len(trajectory)

    if sdf is None:
        sdf = compute_sdf(grid)

    sdf_func = build_sdf_interpolant(sdf, grid)

    opti = casadi.Opti()

    # Decision variables: 2D waypoints as (2, N) matrix
    P = opti.variable(2, n)

    # Reference trajectory as parameter
    P_ref = opti.parameter(2, n)
    opti.set_value(P_ref, original.T)

    # --- Cost ---
    # Smoothness: sum of squared second differences
    accel = P[:, :-2] - 2 * P[:, 1:-1] + P[:, 2:]
    smooth_cost = casadi.sumsqr(accel)

    # Deviation from reference
    dev_cost = casadi.sumsqr(P - P_ref)

    opti.minimize(w_smooth * smooth_cost + w_deviation * dev_cost)

    # --- Constraints ---
    # Obstacle avoidance: sdf(waypoint) >= safety_margin
    for i in range(n):
        opti.subject_to(sdf_func(P[:, i]) >= safety_margin)

    # Fix endpoints
    if fix_endpoints:
        opti.subject_to(P[:, 0] == original[0])
        opti.subject_to(P[:, -1] == original[-1])

    # Keep trajectory within grid bounds
    x_min, x_max, y_min, y_max = grid.world_extent
    margin_cells = grid.resolution  # one cell buffer
    opti.subject_to(opti.bounded(x_min + margin_cells, P[0, :], x_max - margin_cells))
    opti.subject_to(opti.bounded(y_min + margin_cells, P[1, :], y_max - margin_cells))

    # Initial guess
    opti.set_initial(P, original.T)

    # Solver options
    opti.solver(
        "ipopt",
        {},
        {
            "max_iter": max_iter,
            "print_level": 5 if verbose else 0,
            "sb": "yes",
            "tol": 1e-6,
            "acceptable_tol": 1e-4,
        },
    )

    t0 = time.perf_counter()
    try:
        sol = opti.solve()
        success = True
        opt_traj = np.array(sol.value(P)).T
        cost = float(sol.value(opti.f))
        n_iter = int(sol.stats()["iter_count"])
    except RuntimeError:
        success = False
        opt_traj = np.array(opti.debug.value(P)).T
        cost = float(opti.debug.value(opti.f))
        n_iter = max_iter
    solve_time = time.perf_counter() - t0

    return OptimizeResult(
        trajectory=opt_traj,
        original=original,
        cost=cost,
        success=success,
        n_iterations=n_iter,
        solve_time_s=solve_time,
    )
