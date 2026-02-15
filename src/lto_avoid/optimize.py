"""Trajectory optimizer using CasADi Opti + IPOPT."""

import time
from dataclasses import dataclass

import casadi
import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped
from scipy.interpolate import BSpline

from lto_avoid.grid import Grid
from lto_avoid.sdf import compute_sdf
from lto_avoid.trajectory import resample_trajectory, trajectory_length
from lto_avoid.types import SDFArray, Trajectory


@jaxtyped(typechecker=beartype)
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

    trajectory: Trajectory
    original: Trajectory
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


def _build_bspline_basis(n_points: int, n_ctrl: int, degree: int = 3) -> np.ndarray:
    """Build a (n_points, n_ctrl) B-spline basis matrix with clamped uniform knots.

    Clamped knots guarantee the curve passes through the first and last
    control points: B[0, 0] == 1 and B[-1, -1] == 1.

    Args:
        n_points: Number of evaluation points along the curve.
        n_ctrl: Number of control points (must be > degree).
        degree: B-spline degree (default 3 for cubic).

    Returns:
        (n_points, n_ctrl) basis matrix.
    """
    n_knots = n_ctrl + degree + 1
    n_internal = n_knots - 2 * (degree + 1)
    internal = np.linspace(0, 1, n_internal + 2)[1:-1] if n_internal > 0 else []
    knots = np.concatenate([
        np.zeros(degree + 1),
        internal,
        np.ones(degree + 1),
    ])

    t = np.linspace(0, 1, n_points)
    # Nudge last point inward so it falls within the last knot span
    t[-1] = 1.0 - 1e-14

    basis = np.zeros((n_points, n_ctrl))
    for i in range(n_ctrl):
        coeffs = np.zeros(n_ctrl)
        coeffs[i] = 1.0
        basis[:, i] = BSpline(knots, coeffs, degree)(t)

    return basis


def _fit_control_points(trajectory: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Least-squares fit of control points from a trajectory and basis matrix.

    Args:
        trajectory: (N, 2) waypoints.
        basis: (N, M) basis matrix from _build_bspline_basis.

    Returns:
        (M, 2) control points.
    """
    ctrl, _, _, _ = np.linalg.lstsq(basis, trajectory, rcond=None)
    return ctrl


@jaxtyped(typechecker=beartype)
def optimize_trajectory(
    trajectory: Trajectory,
    grid: Grid,
    sdf: SDFArray | None = None,
    w_smooth: float = 5.0,
    w_deviation: float = 0.1,
    w_clearance: float = 2.0,
    safety_margin: float = 0.3,
    max_iter: int = 200,
    fix_endpoints: bool = True,
    verbose: bool = False,
    meters_per_ctrl_pt: float = 3.5,
) -> OptimizeResult:
    """Optimize a trajectory to avoid obstacles using CasADi Opti + IPOPT.

    Optimizes M B-spline control points instead of N waypoints directly.
    The spline is evaluated at N dense sample points for costs and constraints,
    producing a smooth, continuous trajectory by construction.

    Obstacle avoidance is formulated as hard constraints (sdf >= margin),
    not soft penalties. IPOPT handles this natively. A clearance cost
    additionally rewards trajectories that maintain extra distance from
    obstacles beyond the hard constraint minimum.

    Args:
        trajectory: (N, 2) initial waypoints in world coordinates.
        grid: Occupancy grid defining obstacles.
        sdf: Pre-computed SDF. Computed from grid if None.
        w_smooth: Weight for smoothness cost (squared second differences).
        w_deviation: Weight for deviation from original trajectory.
        w_clearance: Weight for clearance cost (penalizes proximity to obstacles).
        safety_margin: Minimum clearance from obstacles in meters.
        max_iter: Maximum IPOPT iterations.
        fix_endpoints: If True, first and last waypoints are fixed.
        verbose: If True, print IPOPT output.
        meters_per_ctrl_pt: Spacing between B-spline control points in meters.
            Lower values give more control points and tighter fits; higher
            values give fewer control points and smoother arcs.

    Returns:
        OptimizeResult with the optimized trajectory and metadata.
    """
    original = trajectory.copy()
    n = len(trajectory)

    if sdf is None:
        sdf = compute_sdf(grid)

    sdf_func = build_sdf_interpolant(sdf, grid)

    # B-spline setup
    arc_len = trajectory_length(original)
    m = max(8, round(arc_len / meters_per_ctrl_pt))
    basis = _build_bspline_basis(n, m)
    B_dm = casadi.DM(basis)

    # Initial control points from least-squares fit
    ctrl_init = _fit_control_points(original, basis)

    opti = casadi.Opti()

    # Decision variables: M control points as (2, M) matrix
    C = opti.variable(2, m)

    # Evaluated trajectory: (2, N) = (2, M) @ (M, N)
    P = C @ B_dm.T

    # Reference trajectory as parameter
    P_ref = opti.parameter(2, n)
    opti.set_value(P_ref, original.T)

    # --- Cost ---
    # Smoothness: sum of squared second differences on control points.
    # Operating on control points (not evaluated points) directly penalizes
    # sharp bends in the spline, producing smoother curves.
    accel = C[:, :-2] - 2 * C[:, 1:-1] + C[:, 2:]
    smooth_cost = casadi.sumsqr(accel)

    # Deviation from reference
    dev_cost = casadi.sumsqr(P - P_ref)

    # Clearance: penalize proximity to obstacles via 1/sdf²
    clearance_cost = 0
    for i in range(n):
        clearance_cost += 1.0 / (sdf_func(P[:, i]) ** 2)

    opti.minimize(
        w_smooth * smooth_cost + w_deviation * dev_cost + w_clearance * clearance_cost
    )

    # --- Constraints ---
    # Obstacle avoidance: sdf(waypoint) >= safety_margin on all evaluated points
    for i in range(n):
        opti.subject_to(sdf_func(P[:, i]) >= safety_margin)

    # Fix endpoints (on control points — clamped knots guarantee these are curve endpoints)
    if fix_endpoints:
        opti.subject_to(C[:, 0] == original[0])
        opti.subject_to(C[:, -1] == original[-1])

    # Keep control points within grid bounds (convex hull property bounds the curve)
    x_min, x_max, y_min, y_max = grid.world_extent
    margin_cells = grid.resolution  # one cell buffer
    opti.subject_to(opti.bounded(x_min + margin_cells, C[0, :], x_max - margin_cells))
    opti.subject_to(opti.bounded(y_min + margin_cells, C[1, :], y_max - margin_cells))

    # Initial guess
    opti.set_initial(C, ctrl_init.T)

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

    # Resample to uniform arc-length spacing (B-spline parameter-space
    # sampling produces non-uniform spacing in world coordinates)
    opt_traj = resample_trajectory(opt_traj, n)

    return OptimizeResult(
        trajectory=opt_traj,
        original=original,
        cost=cost,
        success=success,
        n_iterations=n_iter,
        solve_time_s=solve_time,
    )
