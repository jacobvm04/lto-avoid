import numpy as np
import pytest
from numpy.testing import assert_allclose

from lto_avoid.grid import (
    add_circular_obstacle,
    add_rectangular_obstacle,
    make_empty_grid,
)
from lto_avoid.optimize import (
    OptimizeResult,
    build_sdf_interpolant,
    optimize_trajectory,
)
from lto_avoid.sdf import compute_sdf, sample_sdf
from lto_avoid.trajectory import straight_line_trajectory


# --- Interpolant correctness ---


def test_sdf_interpolant_matches_numpy():
    """CasADi interpolant should produce values close to the numpy SDF at grid points."""
    g = make_empty_grid(50, 50, 0.1)
    g = add_circular_obstacle(g, 2.5, 2.5, 1.0)
    sdf = compute_sdf(g)
    sdf_func = build_sdf_interpolant(sdf, g)

    # Sample at a grid of interior points (skip boundaries where bspline may differ)
    for row in range(2, 48, 5):
        for col in range(2, 48, 5):
            x = g.origin[0] + col * g.resolution
            y = g.origin[1] + row * g.resolution
            expected = sdf[row, col]
            actual = float(sdf_func([x, y]))
            assert abs(actual - expected) < 0.1, (
                f"Mismatch at ({x}, {y}): expected {expected}, got {actual}"
            )


def test_sdf_interpolant_sign_convention():
    """Interpolant should be negative inside obstacles, positive outside."""
    g = make_empty_grid(50, 50, 0.1)
    g = add_circular_obstacle(g, 2.5, 2.5, 1.0)
    sdf = compute_sdf(g)
    sdf_func = build_sdf_interpolant(sdf, g)

    # Inside obstacle
    assert float(sdf_func([2.5, 2.5])) < 0
    # Far from obstacle
    assert float(sdf_func([0.5, 0.5])) > 0


# --- Optimizer behavior ---


def test_optimize_no_obstacles():
    """With no obstacles, trajectory should stay near the original."""
    g = make_empty_grid(100, 100, 0.1)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 20)
    result = optimize_trajectory(traj, g)

    assert isinstance(result, OptimizeResult)
    assert result.success
    assert_allclose(result.trajectory, traj, atol=0.05)


def test_optimize_fixed_endpoints():
    """With fix_endpoints=True, endpoints should not move."""
    g = make_empty_grid(100, 100, 0.1)
    g = add_circular_obstacle(g, 5.0, 5.0, 1.0)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 20)
    result = optimize_trajectory(traj, g, fix_endpoints=True)

    assert_allclose(result.trajectory[0], traj[0], atol=1e-6)
    assert_allclose(result.trajectory[-1], traj[-1], atol=1e-6)


def test_optimize_avoids_obstacle():
    """Trajectory through an obstacle should be pushed to collision-free."""
    g = make_empty_grid(100, 100, 0.1)
    g = add_circular_obstacle(g, 5.0, 5.0, 1.5)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 30)
    sdf = compute_sdf(g)

    result = optimize_trajectory(traj, g, sdf=sdf, safety_margin=0.2)
    assert result.success

    # All waypoints should be outside the obstacle with margin
    sdf_vals = sample_sdf(sdf, g, result.trajectory)
    assert np.all(sdf_vals > 0), f"Collision! min SDF = {sdf_vals.min():.4f}"


def test_optimize_precomputed_sdf():
    """Passing precomputed SDF should give the same result as computing internally."""
    g = make_empty_grid(100, 100, 0.1)
    g = add_circular_obstacle(g, 5.0, 5.0, 1.0)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 20)
    sdf = compute_sdf(g)

    r1 = optimize_trajectory(traj, g, sdf=sdf)
    r2 = optimize_trajectory(traj, g, sdf=None)

    assert_allclose(r1.trajectory, r2.trajectory, atol=1e-4)


def test_optimize_infeasible_graceful():
    """If the problem is infeasible, should return success=False without crashing."""
    g = make_empty_grid(100, 100, 0.1)
    # Fill the entire grid with obstacles — impossible to satisfy margin constraint
    g = add_rectangular_obstacle(g, 0.0, 0.0, 10.0, 10.0)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 10)

    result = optimize_trajectory(traj, g, safety_margin=1.0, max_iter=50)
    assert not result.success
