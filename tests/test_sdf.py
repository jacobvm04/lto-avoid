import numpy as np
from numpy.testing import assert_allclose

from lto_avoid.grid import (
    add_circular_obstacle,
    add_rectangular_obstacle,
    make_empty_grid,
)
from lto_avoid.sdf import compute_sdf, sample_sdf


def test_sdf_free_grid_all_positive():
    """An all-free grid has positive SDF everywhere (no obstacles to be near)."""
    g = make_empty_grid(20, 20, 0.1)
    sdf = compute_sdf(g)
    assert np.all(sdf > 0)


def test_sdf_full_grid_all_negative():
    """An all-occupied grid has negative SDF everywhere."""
    g = make_empty_grid(20, 20, 0.1)
    g = add_rectangular_obstacle(g, 0.0, 0.0, 2.0, 2.0)
    sdf = compute_sdf(g)
    assert np.all(sdf <= 0)


def test_sdf_sign_convention():
    """Positive outside obstacle, negative inside, ~zero at boundary."""
    g = make_empty_grid(100, 100, 0.1)
    g = add_circular_obstacle(g, 5.0, 5.0, 1.0)
    sdf = compute_sdf(g)
    # Center of obstacle (row=50, col=50) should be negative
    assert sdf[50, 50] < 0
    # Far corner should be positive
    assert sdf[0, 0] > 0


def test_sdf_distance_accuracy():
    """SDF values should reflect actual Euclidean distances in meters."""
    g = make_empty_grid(100, 100, 0.1)
    # Rectangle from (4.0, 4.0) to (6.0, 6.0) — a 2m x 2m block
    g = add_rectangular_obstacle(g, 4.0, 4.0, 6.0, 6.0)
    sdf = compute_sdf(g)
    # A point 1m away from the obstacle edge: world (3.0, 5.0)
    # Nearest obstacle edge is at x=4.0, so distance should be ~1.0m
    val = sample_sdf(sdf, g, np.array([[3.0, 5.0]]))
    assert_allclose(val[0], 1.0, atol=0.15)


def test_sdf_resolution_scaling():
    """SDF values should be in meters, not cells."""
    g = make_empty_grid(100, 100, 0.5)  # coarse grid, 50m x 50m
    g = add_rectangular_obstacle(g, 20.0, 20.0, 30.0, 30.0)
    sdf = compute_sdf(g)
    # Point at (15.0, 25.0) is 5m from the obstacle edge at x=20
    val = sample_sdf(sdf, g, np.array([[15.0, 25.0]]))
    assert_allclose(val[0], 5.0, atol=0.6)


def test_sample_sdf_at_grid_points():
    """Sampling at cell centers should match direct array lookup."""
    g = make_empty_grid(50, 50, 0.1)
    g = add_circular_obstacle(g, 2.5, 2.5, 0.5)
    sdf = compute_sdf(g)
    # Sample at the center of cell (row=10, col=20)
    # Cell center in world: x = 0 + (20 + 0.5)*0.1 = 2.05, y = (10 + 0.5)*0.1 = 1.05
    # But world_to_grid maps corner not center, so sample at grid index (10, 20) directly
    x = g.origin[0] + 20 * g.resolution
    y = g.origin[1] + 10 * g.resolution
    val = sample_sdf(sdf, g, np.array([[x, y]]))
    assert_allclose(val[0], sdf[10, 20], atol=1e-10)
