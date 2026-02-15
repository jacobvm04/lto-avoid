import numpy as np
import pytest
from numpy.testing import assert_allclose

from lto_avoid.grid import (
    Grid,
    add_circular_obstacle,
    add_rectangular_obstacle,
    grid_to_world,
    make_empty_grid,
    world_to_grid,
)


def test_make_empty_grid():
    g = make_empty_grid(100, 50, 0.1)
    assert g.obstacles.shape == (50, 100)
    assert g.resolution == 0.1
    assert not g.obstacles.any()


def test_world_extent():
    g = make_empty_grid(100, 50, 0.1, origin=(1.0, 2.0))
    assert g.world_extent == (1.0, 11.0, 2.0, 7.0)


def test_coordinate_roundtrip():
    g = make_empty_grid(100, 100, 0.1, origin=(-5.0, -5.0))
    xy = np.array([[0.0, 0.0], [3.5, -2.1], [-4.9, 4.9]])
    recovered = grid_to_world(g, world_to_grid(g, xy))
    assert_allclose(recovered, xy, atol=1e-12)


def test_world_to_grid_known_values():
    g = make_empty_grid(100, 100, 0.1, origin=(0.0, 0.0))
    # World (1.0, 2.0) → col=10, row=20
    rc = world_to_grid(g, np.array([1.0, 2.0]))
    assert_allclose(rc, [20.0, 10.0], atol=1e-12)


def test_add_rectangular_obstacle():
    g = make_empty_grid(100, 100, 0.1)
    g2 = add_rectangular_obstacle(g, 2.0, 3.0, 4.0, 5.0)
    # Original unchanged
    assert not g.obstacles.any()
    # Interior cell should be occupied
    rc = world_to_grid(g2, np.array([3.0, 4.0]))
    r, c = int(rc[0]), int(rc[1])
    assert g2.obstacles[r, c]
    # Far away cell should be free
    assert not g2.obstacles[0, 0]


def test_add_circular_obstacle():
    g = make_empty_grid(100, 100, 0.1)
    g2 = add_circular_obstacle(g, 5.0, 5.0, 1.0)
    # Original unchanged
    assert not g.obstacles.any()
    # Center should be occupied
    rc = world_to_grid(g2, np.array([5.0, 5.0]))
    r, c = int(rc[0]), int(rc[1])
    assert g2.obstacles[r, c]
    # Point well outside should be free
    rc2 = world_to_grid(g2, np.array([0.5, 0.5]))
    r2, c2 = int(rc2[0]), int(rc2[1])
    assert not g2.obstacles[r2, c2]


def test_grid_frozen():
    g = make_empty_grid(10, 10, 0.1)
    with pytest.raises(AttributeError):
        g.resolution = 0.2
