"""Tests for jaxtyping + beartype shape checking."""

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation
from jaxtyping import TypeCheckError

from lto_avoid.grid import Grid, make_empty_grid, world_to_grid
from lto_avoid.sdf import compute_sdf, sample_sdf
from lto_avoid.trajectory import (
    resample_trajectory,
    trajectory_length,
    trajectory_smoothness,
)


# --- Grid shape checking ---


def test_grid_rejects_1d_obstacles():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        Grid(
            obstacles=np.zeros(10, dtype=bool),
            resolution=0.1,
            origin=np.array([0.0, 0.0]),
        )


def test_grid_rejects_3d_origin():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        Grid(
            obstacles=np.zeros((10, 10), dtype=bool),
            resolution=0.1,
            origin=np.array([0.0, 0.0, 0.0]),
        )


def test_grid_rejects_non_bool_obstacles():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        Grid(
            obstacles=np.zeros((10, 10), dtype=np.float64),
            resolution=0.1,
            origin=np.array([0.0, 0.0]),
        )


def test_grid_accepts_valid_shapes():
    g = Grid(
        obstacles=np.zeros((20, 30), dtype=bool),
        resolution=0.1,
        origin=np.array([1.0, 2.0]),
    )
    assert g.height == 20
    assert g.width == 30


# --- Coordinate transform shape checking ---


def test_world_to_grid_rejects_wrong_last_dim():
    g = make_empty_grid(10, 10, 0.1)
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        world_to_grid(g, np.array([1.0, 2.0, 3.0]))


def test_world_to_grid_accepts_batched():
    g = make_empty_grid(10, 10, 0.1)
    xy = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    result = world_to_grid(g, xy)
    assert result.shape == (3, 2)


# --- Trajectory shape checking ---


def test_trajectory_length_rejects_1d():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        trajectory_length(np.array([1.0, 2.0, 3.0]))


def test_trajectory_length_rejects_wrong_cols():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        trajectory_length(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_trajectory_smoothness_rejects_wrong_shape():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        trajectory_smoothness(np.array([1.0, 2.0, 3.0, 4.0]))


def test_resample_rejects_wrong_shape():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        resample_trajectory(np.array([1.0, 2.0, 3.0]), 10)


def test_resample_allows_different_output_length():
    traj = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    resampled = resample_trajectory(traj, 50)
    assert resampled.shape == (50, 2)


# --- SDF shape checking ---


def test_sample_sdf_rejects_1d_points():
    g = make_empty_grid(10, 10, 0.1)
    sdf = compute_sdf(g)
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        sample_sdf(sdf, g, np.array([0.5, 0.5]))


def test_sample_sdf_accepts_valid():
    g = make_empty_grid(10, 10, 0.1)
    sdf = compute_sdf(g)
    points = np.array([[0.5, 0.5], [0.3, 0.3]])
    result = sample_sdf(sdf, g, points)
    assert result.shape == (2,)


# --- dtype checking ---


def test_grid_rejects_string_origin():
    with pytest.raises((BeartypeCallHintViolation, TypeCheckError)):
        Grid(
            obstacles=np.zeros((10, 10), dtype=bool),
            resolution=0.1,
            origin=np.array(["a", "b"]),
        )
