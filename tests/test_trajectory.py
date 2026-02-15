import numpy as np
from numpy.testing import assert_allclose

from lto_avoid.trajectory import (
    resample_trajectory,
    straight_line_trajectory,
    trajectory_length,
    trajectory_smoothness,
)


def test_straight_line_endpoints():
    traj = straight_line_trajectory((0.0, 0.0), (10.0, 5.0), 20)
    assert traj.shape == (20, 2)
    assert_allclose(traj[0], [0.0, 0.0])
    assert_allclose(traj[-1], [10.0, 5.0])


def test_straight_line_length():
    traj = straight_line_trajectory((0.0, 0.0), (3.0, 4.0), 100)
    assert_allclose(trajectory_length(traj), 5.0, atol=1e-10)


def test_straight_line_smoothness_zero():
    traj = straight_line_trajectory((0.0, 0.0), (10.0, 0.0), 50)
    assert_allclose(trajectory_smoothness(traj), 0.0, atol=1e-20)


def test_kinked_trajectory_smoothness_positive():
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    assert trajectory_smoothness(traj) > 0


def test_resample_preserves_endpoints():
    traj = np.array([[0.0, 0.0], [5.0, 3.0], [10.0, 0.0]])
    resampled = resample_trajectory(traj, 50)
    assert resampled.shape == (50, 2)
    assert_allclose(resampled[0], traj[0])
    assert_allclose(resampled[-1], traj[-1])


def test_resample_preserves_length():
    traj = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 0.0]])
    resampled = resample_trajectory(traj, 200)
    assert_allclose(
        trajectory_length(resampled),
        trajectory_length(traj),
        rtol=0.01,
    )
