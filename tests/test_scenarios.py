"""Integration tests with visual artifacts and speed assertions.

Each scenario builds a grid with obstacles, creates a straight-line trajectory
through them, optimizes, then asserts collision-free results and saves a PNG.
"""

import io
import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lto_avoid.grid import (
    add_circular_obstacle,
    add_rectangular_obstacle,
    make_empty_grid,
)
from lto_avoid.hil import assert_human_in_the_loop
from lto_avoid.optimize import optimize_trajectory
from lto_avoid.sdf import compute_sdf, sample_sdf
from lto_avoid.trajectory import straight_line_trajectory

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def render_scenario(
    grid,
    sdf,
    original,
    optimized,
    safety_margin: float,
    title: str = "",
) -> bytes:
    """Render a scenario visualization and return PNG bytes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    extent = [
        grid.world_extent[0],
        grid.world_extent[1],
        grid.world_extent[2],
        grid.world_extent[3],
    ]

    # SDF heatmap
    ax.imshow(sdf, origin="lower", extent=extent, cmap="RdYlGn", alpha=0.6)
    # Obstacle boundary (SDF=0) and margin contour
    ax.contour(
        sdf,
        levels=[0, safety_margin],
        origin="lower",
        extent=extent,
        colors=["red", "orange"],
        linewidths=[2, 1],
    )

    # Trajectories
    ax.plot(
        original[:, 0],
        original[:, 1],
        "b--o",
        markersize=3,
        linewidth=2,
        label="Original",
    )
    ax.plot(
        optimized[:, 0],
        optimized[:, 1],
        "g-o",
        markersize=3,
        linewidth=2,
        label="Optimized",
    )

    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# --- Scenarios ---


def test_scenario_single_circle():
    """Straight trajectory through a single circular obstacle."""
    g = make_empty_grid(100, 100, 0.1)
    g = add_circular_obstacle(g, 5.0, 5.0, 1.5)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 40)
    sdf = compute_sdf(g)
    safety_margin = 0.3

    result = optimize_trajectory(traj, g, sdf=sdf, safety_margin=safety_margin)
    assert result.success, f"Optimizer failed: {result.n_iterations} iters"

    # All waypoints should be collision-free
    sdf_vals = sample_sdf(sdf, g, result.trajectory)
    assert np.all(sdf_vals > 0), f"Collision! min SDF = {sdf_vals.min():.4f}"

    # Endpoints preserved
    np.testing.assert_allclose(result.trajectory[0], traj[0], atol=1e-5)
    np.testing.assert_allclose(result.trajectory[-1], traj[-1], atol=1e-5)

    image_bytes = render_scenario(
        g, sdf, traj, result.trajectory, safety_margin, title="Single Circle Obstacle"
    )
    assert_human_in_the_loop("scenario_single_circle.png", image_bytes, ARTIFACTS_DIR)


def test_scenario_narrow_corridor():
    """Trajectory must thread a narrow gap between two rectangular obstacles."""
    g = make_empty_grid(100, 100, 0.1)
    # Bottom wall: y from 0 to 4
    g = add_rectangular_obstacle(g, 4.0, 0.0, 6.0, 4.0)
    # Top wall: y from 6 to 10
    g = add_rectangular_obstacle(g, 4.0, 6.0, 6.0, 10.0)
    # Gap is y ∈ [4, 6] — 2m wide
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 40)
    sdf = compute_sdf(g)
    safety_margin = 0.2

    result = optimize_trajectory(traj, g, sdf=sdf, safety_margin=safety_margin)
    assert result.success

    sdf_vals = sample_sdf(sdf, g, result.trajectory)
    assert np.all(sdf_vals > 0), f"Collision! min SDF = {sdf_vals.min():.4f}"

    image_bytes = render_scenario(
        g, sdf, traj, result.trajectory, safety_margin, title="Narrow Corridor"
    )
    assert_human_in_the_loop("scenario_narrow_corridor.png", image_bytes, ARTIFACTS_DIR)


def test_scenario_multiple_circles():
    """Navigate around multiple scattered circular obstacles."""
    g = make_empty_grid(200, 100, 0.1)
    g = add_circular_obstacle(g, 5.0, 5.0, 1.5)
    g = add_circular_obstacle(g, 10.0, 4.0, 1.0)
    g = add_circular_obstacle(g, 10.0, 7.0, 1.0)
    g = add_circular_obstacle(g, 15.0, 5.0, 1.5)
    traj = straight_line_trajectory((1.0, 5.0), (19.0, 5.0), 50)
    sdf = compute_sdf(g)
    safety_margin = 0.3

    result = optimize_trajectory(
        traj, g, sdf=sdf, safety_margin=safety_margin, w_smooth=1.0, w_deviation=0.3
    )
    assert result.success

    sdf_vals = sample_sdf(sdf, g, result.trajectory)
    assert np.all(sdf_vals > 0), f"Collision! min SDF = {sdf_vals.min():.4f}"

    image_bytes = render_scenario(
        g, sdf, traj, result.trajectory, safety_margin, title="Multiple Circular Obstacles"
    )
    assert_human_in_the_loop(
        "scenario_multiple_circles.png", image_bytes, ARTIFACTS_DIR
    )


def test_scenario_no_obstacles():
    """With no obstacles, trajectory should stay very close to reference."""
    g = make_empty_grid(100, 100, 0.1)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 30)
    sdf = compute_sdf(g)

    result = optimize_trajectory(traj, g, sdf=sdf)
    assert result.success

    max_dev = np.max(np.linalg.norm(result.trajectory - traj, axis=1))
    assert max_dev < 0.05, f"Max deviation = {max_dev:.4f}, expected < 0.05"


# --- Speed tests ---


def test_solve_speed_small():
    """A typical small scenario should solve quickly."""
    g = make_empty_grid(100, 100, 0.1)
    g = add_circular_obstacle(g, 5.0, 5.0, 1.5)
    traj = straight_line_trajectory((1.0, 5.0), (9.0, 5.0), 30)
    sdf = compute_sdf(g)

    result = optimize_trajectory(traj, g, sdf=sdf)
    assert result.success
    assert result.solve_time_s < 2.0, (
        f"Solve took {result.solve_time_s:.3f}s, expected < 2s"
    )


def test_solve_speed_medium():
    """A medium scenario (larger grid, more waypoints) should still be fast."""
    g = make_empty_grid(200, 200, 0.1)
    g = add_circular_obstacle(g, 10.0, 10.0, 2.0)
    g = add_circular_obstacle(g, 15.0, 10.0, 1.5)
    traj = straight_line_trajectory((2.0, 10.0), (18.0, 10.0), 50)
    sdf = compute_sdf(g)

    result = optimize_trajectory(traj, g, sdf=sdf)
    assert result.success
    assert result.solve_time_s < 5.0, (
        f"Solve took {result.solve_time_s:.3f}s, expected < 5s"
    )
