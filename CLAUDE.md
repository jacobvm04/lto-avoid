# CLAUDE.md — lto-avoid

## What this project is

A Python library for optimization-based local trajectory adjustment for obstacle avoidance.
Occupancy grid → SDF → CasADi bspline interpolant → IPOPT constrained NLP → collision-free trajectory.

## Running tests

```bash
cd /Users/jacobvm/Dev/lto-avoid
uv run python -m pytest -v
```

Use `uv run python -m pytest` (not `uv run pytest`) — the system pytest via pyenv can shadow the venv's.

**Run tests between every change.** The full suite is 59 tests in ~3s. If tests ever become slow or painful to run, fix that immediately — fast tests are load-bearing for this project. Exception: scenario tests using HIL may block waiting for Telegram approval when artifacts change — this is expected and not a performance problem.

### Testing workflow

- **Run the full test suite after every incremental change** — no exceptions. A change isn't done until the tests pass.
- **Add tests for new or changed behavior.** New functions, new parameters, bug fixes, and behavioral changes all need corresponding test coverage. If you're changing what code does, prove it works with a test.
- **Don't skip tests to move faster.** The suite runs in ~3s. There is no reason to defer testing. Catching regressions immediately is far cheaper than debugging them later.
- **If a test fails, fix the root cause before moving on.** Don't comment out, skip, or weaken assertions to get green. Understand why it failed.

## Dev dependencies

Dev deps live in `[dependency-groups]` in `pyproject.toml`, not `[project.optional-dependencies]`. This is what `uv sync --dev` installs.

## Architecture

```
grid.py        pure numpy — Grid dataclass, coord transforms, obstacle helpers
sdf.py         numpy + scipy — occupancy grid → signed distance field
trajectory.py  pure numpy — straight lines, resampling, arc length, smoothness
optimize.py    casadi — CasADi Opti + IPOPT optimizer
types.py       jaxtyping — shape-checked array type aliases (GridArray, Trajectory, etc.)
hil.py         requests + dotenv — human-in-the-loop snapshot testing via Telegram
```

CasADi is **isolated to `optimize.py`**. Every other module is pure numpy/scipy. Don't leak CasADi imports into grid/sdf/trajectory.

## Key design rules

- **B-spline control point optimization.** The optimizer uses M cubic B-spline control points as decision variables instead of N waypoints directly. The spline is evaluated at N dense sample points via basis matrix multiplication (`C @ B.T`) for costs and constraints. This produces inherently smooth trajectories and prevents inter-waypoint collisions. Control point count is derived from trajectory arc length: `M = max(8, round(arc_length / meters_per_ctrl_pt))`, default 3.5m spacing.
- **Hard constraints, not penalties.** Obstacle avoidance is `sdf(waypoint) >= safety_margin` as an IPOPT constraint. Don't convert this to a soft penalty cost term.
- **No hand-coded gradients.** CasADi's AD differentiates through the bspline interpolant and all cost terms automatically. There is no `cost.py` module and there shouldn't be one.
- **Frozen dataclasses.** `Grid` and `OptimizeResult` are `frozen=True`. Functions that modify a grid return a new `Grid` with copied data. Don't mutate.
- **Pure functions.** Functions take inputs, return outputs, no side effects. No module-level mutable state.

## Coordinate convention

- **World coordinates**: `(x, y)` — x is horizontal, y is vertical.
- **Grid indices**: `(row, col)` — row corresponds to y, col corresponds to x.
- The `(x,y) ↔ (row,col)` swap happens **only** in `world_to_grid` and `grid_to_world`. Every other function works in one coordinate system. Don't do ad-hoc swaps elsewhere.

## SDF → CasADi interpolant data layout (critical)

The `build_sdf_interpolant` function in `optimize.py` converts the numpy SDF to a CasADi bspline interpolant. The data layout must be exactly right:

- `sdf` array shape is `(n_rows, n_cols)` = `(y, x)`
- CasADi `interpolant('sdf', 'bspline', [x_axis, y_axis], data_flat)` expects x varying fastest
- Correct flattening: `sdf.T.ravel(order='F')`
- This is verified by `test_sdf_interpolant_matches_numpy` — if that test breaks, the data layout is wrong

## Test organization

| File | What it tests | Speed |
|------|--------------|-------|
| `test_grid.py` | Grid creation, coord transforms, obstacles, immutability | <0.1s |
| `test_sdf.py` | SDF signs, distances, units, sampling | <0.1s |
| `test_trajectory.py` | Straight lines, resampling, length, smoothness | <0.1s |
| `test_optimize.py` | Interpolant correctness, B-spline basis, optimizer convergence, constraints | <0.5s |
| `test_scenarios.py` | End-to-end scenarios + visual artifacts (via HIL) + speed regression | <2s |
| `test_types.py` | jaxtyping + beartype shape/dtype rejection and acceptance | <0.2s |
| `test_hil.py` | Image comparison, perceptual diff tolerance, diff visualization | <0.5s |

### Visual artifacts & Human-in-the-Loop (HIL) review

Integration tests generate PNGs and compare them against approved snapshots in `artifacts/` (checked into git) using perceptual diff. If an image is unchanged, the test passes silently. If an image is new or changed, it is sent to a Telegram bot for human approval — the test blocks (up to 5 minutes) until the human approves or rejects with feedback.

**Setup:** Copy `.env.example` to `.env` and set `TELEGRAM_BOT_TOKEN`. On first run, send `/start` to the bot to auto-discover `TELEGRAM_CHAT_ID`. Without a token, the fallback saves images directly without review.

The artifacts show SDF heatmaps with obstacle contours, original trajectory (blue dashed), and optimized trajectory (green solid). Always visually inspect these after changing optimizer behavior.

### Speed tests

`test_solve_speed_small` (<2s) and `test_solve_speed_medium` (<5s) catch performance regressions. These bounds are generous for CI; typical solves are well under 1s.

## Style

- **No `from __future__ import annotations`** in source modules — jaxtyping + beartype require real annotations for runtime shape checking
- Type hints on function signatures using jaxtyping aliases from `types.py` (e.g. `Trajectory`, `SDFArray`, `GridArray`)
- Docstrings on public functions (Google style: `Args:` / `Returns:`)
- No comments on obvious code; comments only where the "why" isn't self-evident
- Don't add features, abstractions, or "improvements" beyond what's asked
