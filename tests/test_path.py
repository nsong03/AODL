r"""The ``Path`` move (``trajectory/spec.py``, ``webapp/docs/DESIGN.md`` §1.3).

A :class:`~aodl.trajectory.spec.Path` is a clamped cubic spline through relative waypoints,
knot times proportional to chord length.  The claims checked here, each against a closed form
or an exact identity rather than against the implementation:

1. **it is a position law** — continuous at every knot, through every waypoint, at rest at
   both ends (so it chains with the other rest-to-rest moves), ``C^2`` at interior knots;
2. **a single waypoint is the constant-jerk ramp** — the clamped cubic between two rest points
   is ``3 tau^2 - 2 tau^3`` (Eq. S15), coefficient for coefficient;
3. **it is a first-class move** — ``closed`` returns to the start, duplicate waypoints are
   refused by name, and a ``Lift -> Path -> Hold`` chain compiles, synthesizes (Eq. S19, band
   checked) and plans without the rest of the package noticing anything new.
"""

from __future__ import annotations

import numpy as np
import pytest

from aodl import (
    ArraySpec,
    Hold,
    Lift,
    Path,
    TrajectorySpec,
    default_1030,
    plan_motion,
    synthesize,
)
from aodl.trajectory.ramps import constant_jerk
from aodl.units import um, us

RADIUS = 5.0 * um
N_POINTS = 8
DURATION = 200.0 * us


def _circle(n: int = N_POINTS, radius: float = RADIUS) -> tuple[tuple[float, float, float], ...]:
    """``n`` points on a circle in the XY plane that starts (and ends) at the path origin."""
    points = []
    for k in range(1, n + 1):
        phi = 2.0 * np.pi * k / n
        points.append((radius * np.cos(phi) - radius, radius * np.sin(phi), 0.0))
    points[-1] = (0.0, 0.0, 0.0)  # exactly back at the start (cos 2pi is 1, sin 2pi is not 0)
    return tuple(points)


#: A genuinely three-dimensional open path: a tilted, rising arc.
ARC = (
    (4.0 * um, 1.0 * um, 1.0 * um),
    (6.0 * um, 5.0 * um, 3.0 * um),
    (2.0 * um, 8.0 * um, 2.0 * um),
)


def _compile(path: Path):
    return TrajectorySpec(array=ArraySpec(), moves=(path,)).compile()


def _knots(path: Path) -> np.ndarray:
    relative = np.array([(0.0, 0.0, 0.0), *path.points])
    chords = np.linalg.norm(np.diff(relative, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(chords)])
    return path.duration * cumulative / cumulative[-1]


def _knot_jumps(poly) -> np.ndarray:
    """``|left - right|`` of ``poly`` at each interior knot, exactly from the coefficients.

    In normalized local time a segment's value at ``tau = 1`` is the sum of its coefficients
    and at ``tau = 0`` its constant term — the same closed form ``_check_continuity`` uses.
    """
    return np.abs(poly.coeffs[:-1].sum(axis=1) - poly.coeffs[1:, 0])


# ============================================================================ the spline


def test_value_is_continuous_at_every_knot():
    """Left and right limits agree at each interior knot (the compile-time check passes too)."""
    path = Path(ARC, DURATION)
    polys = _compile(path)
    knots = _knots(path)
    for poly in polys:
        assert poly.n_segments == len(path.points)
        np.testing.assert_allclose(poly.breaks, knots, rtol=1e-12, atol=1e-18)
        scale = float(np.max(np.abs(poly.coeffs)))
        assert np.all(_knot_jumps(poly) <= 1e-9 * scale)


def test_velocity_vanishes_at_both_ends():
    """Clamped boundary conditions: rest at ``t0`` and ``t1``, to 1e-9 of the peak speed."""
    path = Path(ARC, DURATION)
    t = np.linspace(0.0, DURATION, 2001)
    for poly in _compile(path):
        speed = poly.derivative()
        peak = float(np.max(np.abs(speed(t))))
        if peak == 0.0:
            continue
        assert abs(float(speed(0.0))) <= 1e-9 * peak
        assert abs(float(speed(DURATION))) <= 1e-9 * peak


def test_acceleration_is_continuous_at_interior_knots():
    """``C^2`` at the interior knots: velocity and acceleration both match to 1e-6 relative."""
    path = Path(ARC, DURATION)
    for poly in _compile(path):
        speed = poly.derivative()
        accel = speed.derivative()
        for derived in (speed, accel):
            scale = float(np.max(np.abs(derived.coeffs)))
            if scale == 0.0:
                continue
            assert np.all(_knot_jumps(derived) <= 1e-6 * scale)


def test_passes_through_every_waypoint():
    """At each knot the three position laws are the absolute waypoint, to 1e-12 m."""
    path = Path(ARC, DURATION)
    knots = _knots(path)
    polys = _compile(path)
    for i, point in enumerate(path.points):
        for axis in range(3):
            assert float(polys[axis](knots[i + 1])) == pytest.approx(point[axis], abs=1e-12)
    for poly in polys:
        assert float(poly(0.0)) == pytest.approx(0.0, abs=1e-12)


def test_closed_path_returns_to_the_start():
    """``closed=True`` appends the origin, unless the last point already is one."""
    appended = Path(ARC, DURATION, closed=True)
    assert len(appended.points) == len(ARC) + 1
    assert appended.points[-1] == (0.0, 0.0, 0.0)
    for poly in _compile(appended):
        assert float(poly(DURATION)) == pytest.approx(0.0, abs=1e-12)

    already = Path(_circle(), DURATION, closed=True)
    assert len(already.points) == N_POINTS  # nothing appended
    assert already.end == (0.0, 0.0, 0.0)


def test_one_point_path_is_the_constant_jerk_ramp():
    """Two rest points and a clamped cubic: ``y_i + d (3 tau^2 - 2 tau^3)``, Eq. S15."""
    target = (5.0 * um, -2.0 * um, 3.0 * um)
    path = Path((target,), DURATION)
    polys = _compile(path)
    for axis in range(3):
        ramp = constant_jerk(0.0, DURATION, 0.0, target[axis])
        np.testing.assert_allclose(polys[axis].breaks, ramp.breaks, rtol=1e-15)
        np.testing.assert_allclose(
            polys[axis].coeffs, ramp.coeffs, rtol=1e-12, atol=1e-12 * abs(target[axis])
        )


def test_duplicate_waypoint_raises():
    """A zero-length chord is refused by name, at compile time."""
    path = Path(((1.0 * um, 0.0, 0.0), (1.0 * um, 0.0, 0.0), (2.0 * um, 0.0, 0.0)), DURATION)
    with pytest.raises(ValueError, match=r"Path waypoint 1 duplicates waypoint 0"):
        _compile(path)
    # the implicit start counts too: a first point at the origin is a duplicate of it
    with pytest.raises(ValueError, match=r"Path waypoint 0 duplicates"):
        _compile(Path(((0.0, 0.0, 0.0), (1.0 * um, 0.0, 0.0)), DURATION))


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one waypoint"):
        Path((), DURATION)
    with pytest.raises(ValueError, match="finite and positive"):
        Path(ARC, 0.0)
    with pytest.raises(ValueError, match="must be finite"):
        Path(((np.nan, 0.0, 0.0),), DURATION)
    with pytest.raises(ValueError, match="triple"):
        Path(((1.0, 2.0),), DURATION)  # type: ignore[arg-type]
    # points are normalized to float triples
    path = Path(([1, 2, 3],), DURATION)  # type: ignore[arg-type]
    assert path.points == ((1.0, 2.0, 3.0),)


# =========================================================================== as a move


def test_chaining_lift_path_hold_is_continuous_and_synthesizes():
    """``Lift -> Path -> Hold`` compiles continuously and Eq. S19 band-checks the circle."""
    spec = TrajectorySpec(
        array=ArraySpec(),
        moves=(
            Lift(3.0 * um, 30.0 * us),
            Path(_circle(), DURATION, closed=True),
            Hold(20.0 * us),
        ),
    )
    x, y, z = spec.compile()
    assert spec.duration == pytest.approx(30.0 * us + DURATION + 20.0 * us)
    for poly in (x, y, z):
        assert poly.domain == (0.0, spec.duration)
        assert poly.n_segments == 1 + N_POINTS + 1
    # the circle runs at the lifted height, in the plane, and returns home
    t = np.linspace(30.0 * us, 30.0 * us + DURATION, 401)
    np.testing.assert_allclose(z(t), 3.0 * um, rtol=1e-12)
    assert float(x(spec.duration)) == pytest.approx(0.0, abs=1e-12)
    assert float(y(spec.duration)) == pytest.approx(0.0, abs=1e-12)
    assert np.min(x(t)) == pytest.approx(-2.0 * RADIUS, rel=0.05)  # the far side of the circle

    wfs = synthesize(spec, default_1030())  # band check included
    assert set(wfs.channels) == {"Ax", "Bx", "Ay", "By"}
    plan = plan_motion(spec)
    assert plan.report.mode == "s19"
    assert plan.report.worst_margin[1] > 0.0
