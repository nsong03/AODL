r"""M6 §1: the checker's expectation — Table I positions from the request alone.

:class:`aodl.check.expect.Expectation` is the half of a verdict that never touches a
waveform, a pupil or a sample: it takes the requested trajectory and the array, applies the
retardation convention and Table I, and says where every tweezer *should* be.  These tests
build the same answer by hand — ``X + deflection_scale * (n - (M-1)/2) delta_f``, evaluated at
``t - tau/2`` — and pin the two things samples cannot carry: the retardation mode and the fade
whitelist.

``sim_delta`` is exercised against a hand-built stub rather than a real
:class:`aodl.engine.SimResult`, which is the point of it being a ``Protocol``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from aodl.check.expect import Expectation, sim_delta
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us

STORY = TrajectorySpec(
    array=ArraySpec(3, 2, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz),
    moves=(Lift(4 * um, 40 * us), Translate(10 * um, 6 * um, 60 * us), Lift(-4 * um, 40 * us)),
)


# ============================================================== 1. the retardation clamp


@pytest.mark.parametrize("compensated", [False, True])
def test_eval_time_clamps_at_both_ends_in_both_retard_modes(params1030, compensated) -> None:
    """``clamp(t - tau/2, 0, T)``, or ``clamp(t, 0, T)`` when the drive reads ahead."""
    expect = Expectation(spec=STORY, params=params1030, retard_compensated=compensated)
    tau = params1030.channels["Ax"].transit_time
    lead = 0.0 if compensated else 0.5 * tau
    duration = STORY.duration
    assert expect.duration == pytest.approx(duration)

    for t in (0.0, 3.0 * us, 50.0 * us, duration, duration + 3.0 * tau):
        assert expect.eval_time(t) == pytest.approx(min(max(t - lead, 0.0), duration), abs=1e-15)
    # before the drive starts nothing has been asked for, and past T it holds its last state
    assert expect.eval_time(-5.0 * us) == 0.0
    assert expect.eval_time(10.0 * duration) == pytest.approx(duration)
    # the per-channel overload uses that channel's own tau
    assert expect.eval_time(50.0 * us, params1030.channels["By"]) == pytest.approx(
        expect.eval_time(50.0 * us)
    )


def test_the_two_retard_modes_differ_by_exactly_half_a_transit(params1030) -> None:
    """The whole point of ``retard_compensate``: the same request, half a transit apart."""
    tau = params1030.channels["Ax"].transit_time
    lagging = Expectation(spec=STORY, params=params1030)
    ahead = Expectation(spec=STORY, params=params1030, retard_compensated=True)
    t = 40.0 * us
    assert lagging.traps(t + 0.5 * tau).x == pytest.approx(ahead.traps(t).x, abs=1e-18)
    assert lagging.traps(t + 0.5 * tau).z == pytest.approx(ahead.traps(t).z, abs=1e-18)


# ================================================================ 2. Table I trap tables


def test_trap_table_is_the_hand_built_table_i_lattice(params1030) -> None:
    """``X(t_c) + deflection_scale f_x0^(n)`` per column, ``Y`` per row, one common ``Z``."""
    expect = Expectation(spec=STORY, params=params1030)
    tau = params1030.channels["Ax"].transit_time
    t = 90.0 * us
    tc = t - 0.5 * tau
    x_law, y_law, z_law = STORY.compile()
    scale = params1030.deflection_scale

    traps = expect.traps(t)
    columns = float(x_law(tc)) + scale * (np.arange(3) - 1.0) * (1.0 * MHz)
    rows = float(y_law(tc)) + scale * (np.arange(2) - 0.5) * (1.3 * MHz)
    np.testing.assert_allclose(traps.columns, columns, rtol=0, atol=1e-15)
    np.testing.assert_allclose(traps.rows, rows, rtol=0, atol=1e-15)
    assert traps.z == pytest.approx(float(z_law(tc)), abs=1e-18)
    assert traps.n_traps == 6

    # rows vary fastest inside a column, and every trap is (column, row)
    np.testing.assert_array_equal(traps.ix, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(traps.iy, [0, 1, 0, 1, 0, 1])
    np.testing.assert_allclose(traps.x, columns[traps.ix], rtol=0, atol=1e-15)
    np.testing.assert_allclose(traps.y, rows[traps.iy], rtol=0, atol=1e-15)

    # the lateral velocity is the compiled law's own derivative at the same instant
    step = 1e-9
    assert traps.vx == pytest.approx(
        (float(x_law(tc + step)) - float(x_law(tc - step))) / (2.0 * step), rel=1e-5
    )
    assert traps.speed == pytest.approx(max(abs(traps.vx), abs(traps.vy)))


def test_the_pitch_is_the_paper_scale(params1030) -> None:
    """10.3 µm per MHz — the same Table I mapping ``ArraySpec.pitch`` reports."""
    expect = Expectation(spec=STORY, params=params1030)
    traps = expect.traps(30.0 * us)
    pitch_x, pitch_y = STORY.array.pitch(params1030)
    assert float(np.diff(traps.columns)[0]) == pytest.approx(pitch_x, rel=1e-12)
    assert float(np.diff(traps.rows)[0]) == pytest.approx(pitch_y, rel=1e-12)
    assert pitch_x / um == pytest.approx(10.3, abs=0.05)


# ================================================================== 3. lattice and fades


def test_lattice_widens_the_array_by_whole_pitches(params1030) -> None:
    """``extend=1`` is the Shepard extended grid and the commensurate-IM3 whitelist."""
    expect = Expectation(spec=STORY, params=params1030)
    t = 60.0 * us
    traps = expect.traps(t)
    pitch_x, pitch_y = STORY.array.pitch(params1030)

    xs, ys = expect.lattice(t, extend=1)
    assert xs.size == 3 + 2 and ys.size == 2 + 2
    np.testing.assert_allclose(xs[1:-1], traps.columns, rtol=0, atol=1e-15)
    assert xs[0] == pytest.approx(traps.columns[0] - pitch_x, rel=1e-12)
    assert ys[-1] == pytest.approx(traps.rows[-1] + pitch_y, rel=1e-12)

    np.testing.assert_allclose(expect.lattice(t, extend=0)[0], traps.columns, rtol=0, atol=1e-15)
    assert expect.lattice(t, extend=3)[0].size == 3 + 6
    with pytest.raises(ValueError, match="extend must be non-negative"):
        expect.lattice(t, extend=-1)

    # a single-tone axis has no lattice to widen
    single = Expectation(
        spec=TrajectorySpec(array=ArraySpec(1, 1), moves=(Hold(50 * us),)), params=params1030
    )
    assert single.lattice(30 * us, extend=2)[0].size == 1


def test_fade_whitelist_is_time_gated_and_marks_the_drive_as_fading(params1030) -> None:
    """``shadows`` carries the hand-over schedule; ``fade_pad`` decides how wide a window is."""
    offset = params1030.deflection_scale * 1.0 * MHz
    shadows = ((40.0 * us, "x", offset), (70.0 * us, "y", offset))
    expect = Expectation(spec=STORY, params=params1030, shadows=shadows, fade_pad=2.0 * us)
    assert expect.fading and expect.fading_axes == ("x", "y")
    assert expect.in_fade(40.0 * us) and expect.in_fade(41.9 * us)
    assert not expect.in_fade(45.0 * us)
    assert expect.shadow_offsets(40.5 * us) == (("x", offset),)
    assert expect.shadow_offsets(45.0 * us) == ()

    # ... and with no pad, nothing is excluded
    tight = Expectation(spec=STORY, params=params1030, shadows=shadows)
    assert not tight.in_fade(41.9 * us) and tight.fading

    plain = Expectation(spec=STORY, params=params1030)
    assert not plain.fading and plain.fading_axes == () and plain.edge_lines() == ((), ())
    with pytest.raises(ValueError, match="shadow axis"):
        Expectation(spec=STORY, params=params1030, shadows=((1.0, "z", 0.0),))
    with pytest.raises(ValueError, match="fade_pad"):
        Expectation(spec=STORY, params=params1030, fade_pad=-1.0)


def test_edge_lines_and_min_spacing_follow_the_fading_axes(params1030) -> None:
    """A fading axis hands its outermost node over, and its ladder sets the beat comb."""
    offset = params1030.deflection_scale * 1.0 * MHz
    expect = Expectation(spec=STORY, params=params1030, shadows=((40.0 * us, "x", offset),))
    assert expect.edge_lines() == ((0, 2), ())  # mx = 3 fades, my = 2 does not
    # the ladder spacing comes back out of the Eq. S31 shadow offset
    assert expect.min_spacing() == pytest.approx(1.0 * MHz, rel=1e-12)

    plain = Expectation(spec=STORY, params=params1030)
    assert plain.min_spacing() == pytest.approx(1.0 * MHz, rel=1e-12)
    lone = Expectation(
        spec=TrajectorySpec(array=ArraySpec(1, 1), moves=(Hold(50 * us),)), params=params1030
    )
    assert math.isinf(lone.min_spacing())


# ==================================================================== 4. the lab path


def test_from_table_round_trips_a_sampled_trajectory(params1030) -> None:
    """A tabulated ``(t, X, Y, Z)`` stands in for a spec, exactly at its own nodes."""
    x_law, y_law, z_law = STORY.compile()
    times = np.linspace(0.0, STORY.duration, 141)
    table = Expectation.from_table(
        times, x_law(times), y_law(times), z_law(times), STORY.array, params1030
    )
    reference = Expectation(spec=STORY, params=params1030)
    assert table.duration == pytest.approx(STORY.duration)
    assert table.spec.array == STORY.array

    tau = params1030.channels["Ax"].transit_time
    for t in times[::17] + 0.5 * tau:
        got, want = table.traps(float(t)), reference.traps(float(t))
        np.testing.assert_allclose(got.x, want.x, rtol=0, atol=1e-15)
        np.testing.assert_allclose(got.y, want.y, rtol=0, atol=1e-15)
        assert got.z == pytest.approx(want.z, abs=1e-15)

    # between the nodes the cubic Hermite interpolant is close but not exact - it is an
    # interpolation of a quintic min-jerk law, so the error is O(h^4).
    probe = 33.3 * us + 0.5 * tau
    assert table.traps(probe).z == pytest.approx(reference.traps(probe).z, abs=1e-3 * um)

    # options are forwarded, and the table's own conventions are enforced
    ahead = Expectation.from_table(
        times,
        x_law(times),
        y_law(times),
        z_law(times),
        STORY.array,
        params1030,
        retard_compensated=True,
    )
    assert ahead.retard_compensated
    with pytest.raises(ValueError, match=r"times\[0\] == 0"):
        Expectation.from_table(
            times + 1.0, x_law(times), y_law(times), z_law(times), STORY.array, params1030
        )
    scrambled = times.copy()
    scrambled[3], scrambled[4] = scrambled[4], scrambled[3]
    with pytest.raises(ValueError, match="strictly increasing"):
        Expectation.from_table(
            scrambled, x_law(times), y_law(times), z_law(times), STORY.array, params1030
        )
    with pytest.raises(ValueError, match="match times"):
        Expectation.from_table(
            times, x_law(times)[:-1], y_law(times), z_law(times), STORY.array, params1030
        )


# =============================================================== 5. the simulator diff


@dataclass(frozen=True)
class _Spot:
    """A hand-built stand-in for :class:`aodl.field.measure.SpotMetrics` (structural typing)."""

    x: float
    y: float
    z_lab: float
    wx: float
    wy: float
    power: float
    df_opt: float


@dataclass(frozen=True)
class _Run:
    """... and for :class:`aodl.engine.SimResult`."""

    times: np.ndarray
    metrics: list[list[_Spot]]


def test_sim_delta_matches_by_nearest_neighbour_and_stays_report_only(params1030) -> None:
    """Rows are paired with the nearest group inside ``tol_match``; powers compare per frame."""
    waist0 = params1030.optics.waist0
    rows = {
        "time": np.array([0.0, 0.0, 1.0, 1.0]),
        "x": np.array([0.0, 10.0 * um, 0.0, 10.0 * um]),
        "y": np.zeros(4),
        "z_lab": np.zeros(4),
        "wx": np.full(4, waist0),
        "wy": np.full(4, waist0),
        "power": np.array([1.0, 1.0, 1.0, 1.0]),
    }
    spots = [
        _Spot(0.1 * um, 0.0, 0.02 * um, waist0, waist0, 2.0, 0.0),
        _Spot(10.0 * um, 0.0, 0.0, waist0, waist0, 2.0, 1.0),
    ]
    run = _Run(times=np.array([0.0, 1.0]), metrics=[spots, spots])

    delta = sim_delta(rows, run, tol_match=1.0 * um)
    assert delta["n_rows"] == 4.0 and delta["n_matched"] == 4.0
    assert delta["max_dx"] == pytest.approx(0.1 * um, rel=1e-12)
    assert delta["max_dz"] == pytest.approx(0.02 * um, rel=1e-12)
    assert delta["max_dpower"] == pytest.approx(0.0, abs=1e-15)  # both patterns are flat
    assert delta["rms_dxy"] == pytest.approx(0.1 * um / math.sqrt(2.0), rel=1e-9)

    # too tight a match tolerance simply leaves rows unmatched rather than pairing them wrongly
    strict = sim_delta(rows, run, tol_match=0.01 * um)
    assert strict["n_matched"] == 2.0
    assert sim_delta({key: value[:0] for key, value in rows.items()}, run, 1.0 * um)["n_rows"] == 0
