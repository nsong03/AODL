r"""M6 §4: the cross-validation gate — the FFT checker against the analytic simulator.

The two paths share nothing but :mod:`aodl.params` and :mod:`aodl.device.conventions`.  One
reads a parametric :class:`~aodl.waveform.tones.WaveformSet`, expands the retarded phase to
second order in the aperture coordinate (Eqs. S5-S6) and evaluates closed-form Gaussians; the
other reads a **float64 sample buffer**, measures the drive back off it with an FFT, rebuilds
the pupil point by point with no expansion at all, and transforms it with a chirp-z.  In the
``weak`` pupil model they are modelling the *same* physics (Eq. S3 to first order), so they
must agree — and where they agree to 1e-5 on fields, a sign or scale error shared by synthesis
and simulation is the only thing left that could hide.

**Why ``oversample=2``.**  :func:`aodl.check.demod.sample_baseband` interpolates the measured
baseband with a cubic Hermite kernel whose error is ``0.016 (2 pi f_bb / f_s)^3`` — 4.4e-7 at a
3 MHz detuning but 5.5e-5 at the 15 MHz band edge, which is the same size as the gates below.
Doubling the demodulation grid divides that by eight and restores a factor ≥ 16 of headroom
everywhere in the band.  The verdict path (``tests/test_check_verdict.py``) stays at
``oversample=1``: its tolerances are ``0.05 w_0``-scale, three orders of magnitude away.

The frames sit at ``t >= 2 tau``, where the ±4.99 ``w_in`` aperture grid is filled end to end
(:class:`aodl.check.pupil.ApertureGrid`) — except for the last test, which deliberately looks
at a half-filled aperture and compares it loosely.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aodl.check.demod import demodulate
from aodl.check.expect import Expectation
from aodl.check.pupil import ApertureGrid, axis_pupil
from aodl.check.record import from_arrays
from aodl.check.report import check_samples, frame_reach
from aodl.check.transform import zoom_field
from aodl.device.aodl import build_terms
from aodl.engine import simulate
from aodl.field.focal import term_field
from aodl.poly import PiecewisePoly
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us
from aodl.waveform.export import DEFAULT_SAMPLE_RATE, render_samples
from aodl.waveform.shepard import ShepardConfig
from aodl.waveform.synthesis import synthesize
from aodl.waveform.tones import ChannelWaveform, ToneTrack, WaveformSet

RATE = DEFAULT_SAMPLE_RATE
OVERSAMPLE = 2


def _weak_params(params):
    """The same hardware with the strictly linear crystal model the checker's ``weak`` mode is."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _baseband(wfs, params, times, rate=RATE):
    """Render the span the given frames need, and demodulate it at ``oversample=2``."""
    grid = ApertureGrid.design(params, "weak")
    reach = frame_reach(grid, params)
    tau = max(aod.transit_time for aod in params.channels.values())
    t0, t1 = wfs.t_span
    span = (
        max(t0, float(np.min(times)) - 0.5 * tau - reach - 5.0 * us),
        min(t1, float(np.max(times)) - 0.5 * tau + reach + 5.0 * us),
    )
    arrays, scale = render_samples(wfs, rate, span, dtype=np.float64, return_scale=True)
    rec = from_arrays(arrays, rate, params, t_start=span[0], normalization=scale)
    return rec, demodulate(rec, oversample=OVERSAMPLE)


def _checker_field(bb, t, grid, params, xs, ys, z_lab):
    """The checker's complex field on ``xs`` x ``ys``: two axis pupils, two zoom transforms."""
    ux = zoom_field(axis_pupil(bb, "x", t, grid, mode="weak"), grid, params.optics, xs, z_lab)
    uy = zoom_field(axis_pupil(bb, "y", t, grid, mode="weak"), grid, params.optics, ys, z_lab)
    return ux[:, None] * uy[None, :]


def _sim_field(wfs, params, t, xs, ys, z_lab):
    """The simulator's complex field on the same grid — every term summed (Eq. S7)."""
    terms = build_terms(wfs, t, tuple(wfs.channels))
    x2, y2 = np.meshgrid(xs, ys, indexing="ij")
    return term_field(terms, params.optics, x2, y2, z_lab).sum(axis=0)


def _relative(got, want):
    """Largest difference between two fields, each first divided by its own peak value."""
    peak = int(np.argmax(np.abs(want)))
    a = np.asarray(got).ravel() / np.asarray(got).ravel()[peak]
    b = np.asarray(want).ravel() / np.asarray(want).ravel()[peak]
    return float(np.abs(a - b).max() / np.abs(b).max())


# ================================================================= 1. M1: one deflector


@pytest.mark.parametrize(
    ("label", "start", "sweep"), [("static", 3.0 * MHz, 0.0), ("chirped", -2.0 * MHz, 5.0 * MHz)]
)
def test_one_channel_field_matches_the_simulator(params1030, label, start, sweep) -> None:
    """A single ``Ay`` tone, static and linearly chirped: the two paths' fields agree to 1e-4.

    The chirped case is the one that matters — a chirp is a cylindrical lens, so the pupil
    carries the quadratic phase the simulator *expands* to and the checker samples *exactly*.
    A **linear** chirp has no third derivative, so Eqs. S5-S6 are not an approximation at all
    here and the two paths agree at round-off (measured 3e-10); the 1e-4 gate is the checker's
    own reconstruction floor, not the model's.
    """
    params = _weak_params(params1030)
    span = 60.0 * us
    freq = PiecewisePoly.from_segment_coeffs([0.0, span], [[start, sweep]])
    wfs = WaveformSet({"Ay": ChannelWaveform((ToneTrack(freq=freq, phase0=0.4),))}, params)
    tau = params.channels["Ay"].transit_time
    t = 2.0 * tau
    grid = ApertureGrid.design(params, "weak")
    _, bb = _baseband(wfs, params, [t])

    y0 = -params.deflection_scale * float(freq(t - 0.5 * tau))
    waist = params.optics.waist0
    xs = np.linspace(-4.0 * waist, 4.0 * waist, 41)
    ys = np.linspace(y0 - 6.0 * waist, y0 + 6.0 * waist, 81)

    got = _checker_field(bb, t, grid, params, xs, ys, 0.0)
    want = _sim_field(wfs, params, t, xs, ys, 0.0)
    error = _relative(got, want)
    assert error < 1e-4, f"{label}: {error:.3g}"


# ============================================================ 2. M3: the 2x2 user story


@pytest.fixture(scope="module")
def story_2x2():
    """The M3 lift-traverse-lower on a 2x2 array — Eq. S19, all four channels driven."""
    from aodl.params import default_1030

    params = _weak_params(default_1030())
    spec = TrajectorySpec(
        array=ArraySpec(2, 2, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz),
        moves=(
            Lift(5 * um, 40 * us),
            Translate(15 * um, 10 * um, 60 * us),
            Lift(-5 * um, 40 * us),
        ),
    )
    return spec, params, synthesize(spec, params)


def _patch(spec, params, t):
    """A 3-waist-margin image patch around the whole array at frame time ``t``."""
    traps = Expectation(spec=spec, params=params).traps(t)
    waist = params.optics.waist0
    return (
        np.linspace(traps.columns[0] - 3.0 * waist, traps.columns[-1] + 3.0 * waist, 121),
        np.linspace(traps.rows[0] - 3.0 * waist, traps.rows[-1] + 3.0 * waist, 121),
        float(traps.z),
    )


@pytest.mark.parametrize("drive_time", [20.0 * us, 70.0 * us, 120.0 * us])
def test_the_2x2_story_field_matches_the_simulator(story_2x2, drive_time) -> None:
    """Four channels, four traps: fields agree to 1e-4 wherever Eqs. S5-S6 are exact.

    The instants chosen are the **midpoints** of the three min-jerk moves, where the frequency
    law's second derivative vanishes and the simulator's quadratic pupil is therefore not an
    approximation at all — measured agreement 1e-9, five orders inside the gate.  Mid-ramp is a
    different story, and it has its own test below.
    """
    spec, params, wfs = story_2x2
    tau = params.channels["Ax"].transit_time
    t = drive_time + 0.5 * tau
    grid = ApertureGrid.design(params, "weak")
    _, bb = _baseband(wfs, params, [t])
    xs, ys, z_lab = _patch(spec, params, t)

    got = _checker_field(bb, t, grid, params, xs, ys, z_lab)
    want = _sim_field(wfs, params, t, xs, ys, z_lab)
    error = _relative(got, want)
    assert error < 1e-4, f"t_c = {drive_time / us:g} us: {error:.3g}"
    assert error < 1e-7  # the measured floor; a regression would show up here first


def test_the_checker_measures_the_simulators_dropped_coma_term(story_2x2) -> None:
    """Mid-ramp the two paths part company — by exactly the cubic term Eqs. S5-S6 drop.

    ``docs/guide.md`` §6.2 states the model limit ("exact for linear chirps, with the cubic
    (coma) term dropped") without a number for a *curved* chirp.  Here is the number: on the
    2x2 story's min-jerk traverse the aperture-cubic phase reaches ``2 pi fddot (u/v)^3 / 6``
    and the fields differ by 5 %.  It scales as ``fddot ~ 1/T^2``, so doubling the move's
    duration cuts it about fourfold — which is what makes this the *model* talking rather than
    the checker.
    """
    spec, params, wfs = story_2x2
    tau = params.channels["Ax"].transit_time
    t = 55.0 * us + 0.5 * tau  # a quarter of the way through the traverse: |fddot| is near peak
    grid = ApertureGrid.design(params, "weak")
    _, bb = _baseband(wfs, params, [t])
    xs, ys, z_lab = _patch(spec, params, t)
    fast = _relative(
        _checker_field(bb, t, grid, params, xs, ys, z_lab),
        _sim_field(wfs, params, t, xs, ys, z_lab),
    )
    assert 1e-2 < fast < 1e-1

    slow_spec = TrajectorySpec(
        array=spec.array,
        moves=(
            Lift(5 * um, 40 * us),
            Translate(15 * um, 10 * um, 120 * us),
            Lift(-5 * um, 40 * us),
        ),
    )
    slow = synthesize(slow_spec, params)
    t_slow = 70.0 * us + 0.5 * tau  # the same fraction into a traverse twice as long
    _, bb_slow = _baseband(slow, params, [t_slow])
    xs, ys, z_lab = _patch(slow_spec, params, t_slow)
    gentle = _relative(
        _checker_field(bb_slow, t_slow, grid, params, xs, ys, z_lab),
        _sim_field(slow, params, t_slow, xs, ys, z_lab),
    )
    assert gentle < fast / 3.0


def test_the_2x2_story_positions_and_powers_match_the_simulator(story_2x2) -> None:
    """The verdict table against ``simulate``: 1e-3 ``w_0`` on position, 3e-4 on the pattern."""
    spec, params, wfs = story_2x2
    tau = params.channels["Ax"].transit_time
    times = np.array([20.0 * us, 70.0 * us, 120.0 * us]) + 0.5 * tau
    rec, _ = _baseband(wfs, params, times)

    run = simulate(wfs, times)
    report = check_samples(
        rec,
        Expectation(spec=spec, params=params),
        times=times,
        mode="weak",
        sim=run,
        k_subtimes=32,
        oversample=OVERSAMPLE,
    )
    delta = report.sim_delta
    assert delta is not None
    assert delta["n_matched"] == delta["n_rows"] == 12.0
    assert delta["max_dx"] < 1e-3 * params.optics.waist0
    assert delta["max_dy"] < 1e-3 * params.optics.waist0
    assert delta["max_dpower"] < 3e-4
    assert delta["max_dz"] < 1e-2 * params.optics.rayleigh
    assert report.passed, report.summary()


# ============================================================ 3. M4: a Shepard plateau


def test_a_shepard_hold_matches_the_simulator_away_from_a_fade(params1030) -> None:
    """One fading-Shepard tweezer, sampled on a plateau: no hand-over, one term, 1e-4.

    Between hand-overs exactly one rung of each ladder is live, so the scene is a single
    Eq. S7 term and the comparison is as clean as the M1 one — which is the point: it isolates
    the *ladder* bookkeeping from the fading algebra.
    """
    params = _weak_params(params1030)
    spec = TrajectorySpec(
        array=ArraySpec(1, 1),
        moves=(Lift(10 * um, 60 * us), Hold(400 * us), Lift(-10 * um, 60 * us)),
    )
    wfs = synthesize(spec, params, shepard=ShepardConfig(8.0 * MHz, 6.5 * MHz))
    from aodl.api import fade_schedule

    events = [event.time for event in fade_schedule(wfs, params)]
    tau = params.channels["Ax"].transit_time
    # the widest gap between consecutive hand-overs, sampled at its middle (plus the tau/2 lag)
    gaps = np.diff(np.asarray(events))
    best = int(np.argmax(gaps))
    t = 0.5 * (events[best] + events[best + 1]) + 0.5 * tau
    assert min(abs(t - 0.5 * tau - e) for e in events) > 2.0 * tau

    grid = ApertureGrid.design(params, "weak")
    _, bb = _baseband(wfs, params, [t])
    waist = params.optics.waist0
    axis = np.linspace(-5.0 * waist, 5.0 * waist, 81)
    z_lab = float(Expectation(spec=spec, params=params).traps(t).z)

    got = _checker_field(bb, t, grid, params, axis, axis, z_lab)
    want = _sim_field(wfs, params, t, axis, axis, z_lab)
    assert _relative(got, want) < 1e-4


# ========================================================== 4. the transient, loosely


def test_a_half_filled_aperture_agrees_only_loosely(story_2x2) -> None:
    """At ``0.75 tau`` the two paths window the aperture differently — 1e-2, and marked.

    The simulator applies the exact two-sided fill window of a counter-propagating pair to a
    *Taylor-expanded* pupil; the checker applies a hard mask to a pupil it sampled point by
    point, on a grid that runs wider than the crystal.  Both are right about the physics and
    they differ at the percent level on the edges, which is exactly why
    :func:`aodl.check.report.check_samples` marks such frames transient and leaves them out of
    the waist and uniformity gates.
    """
    spec, params, wfs = story_2x2
    tau = params.channels["Ax"].transit_time
    t = 0.75 * tau
    grid = ApertureGrid.design(params, "weak")
    _, bb = _baseband(wfs, params, [t, 2.0 * tau])

    traps = Expectation(spec=spec, params=params).traps(t)
    waist = params.optics.waist0
    xs = np.linspace(traps.columns[0] - 3.0 * waist, traps.columns[-1] + 3.0 * waist, 81)
    ys = np.linspace(traps.rows[0] - 3.0 * waist, traps.rows[-1] + 3.0 * waist, 81)

    got = _checker_field(bb, t, grid, params, xs, ys, float(traps.z))
    want = _sim_field(wfs, params, t, xs, ys, float(traps.z))
    loose = _relative(got, want)
    assert loose < 1e-2
    assert loose > 1e-6, "a filling aperture is *not* expected to match to the tight gate"

    # ... and the report says so
    rec, _ = _baseband(wfs, params, [t])
    report = check_samples(
        rec,
        Expectation(spec=spec, params=params),
        times=[t],
        mode="weak",
        k_subtimes=8,
        oversample=OVERSAMPLE,
    )
    assert np.all(report.table["transient"] > 0.5)
    assert any("transient" in note for note in report.notes)
