r"""M1 acceptance, end to end: one AOD, a moving astigmatic tweezer (``docs/PLAN.md`` §3).

Everything here runs through the real product path — ``ramps`` -> ``WaveformSet`` ->
:func:`aodl.engine.simulate` -> metrics and rendered intensity frames — and every claim is
checked twice: once against the closed-form prediction that :mod:`aodl.field.measure` reports,
and once against the *rendered field*, whose peak position and width know nothing about
Table I.  The four claims are ``PLAN.md``'s M1 ticks:

1. the spot follows ``-deflection_scale * f(t - tau/2)`` — position, and *retardation*;
2. the astigmatic interval is ``Delta F = -lens_scale * fdot(t - tau/2)`` (single y channel:
   ``docs/conventions.md`` §6), i.e. the y focus really sits where the chirp puts it;
3. one AOD makes a *cylindrical* lens, so the spot is visibly elongated in the lab focal
   plane while it moves;
4. the aperture fill transient is there, and lasts about a beam transit.

Plus the bookkeeping trap that the layer boundary invites: the envelope ``A(t_c)`` lives in
the line amplitude and ``alpha`` carries only its *shape*, so drive power must scale as
``A^2`` and never ``A^4`` (``docs/conventions.md`` §3, bookkeeping note).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import erfcinv

from aodl.engine import simulate
from aodl.field.focal import FrameGrid
from aodl.poly import PiecewisePoly
from aodl.trajectory import ramps
from aodl.units import MHz, ms, us
from aodl.waveform.tones import ChannelWaveform, SmoothOnOff, ToneTrack, WaveformSet

#: The M1 move: 0 -> 5 MHz of detuning on Ay, minimum-jerk, in 100 us.
SWEEP_SPAN = 5.0 * MHz
SWEEP_TIME = 100.0 * us


@pytest.fixture
def sweep(params1030):
    """Min-jerk Ay sweep, held past the last frame, plus its exact frequency law."""
    tau = params1030.channels["Ay"].transit_time
    freq = ramps.min_jerk(0.0, SWEEP_TIME, 0.0, SWEEP_SPAN)
    wfs = WaveformSet({"Ay": ChannelWaveform((ToneTrack(freq=freq),))}, params1030).with_hold_until(
        SWEEP_TIME + tau
    )
    return wfs, freq, tau


def _linear_drive(params):
    """``params`` with every channel at ``mixing_order=1`` — the strictly linear model.

    The product default is ``mixing_order=3``: the crystal compresses, so a fundamental's
    amplitude is ``(i/2) m (1 - m^2/8 - ...)`` rather than ``(i/2) m`` (Eqs. S20-S22).  The
    envelope-bookkeeping test below asks whether ``A`` is counted once or twice, which is an
    *exact* first-order statement, so it runs the linear model.
    """
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _static(params, detuning: float = 3.0 * MHz, span: float = 40.0):
    """Constant-detuning Ay tone, programmed over ``span`` transit times."""
    tau = params.channels["Ay"].transit_time
    tone = ToneTrack(freq=PiecewisePoly.constant(detuning, 0.0, span * tau))
    return WaveformSet({"Ay": ChannelWaveform((tone,))}, params), tau


# ------------------------------------------------------------------ rendered-cut helpers


def _cut(result, i, axis: str, centre: tuple[float, float], half: float, plane: float, n=401):
    """1-D intensity cut through ``centre`` along ``axis`` at lab plane ``plane``.

    Uses the engine's own frame evaluator on a three-pixel-wide strip, so the profile comes
    from exactly the code a movie renders with.
    """
    x_c, y_c = centre
    thin = 1e-4 * half
    if axis == "y":
        grid = FrameGrid(x_c - thin, x_c + thin, 3, y_c - half, y_c + half, n)
        return grid.y, result.frame(i, grid, z_lab=plane)[:, 1]
    grid = FrameGrid(x_c - half, x_c + half, n, y_c - thin, y_c + thin, 3)
    return grid.x, result.frame(i, grid, z_lab=plane)[1, :]


def _peak(coord, intensity) -> float:
    """Sub-sample peak position.  Parabolic in ``log I`` — exact for a Gaussian profile."""
    i = int(np.argmax(intensity))
    assert 0 < i < intensity.size - 1, "peak fell on the edge of the cut"
    y0, y1, y2 = np.log(intensity[i - 1 : i + 2])
    return float(coord[i] + 0.5 * (coord[1] - coord[0]) * (y0 - y2) / (y0 - 2.0 * y1 + y2))


def _width(coord, intensity) -> float:
    """Intensity 1/e^2 radius from the second moment (``w = 2 sigma`` for a Gaussian)."""
    total = float(intensity.sum())
    mean = float(intensity @ coord) / total
    var = float(intensity @ (coord - mean) ** 2) / total
    return 2.0 * float(np.sqrt(var))


def _defocused_waist(optics, offset: float) -> float:
    """``waist0 sqrt(1 + (offset / z_R)^2)`` — the textbook width ``offset`` from a focus."""
    return optics.waist0 * float(np.sqrt(1.0 + (offset / optics.rayleigh) ** 2))


# =========================================================== (a) position and retardation


def test_spot_tracks_the_retarded_frequency(sweep, params1030) -> None:
    """``Y_spot(t) = -deflection_scale f(t - tau/2)``, to 1% of a waist once the aperture fills."""
    wfs, freq, tau = sweep
    optics = params1030.optics
    times = np.linspace(0.0, SWEEP_TIME + tau, 61)
    result = simulate(wfs, times)
    table = result.spot_table()

    predicted = -params1030.deflection_scale * freq(times - 0.5 * tau)
    assert np.max(np.abs(table["y"] - predicted)) < 0.01 * optics.waist0
    np.testing.assert_allclose(table["x"], 0.0, atol=1e-18)  # nothing drives x
    assert table["y"][-1] == pytest.approx(-params1030.deflection_scale * SWEEP_SPAN, rel=1e-12)

    # Teeth: the un-retarded law is wrong by many waists in mid-sweep, so the 1% band above is
    # a real statement about tau/2 and not just about the ramp.
    naive = -params1030.deflection_scale * freq(times)
    assert np.max(np.abs(naive - predicted)) > 3.0 * optics.waist0

    # The rendered field agrees: the peak of a cut through the spot lands where predicted.
    for i in np.flatnonzero(times > tau)[:: max(1, int(np.sum(times > tau)) // 4)]:
        i = int(i)
        y_c = float(predicted[i])
        coord, profile = _cut(result, i, "y", (0.0, y_c), 3.0 * optics.waist0, plane=0.0)
        assert abs(_peak(coord, profile) - y_c) < 0.01 * optics.waist0


# ============================================================ (b) astigmatic interval = Table I


def test_astigmatic_interval_matches_the_chirp_rate(sweep, params1030) -> None:
    """``Delta F = Z_x - Z_y = -lens_scale fdot(t_c)``, confirmed by where the y focus lands."""
    wfs, freq, tau = sweep
    optics = params1030.optics
    fdot = freq.derivative()
    probes = np.array([0.25, 0.5, 0.75]) * SWEEP_TIME + 0.5 * tau  # three points along the ramp
    result = simulate(wfs, probes)

    for i, t in enumerate(probes):
        rate = float(fdot(t - 0.5 * tau))
        assert rate > 0.0
        # Only fdot_Ay is nonzero: Z_y,lab = +lens_scale fdot, Z_x,lab = 0, so Table I's
        # Zbar = lens_scale fdot / 2 and Delta F = Z_x - Z_y = -lens_scale fdot.
        z_y = params1030.lens_scale * rate
        metrics = result.metrics[i][0]
        assert metrics.delta_f == pytest.approx(-z_y, rel=0.02)
        assert metrics.z_lab == pytest.approx(0.5 * z_y, rel=0.02)
        assert metrics.sigma_astig == pytest.approx(-z_y / optics.rayleigh, rel=0.02)

        # Rendered check: the y width is minimal at Z_y and grows a Rayleigh range either
        # side, while the x width is minimal at Z = 0 (no chirp on the x pupil at all).
        centre = (0.0, metrics.y)
        at_focus = _width(*_cut(result, i, "y", centre, 3.0 * optics.waist0, plane=z_y))
        assert at_focus == pytest.approx(optics.waist0, rel=0.03)
        for offset in (-optics.rayleigh, +optics.rayleigh):
            half = 3.0 * _defocused_waist(optics, offset)
            off = _width(*_cut(result, i, "y", centre, half, plane=z_y + offset))
            assert off == pytest.approx(_defocused_waist(optics, offset), rel=0.03)
            assert off > 1.3 * at_focus

        x_focus = _width(*_cut(result, i, "x", centre, 3.0 * optics.waist0, plane=0.0))
        assert x_focus == pytest.approx(optics.waist0, rel=0.03)
        x_off = _width(
            *_cut(
                result,
                i,
                "x",
                centre,
                3.0 * _defocused_waist(optics, optics.rayleigh),
                plane=optics.rayleigh,
            )
        )
        assert x_off > 1.3 * x_focus


# ================================================================= (c) visible astigmatism


def test_spot_is_visibly_astigmatic_at_the_chirp_peak(sweep, params1030) -> None:
    """In the lab focal plane the spot is elongated along the *driven* axis (here y).

    ``WO-05`` §5(c) asks for ``wx/wy > 1.05``; with the M1 channel being ``Ay`` the cylindrical
    lens sits on the y pupil, so the elongation is ``wy/wx`` (``docs/conventions.md`` §6, worked
    single-channel case).  The ratio is the astigmatism itself: ``sqrt(1 + sigma_astig^2)``.
    """
    wfs, freq, tau = sweep
    optics = params1030.optics
    peak_rate = float(freq.derivative()(0.5 * SWEEP_TIME))
    result = simulate(wfs, [0.5 * SWEEP_TIME + 0.5 * tau])
    metrics = result.metrics[0][0]
    centre = (0.0, metrics.y)

    z_y = params1030.lens_scale * peak_rate
    expected = _defocused_waist(optics, z_y) / optics.waist0
    assert expected > 2.5  # the min-jerk peak chirp is ~94 MHz/ms: a strongly astigmatic spot

    wy = _width(*_cut(result, 0, "y", centre, 3.0 * _defocused_waist(optics, z_y), plane=0.0))
    wx = _width(*_cut(result, 0, "x", centre, 3.0 * optics.waist0, plane=0.0))
    assert wy / wx > 1.05
    assert wy / wx == pytest.approx(expected, rel=0.03)
    assert wx == pytest.approx(optics.waist0, rel=0.03)
    assert abs(metrics.sigma_astig) == pytest.approx(z_y / optics.rayleigh, rel=1e-9)

    # At the tracked plane the two axes are equally defocused, so the spot is round again -
    # the circle of least confusion, sitting half way between the two line foci.
    round_y = _width(*_cut(result, 0, "y", centre, 4.0 * metrics.wy, plane=metrics.z_lab))
    round_x = _width(*_cut(result, 0, "x", centre, 4.0 * metrics.wx, plane=metrics.z_lab))
    assert round_y / round_x == pytest.approx(1.0, rel=0.03)
    assert round_x == pytest.approx(metrics.wx, rel=0.03)


# ==================================================================== (d) fill transient


def test_fill_transient_lasts_about_one_transit(params1030) -> None:
    """Light builds up as the acoustic column crosses the beam and plateaus at ``t = tau``."""
    wfs, tau = _static(params1030)
    optics = params1030.optics
    times = np.concatenate([np.linspace(0.0, 1.4 * tau, 29), [3.0 * tau]])
    result = simulate(wfs, times)
    power = result.spot_table()["power"]
    full = float(power[-1])

    rising, filled = times[:-1] < tau, times[:-1] >= tau
    assert np.all(np.diff(power[:-1][rising]) > 0.0)  # monotone rise, no overshoot
    assert np.all(np.diff(power[:-1]) > -1e-12 * full)  # and it never falls back
    np.testing.assert_allclose(power[:-1][filled], full, rtol=1e-9)  # flat once full
    assert power[0] < 1e-3 * full  # nothing diffracted before the sound reaches the beam

    grid = FrameGrid(
        x0=-4.0 * optics.waist0,
        x1=4.0 * optics.waist0,
        nx=81,
        y0=result.metrics[0][0].y - 4.0 * optics.waist0,
        y1=result.metrics[0][0].y + 4.0 * optics.waist0,
        ny=81,
    )
    early = simulate(wfs, [0.4 * tau, 3.0 * tau])
    ratio_i = float(early.frame(0, grid).max() / early.frame(1, grid).max())
    ratio_p = early.metrics[0][0].power / early.metrics[1][0].power
    assert 0.05 < ratio_i < 0.95
    assert 0.05 < ratio_p < 0.95

    # The rise is set by the *beam*, not the aperture: the wavefront crosses the beam centre
    # at t = tau/2 and the diffracted power follows the Gaussian's integral,
    # P(t)/P_full = erfc(u_edge sqrt(2) / w_in) / 2 with u_edge = D/2 - v t, so the 10-90%
    # rise takes 2 erfcinv(0.2) w_in / (sqrt(2) v) - a fraction of the transit time across the
    # 1/e^2 diameter quoted in docs/PLAN.md §1.5, and nothing like the aperture transit tau.
    speed = params1030.channels["Ay"].sound_speed
    beam_transit = 2.0 * optics.w_in / speed
    expected_rise = 2.0 * float(erfcinv(0.2)) * optics.w_in / (np.sqrt(2.0) * speed)
    fine = np.linspace(0.0, 1.2 * tau, 481)
    curve = simulate(wfs, fine).spot_table()["power"] / full
    assert fine[np.searchsorted(curve, 0.5)] == pytest.approx(0.5 * tau, rel=0.05)
    width = fine[np.searchsorted(curve, 0.9)] - fine[np.searchsorted(curve, 0.1)]
    assert width == pytest.approx(expected_rise, rel=0.05)
    assert 0.4 * beam_transit < width < beam_transit < tau
    assert tau == pytest.approx(11.54 * us, rel=1e-3)  # docs/PLAN.md §1.5


# ======================================================= envelope bookkeeping (no double count)


def test_envelope_amplitude_is_not_counted_twice(params1030) -> None:
    """Halfway up a gate, ``A = 1/2`` must cost a factor 4 in power - not 16.

    ``device/aod.py`` folds ``A(t_c)`` into the complex line amplitude (Eq. S3) and
    ``device/aodl.py`` therefore multiplies only the *normalized* Eq. S5 shape
    ``(1, -s (A'/A)/v, (A''/A)/(2 v^2))``.  If ``alpha0`` carried ``A`` as well, every power
    would pick up ``A^2`` twice.  The ramp is deliberately slow (1 ms, ~90 transit times) so
    the amplitude-tilt term ``alpha1`` contributes < 1e-4 here and the comparison is clean.
    """
    params = _linear_drive(params1030)  # exact A-scaling: order-1 mixing
    tau = params.channels["Ay"].transit_time
    duration, ramp = 3.0 * ms, 1.0 * ms
    env = SmoothOnOff(t_on=0.0, t_off=duration, ramp=ramp)
    tone = ToneTrack(freq=PiecewisePoly.constant(2.0 * MHz, 0.0, duration), env=env)
    wfs = WaveformSet({"Ay": ChannelWaveform((tone,))}, params)

    # sin^2 reaches 1/2 half way up the rise; remember the frame time is the drive time plus
    # the beam-centre retardation.
    t_half, t_full = 0.5 * ramp, 1.5 * ms
    assert float(env.A(t_half)) == pytest.approx(0.5, rel=1e-12)
    assert float(env.A(t_full)) == pytest.approx(1.0, rel=1e-12)
    result = simulate(wfs, [t_half + 0.5 * tau, t_full + 0.5 * tau])

    half, full = result.metrics[0][0], result.metrics[1][0]
    assert half.power / full.power == pytest.approx(0.25, rel=1e-3)
    assert half.power / full.power != pytest.approx(0.0625, rel=0.1)  # the double-count answer

    # Same statement through the rendered field: the frame integral is the group power
    # (Parseval, see field/measure.py), so it must scale the same way.
    optics = params.optics
    grid = FrameGrid(
        x0=-8.0 * optics.waist0,
        x1=8.0 * optics.waist0,
        nx=161,
        y0=full.y - 8.0 * optics.waist0,
        y1=full.y + 8.0 * optics.waist0,
        ny=161,
    )
    integral = [float(result.frame(i, grid, z_lab=0.0).sum()) * grid.dx * grid.dy for i in (0, 1)]
    assert integral[1] == pytest.approx(full.power, rel=1e-3)
    assert integral[0] / integral[1] == pytest.approx(0.25, rel=1e-3)

    # The envelope shape still reaches the pupil: alpha1 is the Eq. S5 amplitude tilt.
    terms = result.terms(0)
    expected_tilt = (
        -(params.channels["Ay"].sound_speed ** -1)
        * (-1)
        * float(env.dA(t_half))
        / float(env.A(t_half))
    )
    assert terms.alpha[1, 0, 0] == pytest.approx(1.0, rel=1e-12)  # normalized, by construction
    assert complex(terms.alpha[1, 1, 0]).real == pytest.approx(expected_tilt, rel=1e-12)
    assert abs(terms.c[0]) == pytest.approx(
        0.5 * params.channels["Ay"].drive_strength * 0.5, rel=1e-12
    )
