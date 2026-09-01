"""Waveform IR: envelope calculus, exact chirp/phase closure, channel and set validation.

The headline property is phase continuity: ``phase`` is the *exact* antiderivative of the
piecewise-polynomial frequency law, so chaining ramps into a chirp never introduces a
phase jump (which on hardware would be an amplitude glitch and a broadband splatter).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from aodl.params import default_1030
from aodl.poly import PiecewisePoly
from aodl.trajectory import ramps
from aodl.units import MHz, us
from aodl.waveform.tones import (
    TABLE_KEYS,
    ChannelWaveform,
    ConstantEnvelope,
    Envelope,
    SmoothOnOff,
    ToneTrack,
    WaveformSet,
)

TWO_PI = 2.0 * np.pi
T_END = 120.0 * us


def _chirp_chain() -> PiecewisePoly:
    """min-jerk up, constant-accel down, hold: three families, four segments, one law."""
    return PiecewisePoly.concat(
        [
            ramps.min_jerk(0.0, 40.0 * us, 0.0, 3.0 * MHz),
            ramps.constant_accel(40.0 * us, 40.0 * us, 3.0 * MHz, -2.0 * MHz),
            ramps.hold(80.0 * us, 40.0 * us, -2.0 * MHz),
        ]
    )


def _hopping_chain() -> PiecewisePoly:
    """Same span, but the frequency law *jumps* by -5 MHz at 40 us (a tone hop).

    Phase continuity here is a real statement: the integrand is discontinuous, yet the
    integral (and therefore the RF phase) is not.
    """
    return PiecewisePoly.concat(
        [
            ramps.min_jerk(0.0, 40.0 * us, 0.0, 3.0 * MHz),
            ramps.hold(40.0 * us, 40.0 * us, -2.0 * MHz),
            ramps.linear(80.0 * us, 40.0 * us, -2.0 * MHz, 1.0 * MHz),
        ]
    )


# --------------------------------------------------------------------------- envelopes


def test_constant_envelope_values_and_shapes():
    env = ConstantEnvelope(0.75)
    assert env.A(1.0 * us) == 0.75
    assert isinstance(env.A(1.0 * us), float)
    t = np.linspace(0.0, T_END, 17)
    np.testing.assert_array_equal(env.A(t), np.full(t.shape, 0.75))
    np.testing.assert_array_equal(env.dA(t), np.zeros(t.shape))
    np.testing.assert_array_equal(env.d2A(t), np.zeros(t.shape))
    assert env.A(np.zeros((3, 2))).shape == (3, 2)
    assert ConstantEnvelope().amp == 1.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ConstantEnvelope(1.5)


def test_smooth_on_off_shape_is_a_continuous_gate():
    env = SmoothOnOff(t_on=10.0 * us, t_off=100.0 * us, ramp=20.0 * us)
    assert isinstance(env, Envelope)  # runtime-checkable protocol

    assert env.A(5.0 * us) == 0.0
    assert env.A(110.0 * us) == 0.0
    assert env.A(10.0 * us) == pytest.approx(0.0, abs=1e-15)
    assert env.A(20.0 * us) == pytest.approx(0.5, rel=1e-12)  # half-way up the sin^2 rise
    assert env.A(30.0 * us) == 1.0  # end of the rise / start of the plateau
    assert env.A(60.0 * us) == 1.0  # plateau
    assert env.A(80.0 * us) == pytest.approx(1.0, rel=1e-12)  # start of the fall
    assert env.A(90.0 * us) == pytest.approx(0.5, rel=1e-12)  # half-way down
    assert env.A(100.0 * us) == pytest.approx(0.0, abs=1e-15)

    t = np.linspace(-10.0 * us, 130.0 * us, 20001)
    a = env.A(t)
    assert np.all(a >= 0.0) and np.all(a <= 1.0)
    assert np.max(np.abs(np.diff(a))) < 1e-3  # continuous (no step) on a fine grid
    # symmetric gate
    np.testing.assert_allclose(env.A(10.0 * us + t), env.A(100.0 * us - t), atol=1e-14)

    with pytest.raises(ValueError, match="ramp must be positive"):
        SmoothOnOff(0.0, 10.0 * us, 0.0)
    with pytest.raises(ValueError, match="overlap"):
        SmoothOnOff(0.0, 10.0 * us, 6.0 * us)


def test_smooth_on_off_derivatives_match_numerical():
    env = SmoothOnOff(t_on=10.0 * us, t_off=100.0 * us, ramp=20.0 * us)
    # probe inside each branch, away from the corners where d2A is only defined a.e.
    t = np.array([13.0, 17.5, 25.0, 40.0, 70.0, 84.0, 92.5, 97.0]) * us

    h = 1.0e-9
    d1 = (env.A(t + h) - env.A(t - h)) / (2.0 * h)
    d2 = (env.A(t + 1e-8) - 2.0 * env.A(t) + env.A(t - 1e-8)) / 1e-8**2
    c = np.pi / (2.0 * env.ramp)
    np.testing.assert_allclose(env.dA(t), d1, rtol=1e-6, atol=1e-6 * c)
    np.testing.assert_allclose(env.d2A(t), d2, rtol=1e-5, atol=1e-5 * 2.0 * c**2)

    # dA is continuous everywhere (including the four corners), d2A is not
    for corner in (10.0 * us, 30.0 * us, 80.0 * us, 100.0 * us):
        assert env.dA(corner + 1e-12) == pytest.approx(env.dA(corner - 1e-12), abs=1e-3 * c)
    assert env.d2A(10.0 * us) == pytest.approx(2.0 * c**2, rel=1e-12)
    assert env.d2A(10.0 * us - 1e-12) == 0.0


# ------------------------------------------------------------------------- tone tracks


@pytest.mark.parametrize("builder", [_chirp_chain, _hopping_chain], ids=["smooth", "hopping"])
def test_phase_is_continuous_across_the_chirp_chain(builder):
    freq = builder()
    tone = ToneTrack(freq, phase0=0.4)

    # 1. exact: the phase polynomial's segment coefficients agree at every break
    phase_poly = tone._phase_poly
    scale = float(np.max(np.abs(phase_poly(np.linspace(0.0, T_END, 2001)))))
    for k in range(1, phase_poly.n_segments):
        left = float(phase_poly.coeffs[k - 1].sum())
        right = float(phase_poly.coeffs[k, 0])
        assert abs(left - right) < 1e-12 * scale

    # 2. phase(t) is 2 pi * int f dt' + phase0, checked against quadrature
    interior = [float(b) for b in freq.breaks[1:-1]]
    for t_probe in (7.0 * us, 40.0 * us, 63.0 * us, 95.0 * us, T_END):
        points = [b for b in interior if 0.0 < b < t_probe]
        value, _ = quad(freq, 0.0, t_probe, points=points or None, limit=200)
        assert tone.phase(t_probe) == pytest.approx(0.4 + TWO_PI * value, rel=1e-10)

    # 3. the rendered oscillation has no glitch at the breaks: across +/-1 ps the phase
    #    may only advance by 2 pi f_max * 2 eps (here < 4e-5 rad), never step
    eps = 1.0e-12
    f_max = float(np.max(np.abs(freq(np.linspace(0.0, T_END, 2001)))))
    tol = TWO_PI * f_max * 2.0 * eps
    for b in interior:
        assert abs(tone.phase(b + eps) - tone.phase(b - eps)) < 1.5 * tol
        assert np.cos(tone.phase(b + eps)) == pytest.approx(np.cos(tone.phase(b - eps)), abs=tol)


def test_phase_stays_continuous_even_when_the_frequency_hops():
    """The integrand jumps by 5 MHz at 40 us; the phase does not jump at all."""
    freq = _hopping_chain()
    tone = ToneTrack(freq)
    eps = 1.0e-12
    assert freq(40.0 * us - eps) == pytest.approx(3.0 * MHz, rel=1e-9)
    assert freq(40.0 * us + eps) == pytest.approx(-2.0 * MHz, rel=1e-9)
    jump = abs(tone.phase(40.0 * us + eps) - tone.phase(40.0 * us - eps))
    assert jump < TWO_PI * 5.0 * MHz * 2.0 * eps + 1e-9


def test_fdot_matches_numerical_derivative_of_f():
    tone = ToneTrack(_chirp_chain())
    t = np.array([5.0, 12.0, 27.0, 39.0, 45.0, 55.0, 71.0, 79.0, 90.0, 110.0]) * us
    h = 1.0e-9  # 1 ns, far from any break
    numerical = (tone.f(t + h) - tone.f(t - h)) / (2.0 * h)
    np.testing.assert_allclose(tone.fdot(t), numerical, rtol=1e-6, atol=1e-3)

    # scalar in -> float out, matching PiecewisePoly
    assert isinstance(tone.f(10.0 * us), float)
    assert isinstance(tone.fdot(10.0 * us), float)
    assert isinstance(tone.phase(10.0 * us), float)


def test_with_hold_until_freezes_frequency_and_keeps_phase_continuous():
    freq = ramps.min_jerk(0.0, 100.0 * us, 0.0, 2.0 * MHz)
    tone = ToneTrack(freq, phase0=-0.7)
    t1 = 100.0 * us

    # without the hold the *phase* clamps too: it stops advancing past the domain
    assert tone.phase(150.0 * us) == tone.phase(t1)

    held = tone.with_hold_until(150.0 * us)
    assert held.t_span == (0.0, 150.0 * us)
    assert held.freq.n_segments == freq.n_segments + 1
    assert held.phase0 == tone.phase0

    # frequency frozen at the terminal value
    t = np.linspace(t1, 150.0 * us, 101)
    np.testing.assert_allclose(held.f(t), 2.0 * MHz, rtol=1e-12)
    np.testing.assert_allclose(held.fdot(t[1:-1]), 0.0, atol=1e-3)

    # phase continues linearly from where it was, with no jump at the join
    expected = tone.phase(t1) + TWO_PI * 2.0 * MHz * (t - t1)
    np.testing.assert_allclose(held.phase(t), expected, rtol=1e-12)
    # unchanged before the join
    t_before = np.linspace(0.0, t1, 201)
    np.testing.assert_allclose(held.phase(t_before), tone.phase(t_before), rtol=1e-12)
    # a no-op when the request does not extend the track
    assert tone.with_hold_until(50.0 * us) is tone
    assert tone.with_hold_until(t1) is tone


def test_tone_track_validation():
    with pytest.raises(TypeError, match="PiecewisePoly"):
        ToneTrack(freq=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Envelope protocol"):
        ToneTrack(ramps.linear(0.0, 1.0 * us, 0.0, 1.0), env=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        ToneTrack(ramps.linear(0.0, 1.0 * us, 0.0, 1.0), phase0=float("nan"))


# ---------------------------------------------------------------- channel / set level


def test_eval_table_is_one_vectorized_pass_per_quantity():
    tones = (
        ToneTrack(ramps.linear(0.0, T_END, -1.0 * MHz, 1.0 * MHz), ConstantEnvelope(0.6), 0.2),
        ToneTrack(_chirp_chain(), SmoothOnOff(5.0 * us, 115.0 * us, 20.0 * us), -1.1),
    )
    cw = ChannelWaveform(tones)
    assert cw.n_tones == len(cw) == 2
    assert cw.t_span == (0.0, T_END)

    t = np.linspace(0.0, T_END, 257)
    table = cw.eval_table(t)
    assert set(table) == set(TABLE_KEYS)
    for key in TABLE_KEYS:
        assert table[key].shape == (2, t.size)
    for i, tone in enumerate(tones):
        np.testing.assert_array_equal(table["f"][i], tone.f(t))
        np.testing.assert_array_equal(table["fdot"][i], tone.fdot(t))
        np.testing.assert_array_equal(table["A"][i], tone.env.A(t))
        np.testing.assert_array_equal(table["dA"][i], tone.env.dA(t))
        np.testing.assert_array_equal(table["d2A"][i], tone.env.d2A(t))
        np.testing.assert_array_equal(table["phase"][i], tone.phase(t))

    scalar = cw.eval_table(30.0 * us)
    for key in TABLE_KEYS:
        assert scalar[key].shape == (2,)
    assert scalar["f"][0] == pytest.approx(tones[0].f(30.0 * us), rel=1e-15)


def test_waveform_set_validation_and_span(params1030):
    freq = ramps.min_jerk(0.0, T_END, 0.0, 2.0 * MHz)
    cw = ChannelWaveform((ToneTrack(freq),))
    wfs = WaveformSet({"Ay": cw, "By": cw}, params=params1030, description="two rows")
    assert wfs.t_span == (0.0, T_END)
    assert wfs.n_tones == 2
    assert set(wfs.eval_table(1.0 * us)) == {"Ay", "By"}

    with pytest.raises(ValueError, match="unknown channel"):
        WaveformSet({"Cz": cw}, params=params1030)
    with pytest.raises(ValueError, match="at least one channel"):
        WaveformSet({}, params=params1030)
    with pytest.raises(TypeError, match="AODLParams"):
        WaveformSet({"Ay": cw}, params=None)  # type: ignore[arg-type]

    short = ChannelWaveform((ToneTrack(ramps.linear(0.0, 60.0 * us, 0.0, 1.0 * MHz)),))
    with pytest.raises(ValueError, match="with_hold_until"):
        WaveformSet({"Ay": cw, "By": short}, params=params1030)
    # the error message names the fix, and the fix works
    fixed = WaveformSet({"Ay": cw, "By": short.with_hold_until(T_END)}, params=params1030)
    assert fixed.t_span == (0.0, T_END)
    assert fixed.channels["By"].tones[0].f(T_END) == pytest.approx(1.0 * MHz, rel=1e-12)


def test_waveform_set_with_hold_until_extends_every_tone():
    params = default_1030()
    a = ToneTrack(ramps.linear(0.0, 60.0 * us, 0.0, 1.0 * MHz))
    b = ToneTrack(ramps.min_jerk(0.0, 60.0 * us, 0.0, -1.0 * MHz))
    wfs = WaveformSet({"Ax": ChannelWaveform((a,)), "Bx": ChannelWaveform((b,))}, params=params)
    assert wfs.t_span == (0.0, 60.0 * us)
    longer = wfs.with_hold_until(T_END)
    assert longer.t_span == (0.0, T_END)
    assert longer.params is params
    assert longer.channels["Ax"].tones[0].f(T_END) == pytest.approx(1.0 * MHz, rel=1e-12)
    assert longer.channels["Bx"].tones[0].f(T_END) == pytest.approx(-1.0 * MHz, rel=1e-12)
