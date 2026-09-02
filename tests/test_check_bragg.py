r"""M6: the nonlinear pupil model, and the convergence of the frame average.

``bragg_band`` rebuilds the literal crystal — ``T = exp(i C V)`` at every aperture point, with
the ``+1`` order cut out in the aperture's spatial-frequency domain — so it carries the
compression and the intermodulation the weak-drive expansion of Eqs. S20-S22 only approximates.
Three things have to be true of it:

1. **it degenerates to the weak model** as ``C -> 0``, with the ``C^2`` law that says the
   difference is the Bessel expansion's first correction and nothing else;
2. **the compression it reports is Jacobi-Anger's**: the ``m = 1`` term of
   ``exp(i C A cos psi)`` is ``J_1(C A)`` where the linear model has ``C A / 2``, so a single
   driven channel's *intensity* comes out ``(2 J_1(C)/C)^2`` of the weak model's;
3. **the frame average has converged**: doubling the sub-time count or the averaging window
   must not move a verdict metric by a meaningful fraction of its own tolerance, or the number
   in the report is a property of the schedule rather than of the drive.

``scipy.special.j1`` is allowed here — this is a test, not the checker.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j1

from aodl.check import report as report_module
from aodl.check.demod import demodulate
from aodl.check.expect import Expectation
from aodl.check.pupil import ApertureGrid, axis_pupil
from aodl.check.record import from_arrays
from aodl.check.report import Tolerances, check_samples
from aodl.check.transform import zoom_field
from aodl.poly import PiecewisePoly
from aodl.trajectory.spec import ArraySpec, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us
from aodl.waveform.export import DEFAULT_SAMPLE_RATE, render_samples
from aodl.waveform.synthesis import synthesize
from aodl.waveform.tones import ChannelWaveform, SmoothOnOff, ToneTrack, WaveformSet

RATE = DEFAULT_SAMPLE_RATE
SPAN = 60.0 * us
GATE = 8.0 * us
DETUNING = 3.0 * MHz


def _with_strength(params, value):
    """The same hardware at a different ``drive_strength`` (the Eq. S1 modulation index)."""
    return replace(
        params,
        channels={
            name: replace(aod, drive_strength=value) for name, aod in params.channels.items()
        },
    )


def _single_tone(params):
    """One gated ``Ay`` tone, rendered and demodulated — the cleanest possible crystal."""
    tone = ToneTrack(
        freq=PiecewisePoly.constant(DETUNING, 0.0, SPAN),
        env=SmoothOnOff(t_on=0.0, t_off=SPAN, ramp=GATE),
        phase0=0.4,
    )
    wfs = WaveformSet({"Ay": ChannelWaveform((tone,))}, params)
    arrays, scale = render_samples(wfs, RATE, (0.0, SPAN), dtype=np.float64, return_scale=True)
    rec = from_arrays(arrays, RATE, params, normalization=scale)
    return demodulate(rec)


def _peak_intensity(bb, params, mode):
    """Peak focal intensity of the single-tone scene, on the checker's own scale."""
    grid = ApertureGrid.design(params, "bragg_band")
    tau = params.channels["Ay"].transit_time
    t = 2.0 * tau
    y0 = -params.deflection_scale * DETUNING
    waist = params.optics.waist0
    ys = np.linspace(y0 - 2.0 * waist, y0 + 2.0 * waist, 81)
    field = zoom_field(axis_pupil(bb, "y", t, grid, mode=mode), grid, params.optics, ys, 0.0)
    return float(np.abs(field).max() ** 2)


# ================================================== 1. the weak limit, and its C^2 law


def test_a_weakly_driven_crystal_is_the_linear_model(params1030) -> None:
    """At ``C = 0.01`` the two pupil models agree to 2.5e-5, and the gap follows ``C^2``.

    ``(2 J_1(C)/C)^2 = 1 - C^2/4 + O(C^4)``, so the *only* thing separating ``bragg_band`` from
    ``weak`` at small drive is the Bessel expansion's leading correction — quadrupling ``C``
    must quadruple... squared, i.e. multiply the gap by sixteen.
    """
    gaps = {}
    for strength in (0.01, 0.02, 0.04):
        params = _with_strength(params1030, strength)
        bb = _single_tone(params)
        ratio = _peak_intensity(bb, params, "bragg_band") / _peak_intensity(bb, params, "weak")
        gaps[strength] = 1.0 - ratio
        assert ratio == pytest.approx((2.0 * float(j1(strength)) / strength) ** 2, rel=1e-6)

    assert gaps[0.01] < 3e-5
    assert gaps[0.01] == pytest.approx(0.01**2 / 4.0, rel=0.01)
    assert gaps[0.02] / gaps[0.01] == pytest.approx(4.0, rel=0.01)
    assert gaps[0.04] / gaps[0.01] == pytest.approx(16.0, rel=0.02)


# ============================================================== 2. the compression law


@pytest.mark.parametrize("strength", [0.3, 0.8])
def test_the_measured_compression_is_two_j1_over_c_squared(params1030, strength) -> None:
    """One driven channel: ``I_bragg / I_weak = (2 J_1(C)/C)^2``, to 1e-5.

    The product default ``C = 0.30`` costs **2.2 %** of the trap's intensity — the number the
    weak-drive expansion approximates as ``~C^2/4`` and the reason ``bragg_band`` is the
    verdict path.  (An axis with *two* driven channels compresses twice over, and a drive whose
    tones stack raises ``C`` by the crest factor, which is what makes an array's compression far
    larger than this single-tone figure.)
    """
    params = _with_strength(params1030, strength)
    bb = _single_tone(params)
    ratio = _peak_intensity(bb, params, "bragg_band") / _peak_intensity(bb, params, "weak")
    expected = (2.0 * float(j1(strength)) / strength) ** 2
    assert ratio == pytest.approx(expected, rel=1e-5)
    if strength == 0.3:
        assert ratio == pytest.approx(0.9777, abs=5e-4)


# ================================================== 3. the frame average has converged


@pytest.fixture(scope="module")
def mini_story():
    """A 3x3 Eq. S19 lift-traverse-lower and the record its frames need."""
    from aodl.check.report import frame_reach
    from aodl.params import default_1030

    params = default_1030()
    spec = TrajectorySpec(
        array=ArraySpec(3, 3, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz),
        moves=(Lift(4 * um, 40 * us), Translate(10 * um, 6 * um, 60 * us), Lift(-4 * um, 40 * us)),
    )
    wfs = synthesize(spec, params)
    tau = params.channels["Ax"].transit_time
    times = np.array([45.0 * us]) + 0.5 * tau
    reach = frame_reach(ApertureGrid.design(params, "bragg_band"), params)
    t0, t1 = wfs.t_span
    span = (
        max(t0, float(times.min()) - 0.5 * tau - reach - 5.0 * us),
        min(t1, float(times.max()) - 0.5 * tau + reach + 5.0 * us),
    )
    arrays, scale = render_samples(wfs, RATE, span, dtype=np.float64, return_scale=True)
    rec = from_arrays(arrays, RATE, params, t_start=span[0], normalization=scale)
    return spec, params, rec, times


def test_the_frame_average_is_converged_in_k_and_in_the_window(mini_story, monkeypatch) -> None:
    """K 64 -> 128 and W x 2 move no verdict metric by 10 % of its own tolerance.

    That is the statement that the reported residuals belong to the *drive*: a number that
    still moved when the schedule changed would be a property of the schedule.  The window is
    doubled through :data:`aodl.check.report.BEAT_CYCLES`, which is what sets the fallback
    window for a non-fading drive (a fading one gets the beat-commensurate window instead, and
    that one is exact by construction, not converged).
    """
    spec, params, rec, times = mini_story
    expect = Expectation(spec=spec, params=params)
    tol = Tolerances()

    def worst(**kwargs):
        return check_samples(rec, expect, times=times, tolerances=tol, **kwargs).worst()

    base = worst(k_subtimes=64)
    dense = worst(k_subtimes=128)
    monkeypatch.setattr(report_module, "BEAT_CYCLES", 2.0 * report_module.BEAT_CYCLES)
    wide = worst(k_subtimes=64)

    for metric, (value, limit, _) in base.items():
        assert abs(dense[metric][0] - value) < 0.1 * limit, f"{metric} moved with K"
        assert abs(wide[metric][0] - value) < 0.1 * limit, f"{metric} moved with W"
    # ... and the residuals are real numbers, not zeros that would make the test vacuous
    assert base["uniformity"][0] > 1e-3
