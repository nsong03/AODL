r"""M6 §2: recovering the complex drive envelope from literal AWG samples (Eqs. S1-S2).

Everything here goes through the real render path — ``waveform/tones.py`` ->
``waveform/export.render_samples`` -> ``check/record.py`` -> ``check/demod.py`` — and is
compared against the *parametric* tone the samples were rendered from.  Nothing in
``check/`` ever sees that tone; it is the truth the demodulation has to rediscover.

Two error laws are pinned here because everything downstream inherits them, and because
``docs/workorders/WO-21-check-core.md`` states different ones (see
:mod:`aodl.check.demod`'s module docstring):

* the cubic-Hermite gather is **third**-order in the baseband phase step, ``0.0160 theta^3``,
  so ``oversample`` buys ``r^3`` and not ``r^4``;
* analytic-signal edge ringing decays as ``1/(pi f_s dt)`` — with the **sample** rate, not
  the carrier — and a drive whose envelope reaches zero at both record ends has none at all.

Records are rendered ``float64``: ``float32``'s 1e-7 quantization floor is above every
tolerance asserted below.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j1

from aodl.check import demod
from aodl.check.record import SampleRecord, from_arrays, load_samples
from aodl.check.transform import GOLDEN_RATIO
from aodl.poly import PiecewisePoly
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec
from aodl.units import MHz, um, us
from aodl.waveform.export import SAMPLES_SUFFIX, render_samples, save_samples
from aodl.waveform.shepard import ShepardConfig
from aodl.waveform.synthesis import synthesize
from aodl.waveform.tones import ChannelWaveform, SmoothOnOff, ToneTrack, WaveformSet

#: Product-default AWG rate (``docs/PLAN.md`` §1.5).
RATE = 625.0 * MHz

#: Record length used by the single-channel fixtures.
SPAN = 60.0 * us

#: Gate ramp of the "clean" fixtures.  An envelope that reaches zero at both record ends
#: leaves the analytic signal nothing to truncate, so the demodulation is exact to round-off
#: and the tolerances below measure the interpolation alone.
GATE = 8.0 * us


# ------------------------------------------------------------------------ fixtures


def _gated(detuning: float, phase0: float = 0.0, span: float = SPAN) -> ToneTrack:
    """A tone that fades in and out inside the record (see :data:`GATE`)."""
    return ToneTrack(
        freq=PiecewisePoly.constant(detuning, 0.0, span),
        env=SmoothOnOff(t_on=0.0, t_off=span, ramp=GATE),
        phase0=phase0,
    )


def _abrupt(detuning: float, phase0: float = 0.0, span: float = SPAN) -> ToneTrack:
    """A tone that is simply on for the whole record — the hard case for ``P+``."""
    return ToneTrack(freq=PiecewisePoly.constant(detuning, 0.0, span), phase0=phase0)


def _chirp(f0: float, fdot: float, span: float = SPAN) -> ToneTrack:
    """Linear chirp ``f(t) = f0 + fdot t``, gated like :func:`_gated`."""
    return ToneTrack(
        freq=PiecewisePoly.from_segment_coeffs([0.0, span], [[f0, fdot * span]]),
        env=SmoothOnOff(t_on=0.0, t_off=span, ramp=GATE),
    )


def _record(
    params, tracks, channel: str = "Ay", rate: float = RATE, span: float = SPAN, **kwargs
) -> tuple[SampleRecord, ChannelWaveform]:
    """Render ``tracks`` on one channel and wrap the buffers in a :class:`SampleRecord`."""
    cw = ChannelWaveform(tuple(tracks))
    wfs = WaveformSet(channels={channel: cw}, params=params)
    arrays, scale = render_samples(wfs, rate, (0.0, span), dtype=np.float64, return_scale=True)
    return from_arrays(arrays, rate, params, normalization=scale, **kwargs), cw


def _expected(cw: ChannelWaveform, t) -> np.ndarray:
    """The parametric truth ``sum_n A_n(t) exp(i phi_n(t))`` the samples encode."""
    out = np.zeros(np.shape(t), dtype=np.complex128)
    for tone in cw.tones:
        out += np.asarray(tone.env.A(t)) * np.exp(1j * np.asarray(tone.phase(t)))
    return out


# =============================================================== 1. the envelope itself


def test_demodulation_recovers_envelope_and_phase_of_a_two_tone_drive(params1030) -> None:
    """Interior ``z`` equals ``sum A_n e^{i phi_n}`` to 1e-7 (measured ~1e-11).

    Two tones, so the render's global normalization is the drive's crest factor (~2) rather
    than 1 and the factor is actually load-bearing — see
    :func:`test_normalization_is_load_bearing`.
    """
    rec, cw = _record(params1030, (_gated(3.0 * MHz, 0.4), _gated(-2.0 * MHz, 1.1)))
    assert rec.normalization == pytest.approx(2.0, abs=0.05)  # two unit tones, crest ~2
    assert float(np.abs(rec.channels["Ay"]).max()) == pytest.approx(1.0, rel=1e-12)

    bb = demod.demodulate(rec)
    assert bb.sample_rate == RATE
    assert bb.t_start == 0.0
    assert bb.n_samples == rec.n_samples

    t = rec.times()
    want = _expected(cw, t)
    inside = (t > 2.0 * us) & (t < SPAN - 2.0 * us)
    assert np.abs(bb.z["Ay"][inside] - want[inside]).max() < 1e-7
    # The drive really is the real part of the demodulated carrier (Eq. S1 read backwards).
    carrier = np.exp(1j * 2.0 * np.pi * params1030.channels["Ay"].f_center * t)
    rebuilt = np.real(bb.z["Ay"] * carrier)
    assert np.abs(rebuilt[inside] - rec.drive("Ay")[inside]).max() < 1e-7


def test_normalization_is_load_bearing(params1030) -> None:
    """Dropping the factor rescales the linear model and *distorts* the nonlinear one.

    ``render_samples`` divides every channel by one global peak.  A checker that ignores it
    reads a drive that is ``1/crest`` too small: the weak (linear) pupil then differs by a
    constant, which is invisible in any normalized comparison — while the ``exp(i C V)``
    crystal is compressed by the wrong amount, which is not.  That asymmetry is exactly why
    the omission is a silent error and why it is pinned here.
    """
    tracks = (_gated(3.0 * MHz, 0.4), _gated(-2.0 * MHz, 1.1))
    right, cw = _record(params1030, tracks)
    wrong = replace(right, normalization=1.0)
    crest = right.normalization

    z_right = demod.demodulate(right).z["Ay"]
    z_wrong = demod.demodulate(wrong).z["Ay"]
    np.testing.assert_allclose(z_wrong * crest, z_right, rtol=1e-9, atol=1e-12)

    # The modulation index the crystal sees is off by the crest factor, so the fundamental's
    # compression 2 J_1(x)/x is evaluated at the wrong x: a 1.1 % loss instead of 4.4 %.
    index = params1030.channels["Ay"].drive_strength * float(np.abs(z_right).max())
    assert index == pytest.approx(0.6, abs=0.02)
    compression = 2.0 * j1(index) / index
    compression_wrong = 2.0 * j1(index / crest) / (index / crest)
    assert compression == pytest.approx(0.956, abs=0.005)
    assert compression_wrong == pytest.approx(0.989, abs=0.005)
    # 3.5 % of pupil amplitude, on the model that is supposed to be the accurate one.
    assert compression_wrong / compression > 1.03


def test_oversampling_is_exact_at_the_shared_grid_points(params1030) -> None:
    """Spectral zero-padding interpolates: it must not move the samples it already had."""
    rec, _ = _record(params1030, (_gated(3.0 * MHz, 0.4),))
    plain = demod.demodulate(rec).z["Ay"]
    fine = demod.demodulate(rec, oversample=4)
    assert fine.sample_rate == 4.0 * RATE
    assert fine.n_samples == 4 * rec.n_samples
    np.testing.assert_allclose(fine.z["Ay"][::4], plain, rtol=1e-11, atol=1e-13)

    with pytest.raises(ValueError, match="oversample must be a positive integer"):
        demod.demodulate(rec, oversample=0)


# =============================================================== 2. the two error laws


def test_edge_ringing_decays_like_one_over_the_sample_distance(params1030) -> None:
    r"""``P+`` on a truncated drive rings as ``1/(pi f_s dt)`` — sample rate, not carrier.

    ``docs/workorders/WO-21-check-core.md`` quotes ``1/(pi f_center dt)``, which happens to
    agree numerically at the product ``f_s / f_center = 6.25 ~ 2 pi`` but scales with the
    wrong quantity: doubling the sample rate at fixed ``dt`` halves the error, while the
    carrier is untouched.  Both rates are measured here.  The law matters because it is what
    tells WO-22 how early a frame may be trusted.
    """
    for rate in (RATE, 2.0 * RATE):
        rec, cw = _record(params1030, (_abrupt(3.0 * MHz, 0.4),), rate=rate)
        t = rec.times()
        error = np.abs(demod.demodulate(rec).z["Ay"] - _expected(cw, t))
        half = int(round(0.5 * us * rate))
        # The ringing oscillates, so it is the local envelope that follows the law, not any
        # single sample (a probe can land in a null).
        envelope = {
            gap: float(error[int(round(gap * rate)) - half : int(round(gap * rate)) + half].max())
            for gap in (5.0 * us, 10.0 * us, 20.0 * us)
        }
        for gap, measured in envelope.items():
            law = 1.0 / (math.pi * rate * gap)
            assert 0.6 * law < measured < 1.6 * law, f"rate={rate}, gap={gap}, got {measured}"
        # ... and it really does decay, so a late frame is clean where an early one is not.
        assert envelope[20.0 * us] < 0.5 * envelope[5.0 * us]

    # A gated drive truncates nothing: same tone, same record, four orders better.
    gated_rec, gated_cw = _record(params1030, (_gated(3.0 * MHz, 0.4),))
    t = gated_rec.times()
    gated_error = np.abs(demod.demodulate(gated_rec).z["Ay"] - _expected(gated_cw, t))
    assert gated_error[int(round(5.0 * us * RATE))] < 1e-9


def test_cubic_hermite_error_follows_the_measured_cubic_law(params1030) -> None:
    r"""``sample_baseband`` is third-order: ``eps ~ 0.0160 (2 pi f_bb/f_s)^3``.

    Catmull-Rom *is* Keys' cubic convolution with ``a = -1/2``, which is third-order
    accurate; the work order's quartic ``(2 pi f_bb/f_s)^4 / 384`` budget would predict
    1.5e-6 at the band edge where the truth is 5.5e-5.  Measured on a clean gated tone, off
    the sample grid, at two detunings and three oversampling factors.
    """
    for detuning, expected in ((3.0 * MHz, 4.4e-7), (8.0 * MHz, 8.3e-6)):
        rec, cw = _record(params1030, (_gated(detuning, 0.4),))
        # Probe well inside the record, stepping by an irrational number of samples so that
        # every interpolation phase is visited at *every* oversampling factor.  (A rational
        # step would sit on s = 1/2 after oversampling, where the cubic term happens to
        # cancel and the error is anomalously small.)
        probe = 20.0 * us + np.arange(4001) * GOLDEN_RATIO / RATE
        want = _expected(cw, probe)
        errors = []
        for oversample in (1, 2, 4):
            bb = demod.demodulate(rec, oversample=oversample)
            got = demod.sample_baseband(bb, "Ay", probe)
            errors.append(float(np.abs(got - want).max()))
        theta = 2.0 * math.pi * detuning / RATE
        assert errors[0] == pytest.approx(0.0160 * theta**3, rel=0.15)
        assert errors[0] == pytest.approx(expected, rel=0.1)
        # r^3, not r^4: 8x per doubling and 64x from 1 to 4.
        assert errors[0] / errors[1] == pytest.approx(8.0, rel=0.1)
        assert errors[0] / errors[2] == pytest.approx(64.0, rel=0.1)


def test_chirp_baseband_phase_carries_the_quadratic_law(params1030) -> None:
    """A linear chirp demodulates to ``phi(t) = 2 pi (f0 t + fdot t^2/2)`` — Eq. S2's frame."""
    f0, fdot = 1.0 * MHz, 50.0 * MHz / (1e-3)
    rec, cw = _record(params1030, (_chirp(f0, fdot),))
    bb = demod.demodulate(rec)
    t = rec.times()
    inside = (t > 15.0 * us) & (t < 45.0 * us)  # on the gate's plateau

    phase = np.unwrap(np.angle(bb.z["Ay"][inside]))
    quad, lin, _ = np.polynomial.polynomial.polyfit(t[inside], phase, 2)[::-1]
    assert quad == pytest.approx(math.pi * fdot, rel=1e-6)
    assert lin == pytest.approx(2.0 * math.pi * f0, rel=1e-6)
    # ... and the whole envelope matches the tone, phase0 included.
    np.testing.assert_allclose(bb.z["Ay"][inside], _expected(cw, t[inside]), rtol=0.0, atol=1e-9)


# =============================================================== 3. coverage refusal


def test_sample_baseband_is_zero_before_the_drive_and_refuses_to_run_past_it(params1030) -> None:
    """Asymmetric boundaries: silence before ``t_start``, a loud error past the last sample."""
    rec, cw = _record(params1030, (_gated(3.0 * MHz, 0.4),))
    bb = demod.demodulate(rec)

    before = demod.sample_baseband(bb, "Ay", np.array([-5.0 * us, -1e-15]))
    assert np.all(before == 0.0)
    at_end = demod.sample_baseband(bb, "Ay", np.array([rec.t_span[1]]))
    assert np.isfinite(at_end).all()

    with pytest.raises(ValueError, match="record only covers"):
        demod.sample_baseband(bb, "Ay", np.array([SPAN + 1.0 * us]))
    with pytest.raises(KeyError, match="not in this baseband"):
        demod.sample_baseband(bb, "Bx", np.array([1.0 * us]))

    # Shape is preserved (the pupil gathers on a 1-D aperture grid, WO-22 on 2-D batches).
    shaped = demod.sample_baseband(bb, "Ay", np.full((3, 5), 20.0 * us))
    assert shaped.shape == (3, 5)


# =============================================================== 4. the splatter probe


def test_out_of_band_fraction_sees_the_shepard_switching_splatter(params1030) -> None:
    r"""Table II's ``p_B = 0`` rectangles radiate ~-41 dB out of band; a 3 us ramp removes it.

    This is the WO-19 F-3 finding measured from the *samples* rather than from the envelope
    algebra: ``switch_ramp = 0`` steps the ``B`` channels' amplitude at every rung crossing,
    and the step is broadband.  The ``A`` channels' ``cos^p`` windows never step, ramped or
    not (WO-20 §1), so they are the control.
    """
    spec = TrajectorySpec(
        array=ArraySpec(3, 3, 1.0 * MHz, 1.0 * MHz),
        moves=(Lift(6.0 * um, 40.0 * us), Hold(80.0 * us), Lift(-6.0 * um, 40.0 * us)),
    )
    fractions = {}
    for ramp in (0.0, 3.0 * us):
        wfs = synthesize(
            spec, params1030, shepard=ShepardConfig(1.0 * MHz, 1.0 * MHz, switch_ramp=ramp)
        )
        arrays, scale = render_samples(wfs, RATE, dtype=np.float64, return_scale=True)
        rec = from_arrays(arrays, RATE, params1030, normalization=scale)
        fractions[ramp] = demod.out_of_band_fraction(rec)

    plain, ramped = fractions[0.0], fractions[3.0 * us]
    for name in ("Bx", "By"):
        assert 10.0 * math.log10(plain[name]) == pytest.approx(-40.8, abs=1.0)
        assert 10.0 * math.log10(ramped[name]) < -100.0
    for name in ("Ax", "Ay"):  # never step, so nothing to fix
        assert plain[name] < 1e-9
        assert ramped[name] == pytest.approx(plain[name], rel=1e-9)


def test_the_analysis_window_is_not_cosmetic(params1030) -> None:
    """A boxcar's own leakage forges ~-44 dB of splatter on a perfectly smooth drive.

    A rectangular window's sidelobes fall as 1/df in amplitude, so at ±10 MHz from a 100 MHz
    carrier a 60 us record leaks roughly as much power past the band edge as the switching
    steps this diagnostic exists to find.  Hann's fall as 1/df^3 and leak nothing.
    """
    rec, _ = _record(params1030, (_abrupt(3.0 * MHz, 0.4),))
    hann = demod.out_of_band_fraction(rec)["Ay"]
    boxcar = demod.out_of_band_fraction(rec, window="boxcar")["Ay"]
    assert hann < 1e-9
    assert 10.0 * math.log10(boxcar) == pytest.approx(-42.0, abs=6.0)
    assert boxcar > 1e4 * hann

    with pytest.raises(ValueError, match="unknown window"):
        demod.out_of_band_fraction(rec, window="blackman")

    # A silent channel has no out-of-band power, and no division by zero either.
    silent = from_arrays({"Ay": np.zeros(1024)}, RATE, params1030)
    assert demod.out_of_band_fraction(silent) == {"Ay": 0.0}


# =============================================================== 5. the record boundary


def test_record_validates_and_round_trips_through_a_samples_file(params1030, tmp_path) -> None:
    """``load_samples`` reads the schema-1 file and refuses a carrier that is not ours."""
    cw = ChannelWaveform((_gated(3.0 * MHz, 0.4),))
    wfs = WaveformSet(channels={"Ay": cw}, params=params1030, description="check fixture")
    path = save_samples(
        wfs, tmp_path / f"drive{SAMPLES_SUFFIX}", sample_rate=RATE, dtype=np.float64
    )

    rec = load_samples(path, params1030)
    assert rec.sample_rate == RATE
    assert rec.t_start == 0.0
    assert rec.t_span[1] == pytest.approx(SPAN, rel=1e-12)
    assert set(rec.channels) == {"Ay"}
    direct, _ = _record(params1030, (_gated(3.0 * MHz, 0.4),))
    np.testing.assert_allclose(rec.drive("Ay"), direct.drive("Ay"), rtol=0.0, atol=1e-15)

    shifted = replace(
        params1030,
        channels={
            name: replace(aod, f_center=aod.f_center + 1e3)
            for name, aod in params1030.channels.items()
        },
    )
    with pytest.raises(ValueError, match="carrier mismatch"):
        load_samples(path, shifted)
    with pytest.raises(ValueError, match="not a rendered-sample file"):
        load_samples(tmp_path / "drive.npz", params1030)


def test_record_rejects_malformed_buffers(params1030) -> None:
    """The boundary checks the things a wrong buffer would otherwise poison silently."""
    good = np.zeros(64)
    with pytest.raises(ValueError, match="same number of samples"):
        from_arrays({"Ay": good, "Ax": np.zeros(65)}, RATE, params1030)
    with pytest.raises(ValueError, match="unknown channel"):
        from_arrays({"Cz": good}, RATE, params1030)
    with pytest.raises(ValueError, match="must be 1-D"):
        from_arrays({"Ay": np.zeros((8, 8))}, RATE, params1030)
    with pytest.raises(ValueError, match="cubic-Hermite"):
        from_arrays({"Ay": np.zeros(3)}, RATE, params1030)
    with pytest.raises(ValueError, match="must all be finite"):
        from_arrays({"Ay": np.full(64, np.nan)}, RATE, params1030)
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        from_arrays({"Ay": good}, -1.0, params1030)
    with pytest.raises(ValueError, match="normalization must be positive"):
        from_arrays({"Ay": good}, RATE, params1030, normalization=0.0)

    rec = from_arrays({"Ay": good}, RATE, params1030, t_start=1.0 * us, normalization=3.0)
    assert rec.t_span == (1.0 * us, 1.0 * us + 63.0 / RATE)
    with pytest.raises(KeyError, match="not in this record"):
        rec.drive("Bx")
