r"""M6 §3: rebuilding the AODL pupil from samples (Eqs. S1-S4), literally.

The comparison target is the same ``_literal_pupil`` pattern
``tests/test_device_single_aod.py`` uses to validate the Taylor step of Eqs. S5-S6 — the
channel's exact rotating-frame pupil ``(i C / 2) sum_n A_n(t_ret) e^{-i phi_n(t_ret)}`` —
except that here it is rebuilt **from the rendered samples** rather than from the tone
objects.  Two independent paths to the same complex array: one reads a ``PiecewisePoly``,
the other reads a float64 buffer through an FFT.

The frames are taken at ``t >= 2 tau`` on purpose.  The aperture grid runs to ±4.99 ``w_in``
= ±9.98 mm, wider than the 7.5 mm crystal, so it is filled end to end only from
``t = (half_span + D/2)/v = 1.83 tau``; before that the outermost cells are legitimately dark
and the analytic simulator — which models the beam as uncropped and windows only at the fill
edge — would disagree there by the Gaussian tail.

Sign conventions are never restated here: every expectation is built by calling the helper in
``aodl.device.conventions`` that owns it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j1

from aodl.check import pupil as pup
from aodl.check.demod import demodulate
from aodl.check.pupil import ApertureGrid, axis_pupil, band_window, channel_pupil
from aodl.check.record import from_arrays
from aodl.device import conventions
from aodl.device.conventions import geometry
from aodl.params import CHANNELS
from aodl.poly import PiecewisePoly
from aodl.units import MHz, mm, us
from aodl.waveform.export import render_samples
from aodl.waveform.tones import ChannelWaveform, SmoothOnOff, ToneTrack, WaveformSet

RATE = 625.0 * MHz
SPAN = 60.0 * us
GATE = 8.0 * us


# ------------------------------------------------------------------------ fixtures


def _gated(detuning: float, phase0: float = 0.0) -> ToneTrack:
    """A tone gated to zero at both record ends — see ``tests/test_check_demod.py``."""
    return ToneTrack(
        freq=PiecewisePoly.constant(detuning, 0.0, SPAN),
        env=SmoothOnOff(t_on=0.0, t_off=SPAN, ramp=GATE),
        phase0=phase0,
    )


def _baseband(params, channels, oversample: int = 1):
    """Render the given channels and demodulate them."""
    cws = {name: ChannelWaveform(tuple(tracks)) for name, tracks in channels.items()}
    wfs = WaveformSet(channels=cws, params=params)
    arrays, scale = render_samples(wfs, RATE, (0.0, SPAN), dtype=np.float64, return_scale=True)
    rec = from_arrays(arrays, RATE, params, normalization=scale)
    return demodulate(rec, oversample=oversample), cws


def _literal_pupil(cw: ChannelWaveform, aod, geom, t: float, u):
    """``(i C / 2) sum_n A_n(t_ret) e^{-i phi_n(t_ret)}``, windowed by the fill mask.

    The ``tests/test_device_single_aod.py`` pattern, with the hard fill edge of
    :func:`aodl.device.conventions.is_filled` (the checker's grid is far coarser than that
    test's quadrature grid, so no sub-cell ramp is used).  The input Gaussian is *not*
    applied — that is :func:`axis_pupil`'s job.
    """
    u = np.asarray(u, dtype=np.float64)
    t_ret = conventions.retarded_time(t, u, geom, aod)
    drive = np.zeros(u.shape, dtype=np.complex128)
    for tone in cw.tones:
        drive += np.asarray(tone.env.A(t_ret)) * np.exp(-1j * np.asarray(tone.phase(t_ret)))
    drive *= 0.5j * aod.drive_strength
    return np.where(conventions.is_filled(u, t, geom, aod), drive, 0.0)


def _gaussian(params, u):
    return np.exp(-((np.asarray(u) / params.optics.w_in) ** 2))


def _with_drive_strength(params, value: float):
    return replace(
        params,
        channels={
            name: replace(aod, drive_strength=value) for name, aod in params.channels.items()
        },
    )


def _grid_at(params, cells_per_period: int, n: int) -> ApertureGrid:
    """A custom grid with ``du = Lambda / cells_per_period`` — for the alias study."""
    du = params.sound_speed / (cells_per_period * params.channels["Ax"].f_center)
    return ApertureGrid(u=(np.arange(n, dtype=np.float64) - n // 2) * du, du=du)


# =============================================================== 1. the grid itself


def test_aperture_grid_pins_the_lambda_over_eight_rule(params1030) -> None:
    """``du = v / (8 f_center)`` exactly, 24576 cells, ±4.99 w_in, and the weak grid nested."""
    aod = params1030.channels["Ax"]
    acoustic_wavelength = params1030.sound_speed / aod.f_center
    assert acoustic_wavelength == pytest.approx(6.5e-6, rel=1e-12)

    bragg = ApertureGrid.design(params1030, "bragg_band")
    assert bragg.n == pup.BRAGG_CELLS == 24576
    assert bragg.du == pytest.approx(acoustic_wavelength / 8.0, rel=1e-15)
    assert bragg.du == pytest.approx(params1030.sound_speed / (8.0 * aod.f_center), rel=1e-15)
    assert bragg.half_span == pytest.approx(9.984 * mm, rel=1e-12)
    assert bragg.half_span / params1030.optics.w_in == pytest.approx(4.992, rel=1e-6)
    assert bragg.u[bragg.n // 2] == 0.0
    # Four diffraction orders fit inside Nyquist — that is what makes the aliases land on
    # order centres (module docstring).
    assert bragg.nyquist == pytest.approx(4.0 * aod.f_center / params1030.sound_speed, rel=1e-12)

    weak = ApertureGrid.design(params1030, "weak")
    assert weak.n == pup.WEAK_CELLS == 4096
    assert weak.half_span == pytest.approx(bragg.half_span, rel=1e-15)
    np.testing.assert_allclose(weak.u, bragg.u[::6], rtol=1e-15, atol=1e-17)

    # The whole grid is inside the acoustic column only from 1.83 tau (docstring claim).
    full = (bragg.half_span + 0.5 * aod.aperture) / params1030.sound_speed
    assert full / aod.transit_time == pytest.approx(1.831, abs=0.002)
    assert np.all(conventions.is_filled(bragg.u, 2.0 * aod.transit_time, geometry("Ay"), aod))


def test_aperture_grid_refuses_exotic_hardware(params1030) -> None:
    """Too fast, or too low a carrier, and the pinned span would clip the beam: raise."""
    wide = replace(params1030, optics=replace(params1030.optics, w_in=4.0 * mm))
    with pytest.raises(ValueError, match="at least 4.2 w_in"):
        ApertureGrid.design(wide, "bragg_band")

    mixed = replace(
        params1030,
        channels={
            name: replace(aod, f_center=aod.f_center + (1e6 if name == "By" else 0.0))
            for name, aod in params1030.channels.items()
        },
    )
    with pytest.raises(ValueError, match="one common f_center"):
        ApertureGrid.design(mixed, "bragg_band")

    with pytest.raises(ValueError, match="mode must be"):
        ApertureGrid.design(params1030, "literal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="uniformly spaced"):
        ApertureGrid(u=np.array([0.0, 1.0, 3.0, 4.0]), du=1.0)


# =============================================================== 2. the weak pupil


@pytest.mark.parametrize("detuning", [3.0 * MHz, -3.0 * MHz])
def test_weak_pupil_matches_the_literal_closed_form(params1030, detuning) -> None:
    """From samples vs from the tone objects: 2e-6, and it is the interpolation that sets it.

    Measured 4.4e-7 at 3 MHz with ``oversample=1``, falling as ``r^-3``
    (``tests/test_check_demod.py``).
    """
    aod = params1030.channels["Ay"]
    tau = aod.transit_time
    t = 2.0 * tau
    grid = ApertureGrid.design(params1030, "weak")
    bb, cws = _baseband(params1030, {"Ay": (_gated(detuning, 0.4),)})

    got = axis_pupil(bb, "y", t, grid, mode="weak")
    want = _literal_pupil(cws["Ay"], aod, geometry("Ay"), t, grid.u) * _gaussian(params1030, grid.u)
    error = float(np.abs(got - want).max() / np.abs(want).max())
    assert error < 2e-6
    assert error == pytest.approx(4.4e-7, rel=0.25)

    fine, _ = _baseband(params1030, {"Ay": (_gated(detuning, 0.4),)}, oversample=4)
    refined = axis_pupil(fine, "y", t, grid, mode="weak")
    assert float(np.abs(refined - want).max() / np.abs(want).max()) < error / 40.0


def test_weak_pupil_phase_slope_pins_all_four_sound_signs(params1030) -> None:
    """A static detuning gives pupil slope ``s 2 pi f / v`` on **every** channel (Eq. S6).

    This is the whole sign table in one test: demodulating at the retarded time reproduces
    :func:`aodl.device.conventions.theta1_contribution` exactly, sign included, so ``Ax``/
    ``Ay`` tilt one way and ``Bx``/``By`` the other.
    """
    detuning = 3.0 * MHz
    tau = params1030.channels["Ax"].transit_time
    grid = ApertureGrid.design(params1030, "weak")
    signs = {}
    for name in CHANNELS:
        bb, _ = _baseband(params1030, {name: (_gated(detuning, 0.4),)})
        p = channel_pupil(bb, name, 2.0 * tau, grid, mode="weak")
        inner = np.abs(grid.u) < 2.0 * mm
        slope = float(
            np.polynomial.polynomial.polyfit(grid.u[inner], np.unwrap(np.angle(p[inner])), 1)[1]
        )
        want = float(
            conventions.theta1_contribution(detuning, geometry(name), params1030.sound_speed)
        )
        assert slope == pytest.approx(want, rel=1e-9), name
        signs[name] = int(np.sign(slope))
    assert signs == {"Ax": -1, "Bx": +1, "Ay": -1, "By": +1}


def test_fill_mask_is_conventions_is_filled(params1030) -> None:
    """Mid-fill: the weak pupil is exactly zero where the aperture holds no sound."""
    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    tau = aod.transit_time
    t = 0.6 * tau
    grid = ApertureGrid.design(params1030, "weak")
    bb, _ = _baseband(params1030, {"Ay": (_gated(3.0 * MHz, 0.4),)})

    p = channel_pupil(bb, "Ay", t, grid, mode="weak")
    filled = conventions.is_filled(grid.u, t, geom, aod)
    assert filled.any() and (~filled).any()
    np.testing.assert_array_equal(p == 0.0, ~filled)
    # The edge is where docs/conventions.md §7 says: s u <= v t - D/2, transducer side.
    assert grid.u[filled].min() == pytest.approx(-0.1 * aod.aperture, abs=grid.du)
    assert np.all(np.abs(p[filled]) > 0.0)

    # The nonlinear model leaves the unfilled crystal *clear* (T = 1), whose +1-band content
    # is only the ringing of the fill edge itself — small, and not a hard zero.
    bragg_grid = ApertureGrid.design(params1030, "bragg_band")
    q = channel_pupil(bb, "Ay", t, bragg_grid, mode="bragg_band")
    dark = ~conventions.is_filled(bragg_grid.u, t, geom, aod)
    inside = np.abs(bragg_grid.u) < 3.0 * mm
    assert np.abs(q[dark & inside]).max() < 0.05 * np.abs(q[(~dark) & inside]).max()


# =============================================================== 3. the nonlinear pupil


@pytest.mark.parametrize("strength", [0.3, 1.0])
def test_bragg_band_is_the_weak_pupil_compressed_by_two_j1_over_c(params1030, strength) -> None:
    r"""Single tone: ``P_bragg / P_weak = 2 J_1(C)/C``, to 1e-4 (measured 1.4e-8).

    The order cut keeps the ``m = -1`` Jacobi-Anger term of ``exp(i C A cos psi)``, whose
    amplitude is ``J_1(C A)`` where the linear model has ``C A / 2``.  The comparison is made
    over ``|u| < 3 mm``, where the retarded times ``t_c -+ u/v`` all sit on the gate's
    plateau and ``A = 1`` exactly; further out the envelope is ramping and the ratio becomes
    ``2 J_1(C A(u))/(C A(u))``, which is physics, not error.
    """
    params = _with_drive_strength(params1030, strength)
    tau = params.channels["Ay"].transit_time
    t = 2.0 * tau
    grid = ApertureGrid.design(params, "bragg_band")
    bb, _ = _baseband(params, {"Ay": (_gated(3.0 * MHz, 0.4),)})

    bragg = channel_pupil(bb, "Ay", t, grid, mode="bragg_band")
    weak = channel_pupil(bb, "Ay", t, grid, mode="weak")
    inner = np.abs(grid.u) < 3.0 * mm
    ratio = bragg[inner] / weak[inner]

    expected = 2.0 * float(j1(strength)) / strength
    assert float(np.abs(np.abs(ratio) - expected).max()) / expected < 1e-4
    assert float(np.abs(np.abs(ratio) - expected).max()) / expected < 1e-6  # measured 1.4e-8
    # Removing the carrier at the retarded time (not just its spatial ramp) also puts the two
    # models in the same phase origin, so the ratio is real and positive.
    assert float(np.abs(np.angle(ratio)).max()) < 1e-5


def test_band_window_shape_and_order_rejection(params1030) -> None:
    """The window is a flat top with a raised-cosine shoulder, and it cuts orders cleanly.

    The fixture plants four orders at ``p s f_c / v``, ``p = 0, ±1, +2``, on the pinned grid
    and asks the window to return the ``p = +1`` one alone.  Orders sit ``f_c / v`` apart,
    the window's outer edge is ``1.15 * 1.25`` half-bands wide, so the rejection is set by
    the Gaussian's own spectral tail — i.e. total.
    """
    nu = np.array([-1.5, -1.0, 0.0, 1.0, 1.1, 1.125, 1.25, 2.0])
    w = band_window(nu, center=0.0, half=1.0, roll=0.25)
    np.testing.assert_allclose(w[:5], [0.0, 1.0, 1.0, 1.0, 0.654508497187474], rtol=1e-12)
    assert w[5] == pytest.approx(0.5, abs=1e-15)  # halfway down the shoulder
    assert w[6] == pytest.approx(0.0, abs=1e-15)  # the shoulder ends at half (1 + roll)
    assert w[7] == 0.0
    np.testing.assert_array_equal(band_window(nu, 0.0, 1.0, 0.0), [0, 1, 1, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="half-width must be positive"):
        band_window(nu, 0.0, 0.0, 0.25)

    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    grid = ApertureGrid.design(params1030, "bragg_band")
    order = geom.sound_sign * aod.f_center / params1030.sound_speed
    amplitudes = {0: 0.7, 1: 0.5, -1: 0.3, 2: 0.2}
    gauss = _gaussian(params1030, grid.u)
    planted = (
        sum(amp * np.exp(2j * np.pi * p * order * grid.u) for p, amp in amplitudes.items()) * gauss
    )

    half = 1.15 * 0.5 * (aod.band[1] - aod.band[0]) / params1030.sound_speed
    spectrum = np.fft.fft(planted)
    spectrum *= band_window(np.fft.fftfreq(grid.n, grid.du), order, half, 0.25)
    selected = np.fft.ifft(spectrum) * np.exp(-2j * np.pi * order * grid.u)
    assert float(np.abs(selected - amplitudes[1] * gauss).max()) < 1e-12

    # A detuned +1 order (the real case) survives; one 15 MHz out is on the shoulder.
    for detuning, keep in ((9.0 * MHz, 1.0), (15.0 * MHz, 0.0)):
        nu_line = geom.sound_sign * (aod.f_center + detuning) / params1030.sound_speed
        line = np.exp(2j * np.pi * nu_line * grid.u) * gauss
        spectrum = np.fft.fft(line)
        spectrum *= band_window(np.fft.fftfreq(grid.n, grid.du), order, half, 0.25)
        got = float(np.abs(np.fft.ifft(spectrum)).max() / np.abs(line).max())
        assert got == pytest.approx(keep, abs=1e-9)


def test_alias_margin_at_lambda_over_eight(params1030) -> None:
    r"""The planted two-tone alias test: ``Lambda/8`` folds nothing in, ``Lambda/4`` folds a lot.

    Two strong tones are rendered so that the *peak modulation index* ``C |V|`` is 1.0, and
    the pupil is rebuilt on three nested grids — ``Lambda/32`` (reference: the first fold is
    ``p = 33``), ``Lambda/8`` (the pinned grid) and ``Lambda/4``.  The residual against the
    reference *is* the alias contamination.

    The folds onto ``+1`` come from ``p = 9`` and ``p = -7``; the work order names only
    ``p = 9``, but ``|J_7| >> |J_9|``, and the measured numbers follow the ``J_7`` law:
    3.4e-6 at ``C|V| = 1.0``, 1.05e-5 at 1.2 and 4.4e-5 at 1.5 (the work order's 1e-5 bound
    therefore holds up to ``C|V| ~ 1.15``, not quite its quoted 1.2).  At the product default
    ``C = 0.3`` it is 2e-9.
    """
    params = _with_drive_strength(params1030, 0.5)  # two unit tones -> peak index 1.0
    tau = params.channels["Ay"].transit_time
    t = 2.0 * tau
    bb, _ = _baseband(params, {"Ay": (_gated(2.0 * MHz, 0.0), _gated(-2.0 * MHz, 1.1))})
    peak_index = params.channels["Ay"].drive_strength * float(np.abs(bb.z["Ay"]).max())
    assert peak_index == pytest.approx(1.0, abs=0.01)

    reference = channel_pupil(bb, "Ay", t, _grid_at(params, 32, 4 * 24576), mode="bragg_band")
    scale = float(np.abs(reference).max())
    residuals = {}
    for cells, n, step in ((8, 24576, 4), (4, 12288, 8)):
        grid = _grid_at(params, cells, n)
        got = channel_pupil(bb, "Ay", t, grid, mode="bragg_band")
        inner = np.abs(grid.u) < 4.0 * mm
        residuals[cells] = float(np.abs(got[inner] - reference[::step][inner]).max() / scale)

    assert residuals[8] < 1e-5
    assert residuals[8] == pytest.approx(3.4e-6, rel=0.2)
    # Not a vacuous bound: one octave coarser folds |J_3| in and the pupil is wrong by 4 %.
    assert residuals[4] > 1e-2
    assert residuals[4] / residuals[8] > 1e4

    # ... and the pinned grid really is the one design() hands out.
    assert _grid_at(params, 8, 24576).du == ApertureGrid.design(params, "bragg_band").du


def test_bragg_band_refuses_a_grid_that_cannot_resolve_the_orders(params1030) -> None:
    """The weak grid is 6x too coarse for order selection, and says so instead of aliasing."""
    tau = params1030.channels["Ay"].transit_time
    bb, _ = _baseband(params1030, {"Ay": (_gated(3.0 * MHz),)})
    with pytest.raises(ValueError, match="does not resolve"):
        channel_pupil(
            bb, "Ay", 2.0 * tau, ApertureGrid.design(params1030, "weak"), mode="bragg_band"
        )
    with pytest.raises(ValueError, match="mode must be"):
        channel_pupil(
            bb,
            "Ay",
            2.0 * tau,
            ApertureGrid.design(params1030, "weak"),
            mode="raw",  # type: ignore[arg-type]
        )
    with pytest.raises(KeyError, match="not in this baseband"):
        channel_pupil(bb, "Bx", 2.0 * tau, ApertureGrid.design(params1030, "weak"), mode="weak")


# =============================================================== 4. stacking an axis


def test_axis_pupil_multiplies_its_channels_and_illuminates_once(params1030) -> None:
    """Eq. S7: stacked crystals multiply, and the beam is applied once per axis, not per AOD."""
    tau = params1030.channels["Ax"].transit_time
    t = 2.0 * tau
    grid = ApertureGrid.design(params1030, "weak")
    bb, _ = _baseband(
        params1030,
        {
            "Ax": (_gated(2.0 * MHz, 0.31),),
            "Bx": (_gated(-3.0 * MHz, -1.17),),
            "Ay": (_gated(1.0 * MHz),),
        },
    )
    gauss = _gaussian(params1030, grid.u)

    both = axis_pupil(bb, 0, t, grid, mode="weak")
    product = (
        channel_pupil(bb, "Ax", t, grid, mode="weak")
        * channel_pupil(bb, "Bx", t, grid, mode="weak")
        * gauss
    )
    np.testing.assert_allclose(both, product, rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(axis_pupil(bb, "x", t, grid, mode="weak"), both, rtol=0, atol=0)

    # A pair counter-propagates, so its tilt is the *difference* of the two detunings
    # (Table I) — the sum of the two theta1 contributions, taken from the sign authority.
    inner = np.abs(grid.u) < 2.0 * mm
    slope = float(
        np.polynomial.polynomial.polyfit(grid.u[inner], np.unwrap(np.angle(both[inner])), 1)[1]
    )
    want = float(
        conventions.theta1_contribution(2.0 * MHz, geometry("Ax"), params1030.sound_speed)
        + conventions.theta1_contribution(-3.0 * MHz, geometry("Bx"), params1030.sound_speed)
    )
    assert slope == pytest.approx(want, rel=1e-9)
    assert want == pytest.approx(
        2.0 * math.pi * (-3.0 * MHz - 2.0 * MHz) / params1030.sound_speed, rel=1e-12
    )

    # One driven channel on y: its own pupil times the beam.
    single = axis_pupil(bb, "y", t, grid, mode="weak")
    np.testing.assert_allclose(
        single, channel_pupil(bb, "Ay", t, grid, mode="weak") * gauss, rtol=1e-13, atol=1e-18
    )
    with pytest.raises(ValueError, match="unknown axis"):
        axis_pupil(bb, "z", t, grid, mode="weak")


def test_an_undriven_axis_is_the_bare_input_beam(params1030) -> None:
    """No channel on the axis -> the identity factor the simulator uses (Eq. S7)."""
    tau = params1030.channels["Ax"].transit_time
    grid = ApertureGrid.design(params1030, "weak")
    bb, _ = _baseband(params1030, {"Ay": (_gated(3.0 * MHz),)})
    np.testing.assert_allclose(
        axis_pupil(bb, "x", 2.0 * tau, grid, mode="weak"),
        _gaussian(params1030, grid.u),
        rtol=1e-15,
        atol=0.0,
    )
