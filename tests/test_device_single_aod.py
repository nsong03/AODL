r"""M1 acceptance for the device layer: one AOD, from waveform to focal field.

Every test here goes through the real path — ``waveform/tones.py`` -> ``device/aod.py`` ->
``device/aodl.py`` -> the Eq. S11 closed forms of ``field/gaussian.py`` — and the ones that
make a physical claim are cross-checked against ``field/reference.py``, the brute-force
quadrature backend, fed the **literal** windowed pupil built from the channel's phase at
retarded times.  That comparison is what validates the Taylor step of Eqs. S5-S6 itself:
nothing about ``theta1``/``theta2`` is assumed on the reference side.

Both pupils are written in the rotating frame (Eq. S2): the carrier ``f_center`` is dropped
from the drive phase, because its ``s 2 pi f_center u / v`` tilt is a common deflection that
*defines* the optical axis (it would otherwise displace the spot by ~1 mm).  See
``docs/conventions.md`` §3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aodl.device import conventions
from aodl.device.aod import aperture_window, channel_lines, fill_edge
from aodl.device.aodl import build_terms
from aodl.device.conventions import Z_LAB_SIGN, geometry
from aodl.field.gaussian import gauss_moments, gauss_moments_lower, gauss_moments_upper
from aodl.field.reference import reference_field_separable
from aodl.poly import PiecewisePoly
from aodl.units import MHz, mm, ms

tones = pytest.importorskip("aodl.waveform.tones")


# --------------------------------------------------------------- waveform construction


def _tone(freq: PiecewisePoly, phase0: float = 0.0, env=None):
    """A ``ToneTrack`` per WO-02 §2 (``env`` defaults to a flat, fully-on envelope)."""
    if env is None:
        return tones.ToneTrack(freq=freq, phase0=phase0)
    return tones.ToneTrack(freq=freq, env=env, phase0=phase0)


def _static(detuning: float, t_end: float, phase0: float = 0.0, env=None):
    """Constant-detuning tone on ``[0, t_end]``."""
    return _tone(PiecewisePoly.constant(detuning, 0.0, t_end), phase0, env)


def _chirp(f0: float, fdot: float, t_end: float, phase0: float = 0.0, env=None):
    """Linear chirp ``f(t) = f0 + fdot t`` on ``[0, t_end]`` (normalized-time coeffs)."""
    poly = PiecewisePoly.from_segment_coeffs([0.0, t_end], [[f0, fdot * t_end]])
    return _tone(poly, phase0, env)


def _channel(*tracks):
    return tones.ChannelWaveform(tuple(tracks))


def _wfs(params, **channels):
    return tones.WaveformSet(channels=dict(channels), params=params)


# ------------------------------------------------------------------- analytic evaluation


def _axis_factor(optics, alpha, theta1, theta2, coord, z_s11, edge):
    """One separable axis factor of Eq. S11, with the WO-01 §7 ``(a, b)`` mapping.

    ``a = 1/w_in^2 - i (theta2 - k Z / (2 F^2))``, ``b = i (theta1 - k X / F)``; the
    aperture fill edge selects the half-line Gaussian moments.  This mirrors what
    ``field/focal.py`` will do (WO-04) without importing it.
    """
    k, focal = optics.k, optics.focal_length
    a = 1.0 / optics.w_in**2 - 1j * (theta2 - k * np.asarray(z_s11) / (2.0 * focal**2))
    b = 1j * (theta1 - k * np.asarray(coord) / focal)
    if edge is None:
        moments = gauss_moments(a, b)
    elif edge.side == "lower":
        moments = gauss_moments_lower(a, b, edge.u_edge)
    else:
        moments = gauss_moments_upper(a, b, edge.u_edge)
    return alpha[0] * moments[0] + alpha[1] * moments[1] + alpha[2] * moments[2]


def _term_field(terms, optics, x, y, z_s11, index: int = 0):
    """Closed-form field of one pupil term (constant prefactors dropped)."""
    fx = _axis_factor(
        optics,
        terms.alpha[0, :, index],
        terms.theta1[0, index],
        terms.theta2[0, index],
        x,
        z_s11,
        terms.edge[0],
    )
    fy = _axis_factor(
        optics,
        terms.alpha[1, :, index],
        terms.theta1[1, index],
        terms.theta2[1, index],
        y,
        z_s11,
        terms.edge[1],
    )
    return terms.c[index] * fx * fy


# ------------------------------------------------------------------- reference pupils


def _gaussian_pupil(optics):
    """Undriven axis: just the input beam."""
    return lambda u: np.exp(-(np.asarray(u) ** 2) / optics.w_in**2)


def _fill_weight(u, aod, geom, t):
    """Hard fill edge for the quadrature grid.

    The transition is a single-cell ramp, which is exactly the trapezoid weight of a
    half-line integral whose limit sits on a grid point (and an O(h^2) treatment when it
    does not).  Physically it is a hard edge: the ramp spans 3 um of a 4 mm beam.
    """
    edge = fill_edge(aod, geom, t)
    if edge is None:
        return np.ones_like(u)
    h = float(u[1] - u[0])
    if edge.side == "lower":
        return np.clip((u - edge.u_edge) / h + 0.5, 0.0, 1.0)
    return np.clip((edge.u_edge - u) / h + 0.5, 0.0, 1.0)


def _literal_pupil(cw, aod, geom, optics, t):
    """The channel's exact rotating-frame pupil — no Taylor expansion anywhere.

    ``(i C / 2) sum_n A_n(t_ret(u)) exp(-i phase_n(t_ret(u)))`` times the fill window and
    the input Gaussian, with ``t_ret(u) = t - (s u + D/2) / v``.
    """

    def pupil(u):
        u = np.asarray(u, dtype=np.float64)
        t_ret = conventions.retarded_time(t, u, geom, aod)
        drive = np.zeros(u.shape, dtype=np.complex128)
        for tone in cw.tones:
            envelope = np.asarray(tone.env.A(t_ret), dtype=np.float64)
            drive += envelope * np.exp(-1j * np.asarray(tone.phase(t_ret), dtype=np.float64))
        drive *= 0.5j * aod.drive_strength
        return drive * _fill_weight(u, aod, geom, t) * np.exp(-(u**2) / optics.w_in**2)

    return pupil


# ------------------------------------------------------------------------- small tools


def _peak_position(x, y):
    """Sub-sample peak of a sampled profile (parabolic vertex through the top 3 points)."""
    i = int(np.argmax(y))
    assert 0 < i < y.size - 1, "peak fell on the edge of the scan"
    y0, y1, y2 = y[i - 1 : i + 2]
    return float(x[i] + 0.5 * (x[1] - x[0]) * (y0 - y2) / (y0 - 2.0 * y1 + y2))


def _deviation(analytic, reference):
    """Max |difference| of the two complex fields after normalizing at the reference peak."""
    idx = np.unravel_index(int(np.argmax(np.abs(reference))), reference.shape)
    return float(np.abs(analytic / analytic[idx] - reference / reference[idx]).max())


# =============================================================== 1. static tone


def test_static_tone_deflects_per_table_i(params1030) -> None:
    """f = +3 MHz on Ay: one term, ``theta1y = -2 pi f / v``, spot at ``-lambda F f / v``."""
    optics = params1030.optics
    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    tau, v = aod.transit_time, aod.sound_speed
    detuning, t = 3.0 * MHz, 2.0 * tau

    cw = _channel(_static(detuning, 6.0 * tau, phase0=0.4))
    terms = build_terms(_wfs(params1030, Ay=cw), t, channels=("Ay",))

    assert terms.n_terms == 1
    assert terms.theta1[1, 0] == pytest.approx(-2.0 * math.pi * detuning / v, rel=1e-13)
    assert terms.theta1[0, 0] == 0.0
    assert np.all(terms.theta2 == 0.0)
    assert terms.df_opt[0] == pytest.approx(detuning, rel=1e-13)
    assert terms.edge == (None, None)  # fully filled at t = 2 tau
    np.testing.assert_allclose(terms.alpha[:, 0, 0], 1.0)
    np.testing.assert_allclose(terms.alpha[:, 1:, 0], 0.0)

    # c = (i C / 2) A(t_c) exp(-i phase(t_c)), with t_c = t - tau/2 (Eq. S3).
    lines = channel_lines(cw, aod, t)
    expected_amp = 0.5j * aod.drive_strength * np.exp(-1j * cw.tones[0].phase(t - 0.5 * tau))
    assert lines.amp[0] == pytest.approx(complex(expected_amp), rel=1e-13)
    assert terms.c[0] == pytest.approx(lines.amp[0], rel=1e-13)

    y_expect = -params1030.deflection_scale * detuning
    y_axis = y_expect + np.linspace(-4.0, 4.0, 401) * optics.waist0
    analytic = _term_field(terms, optics, 0.0, y_axis, 0.0)
    assert abs(_peak_position(y_axis, np.abs(analytic) ** 2) - y_expect) < 0.01 * optics.waist0

    # The literal windowed pupil agrees — so the Taylor step of Eqs. S5-S6 is exact here
    # (a static tone gives a phase strictly linear in u).
    reference = reference_field_separable(
        _gaussian_pupil(optics),
        _literal_pupil(cw, aod, geom, optics, t),
        optics,
        0.0,
        y_axis,
        0.0,
    )
    assert abs(_peak_position(y_axis, np.abs(reference) ** 2) - y_expect) < 0.01 * optics.waist0
    assert _deviation(analytic, reference) < 1e-6


# =============================================================== 2. linear chirp


def test_linear_chirp_is_a_cylindrical_lens(params1030) -> None:
    """fdot = +50 MHz/ms on Ay: ``theta2y = -pi fdot / v^2``, y-focus at ``-lens_scale fdot``."""
    optics = params1030.optics
    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    tau, v = aod.transit_time, aod.sound_speed
    fdot, t = 50.0 * MHz / ms, 2.0 * tau

    cw = _channel(_chirp(0.0, fdot, 6.0 * tau))
    terms = build_terms(_wfs(params1030, Ay=cw), t, channels=("Ay",))

    assert terms.theta2[1, 0] == pytest.approx(-math.pi * fdot / v**2, rel=1e-13)
    assert terms.theta2[0, 0] == 0.0

    z_expect = -params1030.lens_scale * fdot
    z_from_theta2 = 2.0 * optics.focal_length**2 * terms.theta2[1, 0] / optics.k
    assert z_from_theta2 == pytest.approx(z_expect, rel=1e-12)

    # Scan the on-axis intensity of each axis factor against the Eq. S11 defocus variable.
    z_axis = np.linspace(-2.5, 1.5, 1601) * abs(z_expect)
    y_spot = float(terms.theta1[1, 0]) * optics.focal_length / optics.k
    fy = _axis_factor(
        optics,
        terms.alpha[1, :, 0],
        terms.theta1[1, 0],
        terms.theta2[1, 0],
        y_spot,
        z_axis,
        terms.edge[1],
    )
    assert abs(_peak_position(z_axis, np.abs(fy) ** 2) - z_expect) < 0.02 * abs(z_expect)

    fx = _axis_factor(optics, terms.alpha[0, :, 0], 0.0, 0.0, 0.0, z_axis, terms.edge[0])
    assert abs(_peak_position(z_axis, np.abs(fx) ** 2)) < 0.02 * abs(z_expect)  # x unshifted

    # Restated in lab units.  Only fdot_Ay is nonzero, so Table I gives
    # Zbar = lens_scale fdot / 2 and Delta F = -lens_scale fdot, hence
    # Z_x = Zbar + DeltaF/2 = 0 and Z_y = Zbar - DeltaF/2 = +lens_scale fdot: a single AOD
    # is a pure cylindrical lens, i.e. a pure astigmat.
    lens = params1030.lens_scale
    z_lab = Z_LAB_SIGN * z_expect
    assert z_lab == pytest.approx(lens * fdot, rel=1e-12)
    assert 0.5 * lens * fdot - 0.5 * (-lens * fdot) == pytest.approx(z_lab, rel=1e-12)
    assert 0.5 * lens * fdot + 0.5 * (-lens * fdot) == pytest.approx(0.0, abs=1e-18)

    # Quadrature agrees on a (Y, Z) patch around the y-focus: for a linear chirp the
    # retarded phase is exactly quadratic in u, so the Taylor model should be exact.
    y_axis = y_spot + np.linspace(-3.0, 3.0, 9) * optics.waist0
    z_probe = z_expect + np.linspace(-1.0, 1.0, 9) * optics.rayleigh
    yy, zz = np.meshgrid(y_axis, z_probe, indexing="ij")
    reference = reference_field_separable(
        _gaussian_pupil(optics),
        _literal_pupil(cw, aod, geom, optics, t),
        optics,
        0.0,
        yy,
        zz,
    )
    assert _deviation(_term_field(terms, optics, 0.0, yy, zz), reference) < 1e-6


# =============================================================== 3. retardation


def test_position_tracks_the_beam_center_retarded_time(params1030) -> None:
    """The spot follows ``f(t - tau/2)``; using ``f(t)`` would be off by ``fdot tau / 2``."""
    optics = params1030.optics
    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    tau, v = aod.transit_time, aod.sound_speed
    f0, fdot = 1.0 * MHz, 50.0 * MHz / ms
    t, delta = 2.0 * tau, 0.4 * tau

    cw = _channel(_chirp(f0, fdot, 8.0 * tau))
    wfs = _wfs(params1030, Ay=cw)
    now = build_terms(wfs, t, channels=("Ay",))
    later = build_terms(wfs, t + delta, channels=("Ay",))

    def spot_y(terms):
        return float(terms.theta1[1, 0]) * optics.focal_length / optics.k

    # Ay has sound_sign -1, so an up-chirp walks the spot toward -y.
    assert spot_y(later) - spot_y(now) == pytest.approx(
        -params1030.deflection_scale * fdot * delta, rel=1e-10
    )

    extracted = float(now.theta1[1, 0]) * v / (geom.sound_sign * 2.0 * math.pi)
    assert extracted == pytest.approx(f0 + fdot * (t - 0.5 * tau), rel=1e-12)

    # Teeth: the naive "evaluate at t" answer differs by fdot tau / 2 ...
    naive = f0 + fdot * t
    assert naive - extracted == pytest.approx(fdot * 0.5 * tau, rel=1e-10)
    # ... which is several spot sizes on the screen, so the test can tell them apart.
    offset = params1030.deflection_scale * fdot * 0.5 * tau
    assert offset > 2.0 * optics.waist0

    # Quadrature picks the retarded answer, not the naive one.
    y_expect = -params1030.deflection_scale * extracted
    y_naive = -params1030.deflection_scale * naive
    y_axis = y_expect + np.linspace(-5.0, 5.0, 401) * optics.waist0
    reference = reference_field_separable(
        _gaussian_pupil(optics),
        _literal_pupil(cw, aod, geom, optics, t),
        optics,
        0.0,
        y_axis,
        0.0,
    )
    y_peak = _peak_position(y_axis, np.abs(reference) ** 2)
    assert abs(y_peak - y_expect) < 0.05 * optics.waist0
    assert abs(y_peak - y_naive) > 2.0 * optics.waist0
    assert _deviation(_term_field(now, optics, 0.0, y_axis, 0.0), reference) < 1e-6


# =============================================================== 4. fill transient


def test_fill_edge_hand_computation(params1030) -> None:
    """At ``t = 0.6 tau`` the wavefront is ``0.1 D`` from the center, on the sound's side."""
    aod = params1030.channels["Ay"]
    tau, aperture = aod.transit_time, aod.aperture

    edge = fill_edge(aod, geometry("Ay"), 0.6 * tau)  # sound toward -y
    assert edge is not None
    assert edge.side == "lower"  # filled region is u >= u_edge (toward the transducer)
    assert edge.u_edge == pytest.approx(-0.1 * aperture, rel=1e-12)
    assert edge.u_edge == pytest.approx(-0.75 * mm, rel=1e-12)

    mirrored = fill_edge(aod, geometry("By"), 0.6 * tau)  # sound toward +y
    assert mirrored.side == "upper"
    assert mirrored.u_edge == pytest.approx(+0.1 * aperture, rel=1e-12)

    assert fill_edge(aod, geometry("Ay"), tau) is None
    assert fill_edge(aod, geometry("Ay"), 3.0 * tau) is None
    assert fill_edge(aod, geometry("Ay"), 0.999 * tau) is not None


def test_fill_transient_matches_hard_edged_quadrature(params1030) -> None:
    """Partially filled aperture: edge-windowed closed form vs hard-edged quadrature."""
    optics = params1030.optics
    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    tau = aod.transit_time
    detuning, t = 3.0 * MHz, 0.6 * tau

    cw = _channel(_static(detuning, 6.0 * tau, phase0=-0.9))
    wfs = _wfs(params1030, Ay=cw)
    terms = build_terms(wfs, t, channels=("Ay",))

    assert terms.edge[1] == fill_edge(aod, geom, t)
    assert terms.edge[0] is None  # the undriven x axis is never windowed

    y_expect = -params1030.deflection_scale * detuning
    y_axis = y_expect + np.linspace(-4.0, 4.0, 161) * optics.waist0
    analytic = _term_field(terms, optics, 0.0, y_axis, 0.0)
    reference = reference_field_separable(
        _gaussian_pupil(optics),
        _literal_pupil(cw, aod, geom, optics, t),
        optics,
        0.0,
        y_axis,
        0.0,
    )
    assert _deviation(analytic, reference) < 1e-3

    # A partly filled aperture diffracts less light: the peak intensity is well below the
    # fully filled value (analytically (1 + erf(0.375))^2 / 4 ~ 0.49 of it).
    full = _term_field(build_terms(wfs, 2.0 * tau, channels=("Ay",)), optics, 0.0, y_axis, 0.0)
    ratio = float(np.max(np.abs(analytic) ** 2) / np.max(np.abs(full) ** 2))
    assert 0.3 < ratio < 0.7


def test_aperture_window_is_the_literal_drive_on_the_crystal(params1030) -> None:
    """The diagnostic window reproduces V(t_ret) and is zero ahead of the wavefront."""
    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    tau = aod.transit_time
    t = 0.6 * tau
    cw = _channel(_static(3.0 * MHz, 6.0 * tau, phase0=0.7))

    u, signal = aperture_window(cw, aod, geom, t, n=4001)
    assert u[0] == pytest.approx(-0.5 * aod.aperture)
    assert u[-1] == pytest.approx(+0.5 * aod.aperture)

    edge = fill_edge(aod, geom, t)
    assert np.all(signal[u < edge.u_edge] == 0.0)
    assert np.abs(signal[u > edge.u_edge]).max() > 0.9

    tone = cw.tones[0]
    t_ret = conventions.retarded_time(t, u, geom, aod)
    expected = np.where(
        t_ret >= 0.0,
        tone.env.A(t_ret) * np.cos(2.0 * np.pi * aod.f_center * t_ret + tone.phase(t_ret)),
        0.0,
    )
    np.testing.assert_allclose(signal, expected, atol=1e-12)

    _, filled = aperture_window(cw, aod, geom, 1.5 * tau, n=4001)
    assert np.count_nonzero(filled) > 0.99 * filled.size


# =============================================================== 5. channel products


def test_two_channel_product_is_pure_algebra(params1030) -> None:
    """Ax + Bx static tones: one term with ``theta1x = 2 pi (f2 - f1) / v`` (Table I)."""
    optics = params1030.optics
    tau = params1030.channels["Ax"].transit_time
    v = params1030.sound_speed
    f1, f2 = 2.5 * MHz, -1.5 * MHz
    t = 2.0 * tau

    cw_a = _channel(_static(f1, 6.0 * tau, phase0=0.31))
    cw_b = _channel(_static(f2, 6.0 * tau, phase0=-1.17))
    terms = build_terms(_wfs(params1030, Ax=cw_a, Bx=cw_b), t)

    assert terms.n_terms == 1
    assert terms.theta1[0, 0] == pytest.approx(2.0 * math.pi * (f2 - f1) / v, rel=1e-13)
    assert terms.theta1[1, 0] == 0.0
    assert np.all(terms.theta2 == 0.0)
    assert terms.df_opt[0] == pytest.approx(f1 + f2, rel=1e-13)
    assert terms.edge == (None, None)

    lines_a = channel_lines(cw_a, params1030.channels["Ax"], t)
    lines_b = channel_lines(cw_b, params1030.channels["Bx"], t)
    assert terms.c[0] == pytest.approx(lines_a.amp[0] * lines_b.amp[0], rel=1e-13)

    x_spot = float(terms.theta1[0, 0]) * optics.focal_length / optics.k
    assert x_spot == pytest.approx(params1030.deflection_scale * (f2 - f1), rel=1e-12)


def test_cartesian_product_over_tones(params1030) -> None:
    """Two tones on Ax times three on Bx = six beams (Eq. S7, the Fig. S6 picture)."""
    tau = params1030.channels["Ax"].transit_time
    v = params1030.sound_speed
    t = 2.0 * tau
    fa = (1.0 * MHz, 2.0 * MHz)
    fb = (-0.5 * MHz, 3.0 * MHz, 4.0 * MHz)

    cw_a = _channel(*[_static(f, 6.0 * tau) for f in fa])
    cw_b = _channel(*[_static(f, 6.0 * tau) for f in fb])
    terms = build_terms(_wfs(params1030, Ax=cw_a, Bx=cw_b), t)

    assert terms.n_terms == len(fa) * len(fb)
    np.testing.assert_allclose(terms.df_opt, [a + b for a in fa for b in fb], rtol=1e-13)
    np.testing.assert_allclose(
        terms.theta1[0], [2.0 * math.pi * (b - a) / v for a in fa for b in fb], rtol=1e-13
    )
    np.testing.assert_allclose(terms.theta1[1], 0.0)
    np.testing.assert_allclose(terms.alpha[:, 0, :], 1.0)


def test_undriven_channels_are_identity_factors(params1030) -> None:
    """Selecting a subset leaves the other axis with a bare Gaussian pupil."""
    tau = params1030.channels["Ax"].transit_time
    cw = _channel(_static(2.0 * MHz, 6.0 * tau))
    wfs = _wfs(params1030, Ax=cw)

    only_x = build_terms(wfs, 2.0 * tau, channels=("Ax",))
    assert only_x.theta1[1, 0] == 0.0
    assert only_x.theta2[1, 0] == 0.0
    np.testing.assert_allclose(only_x.alpha[1, :, 0], [1.0, 0.0, 0.0])

    nothing = build_terms(wfs, 2.0 * tau, channels=())
    assert nothing.n_terms == 1
    assert nothing.c[0] == 1.0
    assert np.all(nothing.theta1 == 0.0)
    assert nothing.edge == (None, None)

    with pytest.raises(KeyError, match="not present"):
        build_terms(wfs, 2.0 * tau, channels=("By",))


def test_amplitude_polynomial_from_a_ramping_envelope(params1030) -> None:
    """Eq. S5: alpha carries the envelope *shape*; its value lives in the line amplitude."""
    aod = params1030.channels["Ay"]
    geom = geometry("Ay")
    tau, v = aod.transit_time, aod.sound_speed
    env = tones.SmoothOnOff(t_on=0.5 * tau, t_off=5.0 * tau, ramp=1.5 * tau)
    cw = _channel(_static(2.0 * MHz, 6.0 * tau, env=env))
    t = 1.4 * tau  # t_c = 0.9 tau, midway up the rise
    t_c = t - 0.5 * tau

    terms = build_terms(_wfs(params1030, Ay=cw), t, channels=("Ay",))
    amp, d_amp, d2_amp = float(env.A(t_c)), float(env.dA(t_c)), float(env.d2A(t_c))
    assert 0.0 < amp < 1.0
    assert d_amp > 0.0

    assert terms.alpha[1, 0, 0] == pytest.approx(1.0)
    assert terms.alpha[1, 1, 0] == pytest.approx(-geom.sound_sign * (d_amp / amp) / v, rel=1e-12)
    assert terms.alpha[1, 2, 0] == pytest.approx((d2_amp / amp) / (2.0 * v**2), rel=1e-12)
    np.testing.assert_allclose(terms.alpha[0, :, 0], [1.0, 0.0, 0.0])
    assert abs(terms.c[0]) == pytest.approx(0.5 * aod.drive_strength * amp, rel=1e-12)
