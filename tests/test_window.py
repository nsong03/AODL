r"""Two-sided aperture window (WO-10): ``W_n`` moments, pair fill physics, product guard.

Three things are pinned here.

**The moments.**  ``field/gaussian.gauss_moments_window`` against ``scipy.integrate.quad``,
plus the two identities that tie it to the rest of the family
(``W(u0, u1) + E(u1) + F(u0) = I`` and ``W -> I`` for a wide window) and the documented
cancellation caveat: when both edges sit far out on the same tail the difference of the two
``E_n`` loses relative digits while its *absolute* error stays at roundoff on the full-line
scale, which is the only scale a term is ever compared on.

**The physics.**  A counter-propagating pair fills its axis from both transducers at once,
so the light that crosses both crystals sees the intersection ``[D/2 - v t, v t - D/2]``
(``docs/conventions.md`` §7).  That window is **empty until ``t = tau/2``** — both wavefronts
must reach a point before anything gets through it — so a pair-driven tweezer is strictly
dark for the first half transit, and then grows in.  The mid-fill field is checked against
``field/reference.py`` fed the **literal** two-edged pupil (both channels' drives at their
own retarded times, each with its own hard fill weight), exactly as WO-03's single-AOD tests
do for one edge.

**The guard.**  ``device/aodl.build_terms`` cuts weak lines *before* the Cartesian product
(losslessly — see :func:`aodl.device.aodl._pre_cut_lines`) and refuses to build more than
``max_terms`` of them, which is what a non-commensurate multi-tone drive would otherwise ask
for: nothing in its IM3 set merges, so the line counts multiply out in full.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad
from scipy.special import erf, erfc

from aodl.device import conventions
from aodl.device.aod import channel_lines, fill_edge
from aodl.device.aodl import MAX_TERMS, TERM_PRUNE, FillWindow, build_terms
from aodl.device.conventions import geometry
from aodl.engine import simulate
from aodl.field.focal import FrameGrid, intensity_frame, term_field
from aodl.field.gaussian import (
    gauss_moments,
    gauss_moments_lower,
    gauss_moments_upper,
    gauss_moments_window,
)
from aodl.field.measure import measure
from aodl.field.reference import reference_field_separable
from aodl.poly import PiecewisePoly
from aodl.units import MHz, mm, ms

tones = pytest.importorskip("aodl.waveform.tones")

#: Well-conditioned random draws demanded of the quadrature comparison.
N_DRAWS = 150
#: Largest cancellation ratio ``(\int |integrand|) / |W0|`` a draw may carry.  Above this
#: neither ``quad`` nor the closed-form difference resolves the integral to 1e-9 — the same
#: rejection ``tests/test_gaussian.py`` applies to the half-line families.
COND_MAX = 1e4


# ------------------------------------------------------------------ quadrature reference


def _quad_moment(a, b, n, lo, hi, scale=1.0):
    """``\\int_lo^hi u^n exp(-a u^2 + b u) du`` by quadrature (real and imaginary apart).

    ``scale`` divides the integrand (and multiplies the result back) so extreme exponents
    stay inside double range.  QUADPACK's roundoff/subdivision warnings fire routinely on
    oscillatory integrands that still converge far past the tolerances asserted here.
    """

    def integrand(u, part):
        value = u**n * np.exp(-a * u * u + b * u) / scale
        return value.real if part == 0 else value.imag

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        re, _ = quad(integrand, lo, hi, args=(0,), limit=2000, epsabs=0.0, epsrel=1e-12)
        im, _ = quad(integrand, lo, hi, args=(1,), limit=2000, epsabs=0.0, epsrel=1e-12)
    return (re + 1j * im) * scale


def _draw(rng):
    """One draw: ``a = 10^U(-2,2) e^{i theta}``, ``|b| <= 10 sqrt|a|``, two sorted edges.

    The same box ``tests/test_gaussian.py`` uses for ``E_n`` / ``F_n``: it reaches
    ``|b^2/4a| ~ 100``, i.e. the oscillation-to-width ratio of the real AODL pupil, without
    carrying its units.
    """
    a = 10 ** rng.uniform(-2.0, 2.0) * np.exp(1j * np.deg2rad(rng.uniform(-80.0, 80.0)))
    b = 10.0 * np.sqrt(abs(a)) * rng.uniform(0.0, 1.0) * np.exp(1j * rng.uniform(0.0, 2 * np.pi))
    edges = np.sort(rng.uniform(-3.0, 3.0, 2)) / np.sqrt(abs(a))
    return a, b, float(edges[0]), float(edges[1])


# =============================================================== 1. the window moments


def test_window_moments_match_quadrature(rng) -> None:
    """``W_n`` vs ``quad`` over well-conditioned random draws (rel 1e-9)."""
    accepted = attempts = 0
    while accepted < N_DRAWS:
        attempts += 1
        assert attempts < 40 * N_DRAWS, "draw rejection is out of control"
        a, b, u0, u1 = _draw(rng)
        if u1 - u0 < 1e-3 / np.sqrt(abs(a)):
            continue
        window = gauss_moments_window(a, b, u0, u1)
        i0 = gauss_moments(a, b)[0]
        # The documented cancellation corner: a window this dark is not asked to be
        # relatively accurate (see test_window_cancellation_stays_absolute below).
        if abs(window[0]) <= 1e-10 * abs(i0):
            continue
        l1 = np.sqrt(np.pi / a.real) * np.exp(b.real**2 / (4.0 * a.real))
        if l1 / abs(window[0]) > COND_MAX:
            continue
        accepted += 1
        for n in range(3):
            assert window[n] == pytest.approx(_quad_moment(a, b, n, u0, u1), rel=1e-9)
    assert accepted == N_DRAWS


def test_window_splits_the_line_exactly(rng) -> None:
    """``W_n(u0, u1) + E_n(u1) + F_n(u0) = I_n`` — the three pieces of the real line."""
    for _ in range(50):
        a, b, u0, u1 = _draw(rng)
        full = gauss_moments(a, b)
        window = gauss_moments_window(a, b, u0, u1)
        above = gauss_moments_lower(a, b, u1)
        below = gauss_moments_upper(a, b, u0)
        for n in range(3):
            total = window[n] + above[n] + below[n]
            scale = max(abs(full[n]), abs(window[n]), abs(above[n]), abs(below[n]))
            assert abs(total - full[n]) < 1e-13 * scale


def test_wide_window_reproduces_the_full_line(rng) -> None:
    """A window 40 widths either side of the saddle *is* the full line, to 1e-12."""
    for _ in range(20):
        a, b, _, _ = _draw(rng)
        width = 1.0 / np.sqrt(a.real)
        peak = b.real / (2.0 * a.real)
        window = gauss_moments_window(a, b, peak - 40.0 * width, peak + 40.0 * width)
        full = gauss_moments(a, b)
        for n in range(3):
            assert window[n] == pytest.approx(full[n], rel=1e-12)


def test_window_rejects_an_empty_interval() -> None:
    """``u0 >= u1`` has no moments — the caller must drop the (dark) term instead."""
    for u0, u1 in ((1.0, 1.0), (2.0, -0.5)):
        with pytest.raises(ValueError, match="u0 < u1"):
            gauss_moments_window(1.0, 0.0, u0, u1)
    # Vectorized: one bad row is enough to refuse the whole call.
    with pytest.raises(ValueError, match="u0 < u1"):
        gauss_moments_window(1.0, 0.0, np.array([-1.0, 0.5]), np.array([1.0, 0.5]))
    # ... and Re(a) > 0 is still required.
    with pytest.raises(ValueError, match="Re\\(a\\)"):
        gauss_moments_window(-1.0, 0.0, -1.0, 1.0)


def test_window_is_range_safe_at_the_physical_scale() -> None:
    """AODL magnitudes: the naive ``exp(b^2/4a) [erfc - erfc]`` is nan, ``W_n`` is exact.

    ``a = 2.5e5 - 3.7e5j m^-2`` is (1/w_in^2, chirp + defocus) for a 2 mm beam,
    ``b = 1e6j m^-1`` a few-waist deflection, and the window is the ``t = 0.75 tau`` pair
    fill ``+-1.875 mm``.  Here ``b^2/4a ~ -3.1e5 - 4.6e5j``: ``exp(b^2/4a)`` underflows (the
    full-line ``I0`` is 0 in double precision) while ``erfc`` overflows.
    """
    a, b = 2.5e5 - 3.7e5j, 1e6j
    u0, u1 = -1.875 * mm, 1.875 * mm
    root = np.sqrt(a)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        naive = (
            0.5
            * np.sqrt(np.pi / a)
            * np.exp(b * b / (4.0 * a))
            * (erfc(root * u0 - b / (2.0 * root)) - erfc(root * u1 - b / (2.0 * root)))
        )
    assert not np.isfinite(naive), "the naive form is expected to fail here"

    window = gauss_moments_window(a, b, u0, u1)
    assert np.all(np.isfinite(np.array(window)))
    for n in range(3):
        assert window[n] == pytest.approx(_quad_moment(a, b, n, u0, u1), rel=1e-8)


def test_window_cancellation_stays_absolute() -> None:
    """The documented caveat, quantified: relative digits are lost, absolute ones are not.

    A half-``w_in``-wide window is walked out along one tail of an undeflected Gaussian
    pupil (``a = 1/w_in^2``, ``b = 0``), so both edges end up on the same side and
    ``W0 = E0(u0) - E0(u1)`` becomes a difference of two numbers that are each ``~I0``.

    Measured on this sweep: quadrature agreement stays at 1e-9 relative or better while
    ``|W0| > 1e-8 |I0|`` (the worst case above ``1e-6 |I0|`` is 1.4e-11), and below that the
    *relative* error grows without bound — while the *absolute* error never exceeds
    ``1e-16 |I0|``.  That is the whole claim: such a window passes no light, and the error it
    carries is roundoff on the scale of the fully filled aperture it is compared against.
    """
    w_in = 2.0 * mm
    a, b = 1.0 / w_in**2, 0.0
    i0 = abs(complex(gauss_moments(a, b)[0]))

    seen_dark = False
    for offset in np.arange(0.0, 6.01, 0.25):
        u1 = -offset * w_in
        u0 = u1 - 0.5 * w_in
        got = complex(gauss_moments_window(a, b, u0, u1)[0])
        edge_value = math.exp(-a * u0 * u0)  # rescale so *quad* stays in range too
        ref = _quad_moment(a, b, 0, u0, u1, scale=edge_value)
        assert abs(got - ref) < 1e-14 * i0, f"absolute error blew up at {offset} w_in"
        if abs(got) > 1e-6 * i0:
            assert got == pytest.approx(ref, rel=1e-9)
        if abs(got) < 1e-12 * i0:
            seen_dark = True
    assert seen_dark, "the sweep never reached the cancellation corner"


# ------------------------------------------------------ waveform / pupil helpers (WO-03)


def _tone(freq: PiecewisePoly, phase0: float = 0.0, env=None):
    if env is None:
        return tones.ToneTrack(freq=freq, phase0=phase0)
    return tones.ToneTrack(freq=freq, env=env, phase0=phase0)


def _static(detuning: float, t_end: float, phase0: float = 0.0, env=None):
    """Constant-detuning tone on ``[0, t_end]``."""
    return _tone(PiecewisePoly.constant(detuning, 0.0, t_end), phase0, env)


def _chirp(f0: float, fdot: float, t_end: float, phase0: float = 0.0):
    """Linear chirp ``f(t) = f0 + fdot t`` on ``[0, t_end]`` (normalized-time coeffs)."""
    return _tone(PiecewisePoly.from_segment_coeffs([0.0, t_end], [[f0, fdot * t_end]]), phase0)


def _channel(*tracks):
    return tones.ChannelWaveform(tuple(tracks))


def _wfs(params, **channels):
    return tones.WaveformSet(channels=dict(channels), params=params)


def _linear_drive(params):
    """``params`` with every channel at ``mixing_order=1`` — the strictly linear model.

    The literal reference pupil below is the first-order expansion ``1 + i C V`` of Eq. S3,
    so the analytic side is asked for the same model; one tone per channel then gives exactly
    one pupil term and the comparison is a statement about the *window*, not about mixing.
    """
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _gaussian_pupil(optics):
    """Undriven axis: just the input beam."""
    return lambda u: np.exp(-(np.asarray(u) ** 2) / optics.w_in**2)


def _fill_weight(u, aod, geom, t):
    """Hard fill edge for the quadrature grid (the WO-03 single-cell ramp).

    The ramp is exactly the trapezoid weight of a half-line integral whose limit sits on a
    grid point, and O(h^2) when it does not.  Two counter-propagating channels each apply
    their own, so their product is the two-edged window.
    """
    edge = fill_edge(aod, geom, t)
    if edge is None:
        return np.ones_like(u)
    h = float(u[1] - u[0])
    if edge.side == "lower":
        return np.clip((u - edge.u_edge) / h + 0.5, 0.0, 1.0)
    return np.clip((edge.u_edge - u) / h + 0.5, 0.0, 1.0)


def _literal_pair_pupil(params, channels, t):
    """The exact rotating-frame pupil of *both* channels on one axis — no Taylor step.

    Per channel ``(i C / 2) sum_n A_n(t_ret(u)) exp(-i phase_n(t_ret(u)))`` times its own
    fill window, with ``t_ret(u) = t - (s u + D/2) / v``; the two are multiplied (the AODs are
    stacked, so their pupils multiply, Eq. S7) and the input Gaussian is applied once.
    """
    optics = params.optics

    def pupil(u):
        u = np.asarray(u, dtype=np.float64)
        out = np.ones(u.shape, dtype=np.complex128)
        for name, cw in channels.items():
            geom = geometry(name)
            aod = params.channels[name]
            t_ret = conventions.retarded_time(t, u, geom, aod)
            drive = np.zeros(u.shape, dtype=np.complex128)
            for tone in cw.tones:
                envelope = np.asarray(tone.env.A(t_ret), dtype=np.float64)
                drive += envelope * np.exp(-1j * np.asarray(tone.phase(t_ret), dtype=np.float64))
            out = out * 0.5j * aod.drive_strength * drive * _fill_weight(u, aod, geom, t)
        return out * np.exp(-(u**2) / optics.w_in**2)

    return pupil


def _pair_scene(params):
    """Static tones on ``Ax`` + ``Bx`` (a pure x-axis counter-propagating pair)."""
    tau = params.channels["Ax"].transit_time
    channels = {
        "Ax": _channel(_static(2.0 * MHz, 6.0 * tau, phase0=0.31)),
        "Bx": _channel(_static(-3.0 * MHz, 6.0 * tau, phase0=-1.17)),
    }
    return channels, tones.WaveformSet(channels=channels, params=params)


def _peak_intensity(terms, optics, x_axis):
    """Peak of ``|sum_terms U|^2`` along ``x_axis`` at ``y = z = 0`` (0 with no terms)."""
    if terms.n_terms == 0:
        return 0.0
    field = term_field(terms, optics, x_axis, 0.0, 0.0).sum(axis=0)
    return float(np.max(np.abs(field) ** 2))


def _pair_reference(params, channels, t, x_axis):
    """Brute-force Eq. S11 for the literal two-edged pair pupil (x) and a bare Gaussian (y)."""
    return reference_field_separable(
        _literal_pair_pupil(params, channels, t),
        _gaussian_pupil(params.optics),
        params.optics,
        x_axis,
        0.0,
        0.0,
    )


# =============================================================== 2. pair fill physics


def test_pair_is_strictly_dark_before_half_a_transit(params1030) -> None:
    """t = 0.45 tau: the two wavefronts have not met, so *no* light reaches the image."""
    params = _linear_drive(params1030)
    optics = params.optics
    aod = params.channels["Ax"]
    tau, v, aperture = aod.transit_time, aod.sound_speed, aod.aperture
    _, wfs = _pair_scene(params)

    t = 0.45 * tau
    terms = build_terms(wfs, t)
    lo, hi = terms.edge[0]
    assert isinstance(terms.edge[0], FillWindow)
    assert terms.edge[1] is None  # the undriven y axis is never windowed
    # Ax fills u >= D/2 - v t, Bx fills u <= v t - D/2: the intersection is inverted.
    assert lo == pytest.approx(0.5 * aperture - v * t, rel=1e-12)
    assert hi == pytest.approx(v * t - 0.5 * aperture, rel=1e-12)
    assert lo == pytest.approx(+0.05 * aperture, rel=1e-12)  # +0.375 mm
    assert hi == pytest.approx(-0.05 * aperture, rel=1e-12)  # -0.375 mm
    assert lo > hi

    assert terms.n_terms == 0
    assert terms.pruned_power == 0.0  # darkness is physics, not an approximation

    w0 = optics.waist0
    grid = FrameGrid(-40.0 * w0, 40.0 * w0, 41, -40.0 * w0, 40.0 * w0, 41)
    frame = intensity_frame(terms, optics, grid, 0.0)
    assert np.all(frame == 0.0)
    assert measure(terms, optics) == []

    # The zero-width window at exactly tau/2 is dark too: a point, not an aperture.
    assert build_terms(wfs, 0.5 * tau).n_terms == 0

    # ... and the whole engine survives a dark frame (no groups, no rows, a black frame).
    result = simulate(wfs, [t])
    assert result.metrics == [[]]
    assert result.spot_table()["time"].size == 0
    assert np.all(result.frame(0, grid) == 0.0)


def test_pair_window_matches_the_literal_two_edged_pupil(params1030) -> None:
    """0.55 tau: light appears.  0.75 tau: it matches the hard two-edged quadrature."""
    params = _linear_drive(params1030)
    optics = params.optics
    aod = params.channels["Ax"]
    tau, v, aperture = aod.transit_time, aod.sound_speed, aod.aperture
    channels, wfs = _pair_scene(params)

    x_spot = params.deflection_scale * (-3.0 * MHz - 2.0 * MHz)
    x_axis = x_spot + np.linspace(-4.0, 4.0, 41) * optics.waist0

    full = build_terms(wfs, 2.0 * tau)
    assert full.edge == (None, None)
    peak_full = _peak_intensity(full, optics, x_axis)

    peaks = {}
    for frac in (0.55, 0.75):
        t = frac * tau
        terms = build_terms(wfs, t)
        assert terms.n_terms == 1
        lo, hi = terms.edge[0]
        assert lo == pytest.approx(0.5 * aperture - v * t, rel=1e-12)
        assert hi == pytest.approx(v * t - 0.5 * aperture, rel=1e-12)
        assert lo < hi
        peaks[frac] = _peak_intensity(terms, optics, x_axis)

    # Strictly growing, and still well short of the filled aperture at 0.75 tau.
    assert 0.0 < peaks[0.55] < peaks[0.75] < peak_full
    assert peaks[0.55] / peak_full == pytest.approx(0.043730, rel=1e-4)
    assert peaks[0.75] / peak_full == pytest.approx(0.664392, rel=1e-4)
    # Those two numbers are physics, not a snapshot: a static tone leaves the pupil a pure
    # Gaussian with a linear phase, so cropping it to ``|u| <= h`` scales the on-axis
    # amplitude by ``erf(h / w_in)`` and the peak intensity by its square.
    for frac in (0.55, 0.75):
        h = v * frac * tau - 0.5 * aperture
        assert peaks[frac] / peak_full == pytest.approx(float(erf(h / optics.w_in)) ** 2, rel=1e-4)

    # The literal pupil: both drives at their own retarded times, each hard-windowed.
    t = 0.75 * tau
    terms = build_terms(wfs, t)
    got = np.abs(term_field(terms, optics, x_axis, 0.0, 0.0)[0]) ** 2
    reference = _pair_reference(params, channels, t, x_axis)
    want = np.abs(reference) ** 2
    assert np.abs(got / got.max() - want / want.max()).max() < 1e-3

    # The two-sided window is doing real work: ignoring it is a gross error ...
    unwindowed = np.abs(term_field(full, optics, x_axis, 0.0, 0.0)[0]) ** 2
    assert np.abs(unwindowed / unwindowed.max() - want / want.max()).max() > 0.05
    # ... and so is its *second* edge: one-sided moments alone miss the profile too.
    one_sided = replace(terms, edge=((float(terms.edge[0][0]), "lower"), None))
    half = np.abs(term_field(one_sided, optics, x_axis, 0.0, 0.0)[0]) ** 2
    assert np.abs(half / half.max() - want / want.max()).max() > 0.05


def test_pair_aperture_is_full_after_one_transit(params1030) -> None:
    """At ``t >= tau`` both channels are full: no edge info, and the full-aperture field."""
    params = _linear_drive(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    _, wfs = _pair_scene(params)

    x_spot = params.deflection_scale * (-3.0 * MHz - 2.0 * MHz)
    x_axis = x_spot + np.linspace(-4.0, 4.0, 41) * optics.waist0

    at_tau = build_terms(wfs, tau)
    later = build_terms(wfs, 2.0 * tau)
    assert at_tau.edge == (None, None)
    assert later.edge == (None, None)
    assert _peak_intensity(at_tau, optics, x_axis) == pytest.approx(
        _peak_intensity(later, optics, x_axis), rel=1e-12
    )
    # Just before tau the window is still (barely) two-sided.
    nearly = build_terms(wfs, 0.999 * tau)
    assert isinstance(nearly.edge[0], FillWindow)


def test_four_channel_co_chirp_runs_end_to_end(params1030) -> None:
    """Both axes windowed at ``t = 0.8 tau``: simulate + intensity_frame stay finite.

    Equal chirps on all four channels — the paper's astigmatism-free 3D control — so both
    axes carry a two-sided window at once and the whole engine path (terms -> metrics ->
    frame) runs through :func:`~aodl.field.gaussian.gauss_moments_window` twice per term.
    """
    optics = params1030.optics
    aod = params1030.channels["Ax"]
    tau, v, aperture = aod.transit_time, aod.sound_speed, aod.aperture
    fdot = 40.0 * MHz / ms
    wfs = _wfs(
        params1030,
        **{
            name: _channel(_chirp(0.5 * MHz, fdot, 6.0 * tau, phase0=0.2 * i))
            for i, name in enumerate(("Ax", "Bx", "Ay", "By"))
        },
    )

    t = 0.8 * tau
    result = simulate(wfs, [t])
    terms = result.terms(0)
    for axis in range(2):
        lo, hi = terms.edge[axis]
        assert (lo, hi) == pytest.approx((0.5 * aperture - v * t, v * t - 0.5 * aperture))
    assert terms.n_terms == 1

    (spot,) = result.metrics[0]
    for value in (spot.x, spot.y, spot.z_lab, spot.wx, spot.wy, spot.power, spot.df_opt):
        assert math.isfinite(value)
    assert spot.power > 0.0
    # A co-chirped pair cancels its deflection per axis and adds its lensing (Table I).
    assert spot.x == pytest.approx(0.0, abs=1e-15)
    assert spot.y == pytest.approx(0.0, abs=1e-15)
    assert spot.z_lab == pytest.approx(params1030.lens_scale * 2.0 * fdot, rel=1e-12)
    assert spot.delta_f == pytest.approx(0.0, abs=1e-15)

    w0 = optics.waist0
    grid = FrameGrid(-8.0 * w0, 8.0 * w0, 41, -8.0 * w0, 8.0 * w0, 41)
    frame = result.frame(0, grid)
    assert np.all(np.isfinite(frame))
    assert frame.max() > 0.0
    # Same drive, same plane, window removed: the pair passes only the light inside the
    # window, and for a Gaussian pupil that is ``erf(hi / w_in)`` of the amplitude per axis,
    # i.e. ``erf^4`` of the peak intensity across the two.
    opened = intensity_frame(replace(terms, edge=(None, None)), optics, grid, spot.z_lab)
    passed = float(erf((v * t - 0.5 * aperture) / optics.w_in)) ** 4
    assert passed == pytest.approx(0.62, abs=0.01)
    assert frame.max() / opened.max() == pytest.approx(passed, rel=0.02)


# =============================================================== 3. the product guard


def _faded_pair(params):
    """``Ax`` carrying one strong and one 1e-8-faded tone, against a two-tone ``Bx``.

    The faded fundamental is 1e-8 of its channel's strongest line — below ``term_prune`` —
    so the pre-product cut removes it; ``device/mixing.py`` never would, because programmed
    tones are always kept.
    """
    tau = params.channels["Ax"].transit_time
    faint = tones.ConstantEnvelope(1e-8)
    return _wfs(
        params,
        Ax=_channel(
            _static(1.0 * MHz, 6.0 * tau, phase0=0.3),
            _static(-2.0 * MHz, 6.0 * tau, phase0=1.1, env=faint),
        ),
        Bx=_channel(
            _static(0.5 * MHz, 6.0 * tau, phase0=-0.2),
            _static(3.0 * MHz, 6.0 * tau, phase0=0.9, env=tones.ConstantEnvelope(0.5)),
        ),
    )


@pytest.mark.parametrize("mixing_order", [1, 3])
def test_pre_product_cut_is_lossless(params1030, mixing_order) -> None:
    """Cutting weak lines before the product leaves exactly the same surviving terms.

    ``term_prune = 0`` disables both cuts, so the ``build_terms`` result is the full
    Cartesian product; filtering *that* at the threshold must reproduce the pruned build term
    for term, because ``|c| <= |amp| prod_{other} max|amp|`` bounds every term a cut line can
    reach (see ``device/aodl._pre_cut_lines``).
    """
    params = replace(
        params1030,
        channels={n: replace(a, mixing_order=mixing_order) for n, a in params1030.channels.items()},
    )
    wfs = _faded_pair(params)
    t = 2.0 * params.channels["Ax"].transit_time

    full = build_terms(wfs, t, term_prune=0.0)
    cut = build_terms(wfs, t)
    keep = np.abs(full.c) >= TERM_PRUNE * np.abs(full.c).max()
    assert cut.n_terms == int(keep.sum()) < full.n_terms

    np.testing.assert_allclose(cut.c, full.c[keep], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cut.df_opt, full.df_opt[keep], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cut.theta1, full.theta1[:, keep], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(cut.alpha, full.alpha[:, :, keep], rtol=1e-14, atol=0.0)

    # The power the cut removed is accounted for, on top of whatever mixing already dropped.
    dropped = float(np.sum(np.abs(full.c[~keep]) ** 2))
    assert dropped > 0.0
    assert cut.pruned_power - full.pruned_power == pytest.approx(dropped, rel=1e-9)


#: Detunings whose sums and differences never coincide: ``sqrt(prime)`` values are linearly
#: independent over the rationals, so no IM3 product ``f_j + f_k - f_i`` lands on another
#: line and :func:`aodl.device.mixing.expand_lines` merges nothing.
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _non_commensurate(span: float, scale: float = 1.0, shift: float = 0.0):
    roots = np.sqrt(np.array(_PRIMES, dtype=np.float64))
    centred = roots - roots.mean()
    return span * centred / np.abs(centred).max() * scale + shift


def test_max_terms_guard_stops_the_non_commensurate_blowup(params1030) -> None:
    """12 incommensurate tones on two channels at ``mixing_order = 3``: 804^2 terms.

    Nothing merges and everything stays in band, so each channel carries all
    ``12 + C(12,2)*10 + 12*11 = 12 + 660 + 132 = 804`` lines and the Cartesian product asks
    for 646,416 terms — over the ``MAX_TERMS = 200_000`` default.  The pre-product cut cannot
    help here: the weakest line is 1.5e-2 of the strongest, four orders above ``term_prune``,
    so the count is 804 x 804 with the cut and 804 x 804 without it.  The guard has to be the
    thing that stops it.
    """
    tau = params1030.channels["Ax"].transit_time
    t = 2.0 * tau
    wfs = _wfs(
        params1030,
        Ax=_channel(
            *[
                _static(float(f), 6.0 * tau, phase0=0.11 * i)
                for i, f in enumerate(_non_commensurate(4.0 * MHz))
            ]
        ),
        Bx=_channel(
            *[
                _static(float(f), 6.0 * tau, phase0=-0.07 * i)
                for i, f in enumerate(_non_commensurate(4.0 * MHz, scale=0.87, shift=0.31 * MHz))
            ]
        ),
    )

    counts = []
    for name in ("Ax", "Bx"):
        lines = channel_lines(wfs.channels[name], params1030.channels[name], t)
        magnitude = np.abs(lines.amp)
        after_cut = int(np.sum(magnitude >= TERM_PRUNE * magnitude.max()))
        assert lines.n_lines == 804, f"{name} carried {lines.n_lines} lines, expected 804"
        assert after_cut == lines.n_lines, "the pre-product cut is not expected to bite here"
        counts.append(lines.n_lines)
    product = counts[0] * counts[1]
    assert product == 646_416
    assert product > MAX_TERMS

    with pytest.raises(ValueError, match="above max_terms") as excinfo:
        build_terms(wfs, t)
    message = str(excinfo.value)
    assert f"{product}" in message and "Ax: 804" in message and "Bx: 804" in message
    assert "line_prune" in message and "term_prune" in message

    # Raising the ceiling really does build them all (this is the expensive branch).
    terms = build_terms(wfs, t, max_terms=10**7)
    assert terms.n_terms <= product
    assert terms.n_terms > MAX_TERMS


def test_max_terms_counts_lines_after_the_pre_product_cut(params1030) -> None:
    """The guard is applied to the *cut* line counts, and its message names them."""
    params = replace(
        params1030,
        channels={n: replace(a, mixing_order=1) for n, a in params1030.channels.items()},
    )
    wfs = _faded_pair(params)
    t = 2.0 * params.channels["Ax"].transit_time

    # 2 x 2 lines, of which the pre-product cut leaves 1 x 2 = 2 terms.
    assert build_terms(wfs, t, max_terms=2).n_terms == 2
    with pytest.raises(ValueError, match="Ax: 1, Bx: 2"):
        build_terms(wfs, t, max_terms=1)
    # Without the cut all four survive and the same ceiling is exceeded.
    with pytest.raises(ValueError, match="Ax: 2, Bx: 2"):
        build_terms(wfs, t, term_prune=0.0, max_terms=3)
