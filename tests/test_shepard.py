r"""M4 acceptance for the fading-Shepard waveforms of ``waveform/shepard.py`` (Eqs. S24-S28).

Four levels, in order:

1. **the window** — Eqs. S26/S27 as pure algebra: hand-computed values, the boundary
   triple ``(1, cos^p(pi/4), 0)``, exact derivatives against numerical ones, and the
   ``p_A + p_B = 1`` identity that keeps a tweezer's power constant through a hand-over;
2. **the ladder** — that a hold of ``5x`` the Eq. 1 ceiling (:func:`max_z_integral`), which
   plain Eq. S19 refuses outright, becomes a *finite* ladder whose live tones never leave a
   fixed frequency window, however far ``f_Z`` walks;
3. **the simulation** — the same hold through :func:`aodl.engine.simulate`: a static
   tweezer, tracking ``Zbar``, no astigmatism, and total power flat through many fade
   cycles, probed on the plateaus *and* mid-fade;
4. **the shadows** — Eq. S31's ``+- deflection_scale delta_f`` companions, their intensity
   ratio at the fade centre (derived below), their absence on the plateau, and the ``xi``
   interlacing that keeps the other axis out of the way.

Everything is probed at ``t >= tau``: a pair-driven tweezer is strictly dark before
``tau/2`` and the aperture is full only from ``tau`` on (``docs/conventions.md`` §7).
``mixing_order=1`` everywhere except one order-3 spot check — the statements here are about
the Eq. S24-S28 frequency/amplitude algebra, where one tone means one beam.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from aodl import simulate
from aodl.device.aodl import TermArray, build_terms
from aodl.field.focal import FrameGrid, intensity_frame, spot_params
from aodl.field.measure import measure, track_z
from aodl.params import CHANNELS
from aodl.poly import PiecewisePoly
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec
from aodl.units import MHz, ms, um, us
from aodl.waveform import serialize
from aodl.waveform.shepard import (
    A_FLOOR,
    ETA_DEFAULT,
    SLOPE_CLAMP,
    ChannelFade,
    FadeZoneEnvelope,
    ShepardConfig,
    active_indices,
    clamp_floor,
    fade_window,
    ladder_phases,
    poly_crossings,
    poly_range,
    shepard_band_bound,
    table_ii,
)
from aodl.waveform.synthesis import f_z_ramp, max_z_integral, schroeder_phases, synthesize

#: The sustained hold this file is built around: 10 µm out of the focal plane for 1 ms.
#: ``int Z dt`` is ~5.1x the Eq. 1 ceiling, so plain Eq. S19 cannot express it at all.
HOLD_MOVES = (Lift(10.0 * um, 60.0 * us), Hold(1.0 * ms), Lift(-10.0 * um, 60.0 * us))

#: Ladder spacings of the single-tweezer runs.  Deliberately **unequal**, so that no two
#: simultaneously live term combinations share an optical frequency: with ``delta_f_x /
#: delta_f_y = 16 / 13`` a degeneracy would need index offsets of 13 and 16, while only
#: neighbouring rungs are ever live at once (``docs/conventions.md`` §4, the anti-diagonal
#: degeneracy that equal spacings would create).
#:
#: They are also wide, which is what keeps the *fade* slow compared with the beam transit:
#: ``rho = (w_in / v) / T_fade = 0.037`` here, and the acoustic-irising expansion that
#: :class:`FadeZoneEnvelope` feeds is a degree-2 truncation whose residual scales as
#: ``rho^2`` (see the class docstring, and ``test_the_irising_clamp_...`` below).
DFX, DFY = 8.0 * MHz, 6.5 * MHz


def _linear(params):
    """``params`` with every channel at ``mixing_order=1`` — one tone, one beam."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _hold_spec(mx: int = 1, my: int = 1, dfx: float = 0.0, dfy: float = 0.0) -> TrajectorySpec:
    return TrajectorySpec(array=ArraySpec(mx, my, dfx, dfy), moves=HOLD_MOVES)


def _fade_env(wfs, channel: str) -> FadeZoneEnvelope:
    """A middle rung's envelope — the one whose fade zones fall inside the run."""
    tones = wfs.channels[channel].tones
    return tones[len(tones) // 2].env


def _probe_times(wfs, channel: str, level: float, tau: float) -> np.ndarray:
    """Frame times whose *drive* time sits where ``|g| = level`` on ``channel``.

    Collected over the whole ladder, because one rung crosses its own fade zone exactly
    once: it is the *channel* that hands over again every ``delta_f`` of ``f_Z``, and rung
    ``n`` reaching ``+level`` is the same instant as rung ``n-1`` reaching ``-level`` (their
    fade coordinates differ by ``delta_f``), which is why the union still has one entry per
    fade cycle.  A frame at ``t`` is driven by the waveform at ``t - tau/2``, so the times —
    which live in drive time — are shifted forward by half a transit
    (``docs/conventions.md`` §7); only frames with a full aperture are kept.
    """
    found = np.concatenate([tone.env.crossing_times(level) for tone in wfs.channels[channel].tones])
    times = np.sort(found) + 0.5 * tau
    times = times[(times >= tau) & (times <= wfs.t_span[1])]
    if times.size < 2:
        return times
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    return times[keep]


def _colocated(terms: TermArray, optics, tol: float) -> TermArray:
    """The terms that land on the array centre — the tweezer, without its shadows.

    Frequency grouping alone cannot do this: the two Eq. S31 shadows are *exactly*
    degenerate with each other (module docstring of :mod:`aodl.waveform.shepard`), so they
    form one group whose power-weighted centre sits at ``x = 0`` — right on top of the real
    trap.  Selecting on the per-term deflection first, then grouping, is unambiguous.
    """
    xc, yc, _, _ = spot_params(terms, optics, 0.0)
    keep = (np.abs(xc) < tol) & (np.abs(yc) < tol)
    return TermArray(
        c=terms.c[keep],
        theta1=terms.theta1[:, keep],
        theta2=terms.theta2[:, keep],
        alpha=terms.alpha[:, :, keep],
        df_opt=terms.df_opt[keep],
        edge=terms.edge,
    )


# ================================================================== 1. the fade window


def test_window_values_are_the_s26_piecewise_definition():
    """Plateau, shoulder and stop-band at hand-computed ``g`` (``delta_f = 2``, ``eta = 1/2``).

    With ``m = 1`` the boundaries are ``(1 -+ eta) delta_f / 2 = 0.25`` and ``0.75``, and the
    shoulder is ``cos^p[(pi / 2 eta)(|g| / delta_f - 1/2) + pi/4]`` — at ``|g| = 0.5`` (the
    fade centre) the argument is ``pi/4`` exactly.
    """
    kw = dict(delta_f=2.0, eta=0.5, p=0.5, m=1.0)
    assert fade_window(0.0, **kw) == 1.0
    assert fade_window(0.5, **kw) == 1.0  # |g| = 0.25 delta_f, the inner boundary
    assert fade_window(-0.5, **kw) == 1.0  # symmetric in |g|
    assert fade_window(1.5, **kw) == 0.0  # the outer boundary
    assert fade_window(-1.5, **kw) == 0.0
    assert fade_window(4.0, **kw) == 0.0

    assert fade_window(1.0, **kw) == pytest.approx(math.cos(math.pi / 4) ** 0.5)
    for u in (0.6, 0.8, 1.2, 1.4):
        theta = (math.pi / (2 * 0.5)) * (u / 2.0 - 0.5) + math.pi / 4
        assert fade_window(u, **kw) == pytest.approx(math.cos(theta) ** 0.5, rel=1e-14)
        assert fade_window(-u, **kw) == pytest.approx(math.cos(theta) ** 0.5, rel=1e-14)

    # vectorized, and monotone across the shoulder
    g = np.linspace(0.5, 1.5, 65)
    values = np.asarray(fade_window(g, **kw))
    assert values[0] == 1.0 and values[-1] == 0.0
    assert np.all(np.diff(values) <= 1e-15)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0, 2.0])
def test_window_boundary_triple_for_every_table_ii_exponent(p):
    """``(1, cos^p(pi/4), 0)`` at the inner boundary, the fade centre and the outer one.

    ``p = 0`` is the array ``B`` ladder's rectangle: ``cos^0 = 1`` right up to the outer
    boundary, where the tone stops.  That discontinuity is Table II's, not ours.
    """
    delta_f, eta, m = 3.0e6, 0.5, 2.0
    inner, centre, outer = 0.5 * (m - eta) * delta_f, 0.5 * m * delta_f, 0.5 * (m + eta) * delta_f
    kw = dict(delta_f=delta_f, eta=eta, p=p, m=m)
    assert fade_window(inner, **kw) == 1.0
    assert fade_window(centre, **kw) == pytest.approx(math.cos(math.pi / 4) ** p, rel=1e-14)
    assert fade_window(outer, **kw) == 0.0
    just_inside = float(np.asarray(fade_window(0.999 * outer, **kw)))
    assert just_inside == 1.0 if p == 0.0 else just_inside < 0.1


def test_envelope_derivatives_match_numerical_ones_away_from_the_clamp():
    """``dA``/``d2A`` are the exact chain rule through ``u = |g|`` — not finite differences.

    Probed on the shoulder where the clamp is inactive (``cos theta`` above
    :func:`clamp_floor`), on the plateau and in the stop band, with a quadratic ``g`` so
    that ``gddot != 0`` and the second chain-rule term is actually exercised.
    """
    breaks = np.array([0.0, 1.0e-3])
    # g(t) = 1e6 * (-1.4 + 3.0 tau + 0.4 tau^2) Hz, tau = t / 1 ms: sweeps the whole window
    g = PiecewisePoly(breaks, np.array([[-1.4e6, 3.0e6, 0.4e6]]))
    env = FadeZoneEnvelope(g=g, delta_f=1.0e6, eta=0.5, p=0.5, m=1.0)
    floor = clamp_floor(0.5)

    t = np.linspace(2.0e-5, 9.8e-4, 977)
    theta = np.clip(
        (math.pi / (2 * env.eta * env.delta_f)) * (np.abs(g(t)) - env.g_centre) + math.pi / 4,
        0.0,
        0.5 * math.pi,
    )
    # exclude the branch corners (where d2A jumps) and the clamped tail
    ok = (np.cos(theta) > floor * 1.05) & (np.abs(np.abs(g(t)) - env.g_inner) > 2.0e4)
    ok &= np.abs(np.abs(g(t)) - env.g_outer) > 2.0e4
    assert ok.sum() > 200

    h = 2.0e-9
    d1 = (np.asarray(env.A(t + h)) - np.asarray(env.A(t - h))) / (2 * h)
    d2 = (np.asarray(env.A(t + h)) - 2 * np.asarray(env.A(t)) + np.asarray(env.A(t - h))) / h**2
    scale1 = float(np.max(np.abs(np.asarray(env.dA(t))[ok])))
    scale2 = float(np.max(np.abs(np.asarray(env.d2A(t))[ok])))
    np.testing.assert_allclose(np.asarray(env.dA(t))[ok], d1[ok], rtol=2e-5, atol=1e-6 * scale1)
    np.testing.assert_allclose(np.asarray(env.d2A(t))[ok], d2[ok], rtol=2e-3, atol=1e-4 * scale2)

    # off the shoulder both derivatives are exactly zero, and A is exactly 0 or amp
    off = np.abs(np.asarray(g(t))) >= env.g_outer
    flat = np.abs(np.asarray(g(t))) <= env.g_inner
    assert off.any() and flat.any()
    for mask, value in ((off, 0.0), (flat, 1.0)):
        np.testing.assert_array_equal(np.asarray(env.A(t))[mask], value)
        np.testing.assert_array_equal(np.asarray(env.dA(t))[mask], 0.0)
        np.testing.assert_array_equal(np.asarray(env.d2A(t))[mask], 0.0)

    # scalar in -> float out (the Envelope protocol)
    assert isinstance(env.A(3.0e-4), float)
    assert isinstance(env.dA(3.0e-4), float)
    assert isinstance(env.d2A(3.0e-4), float)


def test_the_irising_clamp_is_a_slope_bound_not_an_amplitude_bound():
    """The log-derivatives freeze at :data:`SLOPE_CLAMP` times the fade's own rate.

    An *amplitude* floor alone (the obvious ``max(A, 1e-3)`` rule) does not bound the
    error, because the Eq. S5 aperture polynomial is normalized: a line's weight goes as
    ``A^2 (1 + (p tan(theta) k gdot w_in / v)^2 / 4)``, which for ``p = 1/2`` *diverges* as
    ``A -> 0`` (:class:`FadeZoneEnvelope`).  This pins the fix: ``|A'/A|`` never exceeds its
    value at ``tan theta = SLOPE_CLAMP``, while ``A`` itself stays exact everywhere.
    """
    assert clamp_floor(0.5) == pytest.approx(1.0 / math.sqrt(1.0 + SLOPE_CLAMP**2))
    assert clamp_floor(0.5) > A_FLOOR ** (1.0 / 0.5)  # the slope rule is the binding one
    assert clamp_floor(10.0) == pytest.approx(A_FLOOR**0.1)  # ... but not for every p
    assert clamp_floor(0.0) == 1.0

    breaks = np.array([0.0, 1.0e-3])
    g = PiecewisePoly(breaks, np.array([[-0.8e6, 1.6e6]]))  # linear sweep across the window
    env = FadeZoneEnvelope(g=g, delta_f=1.0e6, eta=0.5, p=0.5, m=1.0)
    t = np.linspace(0.0, 1.0e-3, 4001)
    a = np.asarray(env.A(t))
    live = a > 0.0
    log1 = np.abs(np.asarray(env.dA(t))[live] / a[live])
    rate = (math.pi / (2 * env.eta * env.delta_f)) * abs(float(g.derivative()(0.0)))
    # sin(theta) keeps creeping to 1 past the floored cos, so the ceiling is
    # p * rate / clamp_floor = p * rate * sqrt(1 + SLOPE_CLAMP^2) -- 5% above the slope at
    # the clamp point itself, and bounded, which is the whole point.
    ceiling = env.p * rate / clamp_floor(env.p)
    assert log1.max() <= ceiling * (1.0 + 1e-12)
    assert log1.max() > 0.999 * ceiling  # ... and the ceiling really is reached
    assert ceiling < 1.06 * env.p * SLOPE_CLAMP * rate

    # A is exact right down to zero: the clamp never touches the value
    edge = env.crossing_times(env.g_outer)
    assert edge.size and np.all(np.asarray(env.A(edge)) == 0.0)
    plateau = env.crossing_times(0.0)  # |g| = 0: the middle of the plateau
    assert plateau.size and np.all(np.asarray(env.A(plateau)) == 1.0)


def test_p_a_plus_p_b_equals_one_keeps_the_tweezer_power_constant():
    r"""The constant-power identity, as algebra on the windows themselves.

    For the single-tweezer row of Table II (``p_A = p_B = 1/2``) the *co-located* product
    ``A_A(g) A_B(g)`` is ``cos theta`` and its partner rung's is ``sin theta``, so the two
    combinations that make the same tweezer carry ``cos^2 theta`` and ``sin^2 theta`` of the
    light and always add to exactly one.  Checked across the whole shoulder, and for the
    array row (``p_A = 1``, ``p_B = 0``), which splits the same unit exponent differently.
    """
    delta_f, eta = 4.0e6, ETA_DEFAULT
    s = np.linspace(0.5 * (1 - eta) * delta_f, 0.5 * (1 + eta) * delta_f, 401)  # rung a's |g|
    partner = delta_f - s  # rung a-1 sits mirrored about the fade centre
    theta = (math.pi / (2 * eta)) * (s / delta_f - 0.5) + math.pi / 4
    np.testing.assert_allclose(theta[[0, -1]], [0.0, 0.5 * math.pi], atol=1e-12)

    for p_a, p_b in ((0.5, 0.5), (1.0, 0.0)):
        kw = dict(delta_f=delta_f, eta=eta, m=1.0)
        dying = np.asarray(fade_window(s, p=p_a, **kw)) * np.asarray(fade_window(s, p=p_b, **kw))
        rising = np.asarray(fade_window(partner, p=p_a, **kw)) * np.asarray(
            fade_window(partner, p=p_b, **kw)
        )
        np.testing.assert_allclose(dying, np.cos(theta), rtol=0, atol=1e-12)
        np.testing.assert_allclose(rising, np.sin(theta), rtol=0, atol=1e-12)
        np.testing.assert_allclose(dying**2 + rising**2, 1.0, rtol=0, atol=1e-12)

    # a split that does *not* obey p_A + p_B = 1 does not conserve power
    kw = dict(delta_f=delta_f, eta=eta, m=1.0)
    half = np.asarray(fade_window(s, p=0.5, **kw))
    bad = half**2 + np.asarray(fade_window(partner, p=0.5, **kw)) ** 2
    assert np.max(np.abs(bad - 1.0)) > 0.4


def test_table_ii_rows_and_the_s28_ladder_phases():
    """Paper Table II per axis, and Eq. S28 on signed rung indices."""
    single = table_ii(ArraySpec(1, 1))
    assert single == {
        "Ax": ChannelFade(1, 0.5, 0.0),
        "Bx": ChannelFade(1, 0.5, 0.0),
        "Ay": ChannelFade(1, 0.5, 0.5),
        "By": ChannelFade(1, 0.5, 0.5),
    }
    array = table_ii(ArraySpec(4, 3, 1.0 * MHz, 1.3 * MHz))
    assert array == {
        "Ax": ChannelFade(1, 1.0, 0.0),
        "Bx": ChannelFade(4, 0.0, 0.0),
        "Ay": ChannelFade(1, 1.0, 0.5),
        "By": ChannelFade(3, 0.0, 0.5),
    }
    for name in CHANNELS:
        for row in (single, array):
            assert row[name].xi == (0.0 if name.endswith("x") else 0.5)
    # p_A + p_B = 1 on every axis of every row
    for row in (single, array):
        assert row["Ax"].p + row["Bx"].p == 1.0
        assert row["Ay"].p + row["By"].p == 1.0

    # Eq. S28 is the one implementation: the counted ladder is the indexed one at n = 0..M-1
    np.testing.assert_allclose(ladder_phases(np.arange(6), 6), schroeder_phases(6), rtol=0)
    np.testing.assert_allclose(ladder_phases([0, 1, 2], 1), 0.0, atol=0.0)  # M = 1 -> all zero
    n = np.array([-3, -2, -1, 0, 1])
    want = np.mod(2 * np.pi * n * (n - 1) / 8.0, 2 * np.pi)
    np.testing.assert_allclose(ladder_phases(n, 4), want)
    assert np.all((ladder_phases(n, 4) >= 0.0) & (ladder_phases(n, 4) < 2 * np.pi))


def test_envelope_and_config_validate_their_inputs():
    g = PiecewisePoly.constant(0.0, 0.0, 1.0e-3)
    with pytest.raises(TypeError, match="must be a PiecewisePoly"):
        FadeZoneEnvelope(g=0.0, delta_f=1e6)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="delta_f must be positive"):
        FadeZoneEnvelope(g=g, delta_f=0.0)
    with pytest.raises(ValueError, match="0 < eta <= m"):
        FadeZoneEnvelope(g=g, delta_f=1e6, eta=1.5, m=1.0)
    with pytest.raises(ValueError, match="p must be non-negative"):
        FadeZoneEnvelope(g=g, delta_f=1e6, p=-1.0)
    with pytest.raises(ValueError, match=r"amp must lie in \[0, 1\]"):
        FadeZoneEnvelope(g=g, delta_f=1e6, amp=2.0)

    with pytest.raises(ValueError, match="delta_f_x must be finite and positive"):
        ShepardConfig(-1.0 * MHz, 1.0 * MHz)
    with pytest.raises(ValueError, match="must be 'auto' or a"):
        ShepardConfig(1.0 * MHz, 1.0 * MHz, config="table3")
    with pytest.raises(ValueError, match="unknown channel name"):
        ShepardConfig(1.0 * MHz, 1.0 * MHz, config={"Cz": ChannelFade(1, 0.5, 0.0)})
    with pytest.raises(ValueError, match="ChannelFade.m must be an integer"):
        ChannelFade(0, 0.5, 0.0)

    # an array axis must not disagree with its own ladder spacing
    spec = ArraySpec(3, 1, 1.0 * MHz)
    with pytest.raises(ValueError, match="conflicts with the array's own delta_f_x"):
        ShepardConfig(2.0 * MHz, 1.0 * MHz).resolve(spec)
    assert ShepardConfig(1.0 * MHz, 1.0 * MHz).resolve(spec)["Bx"].m == 3
    # explicit overrides land on top of Table II (this is the "simultaneous fading" knob)
    override = {"Ay": ChannelFade(1, 0.5, 0.0), "By": ChannelFade(1, 0.5, 0.0)}
    resolved = ShepardConfig(1.0 * MHz, 1.0 * MHz, config=override).resolve(ArraySpec(1, 1))
    assert resolved["Ay"].xi == 0.0 and resolved["Ax"].xi == 0.0


# ==================================================================== 2. ladder bookkeeping


def test_a_hold_five_times_over_the_eq_1_ceiling(params1030):
    """The spec this file is built around really is beyond plain Eq. S19 — by 5x."""
    spec = _hold_spec()
    _, _, z = spec.compile()
    requested = 2.0 * params1030.lens_scale * abs(float(f_z_ramp(z, params1030)(spec.duration)))
    assert requested / max_z_integral(params1030) == pytest.approx(5.15, rel=0.02)

    with pytest.raises(ValueError, match="leaves its usable band"):
        synthesize(spec, params1030)
    with pytest.raises(ValueError, match="leaves its usable band"):
        synthesize(spec, params1030, shepard=None)


def test_the_ladder_is_finite_and_its_size_is_the_f_z_span(params1030):
    r"""``count = (f_Z span) / delta_f + (m + eta)``, to within one rung — and nothing else.

    Rung ``n`` exists iff ``|g_n|`` enters the outer boundary at some instant, so the ladder
    grows one tone per ``delta_f`` of axial integral plus a constant for the window itself.
    """
    spec = _hold_spec()
    _, _, z = spec.compile()
    f_z = f_z_ramp(z, params1030)
    lo, hi = poly_range(f_z)
    span = hi - lo
    assert span / MHz == pytest.approx(51.46, rel=1e-3)  # 10 µm x 1.06 ms of axial integral

    cfg = ShepardConfig(DFX, DFY)
    wfs = synthesize(spec, params1030, shepard=cfg)
    fades = cfg.resolve(spec.array)
    for name in CHANNELS:
        delta_f = cfg.spacing(name)
        predicted = span / delta_f + fades[name].m + cfg.eta
        assert abs(wfs.channels[name].n_tones - predicted) <= 1.0
        # every rung really is audible somewhere, and the silent ones were never built
        for tone in wfs.channels[name].tones:
            assert tone.env.is_active
    assert [wfs.channels[name].n_tones for name in CHANNELS] == [8, 8, 10, 10]

    # a longer hold costs proportionally more rungs, not more bandwidth
    longer = TrajectorySpec(
        array=ArraySpec(1, 1),
        moves=(Lift(10.0 * um, 60.0 * us), Hold(4.0 * ms), Lift(-10.0 * um, 60.0 * us)),
    )
    more = synthesize(longer, params1030, shepard=cfg)
    _, _, z_long = longer.compile()
    long_lo, long_hi = poly_range(f_z_ramp(z_long, params1030))
    assert abs(more.channels["Ax"].n_tones - ((long_hi - long_lo) / DFX + 1.5)) <= 1.0
    assert more.channels["Ax"].n_tones > 3 * wfs.channels["Ax"].n_tones


def test_the_shepard_claim_max_live_frequency_stays_bounded(params1030):
    r"""**The Shepard claim.**  ``max |f| <= (M + eta) delta_f / 2 + max|f_lat| + delta_f``
    over every live tone of the whole run, however far ``f_Z`` walks.

    Measured tone by tone from the waveform set itself, on a dense time grid, counting a
    tone only where its envelope is non-zero — which is the only place the transducer
    launches it.  The tighter bound without the ``delta_f`` slack holds too.
    """
    spec = _hold_spec()
    cfg = ShepardConfig(DFX, DFY)
    wfs = synthesize(spec, params1030, shepard=cfg)
    fades = cfg.resolve(spec.array)
    t = np.linspace(*wfs.t_span, 20001)

    for name in CHANNELS:
        aod = params1030.channels[name]
        delta_f = cfg.spacing(name)
        bound = shepard_band_bound(fades[name], delta_f, cfg.eta, 0.0)
        assert bound == pytest.approx(0.5 * (fades[name].m + cfg.eta) * delta_f)
        worst = 0.0
        unbounded = 0.0
        for tone in wfs.channels[name].tones:
            f = np.asarray(tone.freq(t))
            live = np.asarray(tone.env.A(t)) > 0.0
            unbounded = max(unbounded, float(np.max(np.abs(f))))
            if live.any():
                worst = max(worst, float(np.max(np.abs(f[live]))))
        assert worst <= bound + 1.0  # the claim, tight
        assert worst <= bound + delta_f  # ... and as WO-14 states it
        assert worst == pytest.approx(bound, rel=1e-3)  # the window really is filled
        # the frequency *laws* are unbounded; only the live parts are not
        assert unbounded > 25.0 * MHz
        assert aod.f_center + worst < aod.band[1] and aod.f_center - worst > aod.band[0]

    # ... and the exported drive is in band because A = 0 outside the window
    for name in CHANNELS:
        aod = params1030.channels[name]
        for tone in wfs.channels[name].tones:
            f = np.asarray(tone.freq(t))
            live = np.asarray(tone.env.A(t)) > 0.0
            assert np.all(aod.f_center + f[live] < aod.band[1])
            assert np.all(aod.f_center + f[live] > aod.band[0])


def test_active_indices_matches_a_direct_scan():
    """The index window of :func:`active_indices` agrees with brute force on ``|g| < g_out``."""
    f_z = PiecewisePoly(np.array([0.0, 1.0e-3]), np.array([[0.0, 30.0e6]]))
    fade = ChannelFade(m=2, p=0.0, xi=0.5)
    delta_f = 4.0e6
    got = active_indices(f_z, fade, delta_f)
    t = np.linspace(0.0, 1.0e-3, 20001)
    g_out = 0.5 * (fade.m + ETA_DEFAULT) * delta_f
    brute = [
        n
        for n in range(-40, 40)
        if np.any(np.abs(np.asarray(f_z(t)) + (n + fade.xi) * delta_f) < g_out)
    ]
    np.testing.assert_array_equal(got, brute)


def test_the_shepard_band_check_refuses_an_impossible_ladder(params1030):
    """Under Shepard the band check is the *bounded* excursion, not Eq. 1."""
    spec = _hold_spec()
    with pytest.raises(ValueError) as excinfo:
        synthesize(spec, params1030, shepard=ShepardConfig(20.0 * MHz, 6.5 * MHz))
    message = str(excinfo.value)
    assert "leaves its usable band even with fading-Shepard tones" in message
    assert "does not grow with the hold" in message
    assert "check_band=False" in message

    # the same ladder still synthesizes for plotting
    wide = ShepardConfig(20.0 * MHz, 6.5 * MHz)
    wfs = synthesize(spec, params1030, shepard=wide, check_band=False)
    assert wfs.channels["Ax"].n_tones >= 2


def test_shepard_auto_switches_only_when_eq_1_fails(params1030):
    """``shepard="auto"``: plain Eq. S19 while it fits, fading-Shepard when it does not."""
    short = TrajectorySpec(
        array=ArraySpec(1, 1),
        moves=(Lift(10.0 * um, 60.0 * us), Hold(50.0 * us), Lift(-10.0 * um, 60.0 * us)),
    )
    plain = synthesize(short, params1030, shepard="auto")
    assert "Eq. S19 synthesis" in plain.description
    assert "plain Eq. S19 fits the band" in plain.description
    assert plain.n_tones == 4  # one tone per channel

    long_hold = synthesize(_hold_spec(), params1030, shepard="auto")
    assert "Fading-Shepard synthesis (Eqs. S24-S28)" in long_hold.description
    assert "plain Eq. S19 refused" in long_hold.description
    assert "leaves its usable band" in long_hold.description  # the reason is quoted
    assert long_hold.n_tones > 4

    with pytest.raises(ValueError, match="must be None, 'auto' or a ShepardConfig"):
        synthesize(short, params1030, shepard="always")
    with pytest.raises(TypeError, match="must be None, 'auto' or a ShepardConfig"):
        synthesize(short, params1030, shepard=1.0)  # type: ignore[arg-type]


# ======================================================================= 3. the simulation


def test_the_sustained_hold_tracks_and_keeps_its_power(params1030):
    """The M4 acceptance run: 10 µm held for 1 ms — five Eq. 1 ceilings — as simulated.

    Probed on the plateaus **and** at the fade centres of the x pair.  The tweezer must not
    move laterally, must sit at the requested ``Zbar``, must stay astigmatism-free, and its
    total power — summed over the co-located frequency groups, i.e. over the rungs handing
    over — must be constant through every fade.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = _hold_spec()
    _, _, z = spec.compile()
    wfs = synthesize(spec, params, shepard=ShepardConfig(DFX, DFY))

    centres = _probe_times(wfs, "Ax", _fade_env(wfs, "Ax").g_centre, tau)
    plateaus = _probe_times(wfs, "Ax", 0.0, tau)
    assert centres.size >= 4  # ... four complete fade cycles, as required
    times = np.sort(np.concatenate([centres, plateaus, np.linspace(tau, spec.duration, 40)]))

    tol = 0.25 * params.deflection_scale * DFY
    powers, worst = [], dict(x=0.0, y=0.0, z=0.0, astig=0.0)
    for t in times:
        terms = build_terms(wfs, float(t))
        groups = measure(_colocated(terms, optics, tol), optics)
        assert groups
        total = sum(m.power for m in groups)
        powers.append(total)
        worst["x"] = max(worst["x"], max(abs(m.x) for m in groups))
        worst["y"] = max(worst["y"], max(abs(m.y) for m in groups))
        worst["astig"] = max(worst["astig"], max(abs(m.delta_f) for m in groups))
        z_lab = sum(m.power * m.z_lab for m in groups) / total
        worst["z"] = max(worst["z"], abs(z_lab - z(t - 0.5 * tau)))

    assert worst["x"] < 0.01 * optics.waist0
    assert worst["y"] < 0.01 * optics.waist0
    assert worst["z"] < 0.02 * optics.rayleigh
    assert worst["astig"] < 0.02 * optics.rayleigh

    power = np.array(powers)
    assert (power.max() - power.min()) / power.mean() < 0.01
    assert float(np.max(z(times - 0.5 * tau))) > optics.rayleigh  # the hold is worth measuring


def test_the_hold_is_at_the_requested_height_all_the_way_through(params1030):
    """``Zbar = Z`` exactly, because every rung of every channel chirps at ``fdot_Z``.

    Eq. S24 adds the *same* ``f_Z`` to all four channels and the ladder offsets are
    constants, so Table I's ``Zbar`` is the Eq. S19 answer with no residual at all — the
    fading is an amplitude story, not a frequency one.
    """
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    spec = _hold_spec()
    _, _, z = spec.compile()
    wfs = synthesize(spec, params, shepard=ShepardConfig(DFX, DFY))
    times = np.linspace(tau, spec.duration, 24)

    result = simulate(wfs, times)
    tracked = result.tracked_z()
    np.testing.assert_allclose(tracked, z(times - 0.5 * tau), atol=1e-9 * params.optics.rayleigh)
    for frame in result.metrics:
        assert max(abs(m.delta_f) for m in frame) < 1e-9 * params.optics.rayleigh


def test_one_frame_with_intermodulation_still_finds_the_trap(params1030):
    """Order-3 spot check: IM3 adds ghosts but does not move or dim the tweezer."""
    optics = params1030.optics
    tau = params1030.channels["Ax"].transit_time
    spec = _hold_spec()
    _, _, z = spec.compile()
    wfs = synthesize(spec, params1030, shepard=ShepardConfig(DFX, DFY))
    t = float(_probe_times(wfs, "Ax", _fade_env(wfs, "Ax").g_centre, tau)[0])

    terms = build_terms(wfs, float(t))
    metrics = measure(terms, optics)
    # term-level selection, not group-level: the two Eq. S31 shadows are degenerate with
    # each other, so their *group* reports a power-weighted x of zero (see `_colocated`).
    trap = measure(_colocated(terms, optics, 0.25 * params1030.deflection_scale * DFY), optics)
    assert len(metrics) > len(trap)  # the ghosts are there
    for spot in trap:
        assert abs(spot.x) < 0.01 * optics.waist0
        assert abs(spot.z_lab - z(t - 0.5 * tau)) < 0.05 * optics.rayleigh
        assert abs(spot.delta_f) < 0.05 * optics.rayleigh
    assert sum(m.power for m in trap) / sum(m.power for m in metrics) > 0.3


# ======================================================================== 4. shadow tweezers


def test_shadow_tweezers_at_the_fade_centre(params1030):
    r"""Eq. S31: two companions at ``+- deflection_scale delta_f_x``, each **half** the trap.

    *Derivation.*  During an x hand-over the live rungs are ``a`` (dying, ``cos^{p}theta``)
    and ``a-1`` (rising, ``sin^{p}theta``) on **both** x channels.  A trap's position depends
    only on ``n_Bx - n_Ax``, so the two co-located combinations ``(a, a)`` and ``(a-1, a-1)``
    make the real tweezer with total intensity ``cos^{2(p_A+p_B)} + sin^{2(p_A+p_B)} = 1``,
    while the two *cross* combinations sit one rung off — at ``+- deflection_scale delta_f``
    — with intensities ``cos^{2p_A}theta sin^{2p_B}theta`` and ``sin^{2p_A}theta
    cos^{2p_B}theta``.  At the fade centre ``theta = pi/4`` and every one of the four is
    ``2^{-(p_A+p_B)} = 1/2``, so **each shadow peaks at half the main trap**, for any Table II
    split of the unit exponent.

    Note that this is *not* ``(cos theta sin theta)^2 = 1/4``: that would be the product of
    the two co-located *pair* products, whereas a shadow takes one factor from each pair and
    is their geometric mean ``(cos theta sin theta)^{1/2}`` in amplitude.  Measured on the
    rendered frame at the three exact spot centres, the ratio is 0.50, not 0.25.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    wfs = synthesize(_hold_spec(), params, shepard=ShepardConfig(DFX, DFY))
    offset = params.deflection_scale * DFX

    centres = _probe_times(wfs, "Ax", _fade_env(wfs, "Ax").g_centre, tau)
    plateaus = _probe_times(wfs, "Ax", 0.0, tau)
    assert centres.size >= 4 and plateaus.size >= 4

    # a 3-point grid whose columns are exactly (-offset, 0, +offset): no sampling loss
    grid = FrameGrid(-offset, offset, 3, -optics.waist0, optics.waist0, 3)
    for t in centres[:3]:
        terms = build_terms(wfs, float(t))
        row = intensity_frame(terms, optics, grid, track_z(measure(terms, optics)))[1]
        assert row[2] / row[1] == pytest.approx(0.5, rel=0.02)
        assert row[0] / row[1] == pytest.approx(0.5, rel=0.02)
        assert row[2] / row[1] < 0.4 or row[2] / row[1] > 0.3  # ... and nowhere near 1/4
        assert abs(row[2] / row[1] - 0.25) > 0.2

        # the companions are where Eq. S31 puts them, term by term
        xc, _, _, _ = spot_params(terms, optics, 0.0)
        weight = np.abs(terms.c) ** 2
        loud = xc[weight > 0.01 * weight.max()]
        for want in (-offset, 0.0, offset):
            assert np.min(np.abs(loud - want)) < 1e-9 * optics.waist0
        assert np.max(np.abs(loud)) < offset * (1 + 1e-9)

    for t in plateaus[:3]:
        terms = build_terms(wfs, float(t))
        row = intensity_frame(terms, optics, grid, track_z(measure(terms, optics)))[1]
        assert row[0] == 0.0 and row[2] == 0.0  # no companions outside the fade zone
        assert row[1] > 0.0


def test_the_xi_offset_interlaces_the_two_axes(params1030):
    """During an x hand-over the y pair is on its plateau — that is what ``xi = 1/2`` buys.

    With ``eta = 1/2`` and one common spacing the x fade zones tile the y plateaus exactly,
    so only ever one axis is handing over: the shadows stay a single-axis ``+-delta_f``
    pair instead of the sixteen-ray grid of Fig. S6.
    """
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    wfs = synthesize(_hold_spec(), params, shepard=ShepardConfig(8.0 * MHz, 8.0 * MHz))
    for t in _probe_times(wfs, "Ax", _fade_env(wfs, "Ax").g_centre, tau)[:4]:
        drive = t - 0.5 * tau
        x_live = [tone for tone in wfs.channels["Ax"].tones if tone.env.A(drive) > 0.0]
        y_live = [tone for tone in wfs.channels["Ay"].tones if tone.env.A(drive) > 0.0]
        assert len(x_live) == 2  # mid hand-over ...
        assert len(y_live) == 1  # ... while y sits on its plateau
        assert float(y_live[0].env.A(drive)) == 1.0
        assert build_terms(wfs, float(t)).n_terms == 4  # 2 x 2 x 1 x 1, not 16

    # and on the x plateau the roles are exactly swapped
    for t in _probe_times(wfs, "Ax", 0.0, tau)[:4]:
        drive = t - 0.5 * tau
        assert len([s for s in wfs.channels["Ax"].tones if s.env.A(drive) > 0.0]) == 1
        assert len([s for s in wfs.channels["Ay"].tones if s.env.A(drive) > 0.0]) == 2


def test_an_array_keeps_its_columns_and_grows_two_more_while_fading(params1030):
    """Array row of Table II: the in-array columns never flicker, and the grid is Mx+2.

    ``p_B = 0`` makes the ``B`` ladder a rectangle — it *is* the array ladder — and ``p_A =
    1`` puts the whole hand-over on the single ``A`` tone.  Every interior column is fed by
    both live ``A`` rungs, so its intensity is ``cos^2 + sin^2 = 1`` throughout; the two
    outermost combinations have only one feeder each and make the extended grid.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    delta_f = 1.0 * MHz
    spec = TrajectorySpec(
        array=ArraySpec(3, 1, delta_f),
        moves=(Lift(6.0 * um, 60.0 * us), Hold(150.0 * us), Lift(-6.0 * um, 60.0 * us)),
    )
    wfs = synthesize(spec, params, shepard=ShepardConfig(delta_f, delta_f))
    pitch = params.deflection_scale * delta_f

    def columns(t: float) -> dict[int, float]:
        terms = build_terms(wfs, float(t))
        xc, _, _, _ = spot_params(terms, optics, 0.0)
        out: dict[int, float] = {}
        for x, c in zip(xc, terms.c):
            key = int(round(x / pitch))
            assert abs(x / pitch - key) < 1e-9
            out[key] = out.get(key, 0.0) + float(abs(c) ** 2)
        return out

    interior = []
    for t in np.linspace(tau, spec.duration, 60):
        cols = columns(t)
        assert set(cols) <= {-2, -1, 0, 1, 2}
        assert {-1, 0, 1} <= set(cols)
        reference = max(cols.values())
        interior.append([cols[j] / reference for j in (-1, 0, 1)])
        extra = [cols.get(j, 0.0) / reference for j in (-2, 2)]
        assert sum(extra) == pytest.approx(1.0 if len(cols) == 5 else 0.0, abs=1e-9)
    np.testing.assert_allclose(np.array(interior), 1.0, rtol=1e-12)

    # the extended grid is exactly (Mx + 2) x My, and only during a fade
    widths = {len(columns(t)) for t in np.linspace(tau, spec.duration, 60)}
    assert widths == {3, 5}


# ========================================================================= 5. serialization


def test_schema_v2_round_trips_a_fading_shepard_set(tmp_path, params1030):
    """Bit-exact: the fade envelopes' ``g`` polynomials ride in ``<ch>_env_polys``."""
    wfs = synthesize(
        TrajectorySpec(
            array=ArraySpec(1, 1),
            moves=(Lift(10.0 * um, 60.0 * us), Hold(200.0 * us), Lift(-10.0 * um, 60.0 * us)),
        ),
        params1030,
        shepard=ShepardConfig(DFX, DFY),
    )
    path = serialize.save(wfs, tmp_path / "shepard.npz")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))
        assert meta["schema_version"] == serialize.SCHEMA_VERSION_FADE == 2
        for name in CHANNELS:
            assert data[f"{name}_env_polys"].shape[1] == serialize.SEGMENT_COLUMNS
            assert np.all(data[f"{name}_tones"][:, 2] == serialize.ENV_FADE_ZONE)
            # env_params = (delta_f, eta, p, M); xi lives inside the g-poly's constant term
            row = data[f"{name}_tones"][0, 3:7]
            np.testing.assert_allclose(row, [DFX if name.endswith("x") else DFY, 0.5, 0.5, 1.0])
    assert path.stat().st_size < 100_000  # parameters, not samples

    back = serialize.load(path)
    assert back.description == wfs.description
    assert back.params == wfs.params
    t = np.linspace(*wfs.t_span, 733)
    for name in CHANNELS:
        original, loaded = wfs.channels[name], back.channels[name]
        assert original.n_tones == loaded.n_tones
        for tone_a, tone_b in zip(original.tones, loaded.tones):
            np.testing.assert_array_equal(tone_a.freq.breaks, tone_b.freq.breaks)
            np.testing.assert_array_equal(tone_a.freq.coeffs, tone_b.freq.coeffs)
            assert tone_a.phase0 == tone_b.phase0
            env_a, env_b = tone_a.env, tone_b.env
            assert (env_a.delta_f, env_a.eta, env_a.p, env_a.m) == (
                env_b.delta_f,
                env_b.eta,
                env_b.p,
                env_b.m,
            )
            np.testing.assert_array_equal(np.asarray(env_a.g(t)), np.asarray(env_b.g(t)))
            for method in ("A", "dA", "d2A"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(env_a, method)(t)), np.asarray(getattr(env_b, method)(t))
                )

    # ... and the reloaded drive simulates identically
    tau = params1030.channels["Ax"].transit_time
    before, after = simulate(wfs, [3 * tau]).metrics[0], simulate(back, [3 * tau]).metrics[0]
    assert before == after


def test_a_v1_file_still_loads_and_v1_sets_still_write_v1(tmp_path, params1030):
    """Schema v2 is additive: nothing about a fade-free waveform changes on disk."""
    plain = synthesize(
        TrajectorySpec(array=ArraySpec(2, 1, 1.0 * MHz), moves=(Hold(50.0 * us),)), params1030
    )
    path = serialize.save(plain, tmp_path / "v1.npz")
    with np.load(path, allow_pickle=False) as data:
        version = json.loads(str(data["meta"].item()))["schema_version"]
        assert version == serialize.SCHEMA_VERSION == 1
        assert not [key for key in data.files if key.endswith("_env_polys")]
    back = serialize.load(path)
    assert back.n_tones == plain.n_tones
    np.testing.assert_array_equal(
        back.channels["Bx"].tones[1].freq.coeffs, plain.channels["Bx"].tones[1].freq.coeffs
    )

    # a fade envelope that cannot be expressed is refused, loudly, rather than mangled
    scaled = synthesize(
        TrajectorySpec(
            array=ArraySpec(1, 1),
            moves=(Lift(10.0 * um, 60.0 * us), Hold(200.0 * us), Lift(-10.0 * um, 60.0 * us)),
        ),
        params1030,
        amp=0.5,
        shepard=ShepardConfig(DFX, DFY),
    )
    with pytest.raises(ValueError, match="four env parameter slots"):
        serialize.save(scaled, tmp_path / "scaled.npz")


def test_poly_helpers_are_exact():
    """:func:`poly_crossings` / :func:`poly_range` — segment-exact, not sampled."""
    p = PiecewisePoly(np.array([0.0, 1.0, 2.0]), np.array([[0.0, 1.0, 0.0], [1.0, -3.0, 2.0]]))
    root = 1.0 + 0.25 * (3.0 - math.sqrt(5.0))  # 2 tau^2 - 3 tau + 0.5 = 0 on [1, 2]
    np.testing.assert_allclose(poly_crossings(p, 0.5), [0.5, root], atol=1e-12)
    np.testing.assert_allclose(poly_crossings(p, 0.0), [0.0, 1.5, 2.0], atol=1e-12)
    assert poly_crossings(p, 7.0).size == 0
    lo, hi = poly_range(p)
    dense = np.asarray(p(np.linspace(0.0, 2.0, 100001)))
    assert lo == pytest.approx(float(dense.min()), abs=1e-9)
    assert hi == pytest.approx(float(dense.max()), abs=1e-9)
    assert poly_range(PiecewisePoly.constant(3.0, 0.0, 1.0)) == (3.0, 3.0)
