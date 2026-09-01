r"""M4 acceptance, end to end: fading-Shepard waveforms through the product path
(``docs/PLAN.md`` §3, M4).

``tests/test_shepard.py`` pins the Eqs. S24-S28 *algebra* — window values, ladder
bookkeeping, the Shepard bound.  This file asks the milestone's four questions of the whole
pipeline (``TrajectorySpec`` -> :func:`aodl.waveform.synthesis.synthesize` ->
:func:`aodl.engine.simulate` -> metrics and rendered frames):

1. **a sustained axial offset past Eq. 1.**  10 µm held for 1 ms — five times the one-sided
   band budget, which plain Eq. S19 refuses outright — tracks to the M3 bounds with total
   tweezer power constant to better than 1 %.
2. **shadow tweezers.**  Eq. S31's ``+- deflection_scale delta_f`` companions exist *only*
   while a pair hands over, each carrying half the trap; for an array the grid grows to
   ``(Mx + 2) x My``, and the new column switches on at full brightness rather than fading
   in (``p_B = 0``, Table II).
3. **interlaced vs simultaneous fading.**  Fading both axes at once makes two of the Fig. S6
   rays land on the tweezer *and* share its optical frequency — a static Mach-Zehnder whose
   output depends on an optical path the experiment does not control.  The ``xi = 1/2``
   offset of Table II removes the degeneracy, and :attr:`aodl.field.measure.SpotMetrics.
   power_coherent` is what can see the difference.
4. **the user story, unhurried.**  The 10x10 lift-traverse-lower of ``examples/04`` at
   comfortable durations — band-infeasible under Eq. S19 — synthesized by ``shepard="auto"``.

Positions are compared at the retarded time ``t_c = t - tau/2`` and every probe sits at
``t >= tau``, as in M3.  ``mixing_order=1`` throughout: these are statements about the
Eq. S24-S28 frequency/amplitude algebra reaching the image plane.

**Selection is per term, never per group.**  A fading ladder makes frequency groups that hold
spots at *different* places — the two Eq. S31 shadows are exactly degenerate with each other,
and in an array the rung pairs ``(a, b)`` and ``(a-1, b+1)`` share an optical frequency while
sitting two columns apart — so a group's power-weighted centroid can land where no light is.
Every spatial assertion below therefore starts from :func:`aodl.field.focal.spot_params` on
the terms themselves.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from aodl import simulate
from aodl.device.aodl import TermArray, build_terms
from aodl.field.focal import Z_LAB_SIGN, FrameGrid, intensity_frame, spot_params
from aodl.field.measure import measure, track_z
from aodl.params import CHANNELS
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, ms, um, us
from aodl.waveform.shepard import ChannelFade, ShepardConfig
from aodl.waveform.synthesis import f_z_ramp, max_z_integral, synthesize

#: The sustained hold: 10 µm out of the focal plane for 1 ms, i.e. ~5x the Eq. 1 ceiling.
HOLD_MOVES = (Lift(10.0 * um, 60.0 * us), Hold(1.0 * ms), Lift(-10.0 * um, 60.0 * us))

#: Single-tweezer ladder spacings.  Unequal, so that no two live combinations share an
#: optical frequency (``docs/conventions.md`` §4), and wide, which keeps the fade slow
#: against the beam transit: ``rho = (w_in / v) / T_fade = 0.037`` (WO-14).
DFX, DFY = 8.0 * MHz, 6.5 * MHz

#: The unhurried user story of ``examples/04`` §7 — the durations Eq. S19 refused.
STORY_MOVES = (
    Lift(10.0 * um, 150.0 * us),
    Translate(40.0 * um, 25.0 * um, 250.0 * us),
    Lift(-10.0 * um, 150.0 * us),
)
STORY_ARRAY = ArraySpec(10, 10, 1.0 * MHz, 1.3 * MHz)


def _linear(params):
    """``params`` with every channel at ``mixing_order=1`` — one tone, one beam."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _hold_spec(mx: int = 1, my: int = 1, dfx: float = 0.0, dfy: float = 0.0) -> TrajectorySpec:
    return TrajectorySpec(array=ArraySpec(mx, my, dfx, dfy), moves=HOLD_MOVES)


def _select(terms: TermArray, keep) -> TermArray:
    """The sub-array of ``terms`` selected by a boolean mask (fill edges are frame-wide)."""
    return TermArray(
        c=terms.c[keep],
        theta1=terms.theta1[:, keep],
        theta2=terms.theta2[:, keep],
        alpha=terms.alpha[:, :, keep],
        df_opt=terms.df_opt[keep],
        edge=terms.edge,
    )


def _at_column(terms: TermArray, optics, x_target: float, tol: float) -> TermArray:
    """Terms whose Table I deflection puts them within ``tol`` of ``(x_target, 0)``."""
    xc, yc, _, _ = spot_params(terms, optics, 0.0)
    return _select(terms, (np.abs(xc - x_target) < tol) & (np.abs(yc) < tol))


def _fade_env(wfs, channel: str):
    """A middle rung's envelope — the one whose fade zones fall inside the run."""
    tones = wfs.channels[channel].tones
    return tones[len(tones) // 2].env


def _probe_times(wfs, channel: str, level: float, tau: float) -> np.ndarray:
    """Frame times whose *drive* time sits where ``|g| = level`` on ``channel``.

    ``level = g_centre`` gives the fade centres and ``0`` the plateau centres; the union over
    the ladder has one entry per hand-over, because rung ``n`` reaching ``+level`` is the same
    instant as rung ``n-1`` reaching ``-level``.  Shifted by ``tau/2`` into frame time and
    trimmed to the fully filled aperture.
    """
    found = np.concatenate([tone.env.crossing_times(level) for tone in wfs.channels[channel].tones])
    times = np.sort(found) + 0.5 * tau
    times = times[(times >= tau) & (times <= wfs.t_span[1])]
    if times.size < 2:
        return times
    return times[np.concatenate([[True], np.diff(times) > 1e-9])]


def _lattice(terms: TermArray, optics, centre: tuple[float, float], pitch: tuple[float, float]):
    """Per-term spots as ``{(column, row): power weight}`` plus the worst lattice residual.

    The array is a lattice about ``centre`` with the Table I pitches, so every term must sit
    on an integer node of it; the residual is what says the fading ladder has not disturbed
    the geometry.  Weights are ``|c|^2`` — the terms of one node are not degenerate with each
    other, so their intensities add.
    """
    xc, yc, _, _ = spot_params(terms, optics, 0.0)
    ix = np.round((xc - centre[0]) / pitch[0])
    iy = np.round((yc - centre[1]) / pitch[1])
    residual = max(
        float(np.max(np.abs(xc - centre[0] - ix * pitch[0]))),
        float(np.max(np.abs(yc - centre[1] - iy * pitch[1]))),
    )
    nodes: dict[tuple[int, int], float] = {}
    for i, j, c in zip(ix, iy, terms.c, strict=True):
        key = (int(i), int(j))
        nodes[key] = nodes.get(key, 0.0) + float(abs(c) ** 2)
    return nodes, residual


# ================================================ 1. a sustained offset, past the Eq. 1 budget


def test_a_ten_micron_hold_of_one_millisecond_tracks_and_keeps_its_power(params1030):
    """Five Eq. 1 ceilings of axial offset, held: M3 tracking bounds, power flat to < 1 %.

    Plain Eq. S19 cannot express this drive at all — one tone would have to carry
    ``int Z dt`` on its own and walks 51 MHz out of a 20 MHz band.  Under Eqs. S24-S28 the
    same trajectory is a sliding ladder inside a fixed window, and what has to be shown is
    that the hand-overs are invisible: the tweezer must not move, must sit at the requested
    ``Zbar``, must stay astigmatism-free, and its total power — summed over the co-located
    frequency groups, i.e. over the rungs handing over — must not ripple.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = _hold_spec()
    _, _, z = spec.compile()

    requested = 2.0 * params.lens_scale * abs(float(f_z_ramp(z, params)(spec.duration)))
    assert requested > 3.0 * max_z_integral(params)  # the milestone's "beyond Eq. 1" bar
    with pytest.raises(ValueError, match="leaves its usable band"):
        synthesize(spec, params)

    wfs = synthesize(spec, params, shepard=ShepardConfig(DFX, DFY))
    centres = _probe_times(wfs, "Ax", _fade_env(wfs, "Ax").g_centre, tau)
    assert centres.size >= 4  # ... four complete hand-overs, as PLAN §3 asks
    times = np.sort(
        np.concatenate([centres, _probe_times(wfs, "Ax", 0.0, tau), np.linspace(tau, 1.0 * ms, 60)])
    )

    tol = 0.25 * params.deflection_scale * DFY
    powers = []
    worst = dict(lateral=0.0, axial=0.0, astig=0.0)
    for t in times:
        trap = measure(_at_column(build_terms(wfs, float(t)), optics, 0.0, tol), optics)
        assert trap, f"the tweezer went dark at t = {t / us:.3f} us"
        total = sum(m.power for m in trap)
        powers.append(total)
        worst["lateral"] = max(worst["lateral"], max(max(abs(m.x), abs(m.y)) for m in trap))
        worst["astig"] = max(worst["astig"], max(abs(m.delta_f) for m in trap))
        z_lab = sum(m.power * m.z_lab for m in trap) / total
        worst["axial"] = max(worst["axial"], abs(z_lab - float(z(t - 0.5 * tau))))

    assert worst["lateral"] < 0.01 * optics.waist0
    assert worst["axial"] < 0.02 * optics.rayleigh
    assert worst["astig"] < 0.02 * optics.rayleigh
    power = np.array(powers)
    assert (power.max() - power.min()) / power.mean() < 0.01
    assert float(np.max(z(times - 0.5 * tau))) > 2.0 * optics.rayleigh  # worth measuring


def test_the_sustained_hold_simulate_stays_interactive(params1030):
    """24 probes of the 1 ms hold, metrics only: the notebook does this before it plots."""
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    spec = _hold_spec()
    wfs = synthesize(spec, params, shepard=ShepardConfig(DFX, DFY))

    start = time.perf_counter()
    result = simulate(wfs, np.linspace(tau, spec.duration, 24))
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"1 ms hold over 24 probes took {elapsed:.2f} s"
    assert all(frame for frame in result.metrics)
    assert np.max(np.abs(result.tracked_z())) > 2.0 * params.optics.rayleigh


# ================================================================== 2. shadow tweezers (S31)


def test_shadows_at_plus_minus_delta_f_only_while_a_pair_hands_over(params1030):
    r"""Mid-fade: two companions at ``+- deflection_scale delta_f_x``, each **half** the trap.

    During an x hand-over both x channels carry the dying rung ``a`` (``cos^{p} theta``) and
    the rising rung ``a-1`` (``sin^{p} theta``).  A trap's position depends on the index
    *difference*, so the two co-located combinations ``(a, a)`` and ``(a-1, a-1)`` make the
    tweezer with ``cos^{2(p_A+p_B)} + sin^{2(p_A+p_B)} = 1`` of the light, while the two cross
    combinations sit one rung off and carry ``cos^{2p_A} sin^{2p_B}`` and ``sin^{2p_A}
    cos^{2p_B}``.  At the fade centre all four are ``2^{-(p_A+p_B)} = 1/2``: the *pair* of
    shadows carries as much as the trap and **each shadow is half of it**.

    The two shadows are exactly frequency-degenerate (the Fig. S6 pair) but sit 165 µm apart,
    so they do not interfere — ``power_coherent`` equals ``power`` for their group, which is
    the statement the interlaced scheme relies on and the simultaneous one loses.

    Equal spacings here, because that is what makes interlacing *exact*: with ``eta = 1/2``
    the ``xi = 1/2`` fade zones of one axis tile the plateaus of the other only when the two
    ladders share a ``delta_f``.  The last block below is the price of the unequal spacings an
    array needs.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    delta_f = 8.0 * MHz
    wfs = synthesize(_hold_spec(), params, shepard=ShepardConfig(delta_f, delta_f))
    offset = params.deflection_scale * delta_f
    tol = 0.25 * params.deflection_scale * delta_f

    centres = _probe_times(wfs, "Ax", _fade_env(wfs, "Ax").g_centre, tau)
    plateaus = _probe_times(wfs, "Ax", 0.0, tau)
    assert centres.size >= 4 and plateaus.size >= 4

    for t in centres[:3]:
        terms = build_terms(wfs, float(t))
        trap = sum(m.power for m in measure(_at_column(terms, optics, 0.0, tol), optics))
        for side in (-1.0, 1.0):
            # 1/2 up to the fast-fade aperture correction: a trap combination stacks two
            # envelopes fading the *same* way (tilt terms add) while a shadow mixes a dying
            # rung with a rising one (they cancel), so the two carry different Eq. S5
            # alpha1/alpha2 weights.  That difference is the rho^2 effect of the notebook's
            # §7a, 0.09 % of the ratio at this ladder width — physics, not tolerance.
            shadow = measure(_at_column(terms, optics, side * offset, tol), optics)
            ratio = sum(m.power for m in shadow) / trap
            assert ratio == pytest.approx(0.5, rel=0.005)
            assert abs(ratio - 0.5) > 1e-6  # ... and the correction really is there
        # Nothing else is lit — 2 x 2 x 1 x 1 rays, not 16 — and the degenerate shadow pair
        # is reported as one group whose power-weighted centre sits on the trap, with nothing
        # there: group *position* is meaningless for a degenerate pair, group power is not.
        assert terms.n_terms == 4
        xc, _, _, _ = spot_params(terms, optics, 0.0)
        assert np.max(np.abs(xc)) == pytest.approx(offset, rel=1e-12)
        pair = [m for m in measure(terms, optics) if m.power > 0.9 * trap]
        assert len(pair) == 1
        assert abs(pair[0].x) < 1e-9 * optics.waist0
        assert pair[0].power_coherent == pytest.approx(pair[0].power, rel=1e-6)

    for t in plateaus[:3]:
        terms = build_terms(wfs, float(t))
        xc, _, _, _ = spot_params(terms, optics, 0.0)
        assert np.max(np.abs(xc)) < 1e-9 * optics.waist0  # no companions off the fade zone
        assert measure(terms, optics)

    # Unequal spacings (what an array needs, so that its rows and columns stay distinct) beat
    # the two fade schedules against each other: some hand-overs then happen on both axes at
    # once and the full 2x2x2x2 grid of Fig. S6 lights up.
    beating = synthesize(_hold_spec(), params, shepard=ShepardConfig(DFX, DFY))
    rays = {
        build_terms(beating, float(t)).n_terms
        for t in _probe_times(beating, "Ax", _fade_env(beating, "Ax").g_centre, tau)
    }
    assert rays == {4, 16}


def test_the_array_grid_grows_two_columns_that_switch_on_at_full_brightness(params1030):
    """Table II's array row: ``(Mx + 2) x My`` during an x-fade, interior columns rock steady.

    ``p_B = 0`` makes the ``B`` ladder a rectangle — it *is* the array ladder, and shaping it
    would shape the array — so the whole hand-over sits on the single ``A`` tone (``p_A = 1``).
    Every interior column is fed by both live ``A`` rungs and stays at ``cos^2 + sin^2 = 1``;
    the outermost combinations have one feeder each, and because the ``B`` rectangle switches
    a rung on discontinuously, the new column *appears* at full brightness instead of fading
    up.  That is the scheduling caveat: a pick-up must not be timed inside a fade zone.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    delta_f = 1.0 * MHz
    spec = TrajectorySpec(
        array=ArraySpec(3, 3, delta_f, delta_f),
        moves=(Lift(6.0 * um, 60.0 * us), Hold(150.0 * us), Lift(-6.0 * um, 60.0 * us)),
    )
    wfs = synthesize(spec, params, shepard=ShepardConfig(delta_f, delta_f))
    pitch = params.deflection_scale * delta_f

    widths, brightest_extra = set(), 0.0
    for t in np.linspace(tau, spec.duration, 120):
        nodes, residual = _lattice(build_terms(wfs, float(t)), optics, (0.0, 0.0), (pitch, pitch))
        assert residual < 0.01 * optics.waist0
        columns = {i for i, _ in nodes}
        rows = {j for _, j in nodes}
        assert columns <= {-2, -1, 0, 1, 2} and {-1, 0, 1} <= columns
        assert rows <= {-2, -1, 0, 1, 2} and {-1, 0, 1} <= rows
        widths.add(len(columns))
        reference = max(nodes.values())
        interior = [nodes[(i, j)] / reference for i in (-1, 0, 1) for j in (-1, 0, 1)]
        np.testing.assert_allclose(interior, 1.0, rtol=1e-12)
        extra = [nodes.get((i, 0), 0.0) / reference for i in (-2, 2)]
        assert sum(extra) == pytest.approx(0.0 if len(columns) == 3 else 1.0, abs=1e-9)
        brightest_extra = max(brightest_extra, max(extra))

    assert widths == {3, 5}  # (Mx + 2) x My, and only while a pair hands over
    assert brightest_extra > 0.99  # ... switched on at full brightness, not faded up


# ======================================================== 3. interlaced vs simultaneous fading


def test_only_simultaneous_fading_makes_the_tweezer_phase_sensitive(params1030):
    r"""Fig. S6's static Mach-Zehnder, and what ``xi = 1/2`` does to it.

    Fade both axes together (``xi_y = 0``, equal spacings) and four rays land on the tweezer:
    ``(a,a|a,a)``, ``(a-1,a-1|a,a)``, ``(a,a|a-1,a-1)`` and ``(a-1,a-1|a-1,a-1)``.  Their
    optical frequencies follow the *sum* of the rung indices, so the two mixed rays are
    exactly degenerate — a two-arm interferometer whose relative phase is a per-rung phase of
    one channel, i.e. an acoustic/optical path offset the experiment does not control
    (Eq. S29's ``x_err`` is precisely a linear-in-frequency phase).  Sweeping it swings the
    trap's coherent power by +-50 %, while the incoherent ``power`` cannot see it at all.

    Interlacing (``xi_y = 1/2``) leaves one axis on its plateau, so the tweezer is fed by two
    rays two ladder steps apart in frequency: they add in intensity and the sweep does
    nothing.  That is the whole reason Table II offsets the y pair.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = _hold_spec()
    delta_f = 8.0 * MHz
    tol = 0.25 * params.deflection_scale * delta_f
    simultaneous = {"Ay": ChannelFade(1, 0.5, 0.0), "By": ChannelFade(1, 0.5, 0.0)}

    swing: dict[str, float] = {}
    extremes: dict[str, tuple[float, float]] = {}
    for label, config in (("interlaced", "auto"), ("simultaneous", simultaneous)):
        cfg = ShepardConfig(delta_f, delta_f, config=config)
        reference = synthesize(spec, params, shepard=cfg)
        n_rungs = reference.channels["By"].n_tones
        t = float(_probe_times(reference, "Ax", _fade_env(reference, "Ax").g_centre, tau)[1])

        incoherent, coherent = [], []
        for offset in np.linspace(0.0, 2.0 * np.pi, 9):
            # A linear-in-rung phase is a delay: rung n of By is retarded by n * offset.
            wfs = synthesize(spec, params, shepard=cfg, phases={"By": offset * np.arange(n_rungs)})
            trap = measure(_at_column(build_terms(wfs, t), optics, 0.0, tol), optics)
            incoherent.append(sum(m.power for m in trap))
            coherent.append(sum(m.power_coherent for m in trap))
        incoherent, coherent = np.array(incoherent), np.array(coherent)

        # The incoherent reading is blind to the offset — that is the point of the pair.
        assert (incoherent.max() - incoherent.min()) / incoherent.mean() < 1e-12
        swing[label] = (coherent.max() - coherent.min()) / incoherent.mean()
        assert coherent[0] == pytest.approx(coherent[-1], rel=1e-9)  # 2 pi is a full turn
        extremes[label] = (coherent.max() / incoherent.mean(), coherent.min() / incoherent.mean())

    assert swing["simultaneous"] > 0.10
    assert swing["interlaced"] < 0.001
    # The four rays each carry a quarter of the trap, two of them degenerate: the coherent
    # reading runs from 1/4 + 0 + 1/4 to 1/4 + 1 + 1/4 as the offset turns.
    assert swing["simultaneous"] == pytest.approx(1.0, rel=0.01)
    assert extremes["simultaneous"] == pytest.approx((1.5, 0.5), rel=0.01)
    assert extremes["interlaced"] == pytest.approx((1.0, 1.0), rel=1e-3)


# ======================================================= 4. the user story, at a human pace


def test_the_unhurried_user_story_synthesizes_tracks_and_stays_in_band(params1030):
    """``examples/04``'s refused schedule, run: 10x10, lift 10 µm, traverse, drop — in 550 µs.

    Eq. S19 wants ``4.0e-9 m.s`` of axial integral and the band buys ``2.1e-9`` with nothing
    else in it, so the plain synthesis refuses.  ``shepard="auto"`` notices, says so in the
    description, and hands the axial degree of freedom to the ladders: the array's own
    spacings *are* the ``B`` ladders (Eq. S27), so nothing about the array geometry changes
    and the whole drive stays inside a fixed window.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(array=STORY_ARRAY, moves=STORY_MOVES)
    x, y, z = spec.compile()

    with pytest.raises(ValueError, match="leaves its usable band"):
        synthesize(spec, params)
    wfs = synthesize(spec, params, shepard="auto")
    assert "Fading-Shepard synthesis (Eqs. S24-S28)" in wfs.description
    assert "plain Eq. S19 refused" in wfs.description
    assert wfs.n_tones > 4 * spec.array.mx  # a ladder per channel, not one tone per column

    pitch = spec.array.pitch(params)
    # Eq. S19 centres an M-tone ladder, so an *even* M lands on half-integer multiples of the
    # pitch, and WO-17 §2.1's comb offset puts the Shepard columns there too: the lattice the
    # nodes live on is half a pitch off the array centre itself.  Anchoring `_lattice` on that
    # lattice keeps the ten programmed columns at the same indices as before the correction.
    half = tuple(
        0.5 * p if m % 2 == 0 else 0.0 for p, m in zip(pitch, (STORY_ARRAY.mx, STORY_ARRAY.my))
    )
    times = np.linspace(tau, spec.duration, 12)
    worst = dict(residual=0.0, axial=0.0, astig=0.0, spread=0.0)
    for t in times:
        t_c = float(t) - 0.5 * tau
        terms = build_terms(wfs, float(t))
        centre = (float(x(t_c)) - half[0], float(y(t_c)) - half[1])
        nodes, residual = _lattice(terms, optics, centre, pitch)
        assert {i for i, _ in nodes} >= set(range(-4, 6))  # the 10 programmed columns, always
        assert {j for _, j in nodes} >= set(range(-4, 6))
        worst["residual"] = max(worst["residual"], residual)

        z_axis = Z_LAB_SIGN * 2.0 * optics.focal_length**2 * terms.theta2 / optics.k
        z_lab = 0.5 * (z_axis[0] + z_axis[1])
        worst["axial"] = max(worst["axial"], float(np.max(np.abs(z_lab - float(z(t_c))))))
        worst["astig"] = max(worst["astig"], float(np.max(np.abs(z_axis[0] - z_axis[1]))))
        worst["spread"] = max(worst["spread"], float(z_lab.max() - z_lab.min()))

    assert worst["residual"] < 0.01 * optics.waist0
    assert worst["axial"] < 0.02 * optics.rayleigh
    assert worst["astig"] < 0.02 * optics.rayleigh
    assert worst["spread"] < 0.02 * optics.rayleigh

    # every live rung of every channel stays inside its band, for the whole run.
    scan = np.linspace(*wfs.t_span, 4001)
    for name in CHANNELS:
        aod = params.channels[name]
        for tone in wfs.channels[name].tones:
            live = np.asarray(tone.env.A(scan)) > 0.0
            f = aod.f_center + np.asarray(tone.freq(scan))[live]
            assert np.all(f > aod.band[0]) and np.all(f < aod.band[1])


def test_the_user_story_renders_a_frame_of_a_hundred_traps(params1030):
    """One frame of the unhurried story: the array is there, in the plane it asked for."""
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(array=STORY_ARRAY, moves=STORY_MOVES)
    _, _, z = spec.compile()
    wfs = synthesize(spec, params, shepard="auto")

    t = 300.0 * us
    result = simulate(wfs, [t])
    metrics = result.metrics[0]
    plane = track_z(metrics)
    assert plane == pytest.approx(float(z(t - 0.5 * tau)), rel=1e-9)

    pitch_x, pitch_y = spec.array.pitch(params)
    x_c, y_c, _, _ = spot_params(result.terms(0), optics, plane)
    half_x = 0.5 * (float(x_c.max() - x_c.min())) + 3.0 * optics.waist0
    half_y = 0.5 * (float(y_c.max() - y_c.min())) + 3.0 * optics.waist0
    mid_x, mid_y = 0.5 * float(x_c.max() + x_c.min()), 0.5 * float(y_c.max() + y_c.min())
    grid = FrameGrid(mid_x - half_x, mid_x + half_x, 321, mid_y - half_y, mid_y + half_y, 321)
    frame = result.frame(0, grid)

    # The brightest pixels sit on the lattice the metrics predict, at the tracked plane.
    peak = float(frame.max())
    assert peak > 0.0
    rows, cols = np.nonzero(frame > 0.5 * peak)
    lit_x, lit_y = grid.x[cols], grid.y[rows]
    assert np.max(np.abs(np.round((lit_x - mid_x) / pitch_x) * pitch_x - (lit_x - mid_x))) < (
        2.0 * optics.waist0
    )
    assert np.max(np.abs(np.round((lit_y - mid_y) / pitch_y) * pitch_y - (lit_y - mid_y))) < (
        2.0 * optics.waist0
    )
    # ... and the same frame is dark far outside the array.
    edge = intensity_frame(result.terms(0), optics, grid, plane)[0, 0]
    assert edge < 1e-6 * peak
