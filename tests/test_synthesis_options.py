r"""The M5 synthesis options (``docs/workorders/WO-17-product-core.md`` §2).

Four independent knobs on :func:`aodl.waveform.synthesis.synthesize`, plus the ``spot_table``
column that goes with them.  Every one of them is checked twice over: that it *does* what it
claims, and that switching it off reproduces the pre-M5 drive bit for bit.

1. **§2.1 lattice alignment.**  A Shepard array's columns sit at ``(b + xi_B) - (a + xi_A)``,
   while Eq. S19 centres a counted ladder, which is half-integer multiples of the pitch for
   even ``M``.  :func:`~aodl.waveform.shepard.table_ii` gives an even-``M`` ``B`` ladder
   ``xi_B = xi_A + 1/2``, so the two modes put the traps in the same places *and* every
   requested site is fed by both live ``A`` rungs through a hand-over.  WO-17's frequency-only
   ``delta_f/2`` comb offset got the lattice right but not the window timing (one requested
   column went dark per hand-over, ``R2-DESIGN-ADDENDUM`` §F2); its auto value is now ``0``
   and an explicit ``comb_offset`` mapping is the only way back to it.
2. **§2.2 retardation pre-compensation.**  Reading the trajectory ``tau/2`` ahead makes the
   atom plane match the request at ``t`` instead of at ``t - tau/2``.
3. **§2.3 the ``f_Z`` edge pre-bias.**  Starting the drive half an excursion below the carrier
   centres the ``f_Z`` walk in the band and doubles Eq. 1's budget — ``docs/PLAN.md`` §1.5's
   412 µs — while moving no trap, because the bias is common to all four channels.
4. **§2.4 ``switch_ramp``.**  Raised-cosine on/off ramps for the ``p_B = 0`` rectangles — and
   for nothing else (WO-20 §1): the ``cos^p`` windows switch smoothly already, so ramping them
   bought no continuity and cost flatness.  Scoped that way the ramp leaves every interior
   column *exactly* where it was; only a ladder's two edge columns pay, by the law derived here
   (both parities alike, now that an even-``M`` ``B`` ladder carries the half-rung ``xi_B``).

``mixing_order=1`` throughout: these are statements about the frequency/amplitude algebra, not
about intermodulation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aodl.device.aodl import build_terms
from aodl.engine import SPOT_TABLE_KEYS, simulate
from aodl.field.focal import spot_params
from aodl.params import CHANNELS
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us
from aodl.waveform.shepard import (
    SWITCH_FLATNESS_SHAPE,
    ChannelFade,
    FadeZoneEnvelope,
    ShepardConfig,
    SwitchRamped,
    lattice_comb_offset,
    shepard_band_bound,
)
from aodl.waveform.synthesis import (
    max_z_integral,
    requested_z_integral,
    resolve_f_z_bias,
    synthesize,
)
from aodl.waveform.tones import TIME_TOL, WaveformSet

#: A short lift-hold-drop that fits the band in either mode, so the two can be compared.
SHORT_MOVES = (Lift(2.0 * um, 30.0 * us), Hold(20.0 * us), Lift(-2.0 * um, 30.0 * us))

#: The array pitch used by the lattice checks.  1 MHz -> 10.3 µm at the default hardware.
DF = 1.0 * MHz

#: A free (single-tweezer) y axis wide enough to fit its own fade window in the band.
DFY_FREE = 7.0 * MHz


def _linear(params):
    """``params`` with every channel at ``mixing_order=1`` — one tone, one beam."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _columns(wfs: WaveformSet, optics, t: float) -> np.ndarray:
    """Sorted lab X of every pupil term at frame time ``t`` [m]."""
    xc, _, _, _ = spot_params(build_terms(wfs, float(t)), optics, 0.0)
    return np.sort(np.asarray(xc, dtype=np.float64))


def _mismatch(want: np.ndarray, got: np.ndarray) -> float:
    """Worst distance [m] from a column of ``want`` to the nearest column of ``got``."""
    return float(np.max(np.min(np.abs(want[:, None] - got[None, :]), axis=1)))


def _assert_same_drive(a: WaveformSet, b: WaveformSet, probes: np.ndarray) -> None:
    """The two waveform sets are bit-for-bit the same drive (laws, phases and envelopes)."""
    assert set(a.channels) == set(b.channels)
    for name in a.channels:
        left, right = a.channels[name].tones, b.channels[name].tones
        assert len(left) == len(right), name
        for i, (one, two) in enumerate(zip(left, right, strict=True)):
            np.testing.assert_array_equal(one.freq.breaks, two.freq.breaks, err_msg=f"{name}[{i}]")
            np.testing.assert_array_equal(one.freq.coeffs, two.freq.coeffs, err_msg=f"{name}[{i}]")
            assert one.phase0 == two.phase0, f"{name}[{i}]"
            for quantity in ("A", "dA", "d2A"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(one.env, quantity)(probes), dtype=np.float64),
                    np.asarray(getattr(two.env, quantity)(probes), dtype=np.float64),
                    err_msg=f"{name}[{i}].{quantity}",
                )


def _nodes(
    wfs: WaveformSet,
    optics,
    t: float,
    pitch: tuple[float, float],
    origin: tuple[float, float] = (0.0, 0.0),
) -> dict[tuple[int, int], float]:
    """``{(column, row): summed |c|^2}`` of the lattice at frame time ``t``.

    The terms of one node are not frequency-degenerate with each other (their rung indices
    differ), so their intensities add — which is what makes this the node's brightness.
    ``origin`` is where node ``(0, 0)`` sits: ``(0, 0)`` for an odd-``M`` array, half a pitch
    out for an even one, whose Eq. S19 lattice straddles the array centre.
    """
    terms = build_terms(wfs, float(t))
    xc, yc, _, _ = spot_params(terms, optics, 0.0)
    out: dict[tuple[int, int], float] = {}
    for x, y, c in zip(xc, yc, terms.c, strict=True):
        key = (
            int(round((float(x) - origin[0]) / pitch[0])),
            int(round((float(y) - origin[1]) / pitch[1])),
        )
        out[key] = out.get(key, 0.0) + float(abs(c) ** 2)
    return out


# ============================================== 2.1 the even-M lattice (WO-16 finding F-2)


def _wo17(cfg: ShepardConfig, array: ArraySpec) -> ShepardConfig:
    """``cfg`` with the WO-17 even-``M`` timing: ``xi_B = xi_A`` plus a ``delta_f/2`` comb.

    That is what ``shepard="auto"`` built before the fix — the lattice right, the ``B``
    windows half a rung out of phase with the ``A`` hand-overs (``R2-DESIGN-ADDENDUM`` §F2).
    Odd and single-tweezer axes are left exactly as Table II has them.
    """
    config: dict[str, ChannelFade] = {}
    comb: dict[str, float] = {}
    for b_name, xi_a, count, spacing in (
        ("Bx", 0.0, array.mx, cfg.delta_f_x),
        ("By", 0.5, array.my, cfg.delta_f_y),
    ):
        if count > 1 and count % 2 == 0:
            config[b_name] = ChannelFade(count, 0.0, xi_a)
            comb[b_name] = 0.5 * spacing
    return replace(cfg, config=config, comb_offset=comb)


def _integer_lattice(cfg: ShepardConfig, array: ArraySpec) -> ShepardConfig:
    """``cfg`` with neither correction: ``xi_B = xi_A`` and no comb (the pre-M5 drive)."""
    return replace(_wo17(cfg, array), comb_offset={})


@pytest.mark.parametrize("m_x", [2, 3, 4, 5])
def test_the_shepard_lattice_coincides_with_the_s19_lattice_for_even_and_odd_m(params1030, m_x):
    r"""Same spec, both modes, same trap positions — to float64, for ``M = 2, 3, 4, 5``.

    Eq. S19's ladder is ``f^{(n)} = (n - (M-1)/2) delta_f``: for odd ``M`` that is integer
    multiples of ``delta_f`` and for even ``M`` half-integer ones.  A Shepard array's columns
    come from ``(b + xi_B) - (a + xi_A)``, so with a common ``xi`` they are *always* integers
    and an even-``M`` array used to sit half a pitch sideways (WO-16 F-2).  Table II's
    half-rung ``xi_B`` on an even-``M`` axis closes that gap — exactly as the WO-17 comb offset
    did, which is why the lattice alone could not tell the two apart (the timing can:
    :func:`test_the_half_rung_xi_b_keeps_every_requested_node_lit_and_the_wo17_comb_did_not`).
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    spec = TrajectorySpec(array=ArraySpec(m_x, 1, DF, 0.0), moves=SHORT_MOVES)
    probes = np.linspace(tau, spec.duration, 9)
    cfg = ShepardConfig(DF, DFY_FREE)

    s19 = synthesize(spec, params)
    fixed = synthesize(spec, params, shepard=cfg)
    wo17 = synthesize(spec, params, shepard=_wo17(cfg, spec.array))
    uncorrected = synthesize(spec, params, shepard=_integer_lattice(cfg, spec.array))

    def worst(wfs: WaveformSet) -> float:
        return max(_mismatch(_columns(s19, optics, t), _columns(wfs, optics, t)) for t in probes)

    worst_fixed = worst(fixed)
    assert worst_fixed < 0.01 * optics.waist0
    assert worst_fixed < 1e-9 * pitch  # ... in fact exact to rounding
    assert worst(wo17) < 1e-9 * pitch  # the WO-17 comb had the lattice right too
    if m_x % 2 == 0:
        assert worst(uncorrected) == pytest.approx(0.5 * pitch, rel=1e-9)  # the F-2 offset
    else:
        assert worst(uncorrected) == worst_fixed  # odd M never needed a correction


def test_the_half_rung_xi_b_carries_the_lattice_and_the_comb_offset_is_zero(params1030):
    """``xi_B = xi_A + 1/2`` gives the very frequency laws the WO-17 comb did — and other windows.

    Rung ``n`` of the fixed ``B`` ladder runs at ``f_lat + f_Z + (n + xi_A + 1/2) delta_f``,
    which is rung ``n`` of the WO-17 ladder (``(n + xi_A) delta_f + delta_f/2``) to the last
    bit of the law; what differs is its fade coordinate, ``g`` larger by ``delta_f/2``, i.e.
    the window timing — the one thing the frequency-only offset could not move.  The ``A``
    channels are untouched, ``"auto"`` resolves to zero comb on every channel, and an explicit
    mapping is still honoured verbatim.
    """
    params = _linear(params1030)
    spec = TrajectorySpec(array=ArraySpec(4, 4, DF, 1.3 * MHz), moves=SHORT_MOVES)
    cfg = ShepardConfig(DF, 1.3 * MHz)
    fixed = synthesize(spec, params, shepard=cfg)
    wo17 = synthesize(spec, params, shepard=_wo17(cfg, spec.array))
    probes = np.linspace(*fixed.t_span, 401)

    for name in ("Ax", "Ay"):  # the A ladders are the same drive, bit for bit
        left, right = fixed.channels[name].tones, wo17.channels[name].tones
        assert len(left) == len(right)
        for one, two in zip(left, right, strict=True):
            np.testing.assert_array_equal(one.freq.coeffs, two.freq.coeffs)
            np.testing.assert_array_equal(
                np.asarray(one.env.A(probes)), np.asarray(two.env.A(probes))
            )
    for name, spacing in (("Bx", DF), ("By", 1.3 * MHz)):
        old = {round(float(tone.freq(probes[0])), 3): tone for tone in wo17.channels[name].tones}
        matched = 0
        for tone in fixed.channels[name].tones:
            twin = old.get(round(float(tone.freq(probes[0])), 3))
            if twin is None:
                continue  # a rung only one of the two windows ever opens
            matched += 1
            np.testing.assert_allclose(
                np.asarray(tone.freq(probes)), np.asarray(twin.freq(probes)), rtol=0, atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(tone.env.g(probes)) - np.asarray(twin.env.g(probes)),
                0.5 * spacing,
                rtol=0,
                atol=1e-6,
            )
        assert matched >= spec.array.mx  # the same live comb, give or take an edge rung

    # Table II carries the half pitch now; the comb offset of the auto path is zero everywhere.
    fades = cfg.resolve(spec.array)
    assert fades["Bx"].xi == 0.5 and fades["By"].xi == 1.0
    assert cfg.comb_offsets(fades) == {name: 0.0 for name in CHANNELS}
    assert lattice_comb_offset(ChannelFade(m=3, p=0.0, xi=0.0), DF) == 0.0
    assert lattice_comb_offset(ChannelFade(m=4, p=0.0, xi=0.0), DF) == 0.0
    explicit = replace(cfg, comb_offset={"Bx": 0.25 * DF})
    assert explicit.comb_offsets(fades) == {"Ax": 0.0, "Ay": 0.0, "Bx": 0.25 * DF, "By": 0.0}


def test_the_half_rung_xi_b_keeps_every_requested_node_lit_and_the_wo17_comb_did_not(
    params1030,
):
    r"""The two drives share a lattice; only the fixed one lights it, at every frame.

    With ``xi_B = xi_A + 1/2`` both live ``A`` rungs of a hand-over see the same ``M + 1``
    ``B`` rungs, so every requested node reads ``cos^2 + sin^2 = 1`` of the brightest one —
    exactly, in ``|c|^2`` — while the two extended columns switch.  The WO-17 timing held only
    ``M`` ``B`` rungs during the hand-over, so the lit block walked by a pitch and an end
    column of the requested array went dark (``R2-DESIGN-ADDENDUM`` §F2).
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    spec = TrajectorySpec(
        array=ArraySpec(4, 1, DF, 0.0),
        moves=(Lift(6.0 * um, 40.0 * us), Hold(80.0 * us), Lift(-6.0 * um, 40.0 * us)),
    )
    cfg = ShepardConfig(DF, DFY_FREE)
    fixed = synthesize(spec, params, shepard=cfg)
    wo17 = synthesize(spec, params, shepard=_wo17(cfg, spec.array))
    grid = (pitch, params.deflection_scale * DFY_FREE)
    origin = (-1.5 * pitch, 0.0)  # node (0, 0) is the first requested site of Eq. S19
    requested = [(i, 0) for i in range(spec.array.mx)]

    def row0(nodes: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
        """The array row; rows +-1 are the free y axis' own Eq. S31 shadows."""
        return {key: w for key, w in nodes.items() if key[1] == 0}

    widths, darkest_old = set(), 1.0
    for t in np.linspace(tau, spec.duration, 48):
        new = row0(_nodes(fixed, optics, float(t), grid, origin=origin))
        old = row0(_nodes(wo17, optics, float(t), grid, origin=origin))
        assert set(new) <= {(i, 0) for i in range(-1, spec.array.mx + 1)}
        assert set(old) <= {(i, 0) for i in range(-1, spec.array.mx + 1)}  # same lattice
        brightest = max(new.values())
        for key in requested:
            assert new[key] == pytest.approx(brightest, rel=1e-9), (key, t)
        widths.add(len(new))
        darkest_old = min(darkest_old, min(old.get(k, 0.0) for k in requested) / max(old.values()))
    assert widths == {spec.array.mx, spec.array.mx + 2}  # M, or M + 2 during a hand-over
    assert darkest_old < 0.05  # the defect the half rung closes


def test_the_shepard_band_bound_comb_term_is_zero_unless_set_explicitly(params1030):
    """The live window translates with an explicit comb, so the bound grows by ``|c_comb|``.

    ``xi_B`` lives inside ``g``, so the half rung moves the window *with* the rungs and costs
    no band at all: the auto path's comb term is zero, and only a hand-set offset is charged.
    """
    params = _linear(params1030)
    fade = ChannelFade(m=4, p=0.0, xi=0.5)
    plain = shepard_band_bound(fade, DF, 0.5, 2.0 * MHz)
    shifted = shepard_band_bound(fade, DF, 0.5, 2.0 * MHz, 0.5 * DF)
    assert plain == pytest.approx(0.5 * 4.5 * DF + 2.0 * MHz)
    assert shifted == pytest.approx(plain + 0.5 * DF)

    # ... and a drive that only just fits at the auto (zero) comb is refused with the old one.
    spec = TrajectorySpec(array=ArraySpec(12, 1, 1.5 * MHz, 0.0), moves=SHORT_MOVES)
    cfg = ShepardConfig(1.5 * MHz, DFY_FREE)
    assert set(cfg.comb_offsets(cfg.resolve(spec.array)).values()) == {0.0}
    synthesize(spec, params, shepard=cfg)
    with pytest.raises(ValueError, match="comb offset"):
        synthesize(spec, params, shepard=replace(cfg, comb_offset={"Bx": 0.75 * MHz}))


# ================================================= 2.2 retardation pre-compensation (tau/2)


def _tracking_error(wfs, spec, params, lead: float) -> tuple[float, float]:
    """Worst ``(lateral, axial)`` error [m] against the request at ``t - lead``."""
    x, y, z = spec.compile()
    tau = params.channels["Ax"].transit_time
    times = np.linspace(tau, spec.duration, 13)
    result = simulate(wfs, times)
    lateral = axial = 0.0
    for i, frame in enumerate(result.metrics):
        assert frame, f"frame {i} is dark"
        t_ref = float(times[i]) - lead
        lateral = max(
            lateral,
            abs(float(np.mean([m.x for m in frame])) - float(x(t_ref))),
            abs(float(np.mean([m.y for m in frame])) - float(y(t_ref))),
        )
        axial = max(axial, abs(float(np.mean([m.z_lab for m in frame])) - float(z(t_ref))))
    return lateral, axial


def test_retard_compensation_makes_the_atom_plane_match_the_request_at_t(params1030):
    """``retard_compensate=True`` moves the reference instant from ``t - tau/2`` to ``t``.

    The acoustic sample lighting the beam centre left the transducer half a transit ago
    (``docs/conventions.md`` §7), so a drive written from ``X(t + tau/2)`` replays the request
    on time.  Both readings are checked both ways round, so the test cannot pass by tolerance:
    the compensated drive matches at ``t`` and *misses* at ``t - tau/2``, and vice versa.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(
        array=ArraySpec(2, 2, DF, 1.3 * MHz),
        moves=(Lift(4.0 * um, 40.0 * us), Translate(20.0 * um, 12.0 * um, 50.0 * us)),
    )
    plain = synthesize(spec, params)
    compensated = synthesize(spec, params, retard_compensate=True)

    lateral, axial = _tracking_error(compensated, spec, params, 0.0)
    assert lateral < 0.01 * optics.waist0
    assert axial < 0.02 * optics.rayleigh

    late_lateral, _ = _tracking_error(compensated, spec, params, 0.5 * tau)
    assert late_lateral > optics.waist0  # ... and it is genuinely no longer late

    lateral, axial = _tracking_error(plain, spec, params, 0.5 * tau)
    assert lateral < 0.01 * optics.waist0
    assert axial < 0.02 * optics.rayleigh
    early_lateral, _ = _tracking_error(plain, spec, params, 0.0)
    assert early_lateral > optics.waist0


def test_the_compensated_drive_is_the_trajectory_read_half_a_transit_ahead(params1030):
    """Term by term: ``f_Ax(t) = -v X(t + tau/2) / (2 lambda F) + f_Z(t)`` (Eq. S19, advanced).

    Checked against a closed form built here from the compiled profile, not against the
    uncompensated drive — the advance has to be exactly ``min(t + tau/2, T)``, including the
    clamp at the end of the trajectory that keeps the array at rest where it asked to be.
    """
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    duration = 90.0 * us
    spec = TrajectorySpec(array=ArraySpec(1, 1), moves=(Translate(20.0 * um, 0.0, duration),))
    x, _, _ = spec.compile()
    wfs = synthesize(spec, params, retard_compensate=True)

    t = np.linspace(0.0, duration, 61)
    want = -np.asarray(x(np.minimum(t + 0.5 * tau, duration))) / (2.0 * params.deflection_scale)
    np.testing.assert_allclose(np.asarray(wfs.channels["Ax"].tones[0].freq(t)), want, atol=1e-6)
    # the last requested instant is reached half a transit early and then held
    assert float(wfs.channels["Ax"].tones[0].freq(duration - 0.5 * tau)) == pytest.approx(
        float(wfs.channels["Ax"].tones[0].freq(duration)), rel=1e-12
    )


def test_the_options_default_to_the_pre_m5_drive_bit_for_bit(params1030):
    """Every §2 knob off (the defaults) reproduces the M4-era waveform exactly.

    Both paths: plain Eq. S19, and a fading-Shepard drive on an *odd* ``M`` where the §2.1 fix
    is identically zero.  Bit-for-bit, not to a tolerance — an option that defaults to
    "almost the same" is a silent regression.
    """
    params = _linear(params1030)
    spec = TrajectorySpec(array=ArraySpec(3, 3, DF, 1.3 * MHz), moves=SHORT_MOVES)
    probes = np.linspace(*synthesize(spec, params).t_span, 257)

    _assert_same_drive(
        synthesize(spec, params),
        synthesize(spec, params, retard_compensate=False, f_z_bias=0.0, switch_ramp=None),
        probes,
    )
    cfg = ShepardConfig(DF, 1.3 * MHz)
    _assert_same_drive(
        synthesize(spec, params, shepard=cfg),
        synthesize(spec, params, shepard=replace(cfg, comb_offset={}), switch_ramp=0.0),
        probes,
    )


# ============================================================ 2.3 the f_Z edge pre-bias


def _longest_feasible_hold(params, bias, lift: float = 2.0 * us) -> float:
    """Bisect the longest ``Hold`` a 10 µm lift-hold-drop can carry inside the band [s]."""

    def spec_of(hold: float) -> TrajectorySpec:
        return TrajectorySpec(
            array=ArraySpec(1, 1),
            moves=(Lift(10.0 * um, lift), Hold(hold), Lift(-10.0 * um, lift)),
        )

    def fits(hold: float) -> bool:
        try:
            synthesize(spec_of(hold), params, f_z_bias=bias)
        except ValueError:
            return False
        return True

    lo, hi = 1.0 * us, 2000.0 * us
    assert fits(lo) and not fits(hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return 0.5 * (lo + hi)


def test_the_auto_pre_bias_doubles_the_feasible_hold(params1030):
    r"""Bisected both ways: ``f_z_bias="auto"`` buys exactly twice the axial integral.

    Eq. S19 starts every channel at its carrier, so ``f_Z`` walks off one side of the band and
    only the *one-sided* headroom is usable (``docs/PLAN.md`` §1.5's 206 µs at ``Z = 10 µm``).
    Offsetting the start by ``-max f_Z / 2`` centres the walk, and the whole band is in play:
    412 µs.  The hold itself *overshoots* doubling by a hair (x2.0098) because the two 2 µs
    ramps carry a fixed ``dz * T / 2`` of integral each, which the bias does not scale: the
    hold is ``2 C / Z - r``, not ``2 (C / Z - r)``, so the ratio exceeds 2 for any ``r > 0``.
    """
    params = _linear(params1030)
    plain = _longest_feasible_hold(params, 0.0)
    biased = _longest_feasible_hold(params, "auto")
    assert biased / plain == pytest.approx(2.0, rel=0.02)

    # what actually doubles, exactly, is the axial integral the band can buy
    def integral(hold: float) -> float:
        spec = TrajectorySpec(
            array=ArraySpec(1, 1),
            moves=(Lift(10.0 * um, 2.0 * us), Hold(hold), Lift(-10.0 * um, 2.0 * us)),
        )
        return requested_z_integral(spec, params)

    assert integral(plain) == pytest.approx(max_z_integral(params), rel=1e-4)
    assert integral(biased) == pytest.approx(max_z_integral(params, biased=True), rel=1e-4)
    assert integral(biased) / integral(plain) == pytest.approx(2.0, rel=1e-4)
    assert max_z_integral(params, biased=True) == 2.0 * max_z_integral(params)


def test_the_pre_bias_is_common_to_all_four_channels_and_moves_no_trap(params1030):
    """A constant on all four channels cancels in every Table I quantity — asserted, not argued.

    ``X`` and ``Y`` are *differences* within a pair and ``Zbar``/``Delta F`` are built from
    chirps, so a common constant is invisible to all four.  What it does change is where in the
    band the drive sits, which is the entire point.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(
        array=ArraySpec(3, 2, DF, 1.3 * MHz),
        moves=(Lift(10.0 * um, 30.0 * us), Hold(100.0 * us), Lift(-10.0 * um, 30.0 * us)),
    )
    plain = synthesize(spec, params)
    biased = synthesize(spec, params, f_z_bias="auto")

    _, _, z = spec.compile()
    from aodl.waveform.synthesis import f_z_ramp

    bias = resolve_f_z_bias("auto", f_z_ramp(z, params), params)
    assert bias < 0.0
    probes = np.linspace(*plain.t_span, 129)
    for name in CHANNELS:
        for one, two in zip(plain.channels[name].tones, biased.channels[name].tones, strict=True):
            delta = np.asarray(two.freq(probes)) - np.asarray(one.freq(probes))
            np.testing.assert_allclose(delta, bias, rtol=0, atol=1e-6)

    for t in np.linspace(tau, spec.duration, 9):
        before = simulate(plain, [float(t)]).metrics[0]
        after = simulate(biased, [float(t)]).metrics[0]
        assert len(before) == len(after)
        for a, b in zip(before, after, strict=True):
            assert b.x == pytest.approx(a.x, abs=1e-6 * optics.waist0)
            assert b.y == pytest.approx(a.y, abs=1e-6 * optics.waist0)
            assert b.z_lab == pytest.approx(a.z_lab, abs=1e-6 * optics.rayleigh)

    # ... and the biased walk straddles the carrier instead of leaving it: with X = Y = 0 the
    # drive is f_Z alone, so it ends at +f_Z(T)/2 having started at -f_Z(T)/2.
    plain_end = float(plain.channels["Ax"].tones[0].freq(spec.duration))
    biased_end = float(biased.channels["Ax"].tones[0].freq(spec.duration))
    assert biased_end == pytest.approx(-bias, rel=1e-9)
    assert plain_end == pytest.approx(2.0 * biased_end, rel=1e-9)


def test_a_refusal_quotes_the_doubled_ceiling_and_the_option_that_buys_it(params1030):
    """The Eq. 1 error names both budgets: the one that applied, and the one a bias would give."""
    spec = TrajectorySpec(
        array=ArraySpec(1, 1),
        moves=(Lift(10.0 * um, 30.0 * us), Hold(600.0 * us), Lift(-10.0 * um, 30.0 * us)),
    )
    ceiling = max_z_integral(params1030)
    with pytest.raises(ValueError) as unbiased:
        synthesize(spec, params1030)
    message = str(unbiased.value)
    assert f"{ceiling:.4g} m.s" in message
    assert f"{2.0 * ceiling:.4g} m.s" in message
    assert "f_z_bias='auto'" in message

    with pytest.raises(ValueError) as biased:
        synthesize(spec, params1030, f_z_bias="auto")
    message = str(biased.value)
    assert f"{2.0 * ceiling:.4g} m.s" in message
    assert "already the pre-biased one" in message


def test_the_pre_bias_is_refused_where_it_would_do_nothing(params1030):
    """A fading-Shepard drive never lets ``f_Z`` leave its window, so a bias only eats band."""
    spec = TrajectorySpec(array=ArraySpec(1, 1), moves=SHORT_MOVES)
    with pytest.raises(ValueError, match="f_z_bias applies to plain Eq. S19 only"):
        synthesize(spec, params1030, shepard=ShepardConfig(DFY_FREE, DFY_FREE), f_z_bias="auto")
    with pytest.raises(ValueError, match="must be a number or 'auto'"):
        synthesize(spec, params1030, f_z_bias="centre")


# ================================================================== 2.4 the switch ramp


#: The array the switch-ramp checks use: 3x3 (odd, so no comb offset in the way) lifted far
#: enough that the ladder really slides, with a hold long enough to hold several hand-overs.
RAMP_SPEC = TrajectorySpec(
    array=ArraySpec(3, 3, DF, DF),
    moves=(Lift(6.0 * um, 40.0 * us), Hold(200.0 * us), Lift(-6.0 * um, 40.0 * us)),
)


def _rung_slide_rate(params) -> float:
    """``|gdot| = Z / (2 lens_scale)`` [Hz/s] during ``RAMP_SPEC``'s hold — the fade speed."""
    return 6.0 * um / (2.0 * params.lens_scale)


def _ladder_spec(m_x: int) -> TrajectorySpec:
    """``RAMP_SPEC``'s move on an ``m_x``-column ladder with a free (single-tweezer) y axis.

    One ladder at a time.  The y channels then carry Table II's single-tweezer row (``p = 1/2``
    on both), which has no rectangle and so is never ramped: every x column is scaled by one
    common y factor, which cancels out of the ramped-vs-plain comparisons below.
    """
    return replace(RAMP_SPEC, array=ArraySpec(m_x, 1, DF, 0.0))


def _x_columns(wfs: WaveformSet, optics, t: float, pitch: float, origin: float) -> dict[int, float]:
    """``{column index: summed |c|^2}`` at frame time ``t``, with **nothing** pruned.

    The default term cut drops the weakest lines, which is exactly what a column halfway
    through a ramp is; these comparisons are between two drives at ``1e-9``, so they take the
    whole census.  ``origin`` is where column ``0`` sits — half a pitch out for an even-``M``
    ladder, whose Eq. S19 lattice straddles the array centre.
    """
    terms = build_terms(wfs, float(t), term_prune=0.0)
    xc, _, _, _ = spot_params(terms, optics, 0.0)
    out: dict[int, float] = {}
    for x, c in zip(xc, terms.c, strict=True):
        key = int(round((float(x) - origin) / pitch))
        out[key] = out.get(key, 0.0) + float(abs(c) ** 2)
    return out


def test_the_ramp_wraps_the_p_zero_rectangles_and_nothing_else(params1030):
    """WO-20 §1: the ramp is Table II's rectangles' business, and no other window's.

    A ``cos^p`` window with ``p > 0`` already reaches zero smoothly at its own switching
    instant, so a gate there buys no continuity — and, being the window that does the fading,
    it is the one that would pay for the gate in flatness
    (:func:`test_a_ramp_moves_only_the_columns_the_ramping_rung_feeds`).
    """
    params = _linear(params1030)
    ramp = 3.0 * us
    array = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF, switch_ramp=ramp))

    assert {
        name: sorted({type(tone.env).__name__ for tone in cw.tones})
        for name, cw in array.channels.items()
    } == {
        "Ax": ["FadeZoneEnvelope"],  # Table II array row: p_A = 1, the shaped window
        "Ay": ["FadeZoneEnvelope"],
        "Bx": ["SwitchRamped"],  # ... and p_B = 0, the rectangle that steps
        "By": ["SwitchRamped"],
    }
    assert all(tone.env.ramp == ramp for tone in array.channels["Bx"].tones)
    assert all(tone.env.base.p == 0.0 for tone in array.channels["By"].tones)

    # a single-tweezer axis splits the fade (1/2, 1/2): no rectangle, hence no ramp at all
    single = replace(RAMP_SPEC, array=ArraySpec(1, 1))
    free = synthesize(single, params, shepard=ShepardConfig(DFY_FREE, DFY_FREE, switch_ramp=ramp))
    assert all(
        isinstance(tone.env, FadeZoneEnvelope) for cw in free.channels.values() for tone in cw.tones
    )


def test_the_ramp_removes_the_rectangles_steps_and_the_a_windows_never_had_any(params1030):
    r"""The splatter proxy: a **step** in ``A(t)`` is the broadband event, and only ``p = 0`` steps.

    Counted across every one of a rung's own ``|g| = g_outer`` crossings, as
    ``|A(t + eps) - A(t - eps)|`` against half the rectangle's own unit step (a ``cos^p``
    shoulder is steep there, but continuous, and shrinks with ``eps``).  Table II's rectangles
    step at all of theirs and the ramp removes every one; the ``A`` channels' windows step at
    none, ramped or not — which is why ramping them was never what cut the splatter (WO-19 F-2).
    """
    params = _linear(params1030)
    plain = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF))
    ramped = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF, switch_ramp=3.0 * us))

    def steps(wfs: WaveformSet, channel: str) -> int:
        t0, t1 = wfs.t_span
        count = 0
        for tone in wfs.channels[channel].tones:
            env = tone.env
            for t in env.crossing_times(env.g_outer):
                if not t0 + TIME_TOL < float(t) < t1 - TIME_TOL:
                    continue  # the span's own ends are not switches
                jump = abs(float(env.A(float(t) + 1e-12)) - float(env.A(float(t) - 1e-12)))
                count += int(jump > 0.5)
        return count

    assert steps(plain, "Bx") > 0 and steps(plain, "By") > 0
    assert steps(ramped, "Bx") == steps(ramped, "By") == 0
    for channel in ("Ax", "Ay"):
        assert steps(plain, channel) == steps(ramped, channel) == 0


def test_the_switch_ramp_is_a_raised_cosine_inside_the_rungs_own_switch_instants(params1030):
    r"""``A = A_fade sin^2(pi u / 2r)`` on the way in, mirrored on the way out, ``dA`` continuous.

    The ramps are anchored at the ``|g| = g_outer`` crossings and point *inwards*: the support
    of the envelope — and with it :func:`~aodl.waveform.shepard.shepard_band_bound`, the one
    claim the whole scheme rests on — is exactly what it was.
    """
    params = _linear(params1030)
    ramp = 3.0 * us
    ramped = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF, switch_ramp=ramp))
    plain = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF))

    tones = ramped.channels["Bx"].tones
    env = tones[len(tones) // 2].env
    assert isinstance(env, SwitchRamped)
    reference = plain.channels["Bx"].tones[len(tones) // 2].env
    assert isinstance(reference, FadeZoneEnvelope)

    t_on, t_off, rise, fall = env._gates
    live = np.nonzero(rise > 0.0)[0]
    assert live.size, "the probe rung should switch on inside the run"
    entry = float(t_on[live[0]])
    assert rise[live[0]] == pytest.approx(ramp)

    u = np.linspace(-0.2 * ramp, 1.2 * ramp, 29)
    want = np.where(u < 0.0, 0.0, np.sin(0.5 * np.pi * np.clip(u / ramp, 0.0, 1.0)) ** 2)
    np.testing.assert_allclose(np.asarray(env.A(entry + u)), want, atol=1e-12)
    # continuous value *and* slope at both ends of the ramp; the rectangle had neither
    assert float(env.dA(entry)) == 0.0
    assert float(env.dA(entry + ramp)) == 0.0
    assert float(reference.A(entry - 1e-12)) == 0.0
    assert float(reference.A(entry + 1e-12)) == 1.0

    # support untouched: still silent everywhere the window is closed
    probes = np.linspace(*ramped.t_span, 2001)
    silent = np.asarray(reference.A(probes)) == 0.0
    assert np.all(np.asarray(env.A(probes))[silent] == 0.0)
    assert np.all(np.asarray(env.A(probes)) <= np.asarray(reference.A(probes)) + 1e-15)


@pytest.mark.parametrize("m_x", [3, 4])
def test_the_switching_column_ramps_up_instead_of_switching_on(params1030, m_x):
    r"""Table II's ``p_B = 0`` column *appears*; with a ramp it *arrives*, over ``switch_ramp``.

    The column a rung is arriving at is fed by that rung and one ``A`` rung alone, so its
    brightness is the rung's envelope squared times a factor common to both drives: a step
    without the ramp, a monotone ``sin^4`` rise with it, and *exactly* ``sin^4(pi/4) = 1/4`` of
    the un-ramped column halfway through — the common factor cancels in that ratio, so it is an
    identity, not a tolerance.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    origin = 0.5 * pitch if m_x % 2 == 0 else 0.0
    ramp = 3.0 * us
    spec = _ladder_spec(m_x)
    ramped = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE, switch_ramp=ramp))
    plain = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE))

    tones = ramped.channels["Bx"].tones
    env = tones[len(tones) // 2].env
    t_on, _, rise, _ = env._gates
    entry = float(t_on[np.nonzero(rise > 0.0)[0][0]])
    frames = entry + 0.5 * tau + np.linspace(0.0, ramp, 21)  # the ramp itself, retarded by tau/2

    traces = [_x_columns(ramped, optics, float(t), pitch, origin) for t in frames]
    arriving = max(traces[-1], key=lambda d: traces[-1][d] - traces[0].get(d, 0.0))
    assert traces[0].get(arriving, 0.0) < 1e-9 * max(traces[0].values())  # dark at the switch

    def brightness(wfs, t: float) -> float:
        """The arriving column against the full-depth columns beside it (a common factor)."""
        cols = _x_columns(wfs, optics, t, pitch, origin)
        return cols.get(arriving, 0.0) / max(cols.values())

    with_ramp = np.array([cols[arriving] / max(cols.values()) for cols in traces])
    without = np.array([brightness(plain, float(t)) for t in frames])
    assert np.all(np.diff(with_ramp) >= -1e-12)  # monotone: it fades up, it does not flicker
    assert with_ramp[0] < 1e-9 and with_ramp[-1] == pytest.approx(without[-1], rel=1e-9)
    assert float(np.max(np.diff(with_ramp))) < 0.2  # ... over the whole ramp, not in one step
    assert float(np.min(without[1:])) > 0.9  # the rectangle was there in full from the instant
    # halfway through the ramp the new column carries sin^4(pi/4) = 1/4 of a full one
    middle = entry + 0.5 * tau + 0.5 * ramp
    assert brightness(ramped, middle) == pytest.approx(0.25 * brightness(plain, middle), rel=1e-9)


@pytest.mark.parametrize("m_x, n_paying", [(3, 1), (4, 1)], ids=["odd-M", "even-M"])
def test_a_ramp_moves_only_the_columns_the_ramping_rung_feeds(params1030, m_x, n_paying):
    r"""The WO-20 §1 ruling, measured: every other column is the un-ramped one, to ``1e-9``.

    A ramp is felt only by the columns its own rung feeds while it is inside the ramp.  The
    ``A`` window's outer boundary sits ``(M - 1)/2`` rung spacings inside the rectangle's, and
    with Table II's half-rung ``xi_B`` on an even-``M`` axis a ``B`` rung switches at an ``A``
    hand-over for **both** parities.  It therefore pairs with the entering (or leaving) ``A``
    rung too: **two** columns move, the switching one and the array's own edge column next to
    it, the latter by the law of :func:`test_the_edge_columns_pay_the_documented_rho_law`.
    (Under the WO-17 even-``M`` timing the rung switched mid-plateau and only one column moved —
    the same misphasing that let a requested column go dark.)

    Everything else — the interior of the array, both extended columns on the far side — is the
    un-ramped drive, term for term.  That is the whole point of scoping the ramp to ``p = 0``.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    origin = 0.5 * pitch if m_x % 2 == 0 else 0.0
    ramp = 3.0 * us
    spec = _ladder_spec(m_x)
    plain = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE))
    ramped = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE, switch_ramp=ramp))

    tones = ramped.channels["Bx"].tones
    env = tones[len(tones) // 2].env
    t_on, t_off, rise, fall = env._gates
    switches = np.concatenate([t_on[rise > 0.0], t_off[fall > 0.0]])
    assert switches.size == 2  # this rung enters and leaves inside the run

    for switch in switches:
        frames = float(switch) + 0.5 * tau + np.linspace(-1.25 * ramp, 1.25 * ramp, 21)
        reference = [_x_columns(plain, optics, float(t), pitch, origin) for t in frames]
        measured = [_x_columns(ramped, optics, float(t), pitch, origin) for t in frames]

        worst = {d: 0.0 for frame in reference for d in frame}
        depth = {d: 1.0 for d in worst}
        for want, got in zip(reference, measured, strict=True):
            full = max(want.values())
            for d in worst:
                worst[d] = max(worst[d], abs(got.get(d, 0.0) - want.get(d, 0.0)) / full)
                depth[d] = min(depth[d], want.get(d, 0.0) / full)
        moved = sorted(d for d, v in worst.items() if v > 1e-9)
        for d in set(worst) - set(moved):
            assert worst[d] == 0.0, f"column {d} moved by {worst[d]:.3e}"  # bit-identical
        assert len(moved) == n_paying + 1, f"columns {moved} moved at {switch / us:.3f} us"

        # one of them is the column the rung is switching (dark on one side of the instant);
        # the other is the array's own edge column at full depth that dips a little.
        switching = [d for d in moved if depth[d] < 1e-12]
        paying = [d for d in moved if depth[d] > 0.99]
        assert len(switching) == 1 and len(paying) == n_paying
        for d in paying:
            assert 1e-3 < worst[d] < 0.05  # not passing by being tiny; see the rho_r law


@pytest.mark.parametrize("m_x", [3, 4], ids=["odd-M", "even-M"])
def test_the_edge_columns_pay_the_documented_rho_law(params1030, m_x):
    r"""What a ramp costs on an ``M``-column ladder, measured against a law derived here.

    Both parities pay it: with Table II's half-rung ``xi_B`` on an even-``M`` axis a ``B`` rung
    switches at an ``A`` hand-over exactly as an odd-``M`` one does.

    The rung entering at ``t_on`` is precisely the partner of the ``A`` rung entering there, so
    an **edge** column reads ``cos^2 theta + sin^2 theta s(t)^2`` instead of ``1`` during the
    ramp.  With the ladder sliding at ``gdot`` the dying weight is ``sin^2(pi rho_r x)``, so the
    dip is

        ``max_x sin^2(pi rho_r x) (1 - sin^4(pi x / 2))  <=  (pi rho_r)^2 *``
        :data:`~aodl.waveform.shepard.SWITCH_FLATNESS_SHAPE`

    with ``rho_r = |gdot| r / delta_f`` and ``x`` the fraction of the ramp elapsed.  A
    ``tau``-scale ramp on a 1 MHz ladder is a *fast* fade by that measure (``rho_r = 0.34``) and
    pays 20 %; a 1 µs ramp pays 0.17 %.  Both are checked against the closed form on the same
    ``x`` grid the frames sample — not only against its small-``rho_r`` bound.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    slide = _rung_slide_rate(params)
    spec = _ladder_spec(m_x)
    columns = tuple(range(m_x))
    origin = -0.5 * (m_x - 1) * pitch  # column 0 is the first requested site (Eq. S19)

    # the shape factor, re-derived numerically rather than trusted
    fine = np.linspace(0.0, 1.0, 200001)
    assert float(np.max(fine**2 * (1.0 - np.sin(0.5 * np.pi * fine) ** 4))) == pytest.approx(
        SWITCH_FLATNESS_SHAPE, rel=1e-4
    )

    plain = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE))
    # Table II verbatim: the M columns carry the same light at every instant (p_A + p_B = 1)
    for t in np.linspace(tau, 200.0 * us, 120):
        want = _x_columns(plain, optics, float(t), pitch, origin)
        assert min(want[d] for d in columns) == pytest.approx(
            max(want[d] for d in columns), rel=1e-12
        )

    x = np.linspace(0.0, 1.0, 41)
    measured = {}
    for ramp in (1.0 * us, tau):
        wfs = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE, switch_ramp=ramp))
        tones = wfs.channels["Bx"].tones
        t_on, t_off, rise, fall = tones[len(tones) // 2].env._gates
        frames = np.concatenate([t_on[rise > 0.0][0] + x * ramp, t_off[fall > 0.0][0] - x * ramp])
        worst = 0.0
        for t in frames + 0.5 * tau:
            want = _x_columns(plain, optics, float(t), pitch, origin)
            got = _x_columns(wfs, optics, float(t), pitch, origin)
            worst = max(worst, max(1.0 - got[d] / want[d] for d in columns))
        measured[ramp] = worst

        rho_r = slide * ramp / DF
        law = float(np.max(np.sin(np.pi * rho_r * x) ** 2 * (1.0 - np.sin(0.5 * np.pi * x) ** 4)))
        bound = (np.pi * rho_r) ** 2 * SWITCH_FLATNESS_SHAPE
        assert worst == pytest.approx(law, rel=1e-3), f"ramp {ramp / us:g} us"
        assert worst <= bound + 1e-12, f"ramp {ramp / us:g} us: {worst:.4f} > {bound:.4f}"

    assert measured[1.0 * us] < 0.01  # a short ramp is nearly free
    assert measured[tau] > 0.05  # ... and a tau-scale one on a 1 MHz ladder is not


def test_switch_ramp_zero_is_table_ii_verbatim(params1030):
    """The default builds the plain window object, not a ramp of length zero."""
    params = _linear(params1030)
    plain = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF))
    explicit = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF, switch_ramp=0.0))
    _assert_same_drive(plain, explicit, np.linspace(*plain.t_span, 257))
    assert all(
        isinstance(tone.env, FadeZoneEnvelope)
        for cw in plain.channels.values()
        for tone in cw.tones
    )
    with pytest.raises(ValueError, match="switch_ramp must be finite and non-negative"):
        ShepardConfig(DF, DF, switch_ramp=-1.0)
    with pytest.raises(ValueError, match="SwitchRamped.ramp must be finite and positive"):
        SwitchRamped(base=plain.channels["Bx"].tones[0].env, ramp=0.0)


def test_the_ramps_shrink_to_fit_and_never_run_off_the_programmed_span(params1030):
    """Two edges the gate has to get right: a short live interval, and the ends of the drive.

    A rung live for less than ``2 r`` gets half-length ramps (rise and fall may touch, never
    overlap — the :class:`~aodl.waveform.tones.SmoothOnOff` rule), and an interval bounded by
    the *programmed span* rather than by a ``|g| = g_outer`` crossing is not ramped there: the
    drive itself begins and ends at those instants, so there is no switch to soften.
    """
    params = _linear(params1030)
    ramp = 60.0 * us  # far longer than several of the rungs are live for
    wfs = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF, switch_ramp=ramp))
    t0, t1 = wfs.t_span

    seen = {"clipped": False, "domain_edge": False}
    for tone in wfs.channels["Bx"].tones:
        env = tone.env
        assert isinstance(env, SwitchRamped)
        for t_on, t_off, rise, fall in zip(*env._gates, strict=True):
            width = t_off - t_on
            for edge, length in ((t_on, rise), (t_off, fall)):
                assert length <= min(ramp, 0.5 * width) + 1e-15
                if abs(edge - t0) < TIME_TOL or abs(edge - t1) < TIME_TOL:
                    assert length == 0.0  # the span's own end, not a switch
                    seen["domain_edge"] = True
                else:
                    assert length > 0.0
            if width < 2.0 * ramp:
                seen["clipped"] = True
                if rise > 0.0 and fall > 0.0:
                    # the two half-ramps meet at the midpoint, where the gate reaches 1 exactly
                    assert rise == pytest.approx(fall)
                    assert float(env.A(0.5 * (t_on + t_off))) == pytest.approx(1.0, abs=1e-12)
            probes = np.linspace(t_on, t_off, 97)
            values = np.asarray(env.A(probes))
            assert np.all(values >= 0.0) and np.all(values <= 1.0 + 1e-15)
    assert seen == {"clipped": True, "domain_edge": True}


def test_a_ramped_drive_says_so_rather_than_serializing_silently(params1030, tmp_path):
    """Schema v2 has no slot for a ramp, so :meth:`WaveformSet.save` refuses it by name."""
    params = _linear(params1030)
    ramped = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF, switch_ramp=2.0 * us))
    with pytest.raises(TypeError, match="cannot serialize envelope of type 'SwitchRamped'"):
        ramped.save(tmp_path / "ramped.npz")


# ============================================================ 2.5 the spot_table column


def test_spot_table_carries_both_readings_of_a_group_s_light(params1030):
    """``power`` (incoherent) and ``power_coherent`` (the exact Gram) are both columns now."""
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(array=ArraySpec(2, 2, DF, 1.3 * MHz), moves=SHORT_MOVES)
    result = simulate(synthesize(spec, params), np.linspace(tau, spec.duration, 5))

    table = result.spot_table()
    assert tuple(table) == SPOT_TABLE_KEYS
    assert "power_coherent" in SPOT_TABLE_KEYS
    flat = [m for frame in result.metrics for m in frame]
    np.testing.assert_allclose(table["power"], [m.power for m in flat], rtol=0, atol=0)
    np.testing.assert_allclose(
        table["power_coherent"], [m.power_coherent for m in flat], rtol=0, atol=0
    )
    # a non-degenerate array reads the same either way: nothing overlaps
    np.testing.assert_allclose(table["power_coherent"], table["power"], rtol=1e-9)
