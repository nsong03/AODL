r"""The M5 synthesis options (``docs/workorders/WO-17-product-core.md`` §2).

Four independent knobs on :func:`aodl.waveform.synthesis.synthesize`, plus the ``spot_table``
column that goes with them.  Every one of them is checked twice over: that it *does* what it
claims, and that switching it off reproduces the pre-M5 drive bit for bit.

1. **§2.1 lattice alignment.**  A Shepard array's columns sit at the *index differences* of the
   ``A`` and ``B`` ladders — integer multiples of the pitch — while Eq. S19 centres a counted
   ladder, which is half-integer multiples for even ``M``.  The
   :func:`~aodl.waveform.shepard.lattice_comb_offset` correction adds ``delta_f/2`` to the
   even-``M`` ``B`` rung *frequencies* and to nothing else, so the two modes put the traps in
   the same places.  This one is a fix, not an option: it is on by default (WO-16 finding F-2),
   and ``comb_offset={}`` is the escape hatch back to the uncorrected comb.
2. **§2.2 retardation pre-compensation.**  Reading the trajectory ``tau/2`` ahead makes the
   atom plane match the request at ``t`` instead of at ``t - tau/2``.
3. **§2.3 the ``f_Z`` edge pre-bias.**  Starting the drive half an excursion below the carrier
   centres the ``f_Z`` walk in the band and doubles Eq. 1's budget — ``docs/PLAN.md`` §1.5's
   412 µs — while moving no trap, because the bias is common to all four channels.
4. **§2.4 ``switch_ramp``.**  Raised-cosine on/off ramps for the ``p_B = 0`` rectangles, which
   trade a little interior-column flatness (by a law derived here) for a continuous switch.

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


@pytest.mark.parametrize("m_x", [2, 3, 4, 5])
def test_the_shepard_lattice_coincides_with_the_s19_lattice_for_even_and_odd_m(params1030, m_x):
    r"""Same spec, both modes, same trap positions — to float64, for ``M = 2, 3, 4, 5``.

    Eq. S19's ladder is ``f^{(n)} = (n - (M-1)/2) delta_f``: for odd ``M`` that is integer
    multiples of ``delta_f`` and for even ``M`` half-integer ones.  A Shepard array's columns
    come from index differences and are therefore *always* integers, so ``shepard="auto"`` used
    to move an even-``M`` user array half a pitch sideways.  The comb offset closes that gap,
    and the escape hatch ``comb_offset={}`` shows exactly how wide it was.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    spec = TrajectorySpec(array=ArraySpec(m_x, 1, DF, 0.0), moves=SHORT_MOVES)
    probes = np.linspace(tau, spec.duration, 9)

    s19 = synthesize(spec, params)
    fixed = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE))
    uncorrected = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE, comb_offset={}))

    worst_fixed = max(
        _mismatch(_columns(s19, optics, t), _columns(fixed, optics, t)) for t in probes
    )
    worst_old = max(
        _mismatch(_columns(s19, optics, t), _columns(uncorrected, optics, t)) for t in probes
    )
    assert worst_fixed < 0.01 * optics.waist0
    assert worst_fixed < 1e-12 * pitch  # ... in fact exact: one constant, added to one comb
    if m_x % 2 == 0:
        assert worst_old == pytest.approx(0.5 * pitch, rel=1e-9)  # the F-2 half-pitch offset
    else:
        assert worst_old == worst_fixed  # odd M never needed the correction


def test_the_comb_offset_moves_frequencies_and_leaves_every_other_structure_alone(params1030):
    """``g``, the windows, the schedule, the rung count and the ``A`` channels are untouched.

    That is the whole argument for putting the correction on the frequency instead of on the
    fade coordinate: the hand-over schedule depends on ``g`` alone, and co-location depends on
    index *differences*, so a constant common to every ``B`` rung cannot disturb either.
    """
    params = _linear(params1030)
    spec = TrajectorySpec(array=ArraySpec(4, 4, DF, 1.3 * MHz), moves=SHORT_MOVES)
    fixed = synthesize(spec, params, shepard=ShepardConfig(DF, 1.3 * MHz))
    uncorrected = synthesize(spec, params, shepard=ShepardConfig(DF, 1.3 * MHz, comb_offset={}))
    probes = np.linspace(*fixed.t_span, 401)

    expected = {"Ax": 0.0, "Ay": 0.0, "Bx": 0.5 * DF, "By": 0.5 * 1.3 * MHz}
    for name in CHANNELS:
        left, right = fixed.channels[name].tones, uncorrected.channels[name].tones
        assert len(left) == len(right)  # same ladder, same live rungs
        for one, two in zip(left, right, strict=True):
            # the frequency laws differ by exactly the comb offset, everywhere
            delta = np.asarray(one.freq(probes)) - np.asarray(two.freq(probes))
            np.testing.assert_allclose(delta, expected[name], rtol=0, atol=1e-6)
            # ... and the envelopes are the very same object's values, bit for bit
            np.testing.assert_array_equal(
                np.asarray(one.env.A(probes)), np.asarray(two.env.A(probes))
            )
            np.testing.assert_array_equal(one.env.g.coeffs, two.env.g.coeffs)
            assert one.phase0 == two.phase0

    # Table II's offsets themselves: delta_f/2 on an even-M B ladder, nothing anywhere else.
    fades = ShepardConfig(DF, 1.3 * MHz).resolve(spec.array)
    assert ShepardConfig(DF, 1.3 * MHz).comb_offsets(fades) == expected
    assert lattice_comb_offset(ChannelFade(m=3, p=0.0, xi=0.0), DF) == 0.0
    assert lattice_comb_offset(ChannelFade(m=4, p=0.0, xi=0.0), DF) == 0.5 * DF


def test_the_lattice_correction_leaves_power_flatness_and_shadow_offsets_alone(params1030):
    """Regression against the pre-M5 numbers: the same light, half a pitch across.

    Both the constant-power identity (``p_A + p_B = 1``) and the Eq. S31 shadow offset are
    statements about index *differences*, so a rigid translation of the ``B`` comb cannot
    change either — and the measured per-node brightnesses agree term for term.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    spec = TrajectorySpec(
        array=ArraySpec(4, 1, DF, 0.0),
        moves=(Lift(6.0 * um, 40.0 * us), Hold(80.0 * us), Lift(-6.0 * um, 40.0 * us)),
    )
    fixed = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE))
    uncorrected = synthesize(spec, params, shepard=ShepardConfig(DF, DFY_FREE, comb_offset={}))
    grid = (pitch, params.deflection_scale * DFY_FREE)

    for t in np.linspace(tau, spec.duration, 24):
        # the corrected lattice is the old one shifted by exactly half a pitch, node by node
        new = _nodes(fixed, optics, float(t), grid, origin=(0.5 * pitch, 0.0))
        old = _nodes(uncorrected, optics, float(t), grid)
        assert set(new) == set(old)
        for key, weight in old.items():
            assert new[key] == pytest.approx(weight, rel=1e-12)
        # ... and the constant-power identity still holds: the columns fed by both live A rungs
        # (every one but the two ends of the extended grid) carry exactly the same light.
        brightest = max(new.values())
        flat = [w for w in new.values() if w == pytest.approx(brightest, rel=1e-9)]
        assert len(flat) >= spec.array.mx - 1


def test_the_shepard_band_bound_gains_the_comb_offset(params1030):
    """The live window translates with the comb, so the bound it obeys grows by ``|c_comb|``."""
    params = _linear(params1030)
    fade = ChannelFade(m=4, p=0.0, xi=0.0)
    plain = shepard_band_bound(fade, DF, 0.5, 2.0 * MHz)
    shifted = shepard_band_bound(fade, DF, 0.5, 2.0 * MHz, 0.5 * DF)
    assert plain == pytest.approx(0.5 * 4.5 * DF + 2.0 * MHz)
    assert shifted == pytest.approx(plain + 0.5 * DF)

    # ... and a drive that only just fits without the offset is refused with it.
    spec = TrajectorySpec(array=ArraySpec(12, 1, 1.5 * MHz, 0.0), moves=SHORT_MOVES)
    cfg = ShepardConfig(1.5 * MHz, DFY_FREE)
    synthesize(spec, params, shepard=replace(cfg, comb_offset={}))
    with pytest.raises(ValueError, match="comb offset that puts an even-M ladder"):
        synthesize(spec, params, shepard=cfg)


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


def test_the_extended_column_ramps_up_instead_of_switching_on(params1030):
    r"""Table II's ``p_B = 0`` column *appears*; with a ramp it *arrives*, over ``switch_ramp``.

    The ``(Mx + 2)`` extended column is fed by a single ``(A, B)`` combination whose ``B`` rung
    has just entered the rectangle, so its brightness is that rung's envelope squared: a step
    without the ramp, a monotone ``sin^4`` rise with it.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    ramp = 3.0 * us
    ramped = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF, switch_ramp=ramp))
    plain = synthesize(RAMP_SPEC, params, shepard=ShepardConfig(DF, DF))

    tones = ramped.channels["Bx"].tones
    env = tones[len(tones) // 2].env
    t_on, _, rise, _ = env._gates
    entry = float(t_on[np.nonzero(rise > 0.0)[0][0]])
    frames = entry + 0.5 * tau + np.linspace(-0.25 * ramp, 1.25 * ramp, 25)

    def extra(wfs, t: float) -> float:
        nodes = _nodes(wfs, optics, t, (pitch, pitch))
        interior = max(nodes.get((i, 0), 0.0) for i in (-1, 0, 1))
        return sum(nodes.get((i, 0), 0.0) for i in (-2, 2)) / interior

    with_ramp = np.array([extra(ramped, float(t)) for t in frames])
    without = np.array([extra(plain, float(t)) for t in frames])
    assert np.all(np.diff(with_ramp) >= -1e-12)  # monotone: it fades up, it does not flicker
    assert with_ramp[0] == 0.0 and with_ramp[-1] > 0.9
    assert float(np.max(np.diff(with_ramp))) < 0.2  # ... over the whole ramp, not in one step
    assert float(np.max(np.diff(without))) > 0.9  # the rectangle switches on all at once
    # halfway through the ramp the new column carries sin^4(pi/4) = 1/4 of a full one
    middle = entry + 0.5 * tau + 0.5 * ramp
    assert extra(ramped, middle) == pytest.approx(0.25 * extra(plain, middle), rel=0.02)


def test_a_ramp_costs_interior_flatness_only_by_the_documented_rho_law(params1030):
    r"""What a ramp buys and what it costs, both measured against a law derived here.

    The constant-power identity needs the ``B`` rectangle flat wherever the ``A`` window fades,
    and the rung entering at ``t_on`` is precisely the partner of the ``A`` rung entering there.
    So an interior column reads ``cos^2 theta + sin^2 theta s(t)^2`` during a ramp, and with the
    ladder sliding at ``gdot`` the dip is bounded by

        ``max_x sin^2(pi rho_r x) (1 - sin^4(pi x / 2))  <=  (pi rho_r)^2 *``
        :data:`~aodl.waveform.shepard.SWITCH_FLATNESS_SHAPE`

    with ``rho_r = |gdot| r / delta_f``.  A ``tau``-scale ramp on a 1 MHz ladder is a *fast*
    fade by that measure (``rho_r = 0.34``) and pays 15 %; a 1 µs ramp pays 0.1 %.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    pitch = params.deflection_scale * DF
    slide = _rung_slide_rate(params)

    # the shape factor, re-derived numerically rather than trusted
    x = np.linspace(0.0, 1.0, 200001)
    assert float(np.max(x**2 * (1.0 - np.sin(0.5 * np.pi * x) ** 4))) == pytest.approx(
        SWITCH_FLATNESS_SHAPE, rel=1e-4
    )

    probes = np.linspace(tau, 200.0 * us, 400)
    measured = {}
    for ramp in (0.0, 1.0 * us, tau):
        cfg = ShepardConfig(DF, DF, switch_ramp=ramp)
        wfs = synthesize(RAMP_SPEC, params, shepard=cfg)
        worst = 0.0
        for t in probes:
            nodes = _nodes(wfs, optics, float(t), (pitch, pitch))
            reference = max(nodes.values())
            interior = min(nodes.get((i, j), 0.0) for i in (-1, 0, 1) for j in (-1, 0, 1))
            worst = max(worst, 1.0 - interior / reference)
        measured[ramp] = worst
        bound = (np.pi * slide * ramp / DF) ** 2 * SWITCH_FLATNESS_SHAPE
        assert worst <= bound + 1e-9, f"ramp {ramp / us:g} us: {worst:.4f} > {bound:.4f}"

    assert measured[0.0] < 1e-9  # Table II verbatim is flat, as M4 measured it
    assert measured[1.0 * us] < 0.01  # a short ramp is free
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
