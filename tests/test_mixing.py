r"""M2 acceptance for intra-AOD intermodulation and compression (Eqs. S20-S22).

The arbiter in this file is a **frozen-time projection**, the mixing analogue of
``field/reference.py``: at one frame time the channel's transmission is built *literally*
on a fine aperture grid,

    V(u) = sum_n A_n(t_ret(u)) cos(Phi_n(t_ret(u))),      P(u) = exp(i C V(u)),

with ``t_ret(u) = t_c - s u / v`` and ``Phi_n = 2 pi f_center t' + phase_n(t')`` — no
expansion, no Bessel functions, no Taylor step.  A line whose tone coefficients are
``q_n`` (integers with ``sum q_n = 1``, the ``+1`` diffraction band of
``docs/conventions.md`` §3-4) is then read out by the inner product

    amp = < P, exp(-i Phi_line) > * exp(+i 2 pi f_center t_c),
    Phi_line(u) = sum_n q_n [Phi_n(u) - Phi_n(t_c)]

over a window holding a whole number of periods of every frequency present, which makes
the different bands and lines exactly orthogonal (no FFT, no fitting).  Subtracting
``Phi_n(t_c)`` and restoring the carrier constant put the answer in the gauge
``device/aod.py`` uses for :attr:`~aodl.device.aod.Lines.amp` — the tone phase is measured
at the beam-center retarded time and the common ``exp(-i 2 pi f_center t_c)`` is dropped.

Everything the perturbative expansion claims — the ``i J_1(m)`` compression of a
fundamental, the ``-(i/8) m_i m_j m_k`` IM3 ghosts, the coherent collision of degenerate
products — is checked against that projection, and its residual is shown to fall as
``m^4`` (fundamentals) / ``m^2`` (ghosts), i.e. as the first neglected order.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j1

from aodl.device.aod import channel_lines
from aodl.device.aodl import build_terms
from aodl.device.conventions import geometry, retarded_time
from aodl.device.mixing import MixingConfig, expand_lines
from aodl.params import AODParams
from aodl.poly import PiecewisePoly
from aodl.units import MHz, ms
from aodl.waveform.tones import (
    ChannelWaveform,
    ConstantEnvelope,
    SmoothOnOff,
    ToneTrack,
    WaveformSet,
)

CHANNEL = "Ay"
#: Long enough that every frame time used here sits well inside the programmed span.
SPAN = 20.0


# --------------------------------------------------------------------------- construction


def _static_channel(detunings, phases=None, amps=None, envs=None, span=SPAN, tau=1.0):
    """Constant-frequency tones with the given detunings, phases and envelope amplitudes."""
    n = len(detunings)
    phases = [0.0] * n if phases is None else phases
    amps = [1.0] * n if amps is None else amps
    envs = [None] * n if envs is None else envs
    tracks = []
    for detuning, phase0, amp, env in zip(detunings, phases, amps, envs, strict=True):
        tracks.append(
            ToneTrack(
                freq=PiecewisePoly.constant(detuning, 0.0, span * tau),
                env=ConstantEnvelope(amp) if env is None else env,
                phase0=phase0,
            )
        )
    return ChannelWaveform(tuple(tracks))


def _aod(params, **changes) -> AODParams:
    """The ``Ay`` hardware with fields overridden (frozen dataclass, so a copy)."""
    return replace(params.channels[CHANNEL], **changes)


def _waveform_set(params, cw, aod: AODParams):
    """A one-channel :class:`WaveformSet` whose hardware snapshot is ``aod`` everywhere."""
    return WaveformSet(
        channels={CHANNEL: cw},
        params=replace(params, channels={name: aod for name in params.channels}),
    )


def _line_at(lines, detuning: float, tol: float = 1.0) -> int:
    """Index of the single line at ``detuning`` [Hz] (fails if absent or ambiguous)."""
    hits = np.flatnonzero(np.abs(lines.f - detuning) <= tol)
    assert hits.size == 1, f"expected one line at {detuning / MHz:+.3f} MHz, got {hits.size}"
    return int(hits[0])


# ----------------------------------------------------------------- the reference projection


def _projection_amplitude(cw, aod: AODParams, t: float, coeffs, base: float, n: int = 1 << 15):
    """Complex amplitude of the ``+1``-band line with tone coefficients ``coeffs``.

    Brute-force reference for Eqs. S20-S22 (module docstring): no expansion is used, only
    ``exp(i C V)`` on a grid and one inner product.  Requires static tones and constant
    envelopes so that the pupil is periodic in ``u`` with period ``v / base``, and every
    absolute tone frequency an integer multiple of ``base`` so the window holds a whole
    number of periods of every line *and* of every other diffraction band.
    """
    geom = geometry(CHANNEL)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    assert coeffs.sum() == pytest.approx(1.0), "a +1-band line has sum(q) = 1"
    u = np.arange(n) * (aod.sound_speed / base / n)
    t_ret = retarded_time(t, u, geom, aod)
    t_c = float(t - 0.5 * aod.transit_time)

    harmonics = np.array([(aod.f_center + float(tone.f(t_c))) / base for tone in cw.tones])
    np.testing.assert_allclose(harmonics, np.round(harmonics), atol=1e-9)

    phase = np.array([np.asarray(tone.phase(t_ret), dtype=np.float64) for tone in cw.tones])
    phase += 2.0 * math.pi * aod.f_center * t_ret  # Phi_n(u), carrier included
    envelope = np.array([np.asarray(tone.env.A(t_ret), dtype=np.float64) for tone in cw.tones])
    pupil = np.exp(1j * aod.drive_strength * np.sum(envelope * np.cos(phase), axis=0))
    line_phase = coeffs @ phase
    projected = np.mean(pupil * np.exp(1j * (line_phase - line_phase[0])))  # u = 0 is t_c
    return complex(projected * np.exp(1j * 2.0 * math.pi * aod.f_center * t_c))


def _projected_lines(cw, aod, t, coeffs_by_detuning, base):
    """``{detuning: projected amplitude}`` for a whole line table."""
    return {
        detuning: _projection_amplitude(cw, aod, t, coeffs, base)
        for detuning, coeffs in coeffs_by_detuning.items()
    }


#: Tone coefficients ``q_n`` of every +1-band line of three tones at (-2, 0, +2) MHz, keyed
#: by detuning [MHz].  Degenerate signatures (two ways to reach +/-4 MHz, and the IM3
#: landing on each fundamental) share a key: the entry is the one the frozen-time
#: projection can address, i.e. the line's *total* is what the model must reproduce.
LADDER3_SIGNATURES = {
    -6.0: (2, 0, -1),
    -4.0: (1, 1, -1),
    -2.0: (1, 0, 0),
    0.0: (0, 1, 0),
    +2.0: (0, 0, 1),
    +4.0: (-1, 1, 1),
    +6.0: (-1, 0, 2),
}


# =============================================================== 1. parameter plumbing


def test_mixing_order_is_validated(params1030) -> None:
    """``AODParams.mixing_order`` and ``MixingConfig`` accept only the implemented orders."""
    assert params1030.channels[CHANNEL].mixing_order in (1, 3)
    with pytest.raises(ValueError, match="mixing_order must be 1 or 3"):
        _aod(params1030, mixing_order=2)
    with pytest.raises(ValueError, match="mixing order must be one of"):
        MixingConfig(order=2)
    with pytest.raises(ValueError, match="band_margin"):
        MixingConfig(band_margin=-0.1)
    with pytest.raises(ValueError, match="line_prune"):
        MixingConfig(line_prune=-1e-9)


def test_channel_mixing_order_selects_the_expansion(params1030) -> None:
    """``mixing=None`` follows ``aod.mixing_order``; an explicit config overrides it."""
    aod1 = _aod(params1030, mixing_order=1)
    aod3 = _aod(params1030, mixing_order=3)
    tau = aod1.transit_time
    cw = _static_channel([-1.0 * MHz, 1.0 * MHz], tau=tau)
    t = 2.0 * tau

    assert channel_lines(cw, aod1, t).n_lines == 2
    assert channel_lines(cw, aod3, t).n_lines == 4  # + 2 degenerate IM3 at +/-3 MHz
    assert channel_lines(cw, aod1, t, MixingConfig(order=3)).n_lines == 4
    assert channel_lines(cw, aod3, t, MixingConfig(order=1)).n_lines == 2


# =============================================================== 2. the Bessel identity


@pytest.mark.parametrize("depth", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
def test_single_tone_reproduces_bessel_j1(params1030, depth) -> None:
    """One tone at order 3 gives ``(i/2) m (1 - m^2/8)`` — the first two terms of ``i J_1(m)``."""
    aod = _aod(params1030, drive_strength=depth, mixing_order=3)
    tau = aod.transit_time
    t, phase0 = 0.5 * tau, 0.4  # t_c = 0, so phase(t_c) = phase0
    cw = _static_channel([2.0 * MHz], phases=[phase0], tau=tau)

    lines = channel_lines(cw, aod, t)
    assert lines.n_lines == 1
    assert lines.pruned_power == 0.0

    truncated = 0.5j * depth * (1.0 - depth**2 / 8.0) * np.exp(-1j * phase0)
    assert lines.amp[0] == pytest.approx(complex(truncated), rel=1e-14)

    exact = 1j * j1(depth) * np.exp(-1j * phase0)  # i J_1(m) e^{-i phi}
    assert abs(lines.amp[0] / exact - 1.0) < depth**4 / 100.0


def test_two_equal_tones_compress_each_other(params1030) -> None:
    """Two tones of depth ``m``: the fundamental drops to ``(i/2) m (1 - m^2/8 - m^2/4)``.

    The ``m^2/8`` is self-compression (``i J_1``), the ``m^2/4`` is the neighbour's
    ``J_0(m) = 1 - m^2/4`` — the cross term of Eq. S21.  Checked against the projection.
    """
    depth = 0.2
    aod = _aod(params1030, drive_strength=depth, mixing_order=3)
    tau = aod.transit_time
    t = 0.5 * tau
    phases = [0.31, -1.17]
    cw = _static_channel([-1.0 * MHz, 1.0 * MHz], phases=phases, tau=tau)

    lines = channel_lines(cw, aod, t)
    assert lines.n_lines == 4  # 2 fundamentals + 2 degenerate IM3 at +/-3 MHz

    for detuning, phase0 in zip([-1.0, 1.0], phases, strict=True):
        expected = 0.5j * depth * (1.0 - depth**2 / 8.0 - depth**2 / 4.0) * np.exp(-1j * phase0)
        assert lines.amp[_line_at(lines, detuning * MHz)] == pytest.approx(expected, rel=1e-14)

    # -(i/16) m_i m_j^2 at 2 f_j - f_i, with the phase 2 phi_j - phi_i.
    ghost = -1j / 16.0 * depth**3 * np.exp(-1j * (2.0 * phases[1] - phases[0]))
    assert lines.amp[_line_at(lines, 3.0 * MHz)] == pytest.approx(ghost, rel=1e-14)

    signatures = {-3.0: (2, -1), -1.0: (1, 0), 1.0: (0, 1), 3.0: (-1, 2)}
    projected = _projected_lines(cw, aod, t, signatures, 1.0 * MHz)
    scale = abs(lines.amp[_line_at(lines, 1.0 * MHz)])
    for detuning, reference in projected.items():
        model = lines.amp[_line_at(lines, detuning * MHz)]
        assert abs(model - reference) < 1e-3 * scale


# =============================================================== 3. frozen-time projection


def test_frozen_time_projection_matches_every_line(params1030) -> None:
    """Three tones, three unequal phases: every line agrees with ``exp(i C V)`` itself.

    Tolerance is ``1e-3`` of the strongest fundamental at ``m = 0.2`` (WO-07 §4), which the
    IM3 ghosts meet with room to spare even though their own relative error is ``O(m^2)``.
    """
    depth = 0.2
    aod = _aod(params1030, drive_strength=depth, mixing_order=3)
    tau = aod.transit_time
    t = 0.5 * tau
    cw = _static_channel(
        [-2.0 * MHz, 0.0, 2.0 * MHz], phases=[0.31, -1.17, 2.05], amps=[1.0, 1.0, 1.0], tau=tau
    )

    lines = channel_lines(cw, aod, t)
    assert lines.n_lines == len(LADDER3_SIGNATURES)

    projected = _projected_lines(cw, aod, t, LADDER3_SIGNATURES, 2.0 * MHz)
    scale = max(abs(lines.amp[_line_at(lines, f * MHz)]) for f in (-2.0, 0.0, 2.0))
    worst = 0.0
    for detuning, reference in projected.items():
        model = complex(lines.amp[_line_at(lines, detuning * MHz)])
        worst = max(worst, abs(model - reference))
        assert abs(model - reference) < 1e-3 * scale, f"line at {detuning:+.1f} MHz"
    assert worst > 0.0  # the comparison really ran


def test_projection_residual_falls_as_the_first_neglected_order(params1030) -> None:
    """Halving ``m`` shrinks the truncation residual 16x (``m^4``) — it is the ``m^5`` term.

    This is what pins the *coefficients* rather than merely the line positions: a wrong
    factor in any row of the WO-07 §2 table would leave a residual that does not vanish
    faster than the line it belongs to.
    """
    detunings = [-2.0 * MHz, 0.0, 2.0 * MHz]
    phases = [0.31, -1.17, 2.05]
    residuals = []
    ghost_relative = []
    for depth in (0.2, 0.1):
        aod = _aod(params1030, drive_strength=depth, mixing_order=3)
        tau = aod.transit_time
        t = 0.5 * tau
        cw = _static_channel(detunings, phases=phases, tau=tau)
        lines = channel_lines(cw, aod, t)
        scale = abs(lines.amp[_line_at(lines, 0.0)])
        projected = _projected_lines(cw, aod, t, LADDER3_SIGNATURES, 2.0 * MHz)
        worst = max(
            abs(complex(lines.amp[_line_at(lines, f * MHz)]) - reference)
            for f, reference in projected.items()
        )
        residuals.append(worst / scale)
        ghost = _line_at(lines, 4.0 * MHz)
        ghost_relative.append(
            abs(complex(lines.amp[ghost]) - projected[4.0]) / abs(lines.amp[ghost])
        )

    assert residuals[0] / residuals[1] == pytest.approx(16.0, rel=0.25)  # m^4
    assert ghost_relative[0] / ghost_relative[1] == pytest.approx(4.0, rel=0.25)  # m^2
    assert ghost_relative[0] < 0.05


# =============================================================== 4. ghost bookkeeping


def test_ghost_bookkeeping_of_an_equally_spaced_triplet(params1030) -> None:
    """Three tones at (-2, 0, +2) MHz: ghosts at +/-4, +/-6 MHz and *on* the fundamentals.

    The ladder-edge ghost collects two signatures — ``f_0 + f_{+2} - f_{-2}`` at
    ``-(i/8) m^3`` and ``2 f_{+2} - f_0`` at ``-(i/16) m^3`` — which are the *same*
    frequency and therefore add coherently: ``-(3i/16) m^3`` (WO-07 §4).
    """
    depth = 0.3
    aod = _aod(params1030, drive_strength=depth, mixing_order=3)
    tau = aod.transit_time
    t = 0.5 * tau  # t_c = 0 and all phase0 = 0, so every amplitude is a pure number
    cw = _static_channel([-2.0 * MHz, 0.0, 2.0 * MHz], tau=tau)

    lines = channel_lines(cw, aod, t)

    # 3 fundamentals + 3 distinct-triple IM3 + 6 degenerate IM3 = 12 raw lines, colliding
    # onto 7 distinct frequencies once the degenerate ones are merged.
    assert lines.n_lines == 7
    np.testing.assert_allclose(
        np.sort(lines.f), np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]) * MHz, atol=1e-6
    )

    for edge in (-4.0, 4.0):
        assert lines.amp[_line_at(lines, edge * MHz)] == pytest.approx(
            -3j / 16.0 * depth**3, rel=1e-14
        )
    for outer in (-6.0, 6.0):
        assert lines.amp[_line_at(lines, outer * MHz)] == pytest.approx(
            -1j / 16.0 * depth**3, rel=1e-14
        )

    # The fundamentals carry their compression *and* the IM3 that landed on them:
    #   f = 0     <- (f_{-2} + f_{+2} - f_0), a distinct triple:  -(i/8) m^3
    #   f = +/-2  <- (2 f_0 - f_{-/+2}), degenerate:              -(i/16) m^3
    # Compression of one tone among three: m^2/8 from itself, m^2/4 from each neighbour.
    compressed = 0.5j * depth * (1.0 - depth**2 / 8.0 - 2.0 * depth**2 / 4.0)
    assert lines.amp[_line_at(lines, 0.0)] == pytest.approx(
        compressed - 1j / 8.0 * depth**3, rel=1e-14
    )
    for inner in (-2.0, 2.0):
        assert lines.amp[_line_at(lines, inner * MHz)] == pytest.approx(
            compressed - 1j / 16.0 * depth**3, rel=1e-14
        )

    # ... and the projection agrees with all seven totals.
    projected = _projected_lines(cw, aod, t, LADDER3_SIGNATURES, 2.0 * MHz)
    scale = abs(lines.amp[_line_at(lines, 0.0)])
    for detuning, reference in projected.items():
        assert abs(complex(lines.amp[_line_at(lines, detuning * MHz)]) - reference) < 5e-3 * scale


def test_degenerate_lines_merge_instead_of_duplicating(params1030) -> None:
    """A merged line is one line: same ``df_opt``, one term, amplitudes summed coherently."""
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    t = 2.0 * tau
    cw = _static_channel([-2.0 * MHz, 0.0, 2.0 * MHz], tau=tau)

    terms = build_terms(_waveform_set(params1030, cw, aod), t, channels=(CHANNEL,))
    assert terms.n_terms == 7
    assert np.unique(np.round(terms.df_opt, 3)).size == 7

    # Without the merge the +4 MHz ghost would appear twice, each weaker than the total.
    unmerged = -1j / 8.0 * 0.3**3
    total = terms.c[int(np.argmin(np.abs(terms.df_opt - 4.0 * MHz)))]
    assert abs(total) > abs(unmerged)


# =============================================================== 5. band acceptance


def test_band_acceptance_drops_out_of_band_products(params1030) -> None:
    """Products outside the widened band are never launched; the margin sets the edge."""
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    t = 2.0 * tau
    lo, hi = aod.band
    assert (lo, hi, aod.f_center) == (90.0 * MHz, 110.0 * MHz, 100.0 * MHz)
    cw = _static_channel([3.0 * MHz, 9.0 * MHz, 10.0 * MHz], tau=tau)

    # margin 0.2 of a 20 MHz band -> detunings accepted out to +/-14 MHz.
    wide = channel_lines(cw, aod, t, MixingConfig(order=3, band_margin=0.2))
    np.testing.assert_allclose(
        np.sort(wide.f), np.array([-4.0, -3.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0]) * MHz, atol=1.0
    )
    # 2 f_10 - f_3 = 17, 2 f_9 - f_3 = 15 and f_9 + f_10 - f_3 = 16 MHz are all gone.
    for absent in (15.0, 16.0, 17.0):
        assert not np.any(np.isclose(wide.f, absent * MHz, atol=1.0))

    # Shrinking the margin to zero also rejects the 11 MHz product.
    narrow = channel_lines(cw, aod, t, MixingConfig(order=3, band_margin=0.0))
    np.testing.assert_allclose(
        np.sort(narrow.f), np.array([-4.0, -3.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0]) * MHz, atol=1.0
    )
    assert narrow.pruned_power == 0.0  # band rejection is not an approximation


def test_band_acceptance_never_drops_a_programmed_tone(params1030) -> None:
    """A fundamental outside the band stays: the cut is a statement about *products*.

    Dropping it would make the simulation lose a tweezer the user asked for (and would
    make order 3 stop being a superset of order 1); out-of-band *drive* is the waveform
    synthesizer's business, not the mixing model's.
    """
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    cw = _static_channel([0.0, 20.0 * MHz], tau=tau)

    lines = channel_lines(cw, aod, 2.0 * tau)
    np.testing.assert_allclose(np.sort(lines.f), [0.0, 20.0 * MHz], atol=1.0)


# =============================================================== 6. pruning + diagnostics


def test_line_pruning_reports_the_power_it_dropped(params1030) -> None:
    """Weak products vanish below ``line_prune``, and ``pruned_power`` counts exactly them."""
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    t = 2.0 * tau
    # The third tone is 100x weaker, so products using it twice fall below the cut.
    cw = _static_channel([0.0, 1.0 * MHz, 4.0 * MHz], amps=[1.0, 1.0, 0.01], tau=tau)

    everything = channel_lines(cw, aod, t, MixingConfig(order=3, line_prune=0.0))
    pruned = channel_lines(cw, aod, t, MixingConfig(order=3, line_prune=1e-5))
    assert everything.n_lines == 12  # 3 + 3 + 6, no collisions at these detunings
    assert pruned.n_lines < everything.n_lines
    assert everything.pruned_power == 0.0

    kept = {round(float(f), 3) for f in pruned.f}
    dropped = [i for i, f in enumerate(everything.f) if round(float(f), 3) not in kept]
    assert dropped, "expected at least one product below the cut"
    assert pruned.pruned_power == pytest.approx(
        float(np.sum(np.abs(everything.amp[dropped]) ** 2)), rel=1e-12
    )

    # Every dropped line really was below the threshold, and the survivors are above it.
    threshold = 1e-5 * max(abs(everything.amp[_line_at(everything, f)]) for f in (0.0, 1.0 * MHz))
    assert np.all(np.abs(everything.amp[dropped]) < threshold)
    assert np.all(np.abs(pruned.amp) >= threshold)

    # Relative intensity error of the cut is bounded by ~(prune)^2 per line.
    assert pruned.pruned_power / float(np.sum(np.abs(pruned.amp) ** 2)) < 1e-9


def test_term_pruning_reports_the_power_it_dropped(params1030) -> None:
    """``build_terms`` cuts weak products of lines and accounts for the loss."""
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    t = 2.0 * tau
    cw = _static_channel([0.0, 1.0 * MHz, 4.0 * MHz], tau=tau)
    wfs = _waveform_set(params1030, cw, aod)

    full = build_terms(wfs, t, channels=(CHANNEL,), term_prune=0.0)
    assert full.n_terms == 12
    assert full.pruned_power == 0.0

    cut = build_terms(wfs, t, channels=(CHANNEL,), term_prune=0.05)
    assert cut.n_terms == 3  # only the three fundamentals clear 5% of the strongest term
    power = float(np.sum(np.abs(full.c) ** 2))
    assert cut.pruned_power == pytest.approx(power - float(np.sum(np.abs(cut.c) ** 2)), rel=1e-12)
    assert cut.pruned_power / power < 1e-2

    # Ordering and content of the survivors are untouched by the cut.
    survivors = [i for i, c in enumerate(full.c) if abs(c) >= 0.05 * np.abs(full.c).max()]
    np.testing.assert_allclose(cut.c, full.c[survivors], rtol=1e-14)
    np.testing.assert_allclose(cut.df_opt, full.df_opt[survivors], rtol=1e-14)
    np.testing.assert_allclose(cut.theta1, full.theta1[:, survivors], rtol=1e-14)


def test_term_prune_keeps_a_zero_amplitude_frame_intact(params1030) -> None:
    """A frame whose drive is entirely off still yields a term (nothing is below zero)."""
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    env = SmoothOnOff(t_on=4.0 * tau, t_off=10.0 * tau, ramp=1.0 * tau)
    cw = _static_channel([1.0 * MHz], envs=[env], tau=tau)

    terms = build_terms(_waveform_set(params1030, cw, aod), 1.0 * tau, channels=(CHANNEL,))
    assert terms.n_terms == 1
    assert terms.c[0] == 0.0


def test_line_pruned_power_reaches_the_term_diagnostic(params1030) -> None:
    """Per-channel line pruning is carried through the Cartesian product (Eq. S7)."""
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    t = 2.0 * tau
    cw_y = _static_channel([0.0, 1.0 * MHz, 4.0 * MHz], amps=[1.0, 1.0, 0.01], tau=tau)
    cw_x = _static_channel([2.0 * MHz], tau=tau)
    wfs = WaveformSet(
        channels={CHANNEL: cw_y, "Ax": cw_x},
        params=replace(params1030, channels={name: aod for name in params1030.channels}),
    )

    lines_y = channel_lines(cw_y, aod, t)
    lines_x = channel_lines(cw_x, aod, t)
    assert lines_y.pruned_power > 0.0

    terms = build_terms(wfs, t, channels=(CHANNEL, "Ax"), term_prune=0.0)
    expected = lines_y.pruned_power * float(np.sum(np.abs(lines_x.amp) ** 2))
    assert terms.pruned_power == pytest.approx(expected, rel=1e-12)


# =============================================================== 7. order-1 regression


def test_order_one_matches_the_frozen_m1_snapshot(params1030) -> None:
    """``mixing_order=1`` is the M1 model, digit for digit.

    Two chirping tones on ``Ay`` at ``t = 1.7 tau``; the expected values below are frozen
    literals (not a re-run of the code path), cross-checked against the closed forms of
    ``docs/conventions.md`` §3: ``amp = (i C / 2) A exp(-i phase(t_c))``,
    ``theta1 = s 2 pi f / v``, ``theta2 = -pi fdot / v^2``, ``alpha = (1, 0, 0)``.
    """
    aod = _aod(params1030, mixing_order=1)
    tau, v = aod.transit_time, aod.sound_speed
    span = 6.0 * tau
    t = 1.7 * tau
    t_c = t - 0.5 * tau
    plan = ((1.0 * MHz, 40.0 * MHz / ms, 0.3), (-3.0 * MHz, -25.0 * MHz / ms, -1.2))
    cw = ChannelWaveform(
        tuple(
            ToneTrack(
                freq=PiecewisePoly.from_segment_coeffs([0.0, span], [[f0, fdot * span]]),
                phase0=phase0,
            )
            for f0, fdot, phase0 in plan
        )
    )

    terms = build_terms(_waveform_set(params1030, cw, aod), t, channels=(CHANNEL,))

    expected_c = np.array(
        [-0.14859762349724234 - 0.02046329130349783j, -0.10666223485167269 + 0.10546642904946872j]
    )
    expected_f = np.array([1553846.1538461538, -3346153.846153846])
    expected_fdot = np.array([4.0e10, -2.5e10])
    expected_theta1 = np.array([-15020.158959174867, 32345.391818025088])
    expected_theta2 = np.array([-297428.89028069045, 185893.05642543154])

    assert terms.n_terms == 2
    assert terms.pruned_power == 0.0
    np.testing.assert_allclose(terms.c, expected_c, rtol=1e-13)
    np.testing.assert_allclose(terms.df_opt, expected_f, rtol=1e-13)
    np.testing.assert_allclose(terms.theta1[1], expected_theta1, rtol=1e-13)
    np.testing.assert_allclose(terms.theta1[0], 0.0, atol=0.0)
    np.testing.assert_allclose(terms.theta2[1], expected_theta2, rtol=1e-13)
    np.testing.assert_allclose(terms.alpha[:, 0, :], 1.0, rtol=0.0)
    np.testing.assert_allclose(terms.alpha[:, 1:, :], 0.0, atol=0.0)

    lines = channel_lines(cw, aod, t)
    np.testing.assert_allclose(lines.amp, expected_c, rtol=1e-13)
    np.testing.assert_allclose(lines.fdot, expected_fdot, rtol=1e-13)

    # The snapshot is the physics, not just the code: rebuild it from the closed forms.
    for index, (f0, fdot, phase0) in enumerate(plan):
        phase = 2.0 * math.pi * (f0 * t_c + 0.5 * fdot * t_c**2) + phase0
        assert expected_f[index] == pytest.approx(f0 + fdot * t_c, rel=1e-13)
        assert expected_c[index] == pytest.approx(
            0.5j * aod.drive_strength * np.exp(-1j * phase), rel=1e-12
        )
        assert expected_theta1[index] == pytest.approx(
            -2.0 * math.pi * expected_f[index] / v, rel=1e-13
        )
        assert expected_theta2[index] == pytest.approx(-math.pi * fdot / v**2, rel=1e-13)


def test_order_three_reduces_to_order_one_as_the_drive_weakens(params1030) -> None:
    """Small ``m``: the products fall below the cut and the fundamentals converge to Eq. S3."""
    tau = params1030.channels[CHANNEL].transit_time
    cw = _static_channel([-1.0 * MHz, 0.0, 2.0 * MHz], phases=[0.2, 1.1, -0.7], tau=tau)
    t = 2.0 * tau

    weak1 = channel_lines(cw, _aod(params1030, drive_strength=1e-4, mixing_order=1), t)
    weak3 = channel_lines(cw, _aod(params1030, drive_strength=1e-4, mixing_order=3), t)
    assert weak3.n_lines == weak1.n_lines == 3
    np.testing.assert_allclose(weak3.amp, weak1.amp, rtol=1e-7)
    np.testing.assert_allclose(weak3.f, weak1.f, rtol=1e-14)


# =============================================================== 8. envelopes


def test_mixed_line_envelope_is_the_product_of_its_constituents(params1030) -> None:
    """Irising of a ghost: multiplicities 1 and 2 show up in ``alpha1``/``alpha2`` (Eq. S5).

    Only tone 1 is gated, so ``L1 = mult * l1_1``: the distinct-triple ghost picks it up
    once, the degenerate ghost (which uses tone 1 twice) twice.
    """
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    geom = geometry(CHANNEL)
    tau, v = aod.transit_time, aod.sound_speed
    env = SmoothOnOff(t_on=0.5 * tau, t_off=8.0 * tau, ramp=2.0 * tau)
    detunings = [0.0, 1.0 * MHz, 4.0 * MHz]
    cw = _static_channel(detunings, envs=[None, env, None], tau=tau)
    t = 1.7 * tau  # t_c = 1.2 tau: mid-rise, so A' > 0 and A'' != 0
    t_c = t - 0.5 * tau
    amp, d_amp, d2_amp = float(env.A(t_c)), float(env.dA(t_c)), float(env.d2A(t_c))
    assert 0.0 < amp < 1.0 and d_amp > 0.0

    l1 = d_amp / amp
    l2 = d2_amp / amp
    terms = build_terms(_waveform_set(params1030, cw, aod), t, channels=(CHANNEL,), term_prune=0.0)

    def alpha_at(detuning: float):
        index = int(np.argmin(np.abs(terms.df_opt - detuning)))
        assert abs(terms.df_opt[index] - detuning) < 1.0
        return terms.alpha[geom.axis, :, index]

    for detuning, mult in ((1.0 * MHz + 4.0 * MHz - 0.0, 1), (2.0 * MHz - 0.0, 2)):
        alpha = alpha_at(detuning)
        big1 = mult * l1
        big2 = mult * (l2 - l1**2)
        assert alpha[0] == pytest.approx(1.0, rel=1e-14)
        assert complex(alpha[1]).real == pytest.approx(-geom.sound_sign * big1 / v, rel=1e-12)
        assert complex(alpha[2]).real == pytest.approx((big1**2 + big2) / (2.0 * v**2), rel=1e-12)

    # The gated tone's own fundamental keeps the plain Eq. S5 shape (multiplicity 1).
    alpha = alpha_at(1.0 * MHz)
    assert complex(alpha[1]).real == pytest.approx(-geom.sound_sign * l1 / v, rel=1e-12)
    assert complex(alpha[2]).real == pytest.approx(l2 / (2.0 * v**2), rel=1e-12)


def test_a_faded_tone_takes_its_products_with_it(params1030) -> None:
    """When an envelope closes, every line that used that tone vanishes with it."""
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    env = SmoothOnOff(t_on=4.0 * tau, t_off=12.0 * tau, ramp=1.0 * tau)
    cw = _static_channel([0.0, 1.0 * MHz, 4.0 * MHz], envs=[None, None, env], tau=tau)

    lines = channel_lines(cw, aod, 1.0 * tau)  # t_c = 0.5 tau: tone 2 is still off
    assert lines.amp[_line_at(lines, 4.0 * MHz)] == 0.0

    # Only the three fundamentals and the two products of the *live* tones survive:
    # 2 f_1 - f_0 = +2 MHz and 2 f_0 - f_1 = -1 MHz.  Everything touching tone 2 is zero
    # and falls to the amplitude cut.
    expected = np.array([-1.0, 0.0, 1.0, 2.0, 4.0]) * MHz
    np.testing.assert_allclose(np.sort(lines.f), expected, atol=1.0)
    for detuning in (0.0, 1.0 * MHz, 2.0 * MHz, -1.0 * MHz):
        assert abs(lines.amp[_line_at(lines, detuning)]) > 0.0


# =============================================================== 9. cost


def test_expand_lines_is_a_pure_function_of_the_fundamentals(params1030) -> None:
    """``expand_lines`` needs no phases beyond the ones already inside ``Lines.amp``."""
    aod = _aod(params1030, drive_strength=0.25, mixing_order=1)
    tau = aod.transit_time
    t = 2.0 * tau
    cw = _static_channel([-1.0 * MHz, 0.0, 3.0 * MHz], phases=[0.9, -2.2, 0.1], tau=tau)

    fundamentals = channel_lines(cw, aod, t)
    assert expand_lines(fundamentals, aod, MixingConfig(order=1)) is fundamentals
    expanded = expand_lines(fundamentals, aod, MixingConfig(order=3))
    assert expanded.n_lines == 12
    np.testing.assert_allclose(
        expanded.amp[[_line_at(expanded, f) for f in fundamentals.f]],
        fundamentals.amp * (1.0 - 0.25 * 3.0 * 0.25**2 + 0.125 * 0.25**2),
        rtol=1e-13,
    )


def test_ten_tone_ladder_collapses_onto_its_own_grid(params1030) -> None:
    """M = 10 equal tones: 460 enumerated products merge onto 28 lines of the 1 MHz grid.

    An equally spaced ladder is the worst case for enumeration and the best case for
    merging — every ``f_j + f_k - f_i`` lands back on the ladder, extended by three steps
    at each end (``-13.5 .. +13.5 MHz``), all inside the widened band.
    """
    aod = _aod(params1030, drive_strength=0.3, mixing_order=3)
    tau = aod.transit_time
    detunings = [(n - 4.5) * MHz for n in range(10)]
    cw = _static_channel(detunings, tau=tau)

    lines = channel_lines(cw, aod, 2.0 * tau)
    assert 10 + 10 * 9 * 8 // 2 + 10 * 9 == 460  # fundamentals + IM3 + degenerate IM3
    assert lines.n_lines == 28
    np.testing.assert_allclose(np.sort(lines.f), (np.arange(28) - 13.5) * MHz + 0.0 * MHz, atol=1.0)
    assert lines.pruned_power == 0.0  # equal amplitudes: nothing is small enough to cut
