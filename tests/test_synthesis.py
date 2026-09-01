r"""Array ladders and Schroeder phases (``waveform/synthesis.py``, Eqs. S18/S19, S23/S28).

Three claims, each checked against its closed form rather than against the implementation:

1. :func:`~aodl.waveform.synthesis.schroeder_phases` is the quadratic progression of
   Eq. S23/S28 — and does what the paper wants it for: the ``M``-tone sum stops cresting
   ``M``-fold and settles to ``~sqrt(M)``;
2. :func:`~aodl.waveform.synthesis.array_tones` puts ``M`` tones on the Eq. S18 ladder, so
   the device layer deflects them onto the Table-I grid at pitch ``lambda F Delta f / v``;
3. :func:`~aodl.waveform.synthesis.add_common_ramp` translates that ladder *rigidly*:
   every tone picks up the same ``f(t)`` and the same ``fdot(t)``, spacings unchanged, and
   the phase stays the exact antiderivative (no numerical differentiation anywhere).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from aodl import add_common_ramp, array_tones, schroeder_phases
from aodl.device.aodl import build_terms
from aodl.poly import PiecewisePoly
from aodl.trajectory import ramps
from aodl.units import MHz, us
from aodl.waveform.synthesis import DEFAULT_SPAN
from aodl.waveform.tones import ChannelWaveform, ConstantEnvelope, WaveformSet

T_END = 200.0 * us
TWO_PI = 2.0 * math.pi


def _linear_drive(params):
    """``params`` at ``mixing_order=1``: one emission line per tone, so a ladder of ``M``
    tones gives exactly ``M`` spots to compare against the Table-I grid.  The ghosts the
    product default (order 3) adds are :mod:`aodl.device.mixing`'s business, not the
    ladder geometry's — ``tests/test_integration_m2.py`` pins those."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


# ============================================================== Schroeder phases (S23/S28)


@pytest.mark.parametrize("n_tones", [1, 2, 3, 5, 8, 32])
def test_schroeder_phases_match_the_closed_form(n_tones):
    """``phi_n = mod(2 pi n (n-1) / (2 M), 2 pi)`` — Eq. S23 generalized (Eq. S28)."""
    phases = schroeder_phases(n_tones)
    expected = [(TWO_PI * n * (n - 1) / (2 * n_tones)) % TWO_PI for n in range(n_tones)]
    assert phases.shape == (n_tones,)
    np.testing.assert_allclose(phases, expected, rtol=1e-14, atol=0.0)
    assert np.all((phases >= 0.0) & (phases < TWO_PI))
    assert phases[0] == 0.0 and (n_tones < 2 or phases[1] == 0.0)


def test_schroeder_phases_edge_cases():
    assert schroeder_phases(0).shape == (0,)
    with pytest.raises(ValueError, match="non-negative"):
        schroeder_phases(-1)


def test_schroeder_phases_flatten_the_crest_factor():
    """The point of Eq. S23: the tone sum no longer peaks ``M``-fold.

    With every phase zero the ``M`` tones crest together and ``max |V| = M``; the Schroeder
    progression spreads the crest into a sweep, and the peak drops towards the incoherent
    ``sqrt(M)``.  Checked at ``M = 32`` on a dense time grid over one ``1/Delta f`` period.
    """
    n_tones, delta_f = 32, 1.0 * MHz
    detune = (np.arange(n_tones) - 0.5 * (n_tones - 1)) * delta_f
    t = np.linspace(0.0, 1.0 / delta_f, 20001)
    phase = TWO_PI * detune[:, None] * t[None, :]

    aligned = float(np.max(np.abs(np.cos(phase).sum(axis=0))))
    spread = float(np.max(np.abs(np.cos(phase + schroeder_phases(n_tones)[:, None]).sum(axis=0))))
    assert aligned == pytest.approx(n_tones, rel=1e-6)
    assert spread < 0.35 * aligned
    assert spread < 2.0 * math.sqrt(n_tones)


# ==================================================================== the ladder (Eq. S18)


def test_array_tones_lays_the_eq_s18_ladder():
    """``f_n = center + (n - (M-1)/2) delta_f``, equal amplitudes, one envelope."""
    n_tones, delta_f, center, amp = 5, 1.0 * MHz, 2.0 * MHz, 0.6
    cw = array_tones(n_tones, delta_f, center=center, amp=amp, t0=0.0, t1=T_END)

    assert isinstance(cw, ChannelWaveform)
    assert cw.n_tones == n_tones
    expected = center + (np.arange(n_tones) - 0.5 * (n_tones - 1)) * delta_f
    np.testing.assert_allclose([tone.f(0.0) for tone in cw.tones], expected, rtol=1e-14)
    np.testing.assert_allclose([tone.f(T_END) for tone in cw.tones], expected, rtol=1e-14)
    np.testing.assert_allclose([tone.fdot(0.5 * T_END) for tone in cw.tones], 0.0)
    for tone in cw.tones:
        assert isinstance(tone.env, ConstantEnvelope)
        assert tone.env.amp == pytest.approx(amp)
        assert tone.t_span == (0.0, T_END)
    # centred: the ladder's mean detuning is the requested centre
    assert float(np.mean(expected)) == pytest.approx(center, rel=1e-14)


def test_array_tones_single_tone_and_default_span():
    cw = array_tones(1, 1.0 * MHz, center=3.0 * MHz)
    assert cw.n_tones == 1
    assert cw.tones[0].f(0.0) == pytest.approx(3.0 * MHz)
    assert cw.t_span == (0.0, DEFAULT_SPAN)
    assert cw.tones[0].phase0 == 0.0


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("zero", np.zeros(6)), ("schroeder", schroeder_phases(6))],
)
def test_array_tones_phase_modes(mode, expected):
    cw = array_tones(6, 1.0 * MHz, phases=mode, t1=T_END)
    np.testing.assert_allclose([tone.phase0 for tone in cw.tones], expected, rtol=1e-14)


def test_array_tones_random_phases_are_reproducible_with_a_seeded_rng():
    kwargs = dict(phases="random", t1=T_END)
    a = array_tones(7, 1.0 * MHz, rng=np.random.default_rng(4), **kwargs)
    b = array_tones(7, 1.0 * MHz, rng=np.random.default_rng(4), **kwargs)
    c = array_tones(7, 1.0 * MHz, rng=np.random.default_rng(5), **kwargs)
    phases = [tone.phase0 for tone in a.tones]
    assert phases == [tone.phase0 for tone in b.tones]
    assert phases != [tone.phase0 for tone in c.tones]
    assert all(0.0 <= p < TWO_PI for p in phases)


def test_array_tones_explicit_phases_and_validation():
    explicit = [0.1, 0.2, 0.3]
    cw = array_tones(3, 1.0 * MHz, phases=explicit, t1=T_END)
    np.testing.assert_allclose([tone.phase0 for tone in cw.tones], explicit)

    with pytest.raises(ValueError, match="one entry per tone"):
        array_tones(3, 1.0 * MHz, phases=[0.0, 1.0], t1=T_END)
    with pytest.raises(ValueError, match="must be one of"):
        array_tones(3, 1.0 * MHz, phases="uniform", t1=T_END)
    with pytest.raises(ValueError, match="at least one tone"):
        array_tones(0, 1.0 * MHz)
    with pytest.raises(ValueError, match="t1 > t0"):
        array_tones(3, 1.0 * MHz, t0=1.0 * us, t1=1.0 * us)


def test_ladder_lands_on_the_table_i_grid(params1030):
    """M tones on Ax -> M spots at ``X = -deflection_scale f_n``, pitch ``lambda F df / v``.

    ``Ax`` has ``sound_sign = -1`` and nothing drives ``Bx``, so Table I's
    ``X = deflection_scale (f_Bx - f_Ax)`` reduces to ``-deflection_scale f_Ax``.
    """
    params = _linear_drive(params1030)
    n_tones, delta_f = 5, 1.0 * MHz
    aod = params.channels["Ax"]
    cw = array_tones(n_tones, delta_f, t1=T_END)
    wfs = WaveformSet({"Ax": cw}, params)
    terms = build_terms(wfs, 2.0 * aod.transit_time, channels=("Ax",))

    assert terms.n_terms == n_tones
    x_spots = np.sort(terms.theta1[0] * params.optics.focal_length / params.optics.k)
    detune = (np.arange(n_tones) - 0.5 * (n_tones - 1)) * delta_f
    np.testing.assert_allclose(x_spots, np.sort(-params.deflection_scale * detune), rtol=1e-12)

    pitch = float(np.diff(x_spots).mean())
    assert pitch == pytest.approx(params.deflection_scale * delta_f, rel=1e-12)
    np.testing.assert_allclose(np.diff(x_spots), pitch, rtol=1e-12)  # evenly spaced


# ============================================================== the common ramp (Eq. S19)


def test_add_common_ramp_translates_the_ladder_rigidly():
    """Every tone picks up the same ``f`` and ``fdot``; the spacings never move."""
    n_tones, delta_f = 4, 1.0 * MHz
    cw = array_tones(n_tones, delta_f, phases="schroeder", t0=0.0, t1=T_END)
    ramp = ramps.min_jerk(0.0, T_END, 0.0, 3.0 * MHz)
    moved = add_common_ramp(cw, ramp)

    t = np.linspace(0.0, T_END, 51)
    base = np.array([tone.f(t) for tone in cw.tones])
    after = np.array([tone.f(t) for tone in moved.tones])
    np.testing.assert_allclose(after - base, np.broadcast_to(ramp(t), after.shape), rtol=1e-12)
    np.testing.assert_allclose(np.diff(after, axis=0), delta_f, rtol=1e-9)

    fdot = np.array([tone.fdot(t) for tone in moved.tones])
    np.testing.assert_allclose(fdot, np.broadcast_to(ramp.derivative()(t), fdot.shape), rtol=1e-12)

    # envelopes and phase offsets survive; the phase is still the exact antiderivative
    assert [tone.phase0 for tone in moved.tones] == [tone.phase0 for tone in cw.tones]
    assert [tone.env for tone in moved.tones] == [tone.env for tone in cw.tones]
    fine = np.linspace(0.0, T_END, 20001)
    exact = np.asarray(moved.tones[0].phase(fine))
    values = np.asarray(moved.tones[0].f(fine))
    quad = TWO_PI * np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) * 0.5 * (values[1:] + values[:-1]))]
    )
    assert np.max(np.abs((exact - exact[0]) - quad)) < 1e-7 * np.max(np.abs(quad))


def test_add_common_ramp_rejects_a_mismatched_span():
    cw = array_tones(3, 1.0 * MHz, t0=0.0, t1=T_END)
    with pytest.raises(ValueError, match="same span as the tones"):
        add_common_ramp(cw, ramps.min_jerk(0.0, 0.5 * T_END, 0.0, 1.0 * MHz))
    with pytest.raises(TypeError, match="PiecewisePoly"):
        add_common_ramp(cw, "not a ramp")


def test_add_common_ramp_moves_every_spot_by_the_same_distance(params1030):
    """A common ramp is a rigid array translation (Eq. S19's lateral term)."""
    aod = params1030.channels["Ax"]
    tau = aod.transit_time
    cw = array_tones(4, 1.0 * MHz, t0=0.0, t1=T_END)
    shift = 2.5 * MHz
    moved = add_common_ramp(cw, PiecewisePoly.constant(shift, 0.0, T_END))

    def spots(channel):
        terms = build_terms(WaveformSet({"Ax": channel}, params1030), 2.0 * tau, channels=("Ax",))
        return np.sort(terms.theta1[0] * params1030.optics.focal_length / params1030.optics.k)

    before, after = spots(cw), spots(moved)
    np.testing.assert_allclose(after - before, -params1030.deflection_scale * shift, rtol=1e-12)
