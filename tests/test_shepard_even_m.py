r"""Even-``M`` fading-Shepard arrays keep every requested trap lit (R2 design addendum §F2).

A Shepard array is fed by two live ``A`` rungs during a hand-over (Eq. S26, ``p_A = 1``) and by
the ``B`` rectangle of Eq. S27 (``p_B = 0``), and a requested site stays at full depth only if
*both* ``A`` rungs see it through the same ``B`` rungs — the ``cos^2 theta + sin^2 theta = 1``
identity of :mod:`aodl.waveform.shepard`.  For that the ``B`` ladder must hold ``M + 1`` rungs
exactly while ``A`` hands over.  With a common ladder offset ``xi`` the two phases are set by
``(1 + eta)/2`` and ``(M + eta)/2``, which agree modulo 1 for odd ``M`` and differ by ``1/2`` for
even ``M``: an even-``M`` array then held only ``M`` ``B`` rungs during each hand-over, its lit
block walked by one pitch, and the end column of the requested array went dark.
:func:`~aodl.waveform.shepard.table_ii` now offsets an even-``M`` ``B`` ladder by half a rung
(``xi_B = xi_A + 1/2``), which restores the odd-``M`` phase relation *and* lands the traps on
Eq. S19's half-integer lattice — the ``delta_f/2`` frequency-only comb offset of WO-17 had done
only the latter.

The test measures what an atom would see: per lattice site, the summed Parseval power of every
pupil term binned there (:func:`aodl.field.measure._term_power`, so acoustic irising and the
fill of the aperture are included — which is where the residual ``rho`` ripple of
``docs/guide.md`` §6.4 comes from).  ``mixing_order = 1``: the statement is about the Eq.
S24-S28 frequency/amplitude algebra, where one tone means one beam.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aodl import build_terms, plan_motion
from aodl.field.focal import spot_params
from aodl.field.measure import _term_power
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec
from aodl.units import MHz, um, us
from aodl.waveform.shepard import ChannelFade, FadeZoneEnvelope, ShepardConfig

#: The ``array_lift_hold`` example of the web app: lift 8 µm, hold 500 µs, lower.
MOVES = (Lift(8.0 * um, 80.0 * us), Hold(500.0 * us), Lift(-8.0 * um, 80.0 * us))

#: End of the hold [s]; frames run to ``HOLD_END + tau/2`` (the atom-plane lag).
HOLD_END = 580.0 * us

#: Frames per run (the work order's floor).
N_FRAMES = 40

#: ``(mx, my, delta_f_x, delta_f_y)`` — every even-``M`` shape, and the interlaced 4x4.
CASES = [
    (2, 2, 1.0 * MHz, 1.3 * MHz),
    (4, 4, 1.0 * MHz, 1.3 * MHz),
    (1, 4, 1.0 * MHz, 1.3 * MHz),
    (4, 1, 1.0 * MHz, 1.3 * MHz),
    (4, 4, 1.0 * MHz, 1.0 * MHz),
]


def _linear(params):
    """``params`` with every channel at ``mixing_order=1`` — one tone, one beam."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _spacing(wfs, channel: str) -> float:
    """The ladder spacing [Hz] a synthesized channel was built with (auto-sized if free)."""
    env = wfs.channels[channel].tones[0].env
    assert isinstance(env, FadeZoneEnvelope)
    return env.delta_f


def _wo17(wfs, array: ArraySpec) -> ShepardConfig:
    """The pre-fix even-``M`` timing on the same ladders: ``xi_B = xi_A`` plus a comb ``delta_f/2``.

    Built with the explicit :class:`ShepardConfig` override the API still honours, so the drive
    is exactly what ``shepard="auto"`` synthesized before ``table_ii`` carried the half rung.
    """
    dfx, dfy = _spacing(wfs, "Ax"), _spacing(wfs, "Ay")
    config: dict[str, ChannelFade] = {}
    comb: dict[str, float] = {}
    for b_name, xi_a, count, spacing in (("Bx", 0.0, array.mx, dfx), ("By", 0.5, array.my, dfy)):
        if count > 1 and count % 2 == 0:
            config[b_name] = ChannelFade(count, 0.0, xi_a)
            comb[b_name] = 0.5 * spacing
    return ShepardConfig(dfx, dfy, config=config, comb_offset=comb)


def _site_census(wfs, params, times: np.ndarray, array: ArraySpec):
    """Per-frame ``(requested-site powers, brightest other spot, total power, handing-over)``.

    Every term's centre ``theta1 F / k`` is binned onto Eq. S19's lattice
    ``(n - (M - 1)/2) pitch`` within ``0.25 pitch`` per axis; a requested site is ``0 <= n < M``
    on both axes.  Anything else — the two extended columns of a hand-over, Eq. S31 shadows of
    a free axis, or a term off the lattice altogether — counts toward "other", binned the same
    way.  ``handing_over`` is whether either ``A`` ladder had two live rungs anywhere in the
    transit window ``[t - tau, t]`` that the aperture integrates at frame time ``t``.
    """
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    # an array axis' pitch is its own spacing; a free axis' is its auto-sized ladder spacing,
    # so that its Eq. S31 shadows land on the neighbouring (non-requested) nodes
    spacing = np.array([_spacing(wfs, "Ax"), _spacing(wfs, "Ay")])
    pitch = spacing * params.deflection_scale
    first = -0.5 * (np.array([array.mx, array.my]) - 1) * pitch
    counts = np.array([array.mx, array.my])

    sites, other, total, fading = [], [], [], []
    for t in times:
        terms = build_terms(wfs, float(t))
        xc, yc, _, _ = spot_params(terms, optics, 0.0)
        power = _term_power(terms, optics)
        pos = np.stack([np.asarray(xc), np.asarray(yc)])
        index = (pos - first[:, None]) / pitch[:, None]
        node = np.round(index)
        on_lattice = np.all(np.abs(index - node) < 0.25, axis=0)
        requested = on_lattice & np.all((node >= 0) & (node < counts[:, None]), axis=0)

        grid = np.zeros((array.mx, array.my))
        np.add.at(grid, (node[0, requested].astype(int), node[1, requested].astype(int)), power[requested])
        sites.append(grid.ravel())

        bins: dict[tuple[float, float], float] = {}
        for k in np.nonzero(~requested)[0]:
            key = (float(node[0, k]), float(node[1, k])) if on_lattice[k] else (float(xc[k]), float(yc[k]))
            bins[key] = bins.get(key, 0.0) + float(power[k])
        other.append(max(bins.values(), default=0.0))
        total.append(float(np.sum(power)))

        window = np.linspace(float(t) - tau, float(t), 9)
        live = [
            np.sum([np.asarray(tone.env.A(window)) > 0.0 for tone in wfs.channels[a].tones], axis=0)
            for a in ("Ax", "Ay")
        ]
        fading.append(tuple(int(np.max(n)) - 1 for n in live) + tuple(int(np.ptp(n) > 0) for n in live))
    return np.array(sites), np.array(other), np.array(total), np.array(fading)


@pytest.mark.parametrize(
    "mx, my, dfx, dfy",
    CASES,
    ids=[f"{mx}x{my}@{dfx / MHz:g}/{dfy / MHz:g}MHz" for mx, my, dfx, dfy in CASES],
)
def test_even_m_array_sites_stay_lit_through_hand_overs(params1030, mx, my, dfx, dfy):
    """Every requested site of an even-``M`` array is >= 0.80 of the plateau at every frame.

    Closure: with the half-rung ``xi_B`` the worst requested site reads 0.86-0.94 of the first
    frame's median site (the ``rho`` ripple of a 1 MHz ladder, ``docs/guide.md`` §6.4 — a 3x3
    shows the same), no spot elsewhere outshines a site, and the total power away from
    hand-overs is ``M_x M_y`` sites' worth.  Regression: the same ladders with the WO-17 timing
    (``xi_B = xi_A`` and a frequency-only ``delta_f/2`` comb) put a requested site below 0.05 —
    the defect this closes.
    """
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    array = ArraySpec(mx, my, dfx, dfy)
    spec = TrajectorySpec(array=array, moves=MOVES)
    plan = plan_motion(spec, params)
    assert plan.report.mode == "shepard"
    times = np.linspace(2.0 * tau, HOLD_END + 0.5 * tau, N_FRAMES)

    sites, other, total, fading = _site_census(plan.wfs, params, times, array)
    ref = float(np.median(sites[0]))
    assert ref > 0.0
    worst = float(sites.min() / ref)
    print(f"\n{mx}x{my}: worst {worst:.3f} other {other.max() / ref:.3f} quiet {quiet_dbg(fading)} ")
    assert worst >= 0.80, f"a requested site fell to {worst:.3f} of the plateau"
    assert float(other.max() / ref) <= 1.05
    quiet = ~fading
    if dfx != dfy:  # equal spacings interlace perfectly: some axis is always handing over
        assert quiet.sum() >= 3
    np.testing.assert_allclose(total[quiet] / (mx * my * ref), 1.0, atol=0.05)

    old = plan_motion(spec, params, shepard=_wo17(plan.wfs, array))
    old_sites, _, _, _ = _site_census(old.wfs, params, times, array)
    old_worst = float(old_sites.min() / ref)  # the same ladders, the same plateau
    assert old_worst < 0.05, f"the WO-17 timing kept every site at >= {old_worst:.3f}"


def quiet_dbg(f):
    return int((~f).sum())
