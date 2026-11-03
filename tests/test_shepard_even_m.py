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


def _site_census(wfs, params, times: np.ndarray, array: ArraySpec) -> dict[str, np.ndarray]:
    """Per-frame site powers, the brightest other spot, the total, and the hand-over state.

    Every term's centre ``theta1 F / k`` is binned onto Eq. S19's lattice
    ``(n - (M - 1)/2) pitch`` within ``0.25 pitch`` per axis; a requested site is ``0 <= n < M``
    on both axes.  A free (single-tweezer) axis is binned at its own ladder spacing, so its Eq.
    S31 shadows land on the neighbouring nodes.  Anything not on a requested site — the two
    extended columns of a hand-over, shadows, or a term off the lattice altogether — counts
    toward ``other``, binned the same way.

    ``fading[:, i]`` is the number of extra live ``A`` rungs on axis ``i`` (``1`` while it hands
    over) anywhere in the transit window ``[t - tau, t]`` the aperture integrates at frame time
    ``t``, and ``switching[:, i]`` whether that number changes inside the window.
    """
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    counts = np.array([array.mx, array.my])
    pitch = np.array([_spacing(wfs, "Ax"), _spacing(wfs, "Ay")]) * params.deflection_scale
    first = -0.5 * (counts - 1) * pitch

    out: dict[str, list] = {"sites": [], "other": [], "total": [], "fading": [], "switching": []}
    for t in times:
        terms = build_terms(wfs, float(t))
        xc, yc, _, _ = spot_params(terms, optics, 0.0)
        power = _term_power(terms, optics)
        index = (np.stack([np.asarray(xc), np.asarray(yc)]) - first[:, None]) / pitch[:, None]
        node = np.round(index)
        on_lattice = np.all(np.abs(index - node) < 0.25, axis=0)
        requested = on_lattice & np.all((node >= 0) & (node < counts[:, None]), axis=0)

        grid = np.zeros((array.mx, array.my))
        ij = node[:, requested].astype(int)
        np.add.at(grid, (ij[0], ij[1]), power[requested])
        out["sites"].append(grid.ravel())

        bins: dict[tuple[float, float], float] = {}
        for k in np.nonzero(~requested)[0]:
            key = (node[0, k], node[1, k]) if on_lattice[k] else (index[0, k], index[1, k])
            bins[key] = bins.get(key, 0.0) + float(power[k])
        out["other"].append(max(bins.values(), default=0.0))
        out["total"].append(float(np.sum(power)))

        window = np.linspace(float(t) - tau, float(t), 9)
        live = [
            np.sum([np.asarray(tone.env.A(window)) > 0.0 for tone in wfs.channels[a].tones], axis=0)
            for a in ("Ax", "Ay")
        ]
        out["fading"].append([int(np.max(n)) - 1 for n in live])
        out["switching"].append([bool(np.ptp(n) > 0) for n in live])
    return {key: np.array(value) for key, value in out.items()}


@pytest.mark.parametrize(
    "mx, my, dfx, dfy",
    CASES,
    ids=[f"{mx}x{my}@{dfx / MHz:g}/{dfy / MHz:g}MHz" for mx, my, dfx, dfy in CASES],
)
def test_even_m_array_sites_stay_lit_through_hand_overs(params1030, mx, my, dfx, dfy):
    """Every requested site of an even-``M`` array is >= 0.80 of the plateau at every frame.

    Closure: with the half-rung ``xi_B`` the worst requested site reads 0.97-0.98 of the first
    frame's median site on these 40 frames (0.95 on a 400-frame grid) — the ``rho`` ripple of
    a 1 MHz ladder, ``docs/guide.md`` §6.4, and a 3x3 reads the same 0.967 — no spot elsewhere
    outshines a site, and total power is conserved.  Regression: the same ladders on the WO-17
    timing (``xi_B = xi_A`` and a frequency-only ``delta_f/2`` comb) put a requested site at
    0.0001-0.005 — the defect this closes.
    """
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    array = ArraySpec(mx, my, dfx, dfy)
    spec = TrajectorySpec(array=array, moves=MOVES)
    plan = plan_motion(spec, params)
    assert plan.report.mode == "shepard"
    times = np.linspace(2.0 * tau, HOLD_END + 0.5 * tau, N_FRAMES)

    fixed = _site_census(plan.wfs, params, times, array)
    ref = float(np.median(fixed["sites"][0]))
    assert ref > 0.0
    worst = float(fixed["sites"].min() / ref)
    assert worst >= 0.80, f"a requested site fell to {worst:.3f} of the plateau"
    assert float(fixed["other"].max() / ref) <= 1.05

    # Power is conserved: each hand-over adds exactly one column (row) of light — the switched
    # extended one — and nothing else.  Checked where the state is steady across the transit
    # window and no free axis is fading (whose Eq. S31 shadows carry light of their own).
    array_axis = np.array([mx > 1, my > 1])
    steady = ~np.any(fixed["switching"], axis=1) & ~np.any(
        ~array_axis & (fixed["fading"] > 0), axis=1
    )
    assert steady.sum() >= 2
    lit = np.prod(np.array([mx, my]) + array_axis * fixed["fading"], axis=1)
    np.testing.assert_allclose(fixed["total"][steady] / (lit[steady] * ref), 1.0, atol=0.05)

    # Regression half: the same ladders on the WO-17 timing leave a requested site dark.
    old = plan_motion(spec, params, shepard=_wo17(plan.wfs, array))
    old_worst = float(_site_census(old.wfs, params, times, array)["sites"].min() / ref)
    assert old_worst < 0.05, f"the WO-17 timing kept every site at >= {old_worst:.3f}"
