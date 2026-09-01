r"""M2 acceptance, end to end: two crossed AODs, arrays, IM3 (``docs/PLAN.md`` §3).

Everything runs through the product path — ``array_tones``/``ramps`` -> ``WaveformSet`` ->
:func:`aodl.engine.simulate` -> metrics and rendered frames — and every number is compared
against a closed form derived here, not against the implementation:

1. **diagonal chirp = spherical lens.**  Equal chirps on ``Ax`` and ``Ay`` give
   ``Zbar = lens_scale * fdot(t_c)`` with ``Delta F = 0``: out-of-plane motion with no
   astigmatism (Table I).  The single-axis control is the M1 cylinder, ``|Delta F| =
   lens_scale * fdot``.
2. **array geometry.**  An ``Mx x My`` ladder pair puts spots on the Table-I grid at pitch
   ``deflection_scale * delta_f``, all of equal power.
3. **IM3 ghosts.**  At ``mixing_order=3`` the ladder-edge ghost of Eqs. S20-S22 appears one
   ``delta_f`` beyond the array, at the intensity the ``-(i/8) m^3`` / ``-(i/16) m^3``
   coefficients predict once the contributing index triples are counted.
4. **cost.**  A 5x5 array with mixing on stays fast enough to drive a notebook.
"""

from __future__ import annotations

import time
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from aodl import add_common_ramp, array_tones, simulate
from aodl.field.focal import FrameGrid
from aodl.trajectory import ramps
from aodl.units import MHz, um, us
from aodl.waveform.tones import ChannelWaveform, ToneTrack, WaveformSet

#: The M2 diagonal move: 0 -> 4 MHz on both Ax and Ay, minimum-jerk, in 120 us.
SWEEP_SPAN = 4.0 * MHz
SWEEP_TIME = 120.0 * us

#: Array ladder spacings.  The two axes are *deliberately* different: with the same spacing
#: on both, every anti-diagonal of the array shares one optical frequency ``f_x + f_y`` and
#: the traps become mutually coherent — one grouped "spot" per anti-diagonal rather than one
#: per trap (pinned by :func:`test_equal_spacings_make_anti_diagonals_degenerate`).
DELTA_FX = 1.0 * MHz
DELTA_FY = 1.3 * MHz


def _order(params, order: int):
    """``params`` with every channel at the given ``mixing_order``."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=order) for name, aod in params.channels.items()},
    )


def _chirped(params, channels: tuple[str, ...]):
    """Min-jerk 0 -> SWEEP_SPAN chirp on each named channel, held past the last frame."""
    tau = params.channels[channels[0]].transit_time
    freq = ramps.min_jerk(0.0, SWEEP_TIME, 0.0, SWEEP_SPAN)
    cw = ChannelWaveform((ToneTrack(freq=freq),))
    wfs = WaveformSet({name: cw for name in channels}, params).with_hold_until(SWEEP_TIME + tau)
    return wfs, freq, tau


def _array(params, phases="schroeder", delta_fy: float = DELTA_FY, n_tones: int = 5):
    """Static ``n_tones x n_tones`` array on Ax + Ay, evaluated once the aperture is full."""
    tau = params.channels["Ax"].transit_time
    span = 10.0 * tau
    wfs = WaveformSet(
        {
            "Ax": array_tones(n_tones, DELTA_FX, phases=phases, t1=span),
            "Ay": array_tones(n_tones, delta_fy, phases=phases, t1=span),
        },
        params,
    )
    return wfs, 2.0 * tau


def _im3_paths(detunings, target: float, tol: float = 1.0) -> tuple[int, int]:
    """Count the IM3 index sets landing on ``target``: ``(f_j + f_k - f_i, 2 f_j - f_i)``.

    An independent re-enumeration of the WO-07 §2 signature classes, so the ghost prediction
    below does not borrow :mod:`aodl.device.mixing`'s own bookkeeping.
    """
    n = len(detunings)
    triples = sum(
        abs(detunings[j] + detunings[k] - detunings[i] - target) < tol
        for j, k in combinations(range(n), 2)
        for i in range(n)
        if i not in (j, k)
    )
    degenerate = sum(
        abs(2.0 * detunings[j] - detunings[i] - target) < tol
        for j in range(n)
        for i in range(n)
        if i != j
    )
    return int(triples), int(degenerate)


# ================================================== 1. crossed chirps: spherical vs cylinder


def test_diagonal_equal_chirp_is_a_spherical_lens(params1030):
    """Ax + Ay chirped alike: ``Zbar = lens_scale fdot(t_c)``, ``Delta F = 0`` throughout.

    Table I with only the A channels driven: ``Zbar = (1/2) lens_scale (fdot_Ax + fdot_Ay)``
    and ``Delta F = lens_scale (fdot_Ax - fdot_Ay)``.  Equal chirps therefore double into
    ``Zbar = lens_scale fdot`` and cancel in ``Delta F`` — the PLAN M2 headline: the tweezer
    leaves the focal plane along a *diagonal* while staying round.
    """
    wfs, freq, tau = _chirped(params1030, ("Ax", "Ay"))
    frames = np.linspace(0.0, SWEEP_TIME + tau, 61)
    table = simulate(wfs, frames).spot_table()
    t_c = frames - 0.5 * tau  # docs/conventions.md §7
    optics = params1030.optics

    zbar_predicted = params1030.lens_scale * freq.derivative()(t_c)
    peak = float(np.max(np.abs(zbar_predicted)))
    assert peak > optics.rayleigh  # the excursion is worth measuring
    assert np.max(np.abs(table["z_lab"] - zbar_predicted)) < 0.02 * peak
    assert np.max(np.abs(table["delta_f"])) < 0.02 * optics.rayleigh
    assert np.max(np.abs(table["sigma_astig"])) < 0.02

    # round at the tracked plane, and moving on the diagonal
    np.testing.assert_allclose(table["wx"], table["wy"], rtol=1e-9)
    np.testing.assert_allclose(table["x"], table["y"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        table["x"], -params1030.deflection_scale * freq(t_c), rtol=1e-9, atol=1e-15
    )


def test_single_axis_chirp_is_a_cylindrical_lens(params1030):
    """The control: chirp Ax alone and the astigmatism is back, ``Delta F = lens_scale fdot``."""
    wfs, freq, tau = _chirped(params1030, ("Ax",))
    frames = np.linspace(0.0, SWEEP_TIME + tau, 61)
    table = simulate(wfs, frames).spot_table()
    t_c = frames - 0.5 * tau

    delta_predicted = params1030.lens_scale * freq.derivative()(t_c)
    peak = float(np.max(np.abs(delta_predicted)))
    assert np.max(np.abs(table["delta_f"] - delta_predicted)) < 0.02 * peak
    # x carries the whole lens, y none of it: Zbar is half of Delta F
    np.testing.assert_allclose(table["z_lab"], 0.5 * delta_predicted, rtol=1e-9, atol=1e-15)
    assert np.max(np.abs(table["sigma_astig"])) > 1.0  # visibly astigmatic


def _rendered_widths(result, grid, spot):
    """RMS widths of a rendered frame about ``spot``, ``(sigma_x, sigma_y)`` [m].

    A Gaussian of 1/e^2 *intensity* radius ``w`` has ``sigma = w / 2``, so these are directly
    comparable with :attr:`~aodl.field.measure.SpotMetrics.wx` / ``waist0``.
    """
    frame = result.frame(0, grid, z_lab=0.0)
    cols, rows = frame.sum(axis=0), frame.sum(axis=1)
    sigma_x = np.sqrt(np.sum(cols * (grid.x - spot.x) ** 2) / cols.sum())
    sigma_y = np.sqrt(np.sum(rows * (grid.y - spot.y) ** 2) / rows.sum())
    return float(sigma_x), float(sigma_y)


def test_spherical_defocus_shows_up_in_the_rendered_field(params1030):
    """Not just in ``measure``: at peak chirp the spot is round but swollen at ``Z = 0``.

    A spherical lens moves the *whole* spot out of the lab focal plane, so a camera parked at
    ``Z = 0`` records a symmetric, defocused blob; the same chirp on one axis only stretches
    it along that axis.  Both widths follow the textbook
    ``w(0) = waist0 sqrt(1 + (Z_focus/z_R)^2)`` with ``Z_focus = lens_scale fdot(t_c)``.
    """
    optics = params1030.optics
    tau = params1030.channels["Ax"].transit_time
    frames = [0.5 * SWEEP_TIME + 0.5 * tau]  # peak chirp at the beam-centre retarded time
    freq = ramps.min_jerk(0.0, SWEEP_TIME, 0.0, SWEEP_SPAN)
    z_focus = params1030.lens_scale * float(freq.derivative()(0.5 * SWEEP_TIME))
    swollen = 0.5 * optics.waist0 * np.sqrt(1.0 + (z_focus / optics.rayleigh) ** 2)
    assert swollen > 1.5 * (0.5 * optics.waist0)

    half = 14.0 * optics.waist0
    for channels, round_spot in ((("Ax", "Ay"), True), (("Ax",), False)):
        wfs, _, _ = _chirped(params1030, channels)
        result = simulate(wfs, frames)
        spot = result.metrics[0][0]
        grid = FrameGrid(spot.x - half, spot.x + half, 161, spot.y - half, spot.y + half, 161)
        sigma_x, sigma_y = _rendered_widths(result, grid, spot)

        assert sigma_x == pytest.approx(swollen, rel=0.02)  # x is chirped in both cases
        if round_spot:
            assert sigma_y == pytest.approx(sigma_x, rel=0.02)  # spherical: round at Z = 0
        else:
            assert sigma_y == pytest.approx(0.5 * optics.waist0, rel=0.02)  # y still in focus
            assert sigma_x / sigma_y > 1.5  # cylindrical: visibly elongated


# ============================================================== 2. the array on the Table-I grid


def test_five_by_five_array_sits_on_the_table_i_grid(params1030):
    """25 groups, one per trap, on the ``deflection_scale * delta_f`` grid, equal powers."""
    params = _order(params1030, 1)
    wfs, t = _array(params)
    result = simulate(wfs, [t])
    metrics = result.metrics[0]
    optics = params.optics

    assert result.terms(0).n_terms == 25
    assert len(metrics) == 25

    detune = (np.arange(5) - 2.0) * np.array([[DELTA_FX], [DELTA_FY]])
    expected = {
        (round(-params.deflection_scale * fx, 12), round(-params.deflection_scale * fy, 12))
        for fx in detune[0]
        for fy in detune[1]
    }
    measured = np.array([(m.x, m.y) for m in metrics])
    for x, y in measured:
        nearest = min(expected, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
        assert abs(nearest[0] - x) < 0.01 * optics.waist0
        assert abs(nearest[1] - y) < 0.01 * optics.waist0
    assert len({(round(x, 12), round(y, 12)) for x, y in measured}) == 25  # no trap doubled

    pitch_x = np.diff(np.unique(np.round(measured[:, 0], 12)))
    pitch_y = np.diff(np.unique(np.round(measured[:, 1], 12)))
    np.testing.assert_allclose(pitch_x, params.deflection_scale * DELTA_FX, rtol=1e-9)
    np.testing.assert_allclose(pitch_y, params.deflection_scale * DELTA_FY, rtol=1e-9)

    power = np.array([m.power for m in metrics])
    assert power.std() / power.mean() < 0.01  # per-trap spread, order-1 mixing
    np.testing.assert_allclose([m.z_lab for m in metrics], 0.0, atol=1e-18)
    np.testing.assert_allclose([m.delta_f for m in metrics], 0.0, atol=1e-18)


def test_a_common_ramp_transports_the_whole_array(params1030):
    """Eq. S19's lateral term: the array translates rigidly and defocuses spherically.

    A 3x3 ladder pair plus the *same* min-jerk chirp on both channels — the notebook's movie
    drive.  Every trap moves by ``-deflection_scale f(t_c)``, the pitch never changes, and
    ``Zbar`` follows ``lens_scale fdot(t_c)`` with ``Delta F = 0``.
    """
    params = _order(params1030, 1)
    tau = params.channels["Ax"].transit_time
    freq = ramps.min_jerk(0.0, SWEEP_TIME, 0.0, SWEEP_SPAN)
    wfs = WaveformSet(
        {
            "Ax": add_common_ramp(array_tones(3, DELTA_FX, t1=SWEEP_TIME), freq),
            "Ay": add_common_ramp(array_tones(3, DELTA_FY, t1=SWEEP_TIME), freq),
        },
        params,
    ).with_hold_until(SWEEP_TIME + tau)

    frames = np.array([0.6 * tau + 0.5 * tau, 0.5 * SWEEP_TIME, SWEEP_TIME]) + 0.5 * tau
    result = simulate(wfs, frames)
    t_c = frames - 0.5 * tau
    pitch_x = params.deflection_scale * DELTA_FX
    pitch_y = params.deflection_scale * DELTA_FY

    for i, metrics in enumerate(result.metrics):
        assert len(metrics) == 9
        x = np.unique(np.round([m.x for m in metrics], 12))
        y = np.unique(np.round([m.y for m in metrics], 12))
        np.testing.assert_allclose(np.diff(x), pitch_x, rtol=1e-9)
        np.testing.assert_allclose(np.diff(y), pitch_y, rtol=1e-9)
        assert float(np.mean(x)) == pytest.approx(
            -params.deflection_scale * freq(t_c[i]), abs=1e-12
        )

        z_pred = params.lens_scale * float(freq.derivative()(t_c[i]))
        np.testing.assert_allclose([m.z_lab for m in metrics], z_pred, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose([m.delta_f for m in metrics], 0.0, atol=1e-15)


def test_equal_spacings_make_anti_diagonals_degenerate(params1030):
    """``delta_f_x == delta_f_y`` makes ``f_x + f_y`` collide: 25 traps, 9 coherent classes.

    Not a defect of the grouping rule but of the *drive*: two ladders of the same spacing
    have ``f_x + f_y = (n + m) delta_f``, so the nine anti-diagonals of the array are exactly
    frequency-degenerate (``docs/conventions.md`` §4).  The 25 spots are all still there —
    they just are not separable by optical frequency, which is why the tests above (and the
    notebook) detune the two axes.
    """
    params = _order(params1030, 1)
    wfs, t = _array(params, delta_fy=DELTA_FX)
    result = simulate(wfs, [t])

    terms = result.terms(0)
    assert terms.n_terms == 25
    assert len(result.metrics[0]) == 9  # one per anti-diagonal, not one per trap

    spots = {
        (round(float(x), 9), round(float(y), 9))
        for x, y in zip(
            terms.theta1[0] * params.optics.focal_length / params.optics.k,
            terms.theta1[1] * params.optics.focal_length / params.optics.k,
            strict=True,
        )
    }
    assert len(spots) == 25  # every trap is present in the terms


# ================================================================= 3. IM3 ghosts (Eqs. S20-S22)


def test_ladder_edge_ghost_matches_the_im3_coefficients(params1030):
    """A ghost group appears one ``delta_f`` past the array edge, at the predicted intensity.

    With all tone phases zero every IM3 path into a given ghost arrives in phase, so the
    WO-07 §2 coefficients add arithmetically:

        amp_ghost = N_3 (m^3 / 8) + N_2 (m^3 / 16),    m = C A = 0.3,

    against a first-order trap amplitude ``m / 2``.  The reference trap power is therefore
    taken from the *order-1* run of the same drive, which is exactly ``|m/2|^2`` on the
    common scale — no compression, no IM3-on-fundamental correction to unpick.
    """
    reference = simulate(*_array(_order(params1030, 1), phases="zero")).spot_table()
    result = simulate(*_array(_order(params1030, 3), phases="zero"))
    table = result.spot_table()
    optics = params1030.optics
    m = params1030.channels["Ax"].drive_strength
    assert m == pytest.approx(0.3)

    trap_power = float(reference["power"].mean())
    np.testing.assert_allclose(reference["power"], trap_power, rtol=1e-12)

    ladder_x = (np.arange(5) - 2.0) * DELTA_FX
    ladder_y = (np.arange(5) - 2.0) * DELTA_FY
    ghost_fx = 3.0 * DELTA_FX  # one step beyond the +2 delta_f edge
    n3, n2 = _im3_paths(ladder_x, ghost_fx)
    assert (n3, n2) == (4, 2)
    predicted = ((n3 * m**3 / 8.0 + n2 * m**3 / 16.0) / (0.5 * m)) ** 2

    # the ghost sits at the edge trap's position plus one array pitch, on the array's centre row
    x_ghost = -params1030.deflection_scale * ghost_fx
    near = (np.abs(table["x"] - x_ghost) < 0.05 * optics.waist0) & (
        np.abs(table["y"]) < 0.05 * optics.waist0
    )
    assert near.sum() == 1
    measured = float(table["power"][near][0]) / trap_power
    assert measured == pytest.approx(predicted, rel=0.5)  # "within a factor 2"
    assert 0.5 < measured / predicted < 2.0

    # the ghost row/column is a real, resolvable extra spot outside the array
    assert x_ghost < np.min(reference["x"]) - 0.5 * params1030.deflection_scale * DELTA_FX
    assert set(np.round(ladder_y / MHz, 6)) == set(np.round(np.unique(ladder_y) / MHz, 6))


def test_pruned_power_is_negligible(params1030):
    """The 5x5 array at order 3 drops < 1e-6 of its power to the amplitude cuts."""
    result = simulate(*_array(_order(params1030, 3)))
    terms = result.terms(0)
    kept = float(np.sum(np.abs(terms.c) ** 2))
    assert kept > 0.0
    assert terms.pruned_power / kept < 1e-6


def test_schroeder_phases_beat_zero_and_random_on_trap_uniformity(params1030, rng):
    """IM3 products land back *on* the fundamentals; the phases decide how much they hurt.

    An equally spaced ladder is closed under ``f_j + f_k - f_i``, so every trap picks up
    third-order light coherently (Eqs. S20-S22) and the per-trap intensity spreads.  The
    Schroeder progression (Eq. S23/S28) scatters those contributions; zero phases add them
    all in step.
    """
    params = _order(params1030, 3)
    tau = params.channels["Ax"].transit_time
    detunings = (np.arange(8) - 3.5) * DELTA_FX

    spreads = {}
    for mode in ("schroeder", "zero", "random"):
        ladder = array_tones(8, DELTA_FX, phases=mode, t1=10.0 * tau, rng=rng)
        table = simulate(WaveformSet({"Ax": ladder}, params), [2.0 * tau]).spot_table()
        fundamental = np.min(np.abs(table["df_opt"][:, None] - detunings[None, :]), axis=1) < 1.0
        assert fundamental.sum() == 8  # the eight programmed traps, ghosts excluded
        power = table["power"][fundamental]
        spreads[mode] = float(power.std() / power.mean())

    assert spreads["schroeder"] < 0.05
    assert spreads["schroeder"] < 0.25 * spreads["zero"]
    # Schroeder beats *zero* by a large, structural factor (every contribution to a given
    # ghost arrives in step there), but against a random draw the honest claim is only that
    # it wins: a lucky seed can land within a factor of two, so the old `< 0.5 * random`
    # was an over-claim on this fixture's seed rather than a tolerance (WO-09 finding 4).
    assert spreads["schroeder"] < spreads["random"]


# ================================================================================ 4. cost


def test_five_by_five_with_mixing_stays_fast(params1030):
    """``simulate`` + one 512^2 frame under 2 s: notebooks have to stay usable."""
    wfs, t = _array(_order(params1030, 3))
    half = 40.0 * um
    grid = FrameGrid(-half, half, 512, -half, half, 512)

    start = time.perf_counter()
    result = simulate(wfs, [t])
    frame = result.frame(0, grid)
    elapsed = time.perf_counter() - start

    assert frame.shape == (512, 512)
    assert frame.max() > 0.0
    assert elapsed < 2.0, f"5x5 array with mixing_order=3 took {elapsed:.2f} s"
