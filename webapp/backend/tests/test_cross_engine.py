"""The load-bearing test: the FFT engine against ``aodl``'s closed forms at zero misalignment.

This file *may* import ``aodl.engine`` / ``aodl.device`` / ``aodl.field`` — it is a test — and
it is the only place where the two physics paths meet.  ``aodl``'s ``mixing_order = 3``
compresses every fundamental by ``1 - C^2/8``; the engine's ``"weak"`` law is the strictly
first-order Eq. S3 and is therefore compared against ``mixing_order = 1``, while ``"j1"``
(all orders in ``C``) is compared against order 3.
"""

from __future__ import annotations

import numpy as np
import pytest
from _advanced_fixtures import (
    DEFLECTION,
    PARAMS,
    TAU,
    W0,
    Z_R,
    aodl_spot,
    aodl_spots,
    fading_hold,
    local_maxima,
    pick_and_place,
    reparam,
    run_frame,
    static_array,
    window_power,
    with_order,
)
from scipy.special import j1

from aodl.field.focal import FrameGrid, intensity_frame
from aodl.units import MHz, um, us
from aodl_advanced.engine import GridSpec, XzSpec

ORDER1 = with_order(PARAMS, 1)
TIMES = (2 * TAU, 70 * us + 0.5 * TAU)


@pytest.fixture(scope="module")
def pick_place_frames():
    wfs = reparam(pick_and_place().wfs, ORDER1)
    return wfs, [(t, aodl_spot(wfs, t), run_frame(wfs, t, order_model="weak")) for t in TIMES]


def test_pick_and_place_centroid_waists_and_focus(pick_place_frames) -> None:
    _, frames = pick_place_frames
    for t, spot, frame in frames:
        m = frame.metrics
        assert abs(m.x - spot.x) < 0.02 * W0, t
        assert abs(m.y - spot.y) < 0.02 * W0, t
        assert m.wx == pytest.approx(spot.wx, rel=0.03)
        assert m.wy == pytest.approx(spot.wy, rel=0.03)
        assert m.z_best_x is not None and m.z_best_y is not None
        assert abs(m.z_best_x - spot.z_lab) < 0.1 * Z_R
        assert abs(m.z_best_y - spot.z_lab) < 0.1 * Z_R
        assert frame.n_groups == 1 and frame.groups_dropped == 0


def test_pick_and_place_power_ratio_and_absolute_scale(pick_place_frames) -> None:
    wfs, frames = pick_place_frames
    (_, s0, f0), (_, s1, f1) = frames
    assert (f1.metrics.power / f0.metrics.power) == pytest.approx(s1.power / s0.power, rel=0.02)
    # bonus: both engines drop the same prefactors, so the scale is shared absolutely
    for t, spot, frame in frames:
        assert frame.metrics.power == pytest.approx(spot.power, rel=0.01)
        grid = FrameGrid(spot.x - 6 * um, spot.x + 6 * um, 96, spot.y - 6 * um, spot.y + 6 * um, 96)
        from aodl.device.aodl import build_terms

        exact = intensity_frame(build_terms(wfs, t), PARAMS.optics, grid, spot.z_lab)
        assert frame.metrics.peak == pytest.approx(exact.max(), rel=0.01)
        assert np.max(np.abs(frame.xy - exact)) < 0.01 * exact.max()


def test_two_by_two_array_spots() -> None:
    wfs = reparam(static_array(2, 2).wfs, ORDER1)
    t = 2 * TAU
    spots = aodl_spots(wfs, t)
    assert len(spots) == 4
    frame = run_frame(
        wfs, t, order_model="weak", grid=GridSpec(10 * um, 12 * um, 192), xz=None, centre=(0.0, 0.0)
    )
    found = local_maxima(frame)
    assert len(found) == 4
    for spot in spots:
        d = min(np.hypot(x - spot.x, y - spot.y) for x, y, _ in found)
        assert d < 0.05 * W0


def test_fading_shadow_pair_has_half_the_main_power() -> None:
    plan = fading_hold(delta_f=3 * MHz)
    wfs = reparam(plan.wfs, ORDER1)
    event = [e for e in plan.report.fade_events if e.axis == "x"][1]
    t = event.time + 0.5 * TAU
    shadow = event.shadow
    assert shadow == pytest.approx(DEFLECTION * 3 * MHz)
    spots = aodl_spots(wfs, t)
    assert len(spots) == 3  # (a,a), (a-1,a-1) at the centre and the degenerate shadow pair
    by_x = sorted(spots, key=lambda m: abs(m.df_opt))
    shadow_group = by_x[0]  # the pair's tag is the mean of the two neighbours' tags
    main_power = sum(m.power for m in spots) - shadow_group.power
    frame = run_frame(
        wfs,
        t,
        order_model="weak",
        grid=GridSpec(38 * um, 6 * um, 192),
        xz=None,
        centre=(0.0, 0.0),
    )
    assert frame.n_groups == 3
    left = window_power(frame, -shadow, 8 * um)
    right = window_power(frame, shadow, 8 * um)
    centre = window_power(frame, 0.0, 8 * um)
    assert left + right == pytest.approx(shadow_group.power, rel=0.02)
    assert centre == pytest.approx(main_power, rel=0.02)
    assert left == pytest.approx(0.5 * centre, rel=0.05)
    assert right == pytest.approx(0.5 * centre, rel=0.05)


def test_j1_versus_weak_single_tone_peak_ratio() -> None:
    wfs = reparam(static_array(1, 1).wfs, ORDER1)
    t = 2 * TAU
    weak = run_frame(wfs, t, order_model="weak", xz=None)
    strong = run_frame(wfs, t, order_model="j1", xz=None)
    c = PARAMS.channels["Ax"].drive_strength
    expected = (2.0 * j1(c) / c) ** 8  # four channels, intensity
    assert c == 0.3
    assert strong.metrics.peak / weak.metrics.peak == pytest.approx(expected, rel=1e-3)
    # the single-channel law itself: (2 J1(C)/C)^2 per channel
    assert (strong.metrics.peak / weak.metrics.peak) ** 0.25 == pytest.approx(
        (2.0 * j1(c) / c) ** 2, rel=1e-3
    )


@pytest.mark.parametrize(("drive_strength", "tolerance"), [(0.15, 0.05), (0.3, 0.10)])
def test_im3_ghost_power_against_mixing_order_3(drive_strength: float, tolerance: float) -> None:
    """A 4-tone Schroeder ladder's IM3 ghost one pitch outside the array.

    ``aodl`` truncates the expansion at ``C^3``; the ``J1`` law carries every order, and the
    ``C^5`` terms are ~7 % of the ghost at ``C = 0.3`` — so the 5 % agreement is asked at
    ``C = 0.15`` and the known saturation difference is bounded at the default drive.
    """
    params3 = with_order(PARAMS, 3, drive_strength)
    wfs = reparam(static_array(4, 1).wfs, params3)
    t = 2 * TAU
    spots = aodl_spots(wfs, t)
    pitch = DEFLECTION * 1 * MHz
    columns = sorted(m.x for m in spots if m.power > 0.1 * max(s.power for s in spots))
    assert len(columns) == 4
    ghost_x = columns[-1] + pitch
    ghost = min(spots, key=lambda m: abs(m.x - ghost_x))
    assert abs(ghost.x - ghost_x) < 1e-9
    frame = run_frame(
        wfs,
        t,
        order_model="j1",
        comb_dft_points=16,
        grid=GridSpec(40 * um, 6 * um, 192),
        xz=None,
        centre=(0.0, 0.0),
    )
    mine = window_power(frame, ghost_x, 0.45 * pitch)
    assert mine == pytest.approx(ghost.power, rel=tolerance)
    # the fundamentals agree to the compression difference between J1 and its C^3 truncation
    main = max(spots, key=lambda m: m.power)
    assert window_power(frame, main.x, 0.45 * pitch) == pytest.approx(main.power, rel=0.01)


def test_xz_slice_agrees_with_aodl_on_a_defocused_frame() -> None:
    wfs = reparam(pick_and_place().wfs, ORDER1)
    t = 2 * TAU
    spot = aodl_spot(wfs, t)
    frame = run_frame(wfs, t, order_model="weak", xz=XzSpec(24, 8 * um))
    assert frame.xz is not None and frame.xz_z is not None
    from aodl.device.aodl import build_terms
    from aodl.field.focal import intensity_slice_xz

    exact = intensity_slice_xz(build_terms(wfs, t), PARAMS.optics, frame.x, frame.xz_z, frame.row_y)
    assert np.max(np.abs(frame.xz - exact)) < 0.01 * exact.max()
    assert spot.z_lab == pytest.approx(frame.plane_z, rel=1e-9)
