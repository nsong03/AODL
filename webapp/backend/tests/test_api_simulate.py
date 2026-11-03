"""``POST /api/simulate`` and ``POST /api/frame`` (API.md §5.1-5.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import as_json

from aodl.units import um
from aodl_webapp.examples import example_request
from aodl_webapp.planner import get_plan
from aodl_webapp.schemas import PackedImage
from aodl_webapp.simulation import pack_image, unpack_image

TAU_US = 7.5e-3 / 650.0 / 1e-6  # D / v
SPOT_KEYS = (
    "x_um",
    "y_um",
    "z_um",
    "delta_f_um",
    "sigma_astig",
    "wx_um",
    "wy_um",
    "power",
    "power_coherent",
    "df_opt_MHz",
    "group",
    "n_terms",
)


def _simulate(client, req, **kwargs):
    response = client.post("/api/simulate", json={"plan": as_json(req), **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def test_frame_count_defaults_and_struct_of_arrays(client, pick_and_place):
    body = _simulate(client, pick_and_place, n_frames=60)
    assert len(body["times_us"]) == 60 == len(body["frames"]) == len(body["tracked_z_um"])
    assert body["times_us"][0] == 0.0
    # default end: min(T + tau, t_span[1] - tau/2) = 140 + 11.54
    assert body["times_us"][-1] == pytest.approx(140.0 + TAU_US, abs=1e-6)
    for frame in body["frames"]:
        assert set(frame) == set(SPOT_KEYS)
        n = len(frame["x_um"])
        assert all(len(frame[key]) == n for key in SPOT_KEYS)
    assert body["constants"]["tau_us"] == pytest.approx(TAU_US)
    assert body["power_ref"] > 0.0


def test_power_ref_is_the_plateau_trap_power(client, pick_and_place, single_lift_hold):
    """``power_ref`` normalises a trap to 1: on an Eq. S19 drive it equals the lone spot's
    plateau power, and on a fading drive the two coincident half-power groups of a hand-over
    merge back to ~1 while the Eq. S31 shadows read ~0.5 (API.md §5.1)."""
    body = _simulate(client, pick_and_place, n_frames=40)
    tau = body["constants"]["tau_us"]
    late = [f for t, f in zip(body["times_us"], body["frames"]) if t > 2 * tau and f["power"]]
    plateau = np.median([max(f["power"]) for f in late])
    assert body["power_ref"] == pytest.approx(plateau, rel=0.02)

    plan = client.post("/api/plan", json=as_json(single_lift_hold)).json()
    event = next(e for e in plan["fade_events"] if e["axis"] == "x")
    t = event["time_us"] + 0.5 * plan["constants"]["tau_us"]
    body = _simulate(client, single_lift_hold, n_frames=2, t_start_us=t, t_end_us=t + 1e-3)
    frame = body["frames"][0]
    x = np.asarray(frame["x_um"])
    power = np.asarray(frame["power"]) / body["power_ref"]
    main = np.abs(x) < 0.05
    assert main.sum() == 2  # the two coincident main-trap groups
    assert power[main].sum() == pytest.approx(1.0, abs=0.06)
    assert np.all(np.abs(power[~main] - 0.5) < 0.06)


def test_frames_before_half_tau_are_empty(client, pick_and_place):
    body = _simulate(client, pick_and_place, n_frames=40, t_start_us=0.0, t_end_us=2.0 * TAU_US)
    for t, frame in zip(body["times_us"], body["frames"], strict=True):
        if t < 0.5 * TAU_US:
            assert frame["x_um"] == [], f"a frame at {t} us < tau/2 must be dark"
        elif t > 0.6 * TAU_US:
            assert len(frame["x_um"]) == 1
    assert any("dark" in note for note in body["notes"])


def test_pick_and_place_arrives_at_20_um(client, pick_and_place):
    """At ``t = T + tau/2`` the beam centre sees the drive at ``T``: the spot is at 20 µm."""
    t = 140.0 + 0.5 * TAU_US
    body = _simulate(client, pick_and_place, n_frames=2, t_start_us=t, t_end_us=t)
    for frame in body["frames"]:
        assert len(frame["x_um"]) == 1
        assert frame["x_um"][0] == pytest.approx(20.0, abs=1e-3)
        assert frame["y_um"][0] == pytest.approx(0.0, abs=1e-3)
        assert frame["z_um"][0] == pytest.approx(0.0, abs=1e-3)


def test_shadow_pair_appears_as_two_spots(client, single_lift_hold):
    """At an x hand-over the Eq. S31 shadows are two spots at ``+-10.3 Delta f`` µm with half
    the main spot's power, and the main spot keeps ~unit power (no collapsed phantom)."""
    plan = client.post("/api/plan", json=as_json(single_lift_hold)).json()
    x_events = [e for e in plan["fade_events"] if e["axis"] == "x"]
    assert x_events
    event = x_events[0]
    t = event["time_us"] + 0.5 * TAU_US
    body = _simulate(client, single_lift_hold, n_frames=2, t_start_us=t, t_end_us=t)
    frame = body["frames"][0]
    x = np.asarray(frame["x_um"])
    power = np.asarray(frame["power"])
    assert event["shadow_um"] == pytest.approx(10.3 * 12.0, rel=1e-3)
    # Four spots, not three: the main trap is the dying (a, a) and rising (a-1, a-1) rung
    # pairs, 2 delta_f apart in optical frequency and therefore two coincident groups whose
    # powers add to the trap's; the shadows (a, a-1) and (a-1, a) share one tag (one group,
    # two clusters).  A collapsed group would instead put a phantom of the shadows' power on
    # the trap and no spot at +-123.6 um.
    shadow = event["shadow_um"]
    main = np.abs(x) < 0.05
    left = np.abs(x + shadow) < 0.05
    right = np.abs(x - shadow) < 0.05
    assert main.sum() == 2 and left.sum() == 1 and right.sum() == 1, frame
    assert x.size == 4
    # unit power is a plateau spot's power: a frame well inside the hold, away from any fade
    plateau_t = 0.5 * (event["time_us"] + x_events[-1]["time_us"]) + 0.5 * TAU_US
    plateau_t = min(plateau_t, event["time_us"] + 60.0)
    plateau = _simulate(
        client, single_lift_hold, n_frames=2, t_start_us=plateau_t, t_end_us=plateau_t
    )["frames"][0]
    unit = max(plateau["power"])
    assert power[main].sum() == pytest.approx(unit, rel=0.05)
    assert power[left][0] == pytest.approx(0.5 * unit, rel=0.1)
    assert power[right][0] == pytest.approx(0.5 * unit, rel=0.1)
    groups = np.asarray(frame["group"])
    assert groups[left][0] == groups[right][0]  # one frequency group, two clusters
    assert groups[main][0] != groups[main][1]  # the co-located pair are two groups
    assert all(n == 1 for n in frame["n_terms"])


def test_rounding_and_power_cut(client, array_lift_hold):
    body = _simulate(client, array_lift_hold, n_frames=12, t_start_us=3 * TAU_US, t_end_us=300.0)
    for frame in body["frames"]:
        for key in SPOT_KEYS[:-2]:
            for value in frame[key]:
                if value != 0.0:
                    assert float(f"{value:.6g}") == value, (key, value)
        if frame["power"]:
            peak = max(frame["power"])
            assert min(frame["power"]) >= 1e-3 * peak * (1 - 1e-6)


def test_movie_mixing_order_fallback_and_override(client):
    flagship = example_request("flagship_10x10")
    body = _simulate(client, flagship, n_frames=3, t_start_us=2 * TAU_US, t_end_us=3 * TAU_US)
    assert any("fell back to mixing order 1" in note for note in body["notes"])
    n_order1 = len(body["frames"][0]["x_um"])
    body = _simulate(
        client,
        flagship,
        n_frames=2,
        t_start_us=2 * TAU_US,
        t_end_us=3 * TAU_US,
        movie_mixing_order=3,
    )
    # pinned to the hardware's own order: no fallback, no note, and the IM3 ghosts appear
    assert not any("mixing order" in note for note in body["notes"])
    assert len(body["frames"][0]["x_um"]) > n_order1
    body = _simulate(
        client,
        example_request("array_arbitrary_path"),
        n_frames=2,
        t_start_us=2 * TAU_US,
        t_end_us=3 * TAU_US,
        movie_mixing_order=1,
    )
    assert any("mixing order 1 as requested" in note for note in body["notes"])
    assert len(body["frames"][0]["x_um"]) == 9
    # a 3x3 at order 3 holds 49 groups (9 traps + 40 IM3 ghosts): no fallback, no note.  The
    # ghosts carry ~1e-4 of a trap's power at drive_strength 0.3, so the movie's 1e-3 cut
    # drops them (9 spots), while the uncut spot list of /api/frame shows all 49.
    three = example_request("array_arbitrary_path")
    body = _simulate(client, three, n_frames=3, t_start_us=2 * TAU_US, t_end_us=3 * TAU_US)
    assert not any("mixing order" in note for note in body["notes"])
    assert len(body["frames"][0]["x_um"]) == 9
    frame = client.post(
        "/api/frame",
        json={
            "plan": as_json(three),
            "time_us": 2 * TAU_US,
            "grid": {"x0_um": -15, "x1_um": 15, "nx": 16, "y0_um": -15, "y1_um": 15, "ny": 16},
        },
    ).json()
    assert len(frame["spots"]) == 49
    ghosts = sorted(s["power"] for s in frame["spots"])[:40]
    traps = sorted(s["power"] for s in frame["spots"])[40:]
    assert max(ghosts) < 1e-3 * min(traps)


def test_bounds_come_from_the_requested_trajectory(client, single_lift_hold):
    """Shadows at +-123 µm must not blow the canvas up: bounds follow the request."""
    body = _simulate(client, single_lift_hold, n_frames=4)
    b = body["bounds_um"]
    w0 = body["constants"]["waist0_um"]
    assert b["x0"] == pytest.approx(-6 * w0) and b["x1"] == pytest.approx(6 * w0)
    assert b["z0"] == pytest.approx(-body["constants"]["rayleigh_um"])
    assert b["z1"] == pytest.approx(10.0 + body["constants"]["rayleigh_um"])


def test_late_frame_is_a_coverage_error(client, pick_and_place):
    response = client.post(
        "/api/simulate",
        json={
            "plan": as_json(pick_and_place),
            "n_frames": 2,
            "t_start_us": 1000.0,
            "t_end_us": 1000.0,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "coverage"
    response = client.post(
        "/api/frame",
        json={
            "plan": as_json(pick_and_place),
            "time_us": 1000.0,
            "grid": {"x0_um": -5, "x1_um": 25, "nx": 32, "y0_um": -5, "y1_um": 5, "ny": 16},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "coverage"
    assert "needs drive time" in response.json()["error"]["message"]


def test_n_frames_is_bounded(client, pick_and_place):
    assert (
        client.post(
            "/api/simulate", json={"plan": as_json(pick_and_place), "n_frames": 1}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/simulate", json={"plan": as_json(pick_and_place), "n_frames": 241}
        ).status_code
        == 422
    )


# ------------------------------------------------------------------------------- /frame


def _frame(client, req, **kwargs):
    payload = {
        "plan": as_json(req),
        "time_us": 3 * TAU_US,
        "plane": "tracked",
        "grid": {"x0_um": -5.0, "x1_um": 5.0, "nx": 96, "y0_um": -4.0, "y1_um": 4.0, "ny": 64},
        "xz": {"nx": 48, "nz": 40, "z_half_um": 8.0, "row_y_um": None},
    }
    payload.update(kwargs)
    response = client.post("/api/frame", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_frame_images_have_the_requested_shape_and_pack_exactly(client, static_tweezer):
    body = _frame(client, static_tweezer)
    xy = body["xy"]
    assert xy["image"]["shape"] == [64, 96]
    assert len(xy["x_um"]) == 96 and len(xy["y_um"]) == 64
    assert xy["image"]["peak"] > 0.0
    xz = body["xz"]
    assert xz["image"]["shape"] == [40, 48]
    assert len(xz["z_um"]) == 40 and xz["z_um"][0] == pytest.approx(body["plane_z_um"] - 8.0)
    assert body["plane_z_um"] == pytest.approx(0.0, abs=1e-3)  # at rest on the focal plane
    image = unpack_image(PackedImage(**xy["image"]))
    assert image.shape == (64, 96)
    assert image.max() == pytest.approx(xy["image"]["peak"])

    rng = np.random.default_rng(3)
    arr = rng.uniform(0.0, 2.5, (7, 11))
    packed = pack_image(arr)
    assert packed.peak == pytest.approx(arr.max())
    np.testing.assert_allclose(unpack_image(packed), arr, atol=0.5 * packed.peak / 65535 + 1e-15)


def test_exact_peak_matches_the_spot_metrics_scale(client, static_tweezer):
    """A static tweezer at ``t = 3 tau``: the frame's peak is ``2 P / (pi wx wy)`` to 2 %."""
    body = _frame(
        client,
        static_tweezer,
        time_us=3 * TAU_US,
        grid={"x0_um": -3.0, "x1_um": 3.0, "nx": 241, "y0_um": -3.0, "y1_um": 3.0, "ny": 241},
        xz=None,
    )
    spot = max(body["spots"], key=lambda s: s["power"])
    expected = 2.0 * spot["power"] / (math.pi * spot["wx_um"] * um * spot["wy_um"] * um)
    assert body["xy"]["image"]["peak"] == pytest.approx(expected, rel=0.02)
    assert body["xz"] is None
    assert len(body["cuts"]["z_um"]) == 96  # the default axial cut


def test_cuts_peak_at_the_brightest_spot(client, pick_and_place):
    t = 70.0 + 0.5 * TAU_US  # mid-translate: X ~ 10 µm, lifted 5 µm
    body = _frame(
        client,
        pick_and_place,
        time_us=t,
        grid={"x0_um": 0.0, "x1_um": 20.0, "nx": 201, "y0_um": -3.0, "y1_um": 3.0, "ny": 61},
    )
    bright = max(body["spots"], key=lambda s: s["power"])
    assert bright["x_um"] == pytest.approx(10.0, abs=0.05)
    assert body["plane_z_um"] == pytest.approx(5.0, abs=0.05)
    cuts = body["cuts"]
    assert cuts["through"]["x_um"] == pytest.approx(bright["x_um"])
    i_x = np.asarray(cuts["i_x"])
    assert cuts["x_um"][int(np.argmax(i_x))] == pytest.approx(bright["x_um"], abs=0.11)
    i_z = np.asarray(cuts["i_z"])
    assert cuts["z_um"][int(np.argmax(i_z))] == pytest.approx(bright["z_um"], abs=0.45)
    assert body["xz"]["row_y_um"] == pytest.approx(bright["y_um"], abs=1e-3)


def test_fixed_and_explicit_planes(client, pick_and_place):
    t = 70.0 + 0.5 * TAU_US  # mid-translate: the trap is at X ~ 10 µm, Z = 5 µm
    grid = {"x0_um": 5.0, "x1_um": 15.0, "nx": 96, "y0_um": -4.0, "y1_um": 4.0, "ny": 64}
    fixed = _frame(client, pick_and_place, time_us=t, plane="fixed", grid=grid, xz=None)
    assert fixed["plane_z_um"] == 0.0
    explicit = _frame(client, pick_and_place, time_us=t, plane={"z_um": 2.5}, grid=grid, xz=None)
    assert explicit["plane_z_um"] == 2.5
    tracked = _frame(client, pick_and_place, time_us=t, grid=grid, xz=None)
    assert tracked["plane_z_um"] == pytest.approx(5.0, abs=0.05)
    # the trap is at Z = 5 µm: the tracked plane is the brightest of the three
    assert (
        tracked["xy"]["image"]["peak"]
        > explicit["xy"]["image"]["peak"]
        > fixed["xy"]["image"]["peak"]
    )


def test_frame_spots_match_the_simulate_frame(client, single_lift_hold):
    plan = client.post("/api/plan", json=as_json(single_lift_hold)).json()
    # the first *x* hand-over: fade_events[0] is a y event at t = 0 (xi_y = 1/2 puts a y
    # fade centre exactly where f_Z = 0), and t = tau/2 is still dark
    event = next(e for e in plan["fade_events"] if e["axis"] == "x")
    t = event["time_us"] + 0.5 * TAU_US
    body = _frame(client, single_lift_hold, time_us=t, xz=None)
    assert len(body["spots"]) == 4
    sim = _simulate(client, single_lift_hold, n_frames=2, t_start_us=t, t_end_us=t)["frames"][0]
    assert sorted(round(s["x_um"], 3) for s in body["spots"]) == sorted(
        round(x, 3) for x in sim["x_um"]
    )
    plan_obj = get_plan(single_lift_hold)
    assert plan_obj.report.mode == "shepard"
