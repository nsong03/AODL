"""``Path`` moves through the API: the examples that use them, and a user-drawn loop."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import as_json

from aodl_webapp.examples import EXAMPLES_BY_ID, spherical_spiral, tilted_circle

TAU_US = 7.5e-3 / 650.0 / 1e-6


def test_generators_have_the_documented_geometry():
    spiral = spherical_spiral()
    assert len(spiral) == 24
    centre = np.array([0.0, 0.0, 6.0])
    for point in spiral:
        assert np.linalg.norm(np.asarray(point) - centre) == pytest.approx(6.0)
    assert spiral[-1] == pytest.approx((0.0, 0.0, 12.0), abs=1e-12)  # the north pole

    circle = tilted_circle()
    assert len(circle) == 12 and circle[-1] == (0.0, 0.0, 0.0)
    centre = np.array([-8.0, 0.0, 0.0])
    for x, y, z in circle[:-1]:
        assert np.hypot(x - centre[0], np.hypot(y, z)) == pytest.approx(8.0)
        assert np.degrees(np.arctan2(z, y)) % 180 == pytest.approx(30.0) or (y == 0 and z == 0)


def test_closed_path_examples_return_home(client):
    for example_id in ("sphere_loop", "array_arbitrary_path"):
        request = EXAMPLES_BY_ID[example_id].request
        assert request.moves[0 if example_id == "sphere_loop" else 1].closed
        body = client.post("/api/plan", json=as_json(request)).json()
        traj = body["trajectory"]
        for key in ("x_um", "y_um", "z_um"):
            assert traj[key][-1] == pytest.approx(0.0, abs=1e-9), (example_id, key)


def test_user_drawn_circle_is_followed_by_the_simulated_spot(client):
    """A 24-point circle of radius 10 µm in XY over 300 µs: away from the ends the spot at
    ``t + tau/2`` sits on the requested circle to 0.1 µm (DESIGN §5, W4 check 3)."""
    n, radius = 24, 10.0
    points = []
    for k in range(1, n + 1):
        phi = 2 * np.pi * k / n
        points.append([radius * np.cos(phi) - radius, radius * np.sin(phi), 0.0])
    points[-1] = [0.0, 0.0, 0.0]
    body = {
        "moves": [{"kind": "path", "points_um": points, "duration_us": 300.0, "closed": True}],
    }
    plan = client.post("/api/plan", json=body)
    assert plan.status_code == 200, plan.text
    plan = plan.json()
    assert plan["mode"] == "s19"
    traj = plan["trajectory"]
    sim = client.post(
        "/api/simulate",
        json={"plan": body, "n_frames": 25, "t_start_us": 40.0, "t_end_us": 260.0},
    ).json()
    for t, frame in zip(sim["times_us"], sim["frames"], strict=True):
        assert len(frame["x_um"]) == 1
        t_drive = t - 0.5 * TAU_US
        x_req = np.interp(t_drive, traj["t_us"], traj["x_um"])
        y_req = np.interp(t_drive, traj["t_us"], traj["y_um"])
        assert frame["x_um"][0] == pytest.approx(x_req, abs=0.1)
        assert frame["y_um"][0] == pytest.approx(y_req, abs=0.1)
        # and the spot really is on the circle
        assert np.hypot(frame["x_um"][0] + radius, frame["y_um"][0]) == pytest.approx(
            radius, abs=0.15
        )
    # the band check saw the full lateral span: Eq. S19 puts half the 20 µm of x on each
    # member of the pair, 20 / 10.3 / 2 MHz of live span per channel
    assert plan["band_usage"]["Ax"]["f_max_MHz"] - plan["band_usage"]["Ax"][
        "f_min_MHz"
    ] == pytest.approx(20.0 / 10.3 / 2.0, abs=0.02)
