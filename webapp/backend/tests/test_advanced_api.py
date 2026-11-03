"""``/api/advanced/*`` through the FastAPI test client (W2 §5, API.md §6).

Needs the core package's frozen helpers (``aodl_webapp.simulation``); the module is skipped
with a clear reason while those are not importable yet.
"""

from __future__ import annotations

import base64
import math

import numpy as np
import pytest

pytest.importorskip(
    "aodl_webapp.simulation",
    reason="aodl_webapp.simulation (W1's frozen pack_image / tracked_plane / frame_spots) "
    "is not importable yet",
)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from aodl_webapp.errors import install_handlers  # noqa: E402
from aodl_webapp.planner import get_plan  # noqa: E402
from aodl_webapp.routes import advanced  # noqa: E402
from aodl_webapp.schemas import PlanRequest  # noqa: E402

PICK_AND_PLACE = {
    "hardware": {},
    "array": {"mx": 1, "my": 1},
    "moves": [
        {"kind": "lift", "dz_um": 5.0, "duration_us": 40.0},
        {"kind": "translate", "dx_um": 20.0, "dy_um": 0.0, "duration_us": 60.0},
        {"kind": "lift", "dz_um": -5.0, "duration_us": 40.0},
    ],
}
TAU_US = 7.5e-3 / 650.0 * 1e6


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    install_handlers(app)
    app.include_router(advanced.router, prefix="/api")
    return TestClient(app)


def _unpack(image: dict) -> np.ndarray:
    raw = base64.b64decode(image["data"])
    codes = np.frombuffer(raw, dtype="<u2").reshape(image["shape"])
    return image["peak"] * codes / 65535.0


def test_efficiency_route(client: TestClient) -> None:
    r = client.post("/api/advanced/efficiency", json={"hardware": {}, "model": {}, "n": 101})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["f_MHz"]) == len(body["eta_first"]) == len(body["deflection_mrad"]) == 101
    assert body["eta_second_at_center"] == pytest.approx(body["eta_first"])  # f_tpm = f_center
    assert body["bandwidth_3dB_MHz"] == pytest.approx(27.9, abs=0.5)
    assert 5.0 < body["first_aod_usable_MHz"] < 30.0
    assert body["band_MHz"] == [90.0, 110.0] and body["f_tpm_MHz"] == 100.0
    assert len(body["pair"]["eta"]) == 121 and len(body["pair"]["eta"][0]) == 121
    assert len(body["ridge"]["eta_max"]) == 121
    assert max(body["eta_first"]) == pytest.approx(0.85, rel=1e-6)
    assert "Bragg" in body["assumptions"]
    assert client.post("/api/advanced/efficiency", json={"n": 2}).status_code == 422


def test_trajectory_efficiency_route(client: TestClient) -> None:
    r = client.post(
        "/api/advanced/trajectory_efficiency", json={"plan": PICK_AND_PLACE, "model": {}, "n": 50}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["t_us"]) == 50 and set(body["per_channel"]) == {"Ax", "Bx", "Ay", "By"}
    assert len(body["product"]) == 50
    assert body["min_product"] == pytest.approx(min(body["product"]))
    assert body["product"][0] == pytest.approx(0.85**4, rel=1e-6)
    assert body["min_product"] < 0.85**4


def test_simulate_route_shapes_and_reference(client: TestClient) -> None:
    advanced._REFERENCE.clear()
    req = {
        "plan": PICK_AND_PLACE,
        "misalignment": {"Bx": {"rotation_mrad": 10.0}},
        "times_us": [2 * TAU_US, 70.0 + 0.5 * TAU_US],
        "grid": {"half_x_um": 6.0, "half_y_um": 6.0, "n": 64, "center": "tracked"},
        "xz": {"nz": 16, "z_half_um": 8.0},
        "model": {"pupil_n": 128, "order_model": "weak", "comb_dft_points": 8, "max_groups": 32},
        "reference": True,
    }
    r = client.post("/api/advanced/simulate", json=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["frames"]) == 2 and len(body["plan_id"]) == 16
    frame = body["frames"][1]
    assert frame["time_us"] == pytest.approx(70.0 + 0.5 * TAU_US)
    assert frame["plane_z_um"] == pytest.approx(5.0, abs=0.05)
    xy = _unpack(frame["xy"]["image"])
    assert xy.shape == (64, 64) and len(frame["xy"]["x_um"]) == 64
    assert frame["xz"] is not None and _unpack(frame["xz"]["image"]).shape == (16, 64)
    m, ref = frame["metrics"], frame["reference_metrics"]
    assert ref["strehl"] == 1.0 and ref["peak_rel"] == 1.0
    assert m["x_um"] == pytest.approx(10.0, abs=0.3)
    # rotation of Bx moves Y by tan(theta) * deflection_scale * f_Bx(t_c), f_Bx carrying the
    # lateral half-split X/(2 deflection_scale) *and* the f_Z co-chirp of the sustained lift
    plan = get_plan(PlanRequest(**PICK_AND_PLACE))
    t_c = (70.0 + 0.5 * TAU_US) * 1e-6 - 0.5 * TAU_US * 1e-6
    f_bx = float(plan.wfs.channels["Bx"].tones[0].f(t_c))
    expected_dy = math.tan(10e-3) * plan.params.deflection_scale * f_bx * 1e6
    assert expected_dy > 0.1
    assert m["y_um"] - ref["y_um"] == pytest.approx(expected_dy, rel=0.05)
    assert m["z_best_x_um"] is not None and m["astig_um"] is not None
    assert 0.95 < m["strehl"] <= 1.0005  # sub-pixel refined peak: a pure rotation barely dims
    assert body["static_offset_um"]["Bx"]["y"] == pytest.approx(10.3, rel=0.01)
    assert body["static_offset_um"]["Ax"] == {"x": 0.0, "y": 0.0}
    assert any("brightest spot" in note for note in body["notes"])
    # the aligned reference is cached and shared by requests differing only in misalignment
    assert len(advanced._REFERENCE) == 1
    req2 = dict(req, misalignment={"Ax": {"delay_ns": 100.0}})
    assert client.post("/api/advanced/simulate", json=req2).status_code == 200
    assert len(advanced._REFERENCE) == 1
    assert advanced.reference_key(
        advanced.AdvancedSimulateRequest(**req)
    ) == advanced.reference_key(advanced.AdvancedSimulateRequest(**req2))


def test_simulate_route_without_xz_and_reference(client: TestClient) -> None:
    req = {
        "plan": PICK_AND_PLACE,
        "times_us": [2 * TAU_US],
        "grid": {"n": 32, "center": {"x_um": 0.0, "y_um": 0.0}},
        "model": {"pupil_n": 128, "order_model": "j1", "comb_dft_points": 1},
        "reference": False,
    }
    r = client.post("/api/advanced/simulate", json=req)
    assert r.status_code == 200, r.text
    frame = r.json()["frames"][0]
    assert frame["xz"] is None and frame["reference_metrics"] is None
    assert frame["metrics"]["z_best_x_um"] is None and frame["metrics"]["strehl"] is None


def test_simulate_guards_and_coverage(client: TestClient) -> None:
    base = {"plan": PICK_AND_PLACE, "times_us": [2 * TAU_US], "grid": {"n": 32}}
    assert (
        client.post("/api/advanced/simulate", json=dict(base, times_us=[1.0] * 25)).status_code
        == 422
    )
    assert (
        client.post("/api/advanced/simulate", json=dict(base, grid={"n": 193})).status_code == 422
    )
    assert (
        client.post("/api/advanced/simulate", json=dict(base, model={"pupil_n": 300})).status_code
        == 422
    )
    assert client.post("/api/advanced/simulate", json=dict(base, xz={"nz": 33})).status_code == 422
    assert (
        client.post(
            "/api/advanced/simulate", json=dict(base, model={"comb_dft_points": 65})
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/advanced/simulate", json=dict(base, model={"max_groups": 257})
        ).status_code
        == 422
    )
    late = client.post(
        "/api/advanced/simulate", json=dict(base, times_us=[5000.0], model={"pupil_n": 128})
    )
    assert late.status_code == 400 and late.json()["error"]["code"] == "coverage"


def test_spectrum_route(client: TestClient) -> None:
    r = client.post(
        "/api/advanced/spectrum",
        json={"plan": PICK_AND_PLACE, "channel": "Bx", "t_center_us": 50.0, "window_us": 20.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["f_MHz"]) == len(body["psd_dB"]) > 1000
    assert body["crest_factor"] == pytest.approx(2**0.5, rel=0.02)
    assert body["peak_depth_rad"] == pytest.approx(0.3, rel=0.02)
    assert body["band_MHz"] == [90.0, 110.0]
    assert max(body["psd_dB"]) == 0.0
    bad = client.post(
        "/api/advanced/spectrum",
        json={"plan": PICK_AND_PLACE, "channel": "Ax", "t_center_us": 5000.0},
    )
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "coverage"


def test_simulate_route_aligned_request_is_its_own_reference(client: TestClient) -> None:
    req = {
        "plan": PICK_AND_PLACE,
        "times_us": [2 * TAU_US],
        "grid": {"n": 32},
        "model": {"pupil_n": 128, "order_model": "weak"},
    }
    r = client.post("/api/advanced/simulate", json=req)
    assert r.status_code == 200, r.text
    frame = r.json()["frames"][0]
    assert frame["metrics"]["strehl"] == 1.0 and frame["metrics"]["power_rel"] == 1.0
    assert frame["reference_metrics"]["x_um"] == frame["metrics"]["x_um"]
    assert frame["n_groups"] == 1 and frame["groups_dropped"] == 0
