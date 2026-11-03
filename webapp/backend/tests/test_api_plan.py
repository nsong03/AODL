"""``POST /api/plan`` and its companions (API.md §1, §2, §4, §5.4, §5.5)."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from conftest import EXAMPLE_IDS, as_json

from aodl import band_usage
from aodl.units import MHz
from aodl.waveform.tones import WaveformSet
from aodl_webapp.examples import EXAMPLES, EXPECTED_MODES, example_request
from aodl_webapp.planner import get_plan, plan_id


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["aodl_version"] == "0.1.0"
    assert body["version"]


def test_presets_list_the_seven_examples(client):
    body = client.get("/api/presets").json()
    assert set(body["hardware_presets"]) == {"default_1030", "paper_808"}
    assert body["hardware_presets"]["paper_808"]["wavelength_nm"] == pytest.approx(808.0)
    assert body["profiles"] == [
        "min_jerk",
        "constant_jerk",
        "constant_accel",
        "switching_constant_jerk",
        "linear",
    ]
    assert [e["id"] for e in body["examples"]] == list(EXAMPLE_IDS)
    assert len(EXAMPLES) == 7


@pytest.mark.parametrize("example_id", EXAMPLE_IDS)
def test_every_example_plans_in_its_expected_mode(client, example_id):
    """Status 200, the mode the review predicts, and a response with every §4 field."""
    response = client.post("/api/plan", json=as_json(example_request(example_id)))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == EXPECTED_MODES[example_id]
    assert body["plan_id"] == plan_id(example_request(example_id))
    assert set(body["band_usage"]) == {"Ax", "Bx", "Ay", "By"}
    assert body["summary_text"].startswith("AODL motion plan")
    assert len(body["trajectory"]["t_us"]) == 400
    assert body["seams_us"][0] == 0.0 and body["seams_us"][-1] == pytest.approx(body["duration_us"])
    assert body["t_span_us"][1] > body["duration_us"]
    assert set(body["waveform_ir"]["channels"]) == {"Ax", "Bx", "Ay", "By"}
    assert body["constants"]["beam_transit_us"] == pytest.approx(6.1538, abs=1e-3)
    assert body["constants"]["eq1_hold_10um_us"] == pytest.approx(206.0, abs=1e-6)
    assert "plan_motion(" in body["code_python"]
    if body["mode"] == "shepard":
        assert body["fade_events"], "a fading drive has a hand-over schedule"
        assert body["z_budget"]["ceiling_m_s"] is None
        assert body["z_budget"]["fraction"] > 1.0
    else:
        assert body["fade_events"] == []
        assert 0.0 < body["z_budget"]["fraction"] <= 1.0


def test_plan_id_is_stable_and_equals_the_formula(client, pick_and_place):
    body = as_json(pick_and_place)
    first = client.post("/api/plan", json=body).json()["plan_id"]
    second = client.post("/api/plan", json=body).json()["plan_id"]
    assert first == second
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert first == hashlib.sha1(canonical.encode()).hexdigest()[:16]
    # field order on the wire does not matter, option values do
    shuffled = {k: body[k] for k in reversed(list(body))}
    assert client.post("/api/plan", json=shuffled).json()["plan_id"] == first
    body["options"]["amp"] = 0.9
    assert client.post("/api/plan", json=body).json()["plan_id"] != first


def test_band_usage_equals_aodl_band_usage_in_mhz(client, array_lift_hold):
    body = client.post("/api/plan", json=as_json(array_lift_hold)).json()
    reference = band_usage(get_plan(array_lift_hold).wfs)
    for name, (lo, hi, margin) in reference.items():
        got = body["band_usage"][name]
        assert got["f_min_MHz"] == pytest.approx(lo / MHz, rel=1e-12)
        assert got["f_max_MHz"] == pytest.approx(hi / MHz, rel=1e-12)
        assert got["margin_MHz"] == pytest.approx(margin / MHz, rel=1e-12)


def test_out_of_band_request_is_a_band_violation(client, pick_and_place):
    """Insist on Eq. S19 for a hold three times the budget: 400 with the aodl message."""
    req = example_request("single_lift_hold")
    req.options.shepard = "never"
    response = client.post("/api/plan", json=as_json(req))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "band_violation"
    assert "leaves its usable band" in error["message"]
    assert "Eq. 1" in error["message"]


def test_shepard_always_without_spacings_is_a_422(client, pick_and_place):
    body = as_json(pick_and_place)
    body["options"]["shepard"] = "always"
    assert client.post("/api/plan", json=body).status_code == 422
    body["options"]["shepard_delta_f_MHz"] = [8.0, 6.5]
    response = client.post("/api/plan", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "shepard"


def test_unknown_field_is_a_422(client, pick_and_place):
    body = as_json(pick_and_place)
    body["options"]["switch_ramp_us"] = 3.0
    assert client.post("/api/plan", json=body).status_code == 422


def test_path_with_a_duplicate_point_is_invalid_spec(client):
    body = as_json(example_request("pick_and_place"))
    body["moves"] = [
        {"kind": "path", "points_um": [[5, 0, 0], [5, 0, 0], [0, 0, 0]], "duration_us": 100.0}
    ]
    response = client.post("/api/plan", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_spec"
    assert "Path waypoint 1 duplicates waypoint 0" in error["message"]


def test_zero_spacing_on_a_ladder_axis_is_invalid_spec(client, array_lift_hold):
    body = as_json(array_lift_hold)
    body["array"]["delta_f_x_MHz"] = 0.0
    response = client.post("/api/plan", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_spec"
    assert "delta_f_x must be non-zero" in response.json()["error"]["message"]


def test_npz_download_round_trips_through_waveformset_load(client, pick_and_place):
    plan = client.post("/api/plan", json=as_json(pick_and_place)).json()
    response = client.get(f"/api/plan/{plan['plan_id']}/waveform.npz")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert f"aodl_{plan['plan_id']}.npz" in response.headers["content-disposition"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "download.npz"
        path.write_bytes(response.content)
        loaded = WaveformSet.load(path)
    original = get_plan(pick_and_place).wfs
    assert set(loaded.channels) == set(original.channels)
    for name, cw in original.channels.items():
        for tone, back in zip(cw.tones, loaded.channels[name].tones, strict=True):
            np.testing.assert_array_equal(tone.freq.coeffs, back.freq.coeffs)
            np.testing.assert_array_equal(tone.freq.breaks, back.freq.breaks)
            assert tone.phase0 == back.phase0


def test_npz_download_of_an_unknown_plan_is_404(client):
    response = client.get("/api/plan/0123456789abcdef/waveform.npz")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "plan_not_cached"


def test_npz_download_of_a_shepard_drive_with_amp_below_one_is_unsupported(client):
    req = example_request("single_lift_hold")
    req.options.amp = 0.8
    plan = client.post("/api/plan", json=as_json(req))
    assert plan.status_code == 200, plan.text
    assert plan.json()["mode"] == "shepard"
    assert any("waveform download is refused" in note for note in plan.json()["notes"])
    response = client.get(f"/api/plan/{plan.json()['plan_id']}/waveform.npz")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported"
    assert "cannot serialize a FadeZoneEnvelope with amp" in response.json()["error"]["message"]


def test_code_route_matches_the_plan_response(client, pick_and_place):
    body = as_json(pick_and_place)
    plan = client.post("/api/plan", json=body).json()
    code = client.post("/api/code", json=body).json()
    assert code["code_python"] == plan["code_python"]


def test_random_phases_are_seeded_and_reproducible(client, array_lift_hold):
    body = as_json(array_lift_hold)
    body["options"]["phases"] = "random"
    body["options"]["phase_seed"] = 7
    first = client.post("/api/plan", json=body).json()["waveform_ir"]
    second = client.post("/api/plan", json=body).json()["waveform_ir"]
    assert first == second
    body["options"]["phase_seed"] = 8
    third = client.post("/api/plan", json=body).json()["waveform_ir"]
    assert (
        first["channels"]["Bx"]["tones"][0]["phase0"]
        != third["channels"]["Bx"]["tones"][0]["phase0"]
    )


def test_t_pad_zero_is_noted(client, pick_and_place):
    body = as_json(pick_and_place)
    body["options"]["t_pad_us"] = 0.0
    plan = client.post("/api/plan", json=body).json()
    assert plan["t_span_us"][1] == pytest.approx(plan["duration_us"])
    assert any("t_pad_us = 0" in note for note in plan["notes"])


def test_eta_below_half_under_auto_sizes_the_ladders_at_that_duty(client):
    req = example_request("single_lift_hold")
    req.options.eta = 0.4
    body = client.post("/api/plan", json=as_json(req))
    assert body.status_code == 200, body.text
    plan = body.json()
    assert plan["mode"] == "shepard"
    envs = [t["env"] for t in plan["waveform_ir"]["channels"]["Ax"]["tones"]]
    assert all(e["kind"] == "fade_zone" and e["eta"] == pytest.approx(0.4) for e in envs)
    assert any("eta = 0.4" in note for note in plan["notes"])
    # the window is narrower, so the ladder is wider than at eta = 1/2 (12 MHz)
    assert envs[0]["delta_f_MHz"] > 12.0
