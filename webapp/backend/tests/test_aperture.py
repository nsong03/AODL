"""``POST /api/aperture`` (API.md §5.3) against direct ``ToneTrack``/``conventions`` evaluation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import as_json

from aodl.device.aod import aperture_window
from aodl.device.conventions import geometry, is_filled, retarded_time
from aodl.units import MHz, mm, us
from aodl_webapp.aperture import voltage_points
from aodl_webapp.planner import get_plan

TAU_US = 7.5e-3 / 650.0 / 1e-6


def _aperture(client, req, **kwargs):
    response = client.post("/api/aperture", json={"plan": as_json(req), **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("time_factor", [0.6, 2.0])
def test_envelope_frequency_fill_and_retarded_time(client, array_lift_hold, time_factor):
    plan = get_plan(array_lift_hold)
    t_us = time_factor * TAU_US
    body = _aperture(client, array_lift_hold, time_us=t_us, n=256)
    assert body["time_us"] == pytest.approx(t_us)
    for name, cw in plan.wfs.channels.items():
        aod = plan.params.channels[name]
        geom = geometry(name)
        chan = body["channels"][name]
        u = np.asarray(chan["u_mm"]) * mm
        assert (
            u.size == 256
            and u[0] == pytest.approx(-0.5 * aod.aperture)
            and u[-1] == pytest.approx(0.5 * aod.aperture)
        )
        t_ret = retarded_time(t_us * us, u, geom, aod)
        np.testing.assert_allclose(np.asarray(chan["t_ret_us"]) * us, t_ret, rtol=1e-12, atol=1e-18)
        filled = is_filled(u, t_us * us, geom, aod)
        assert chan["filled"] == filled.tolist()
        amps = np.array([tone.env.A(t_ret) for tone in cw.tones])
        freqs = np.array([tone.freq(t_ret) for tone in cw.tones])
        envelope = np.where(filled, amps.sum(axis=0), 0.0)
        np.testing.assert_allclose(chan["envelope"], envelope, atol=1e-12)
        live = envelope > 0.0
        expected_f = aod.f_center + (amps * freqs).sum(axis=0)[live] / envelope[live]
        got_f = chan["f_inst_MHz"]
        assert all(v is None for v, on in zip(got_f, live, strict=True) if not on)
        got_live = np.asarray([v for v, on in zip(got_f, live, strict=True) if on], dtype=float)
        np.testing.assert_allclose(got_live * MHz, expected_f, rtol=1e-12)
        if time_factor < 1.0:
            # mid-fill: the transducer side (u = -s D/2) carries content, the far side does not
            assert filled.any() and not filled.all()
            transducer_end = 0 if geom.sound_sign > 0 else -1
            assert filled[transducer_end] and not filled[-1 - transducer_end]
            assert chan["envelope"][-1 - transducer_end] == 0.0
            assert got_f[-1 - transducer_end] is None
        else:
            assert filled.all()
            assert chan["voltage"] is None


def test_voltage_equals_aperture_window_on_the_same_points(client, pick_and_place):
    """Asking for the whole aperture reproduces ``aperture_window`` exactly."""
    plan = get_plan(pick_and_place)
    t_us = 2.0 * TAU_US
    half_mm = 0.5 * plan.params.channels["Ax"].aperture / mm
    body = _aperture(
        client,
        pick_and_place,
        time_us=t_us,
        n=64,
        include_voltage=True,
        voltage_window_mm=[-half_mm, half_mm],
    )
    for name, cw in plan.wfs.channels.items():
        aod = plan.params.channels[name]
        voltage = body["channels"][name]["voltage"]
        m = len(voltage["u_mm"])
        assert m == voltage_points(aod, (-half_mm * mm, half_mm * mm)) >= 4096
        u, v = aperture_window(cw, aod, geometry(name), t_us * us, m)
        # the same points (u / mm is the one conversion applied) and bit-identical voltages
        np.testing.assert_array_equal(np.asarray(voltage["u_mm"]), u / mm)
        np.testing.assert_array_equal(np.asarray(voltage["v"]), v)


def test_default_voltage_zoom_is_the_central_tenth_of_a_millimetre(client, pick_and_place):
    plan = get_plan(pick_and_place)
    t_us = 0.6 * TAU_US  # mid-fill: part of the window is still empty on some channels
    body = _aperture(client, pick_and_place, time_us=t_us, include_voltage=True)
    for name, cw in plan.wfs.channels.items():
        aod = plan.params.channels[name]
        geom = geometry(name)
        voltage = body["channels"][name]["voltage"]
        u = np.asarray(voltage["u_mm"]) * mm
        assert u.size == 4096
        assert u[0] == pytest.approx(-0.05 * mm) and u[-1] == pytest.approx(0.05 * mm)
        t_ret = retarded_time(t_us * us, u, geom, aod)
        expected = np.zeros_like(u)
        for tone in cw.tones:
            expected += np.asarray(tone.env.A(t_ret)) * np.cos(
                2.0 * math.pi * aod.f_center * t_ret + np.asarray(tone.phase(t_ret))
            )
        expected = np.where(is_filled(u, t_us * us, geom, aod), expected, 0.0)
        np.testing.assert_allclose(voltage["v"], expected, atol=1e-12)
        # the 100 MHz carrier is resolved: 32+ samples per acoustic wavelength
        wavelength = aod.sound_speed / aod.f_center
        assert (u[1] - u[0]) <= wavelength / 32


def test_aperture_request_validation(client, pick_and_place):
    base = {"plan": as_json(pick_and_place), "time_us": 20.0}
    assert client.post("/api/aperture", json={**base, "n": 8}).status_code == 422
    assert (
        client.post("/api/aperture", json={**base, "voltage_window_mm": [1.0, -1.0]}).status_code
        == 422
    )
