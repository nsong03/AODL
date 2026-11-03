"""``aodl_webapp.ir``: the JSON IR and its reference evaluator against the ``aodl`` objects."""

from __future__ import annotations

import numpy as np
import pytest

from aodl import (
    ArraySpec,
    ChannelWaveform,
    Hold,
    Lift,
    ToneTrack,
    TrajectorySpec,
    WaveformSet,
    default_1030,
    plan_motion,
)
from aodl.poly import PiecewisePoly
from aodl.units import MHz, um, us
from aodl.waveform.export import render_samples, sample_times
from aodl.waveform.shepard import FadeZoneEnvelope, SwitchRamped
from aodl.waveform.tones import SmoothOnOff
from aodl_webapp.errors import ApiError
from aodl_webapp.examples import example_request
from aodl_webapp.ir import eval_envelope, eval_poly, evaluate_ir, to_ir
from aodl_webapp.planner import get_plan
from aodl_webapp.schemas import SmoothOnOffEnv

SEED = 20261102


def _drives():
    return {
        "s19": get_plan(example_request("pick_and_place")).wfs,
        "fading": get_plan(example_request("single_lift_hold")).wfs,
        "fading_array": get_plan(example_request("array_lift_hold")).wfs,
    }


@pytest.mark.parametrize("which", ["s19", "fading", "fading_array"])
def test_evaluate_ir_matches_tonetrack_on_random_times(which):
    """``f`` 1e-9 relative, ``A`` 1e-12 absolute, ``phase`` 1e-9 relative of the largest."""
    wfs = _drives()[which]
    ir = to_ir(wfs)
    rng = np.random.default_rng(SEED)
    t0, t1 = wfs.t_span
    t = rng.uniform(t0 - 5 * us, t1 + 5 * us, 200)  # beyond the domain too: clamp-hold
    for name, cw in wfs.channels.items():
        got = evaluate_ir(ir, name, t / us)
        table = cw.eval_table(t)
        assert got["f_MHz"].shape == (cw.n_tones, 200)
        f_scale = max(float(np.max(np.abs(table["f"]))), 1e-30)
        assert np.max(np.abs(got["f_MHz"] * MHz - table["f"])) <= 1e-9 * f_scale
        assert np.max(np.abs(got["A"] - table["A"])) <= 1e-12
        ph_scale = max(float(np.max(np.abs(table["phase"]))), 1e-30)
        assert np.max(np.abs(got["phase_rad"] - table["phase"])) <= 1e-9 * ph_scale


@pytest.mark.parametrize("which", ["s19", "fading"])
def test_voltage_matches_render_samples(which):
    """``V`` against ``render_samples(dtype=float64, return_scale=True)``, ``V / scale`` at 1e-9."""
    wfs = _drives()[which]
    ir = to_ir(wfs)
    span = (0.3 * wfs.t_span[1], 0.3 * wfs.t_span[1] + 2.0 * us)
    rate = 625e6
    samples, scale = render_samples(wfs, rate, span, dtype=np.float64, return_scale=True)
    n, start = sample_times(span, rate)
    t = start + np.arange(n) / rate
    for name, rendered in samples.items():
        got = evaluate_ir(ir, name, t / us)["V"]
        assert np.max(np.abs(got / scale - rendered)) <= 1e-9


def test_ir_units_and_metadata(array_lift_hold):
    """Frequencies in MHz, phase in rad (already ``2 pi int f``), breaks in µs, signs from
    ``conventions``; a fade envelope carries its ``g`` polynomial in MHz."""
    wfs = get_plan(array_lift_hold).wfs
    ir = to_ir(wfs)
    assert ir.t_span_us == pytest.approx(tuple(v / us for v in wfs.t_span))
    for name, chan in ir.channels.items():
        aod = wfs.params.channels[name]
        assert chan.f_center_MHz == pytest.approx(aod.f_center / MHz)
        assert chan.tau_us == pytest.approx(aod.transit_time / us)
        assert chan.sound_sign == (-1 if name.startswith("A") else 1)
        assert chan.axis == name[-1]
        assert chan.band_MHz == pytest.approx((90.0, 110.0))
        tone = wfs.channels[name].tones[0]
        ir_tone = chan.tones[0]
        np.testing.assert_allclose(ir_tone.freq.breaks_us, tone.freq.breaks / us, rtol=1e-15)
        np.testing.assert_allclose(ir_tone.freq.coeffs, tone.freq.coeffs / MHz, rtol=1e-15)
        expected_phase = tone.freq.antiderivative().scale(2.0 * np.pi)
        np.testing.assert_allclose(ir_tone.phase.coeffs, expected_phase.coeffs, rtol=1e-15)
        assert ir_tone.phase0 == tone.phase0
        env = tone.env
        assert isinstance(env, FadeZoneEnvelope)
        assert ir_tone.env.kind == "fade_zone"
        assert ir_tone.env.delta_f_MHz == pytest.approx(env.delta_f / MHz)
        np.testing.assert_allclose(ir_tone.env.g.coeffs, env.g.coeffs / MHz, rtol=1e-15)


def test_eval_poly_clamp_holds_outside_the_domain():
    poly = PiecewisePoly(np.array([0.0, 10.0, 30.0]) * us, np.array([[1.0, 2.0], [3.0, -1.0]]))
    ir_poly = (
        to_ir(WaveformSet({"Ax": ChannelWaveform((ToneTrack(freq=poly),))}, default_1030()))
        .channels["Ax"]
        .tones[0]
        .freq
    )
    t_us = np.array([-5.0, 0.0, 5.0, 10.0, 20.0, 30.0, 45.0])
    expected = np.asarray(poly(t_us * us)) / MHz
    np.testing.assert_allclose(eval_poly(ir_poly, t_us), expected, rtol=1e-15, atol=1e-18)
    # the last break belongs to the last segment, exactly as aodl.poly does it
    assert eval_poly(ir_poly, np.array([30.0]))[0] == pytest.approx((3.0 - 1.0) / MHz)


def test_smooth_on_off_envelope_matches_aodl():
    env = SmoothOnOff(t_on=10 * us, t_off=100 * us, ramp=20 * us)
    ir_env = SmoothOnOffEnv(t_on_us=10.0, t_off_us=100.0, ramp_us=20.0)
    t_us = np.linspace(-5.0, 120.0, 501)
    np.testing.assert_allclose(eval_envelope(ir_env, t_us), env.A(t_us * us), atol=1e-14)


def test_switch_ramped_envelope_is_refused():
    spec = TrajectorySpec(
        array=ArraySpec(3, 3, 1.0 * MHz, 1.3 * MHz), moves=(Lift(8 * um, 80 * us), Hold(300 * us))
    )
    plan = plan_motion(spec, default_1030(), switch_ramp=3 * us)
    assert any(
        isinstance(t.env, SwitchRamped) for cw in plan.wfs.channels.values() for t in cw.tones
    )
    with pytest.raises(ApiError) as info:
        to_ir(plan.wfs)
    assert info.value.code == "unsupported"
