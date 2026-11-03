"""Per-AOD misalignment: each knob moves the spot exactly as the conventions predict (W2 §5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _advanced_fixtures import (
    DEFLECTION,
    LENS,
    PARAMS,
    TAU,
    W0,
    lift_hold,
    reparam,
    run_frame,
    static_offset_trap,
    with_order,
)

from aodl.device.conventions import geometry, theta1_contribution
from aodl.poly import PiecewisePoly
from aodl.units import MHz, um, us
from aodl.waveform.tones import ChannelWaveform, ToneTrack, WaveformSet
from aodl_advanced.engine import GridSpec
from aodl_advanced.misalignment import AODMisalignment, Misalignment
from aodl_advanced.pupil import static_offset

ORDER1 = with_order(PARAMS, 1)
OPTICS = PARAMS.optics


def _single_tone(channel: str, detuning: float) -> WaveformSet:
    tone = ToneTrack(freq=PiecewisePoly.constant(detuning, 0.0, 8 * TAU))
    return WaveformSet({channel: ChannelWaveform((tone,))}, ORDER1)


def test_aligned_misalignment_reproduces_the_reference_bit_exactly() -> None:
    wfs = reparam(static_offset_trap().wfs, ORDER1)
    t = 60 * us + 50 * us + 0.5 * TAU
    reference = run_frame(wfs, t, None, xz=None)
    explicit = run_frame(wfs, t, Misalignment({"Bx": AODMisalignment()}), xz=None)
    assert Misalignment().is_aligned and Misalignment.aligned() == Misalignment()
    np.testing.assert_array_equal(explicit.xy, reference.xy)
    assert explicit.metrics == reference.metrics
    assert explicit.reference_metrics == explicit.metrics


def test_delay_on_ax_during_a_chirp_moves_the_spot_by_the_conventions_sign() -> None:
    wfs = reparam(lift_hold().wfs, ORDER1)
    t = 60 * us + 40 * us + 0.5 * TAU  # mid-hold: every channel chirps at fdot
    fdot = float(wfs.channels["Ax"].tones[0].fdot(t - 0.5 * TAU))
    assert fdot > 0.0
    delay = 200e-9
    frame = run_frame(wfs, t, Misalignment({"Ax": AODMisalignment(delay=delay)}), xz=None)
    ref = frame.reference_metrics
    assert ref is not None
    # the delayed channel reads f(t_c - delay) = f(t_c) - fdot*delay at the beam centre; the
    # spot moves by theta1 F / k of that frequency change (sign from conventions, not by hand)
    theta1 = float(theta1_contribution(-fdot * delay, geometry("Ax"), PARAMS.sound_speed))
    expected = theta1 * OPTICS.focal_length / OPTICS.k
    assert expected == pytest.approx(-geometry("Ax").sound_sign * DEFLECTION * fdot * delay)
    assert (frame.metrics.x - ref.x) == pytest.approx(expected, rel=0.02)
    assert abs(frame.metrics.y - ref.y) < 1e-3 * W0


def test_delay_does_nothing_to_a_static_tone() -> None:
    wfs = _single_tone("Ax", 2 * MHz)
    t = 3 * TAU
    frame = run_frame(wfs, t, Misalignment({"Ax": AODMisalignment(delay=500e-9)}), xz=None)
    ref = frame.reference_metrics
    assert ref is not None
    assert abs(frame.metrics.x - ref.x) < 1e-4 * W0
    assert frame.metrics.power == pytest.approx(ref.power, rel=1e-9)


def test_lateral_offset_keeps_the_position_but_moves_the_fill_edge() -> None:
    wfs = _single_tone("Ax", 2 * MHz)
    mis = Misalignment({"Ax": AODMisalignment(offset_par=0.5e-3)})
    full = run_frame(wfs, 3 * TAU, mis, xz=None)
    assert abs(full.metrics.x - full.reference_metrics.x) < 1e-6 * W0
    assert full.metrics.power == pytest.approx(full.reference_metrics.power, rel=1e-9)
    filling = run_frame(wfs, 0.6 * TAU, mis, xz=None, centre=(full.metrics.x, 0.0))
    ref = filling.reference_metrics
    assert ref is not None
    assert abs(filling.metrics.x - ref.x) < 0.01 * W0
    assert not math.isclose(filling.metrics.power, ref.power, rel_tol=0.05)
    assert filling.metrics.power < ref.power  # the edge moved away from the transducer side


def test_velocity_scale_on_bx_only_gives_astigmatism_and_an_x_scale_error() -> None:
    wfs = reparam(lift_hold().wfs, ORDER1)
    t = 60 * us + 40 * us + 0.5 * TAU
    fdot = float(wfs.channels["Ax"].tones[0].fdot(t - 0.5 * TAU))
    m = 1.02
    frame = run_frame(wfs, t, Misalignment({"Bx": AODMisalignment(velocity_scale=m)}))
    ref = frame.reference_metrics
    assert ref is not None and frame.metrics.astig is not None and ref.astig is not None
    expected_df = LENS * fdot * (1.0 / m**2 - 1.0)
    assert expected_df == pytest.approx(-0.0388 * LENS * fdot, rel=0.01)
    assert expected_df == pytest.approx(-0.194 * um, rel=0.01)
    assert abs(ref.astig) < 0.02 * abs(expected_df)
    assert frame.metrics.astig == pytest.approx(expected_df, rel=0.15)
    # position: only one pair member is scaled, so X moves by deflection_scale f_Bx (1/m - 1)
    f_bx = float(wfs.channels["Bx"].tones[0].f(t - 0.5 * TAU))
    assert (frame.metrics.x - ref.x) == pytest.approx(DEFLECTION * f_bx * (1.0 / m - 1.0), rel=0.05)
    # the pure scale error on a trap at rest at X = 20 um, where f_Bx = -f_Ax = X / (2 deflection)
    rest = reparam(static_offset_trap(x=20 * um).wfs, ORDER1)
    t_rest = 60 * us + 50 * us + 0.5 * TAU
    frame_rest = run_frame(
        rest, t_rest, Misalignment({"Bx": AODMisalignment(velocity_scale=m)}), xz=None
    )
    ref_rest = frame_rest.reference_metrics
    assert ref_rest is not None
    scale_error = (frame_rest.metrics.x - ref_rest.x) / ref_rest.x
    assert scale_error == pytest.approx((1.0 / m - 1.0) / 2.0, rel=0.10)
    assert scale_error == pytest.approx(-0.0098, rel=0.10)


def test_rotation_of_bx_moves_y_by_tan_theta_times_half_x_and_reports_the_static_offset() -> None:
    wfs = reparam(static_offset_trap(x=20 * um).wfs, ORDER1)
    t = 60 * us + 50 * us + 0.5 * TAU
    theta = 10e-3
    mis = Misalignment({"Bx": AODMisalignment(rotation=theta)})
    frame = run_frame(wfs, t, mis, xz=None)
    ref = frame.reference_metrics
    assert ref is not None
    f_bx = float(wfs.channels["Bx"].tones[0].f(t - 0.5 * TAU))
    expected = math.tan(theta) * DEFLECTION * f_bx
    assert expected == pytest.approx(math.tan(theta) * 20 * um / 2.0, rel=1e-6)
    assert expected == pytest.approx(0.1 * um, rel=0.01)
    assert (frame.metrics.y - ref.y) == pytest.approx(expected, rel=0.10)
    assert abs(frame.metrics.x - ref.x) < 0.01 * W0
    offsets = static_offset(PARAMS, mis)
    f_c = PARAMS.channels["Bx"].f_center
    assert offsets["Bx"][1] == pytest.approx(math.sin(theta) * DEFLECTION * f_c, rel=1e-9)
    assert offsets["Bx"][1] == pytest.approx(10.3 * um, rel=0.01)
    assert offsets["Bx"][0] == pytest.approx((math.cos(theta) - 1.0) * DEFLECTION * f_c, rel=1e-9)
    for name in ("Ax", "Ay", "By"):
        assert offsets[name] == (0.0, 0.0)


def test_pointing_tilt_shifts_the_whole_image_by_f_alpha() -> None:
    wfs = reparam(static_offset_trap(x=0.0).wfs, ORDER1)
    t = 60 * us + 50 * us + 0.5 * TAU
    alpha = 0.3e-3
    mis = Misalignment({"Ay": AODMisalignment(tilt_x=alpha)})
    frame = run_frame(wfs, t, mis, xz=None, grid=GridSpec(6 * um, 6 * um, 128))
    ref = frame.reference_metrics
    assert ref is not None
    assert (frame.metrics.x - ref.x) == pytest.approx(OPTICS.focal_length * alpha, rel=0.02)


def test_misalignment_validation() -> None:
    with pytest.raises(ValueError):
        AODMisalignment(velocity_scale=0.0)
    with pytest.raises(ValueError):
        AODMisalignment(rotation=float("nan"))
    with pytest.raises(KeyError):
        Misalignment({"Cz": AODMisalignment()})
    assert Misalignment({"Bx": AODMisalignment(rotation=1e-3)}).any_rotated
