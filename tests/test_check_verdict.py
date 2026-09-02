r"""M6 §4: the verdict — a good drive passes, and three broken ones fail *by name*.

A checker that only ever says "PASS" is worth nothing.  These tests take one small, entirely
ordinary Eq. S19 drive — a 3x3 array lifted, traversed and lowered — put it through
:meth:`aodl.api.MotionPlan.check`, and then break it in three different ways, each of which a
different gate has to catch and *name*:

============================  =========================================================
corruption                    what should fail
============================  =========================================================
``Ax`` chirp sign flipped     ``lateral`` (the pair no longer differences to ``X``) and
                              ``astigmatism`` (its chirp no longer cancels in ``Delta F``)
one ``Bx`` ladder tone gone   ``missing trap`` — a whole column of the array is dark
one ``Bx`` tone 5 % louder    ``uniformity`` — and *only* uniformity
============================  =========================================================

The third is deliberately a **single tone** and not a whole channel: a uniform per-channel gain
survives :func:`aodl.waveform.export.render_samples`' global normalization as nothing at all,
and is optically invisible in any case, so no checker can see it.  That is the blind spot the
report states in its notes, and this test is where it is pinned.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from aodl.api import MotionPlan, plan_motion
from aodl.check.report import Tolerances
from aodl.trajectory.spec import ArraySpec, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us
from aodl.waveform.tones import ChannelWaveform, ConstantEnvelope, ToneTrack

STORY = TrajectorySpec(
    array=ArraySpec(3, 3, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz),
    moves=(Lift(4 * um, 40 * us), Translate(10 * um, 6 * um, 60 * us), Lift(-4 * um, 40 * us)),
)
FRAMES = 3


@pytest.fixture(scope="module")
def plan():
    """The mini-story, planned once (synthesis is cheap; rendering is not)."""
    from aodl.params import default_1030

    return plan_motion(STORY, default_1030())


def _frames(plan):
    """Three interior frames — the midpoint of each move, shifted by the ``tau/2`` lag."""
    tau = max(aod.transit_time for aod in plan.params.channels.values())
    return np.array([20.0 * us, 70.0 * us, 120.0 * us]) + 0.5 * tau


def _broken(plan, channel, rebuild):
    """The same plan with one channel's tones rebuilt — everything else untouched."""
    cw = plan.wfs.channels[channel]
    wfs = replace(
        plan.wfs, channels={**plan.wfs.channels, channel: ChannelWaveform(rebuild(cw.tones))}
    )
    return MotionPlan(
        spec=plan.spec, params=plan.params, wfs=wfs, report=plan.report, options=plan.options
    )


def _metrics(report):
    """The metric each failure line names — every message leads with it."""
    return {re.match(r"([a-z ]+):", line).group(1) for line in report.failures}


# ================================================================== 1. the good drive


def test_the_mini_story_passes(plan) -> None:
    """A 3x3 Eq. S19 lift-traverse-lower checks out at the default tolerances."""
    report = plan.check(times=_frames(plan), k_subtimes=32)
    assert report.passed, report.summary()
    assert report.n_rows == FRAMES * 9
    assert report.mode == "bragg_band"
    assert set(report.out_of_band) == set(plan.wfs.channels)

    worst = report.worst()
    assert worst["lateral"][0] < 0.01  # measured 3e-3 w0
    assert worst["axial"][0] < 0.01
    assert worst["astigmatism"][0] < 1e-3  # Table I says exactly zero
    assert worst["waist"][0] < 5e-3
    assert 1e-3 < worst["uniformity"][0] < 0.03  # real IM3 spread, inside the gate
    assert not [blob for blob in report.blobs if not blob.on_lattice]
    assert "blind spot" in " ".join(report.notes)
    assert report.summary().startswith("AODL sample check - PASS")


def test_the_verdict_is_diffable_against_the_simulator(plan) -> None:
    """``sim=`` fills the report's ``sim_delta`` block, and it stays report-only."""
    times = _frames(plan)
    report = plan.check(times=times, k_subtimes=32, sim=plan.simulate(times))
    delta = report.sim_delta
    assert delta is not None
    assert delta["n_matched"] == delta["n_rows"] == FRAMES * 9
    # the analytic mixing_order=3 census and the literal exp(iCV) rebuild agree on the pattern
    assert delta["max_dx"] < 0.02 * plan.params.optics.waist0
    assert delta["max_dpower"] < 1e-3
    assert report.passed
    assert "vs simulator" in report.summary()


def test_tolerance_overrides_are_respected(plan) -> None:
    """The same measurements fail against an absurd gate and pass against a loose one."""
    times = _frames(plan)
    strict = plan.check(times=times, k_subtimes=32, tolerances=Tolerances(lateral=1e-6))
    assert not strict.passed
    assert _metrics(strict) == {"lateral"}
    assert strict.tolerances.lateral == 1e-6
    assert plan.check(times=times, k_subtimes=32, tolerances=Tolerances(uniformity=0.5)).passed


# =============================================================== 2. breaking it on purpose


def test_a_flipped_ax_chirp_fails_laterally_and_astigmatically(plan) -> None:
    """Negate ``Ax``'s frequency law: the pair stops differencing, and ``Delta F`` opens up.

    Eq. S19 splits the lateral term antisymmetrically across a counter-propagating pair and
    puts the *same* ``f_Z`` on all four channels, so ``X = deflection_scale (f_Bx - f_Ax)``
    comes out right while the chirps cancel in ``Delta F``.  Flip one member's sign and both
    statements break at once — which is exactly the class of error a checker sharing no code
    with the synthesizer exists to catch.
    """
    broken = _broken(
        plan,
        "Ax",
        lambda tones: tuple(
            ToneTrack(freq=tone.freq.scale(-1.0), env=tone.env, phase0=tone.phase0)
            for tone in tones
        ),
    )
    report = broken.check(times=_frames(plan), k_subtimes=16)
    assert not report.passed
    assert {"lateral", "astigmatism"} <= _metrics(report)
    assert report.failures[0].startswith(("lateral:", "axial:", "astigmatism:"))
    lateral = next(line for line in report.failures if line.startswith("lateral:"))
    astig = next(line for line in report.failures if line.startswith("astigmatism:"))
    assert "w0 off" in lateral and "Table I says 0" in astig
    assert report.worst()["astigmatism"][0] > 0.05


def test_a_dropped_bx_ladder_tone_fails_as_a_missing_trap(plan) -> None:
    """Delete one ``Bx`` rung and a whole column of the array goes dark."""
    broken = _broken(plan, "Bx", lambda tones: tones[:-1])
    report = broken.check(times=_frames(plan), k_subtimes=16)
    assert not report.passed
    assert "missing trap" in _metrics(report)
    missing = [line for line in report.failures if line.startswith("missing trap:")]
    assert len(missing) == FRAMES * 3  # one column, three rows, three frames
    assert "peaks at" in missing[0] and "threshold 0.25" in missing[0]
    # the two surviving columns are still where they should be
    table = report.table
    still_there = table["present"] > 0.5
    assert int(still_there.sum()) == FRAMES * 6


def test_five_percent_on_one_bx_tone_fails_uniformity_only(plan) -> None:
    """One rung 5 % off in amplitude: the *pattern* moves, and nothing else does.

    Amplitude lives in the envelope, so this is a pure intensity fault — the trap does not
    move, does not defocus and does not change shape.  Only ``uniformity`` may fire.  (The
    envelope is capped at 1, so the 5 % goes downwards; the sign of the fault is immaterial.)
    """

    def detuned_amplitude(tones):
        head = tones[0]
        return (
            ToneTrack(
                freq=head.freq, env=ConstantEnvelope(amp=0.95 * head.env.amp), phase0=head.phase0
            ),
            *tones[1:],
        )

    broken = _broken(plan, "Bx", detuned_amplitude)
    report = broken.check(times=_frames(plan), k_subtimes=16)
    assert not report.passed
    assert _metrics(report) == {"uniformity"}
    line = report.failures[0]
    assert line.startswith("uniformity:") and "off the frame's median peak" in line
    # ~10 % in intensity for 5 % in amplitude, on the three traps of that column
    assert report.worst()["uniformity"][0] == pytest.approx(0.10, abs=0.03)
    offenders = {
        int(report.table["ix"][i]) for i, ok in enumerate(report.table["verdict_trap"]) if ok < 0.5
    }
    assert offenders == {0}


def test_a_uniform_channel_gain_is_the_documented_blind_spot(plan) -> None:
    """Halve *every* ``Bx`` tone and the checker sees nothing — because there is nothing to see.

    ``render_samples`` divides all four channels by one global peak, and a common gain only
    rescales the image, so this drive is optically identical to the original.  The report says
    so in its notes; this test is the proof that it means it.
    """
    quiet = _broken(
        plan,
        "Bx",
        lambda tones: tuple(
            ToneTrack(
                freq=tone.freq, env=ConstantEnvelope(amp=0.5 * tone.env.amp), phase0=tone.phase0
            )
            for tone in tones
        ),
    )
    report = quiet.check(times=_frames(plan), k_subtimes=16)
    assert report.passed, report.summary()
    assert any("blind spot" in note for note in report.notes)


# ============================================================ 3. the record has to reach


def test_default_frames_leave_room_for_the_averaging_window(plan) -> None:
    """``times=None`` picks frames the record can actually be gathered for — window included.

    A frame reads drive over ``t -+ (W/2 + tau/2 + grid.half_span/v)``: the aperture grid runs
    wider than the crystal *and* the frame averages over a beat window.  Leave the window out of
    that bound and the last default frame asks
    :func:`aodl.check.demod.sample_baseband` for drive past the end of the record, which is
    refused — rightly, since clamp-holding it would render a dead aperture.
    """
    from aodl.check import check_samples
    from aodl.check.pupil import ApertureGrid
    from aodl.check.record import from_arrays
    from aodl.check.report import averaging_window, frame_reach
    from aodl.waveform.export import DEFAULT_SAMPLE_RATE, render_samples

    params = plan.params
    arrays, scale = render_samples(
        plan.wfs, DEFAULT_SAMPLE_RATE, dtype=np.float64, return_scale=True
    )
    record = from_arrays(arrays, DEFAULT_SAMPLE_RATE, params, normalization=scale)
    expect = plan.expectation()

    report = check_samples(record, expect, k_subtimes=8, n_z=3)
    assert report.passed, report.summary()
    assert report.times.size == 9

    tau = max(aod.transit_time for aod in params.channels.values())
    reach = frame_reach(ApertureGrid.design(params, "bragg_band"), params)
    window = averaging_window(expect, 8)
    assert window > 0.0  # a 3x3 array has beats, so this bound is not vacuous
    assert report.times[-1] + 0.5 * window - 0.5 * tau + reach <= record.t_span[1] + 1e-12

    # ... and a record that starts *after* the drive did refuses frames that reach before it
    late = from_arrays(
        {name: values[len(values) // 2 :] for name, values in arrays.items()},
        DEFAULT_SAMPLE_RATE,
        params,
        t_start=record.times()[len(record.times()) // 2],
        normalization=scale,
    )
    with pytest.raises(ValueError, match="reaches drive earlier than"):
        check_samples(late, expect, times=[2.0 * tau], k_subtimes=8, n_z=3)
