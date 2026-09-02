r"""M6: the CI gate — the guide's flagship drive, checked from its own samples.

Two drives, both of them the product's own:

* **the flagship** of ``docs/guide.md`` §2 — a 10x10 array lifted 10 µm, traversed 40 x 25 µm
  and lowered, which the band refuses as Eq. S19 and ``plan_motion`` therefore builds from the
  fading-Shepard ladders of Eqs. S24-S28.  Ninety-three tones, thirty-four hand-overs, 100
  tweezers, checked at the seven deterministic frames of
  :meth:`aodl.api.MotionPlan.check_times`;
* **the hurried Eq. S19 variant** — the same manoeuvre at the 25/30/25 µs the user story
  originally asked for, shrunk to a 2x2 so that it fits the band, and gated harder.

The flagship's *position* gates run at the defaults; its waist and uniformity gates do not, and
the reason is the drive rather than the checker.  A 30-rung ladder at ``drive_strength = 0.30``
renders with a **normalization factor** (peak over single-tone amplitude) of 4.59, so the
crystal's peak modulation index is 1.4 rad and the intermodulation of Eqs. S20-S22 spreads the
per-trap intensity by about a fifth.  That spread is not purely ``C^2``: about **82 %** of it
scales as ``C^2`` (``tests/test_check_bragg.py``) and the rest is a **~3.8 % C-independent
floor** left by the fade-speed apodization, measured on this array down to
``drive_strength = 0.003`` (``docs/guide.md`` §5.5).  The same fade-speed effect (``rho = 0.30``
for this array, §6.4) widens the spots by up to 8 % mid-hand-over.  Both are physics; the test
pins them from *both* sides so that a regression in either direction is a failure.

The last section takes the flagship apart.  WO-24 §1 asked whether a *time* statistic — each
trap's median deviation over the frames — separates that physics from a real fault, on the
argument that hand-over ripple moves with the fade phase while a fault does not.  On this drive
it does not, and the measurement showing why is here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aodl.api import MotionPlan, plan_motion
from aodl.check.report import Tolerances
from aodl.trajectory.spec import ArraySpec, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us
from aodl.waveform.tones import ChannelWaveform, ToneTrack

#: The quickstart of ``docs/guide.md`` §2, verbatim.
FLAGSHIP = TrajectorySpec(
    array=ArraySpec(10, 10, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz),
    moves=(
        Lift(10 * um, 150 * us),
        Translate(40 * um, 25 * um, 250 * us),
        Lift(-10 * um, 150 * us),
    ),
)

#: The same manoeuvre at the user story's original, band-infeasible pace, on an array small
#: enough that plain Eq. S19 can carry it (``docs/ORCHESTRATION.md``, Wave I).
HURRIED = TrajectorySpec(
    array=ArraySpec(2, 2, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz),
    moves=(
        Lift(10 * um, 25 * us),
        Translate(40 * um, 25 * um, 30 * us),
        Lift(-10 * um, 25 * us),
    ),
)

#: The unhurried counterpart of ``HURRIED``: the same 2x2, given time to move.
GENTLE = TrajectorySpec(
    array=HURRIED.array,
    moves=(Lift(5 * um, 40 * us), Translate(15 * um, 10 * um, 60 * us), Lift(-5 * um, 40 * us)),
)


@pytest.fixture(scope="module")
def flagship_report():
    """One flagship check, shared by every assertion below (the render is the expensive part)."""
    from aodl.params import default_1030

    plan = plan_motion(FLAGSHIP, default_1030())
    assert plan.report.mode == "shepard"
    return plan, plan.check(k_subtimes=48, tolerances=Tolerances(waist=0.12, uniformity=0.30))


# ============================================================== 1. the flagship passes


def test_the_guide_flagship_checks_out(flagship_report) -> None:
    """PASS, at seven frames spanning the whole 550 µs manoeuvre."""
    plan, report = flagship_report
    assert report.passed, report.summary()
    assert report.mode == "bragg_band"
    assert report.times.size == 7
    assert report.n_rows == 7 * 100
    np.testing.assert_allclose(report.times, plan.check_times(), rtol=0, atol=0)
    assert report.times[0] == pytest.approx(2.0 * plan.params.channels["Ax"].transit_time)
    assert report.times[-1] == pytest.approx(
        FLAGSHIP.duration + 0.5 * plan.params.channels["Ax"].transit_time
    )


def test_the_flagship_positions_hold_the_default_tolerances(flagship_report) -> None:
    """Lateral, axial and astigmatic residuals sit well inside the 0.05 defaults.

    These are the verdict-bearing numbers of a 3D trajectory, and this is the claim the paper
    makes: a hundred tweezers tracked through a 550 µs lift-traverse-lower stay within a
    hundredth of a waist of where they were asked to be, with ``Delta F`` under 0.02 ``z_R``.
    """
    _, report = flagship_report
    worst = report.worst()
    assert worst["lateral"][0] < 0.02
    assert worst["axial"][0] < 0.02
    assert worst["astigmatism"][0] < 0.03
    # ... and they are real measurements, not zeros
    assert worst["lateral"][0] > 1e-4


def test_the_flagship_intensity_spread_is_the_crystals_own(flagship_report) -> None:
    """The waist and uniformity ceilings, pinned from both sides.

    ``uniformity`` here is the Eqs. S20-S22 intermodulation of a 30-rung ladder whose tones
    stack to a modulation index of 1.4 rad, plus the C-independent fade-speed floor; ``waist``
    is the ``rho = 0.30`` fade-speed apodization of ``docs/guide.md`` §6.4.  Neither is a drive
    error and neither is the checker — ~82 % of the spread scales with ``C^2`` and the rest
    with the hand-over rate.
    """
    _, report = flagship_report
    worst = report.worst()
    assert 0.08 < worst["uniformity"][0] < 0.30
    assert 0.02 < worst["waist"][0] < 0.12


def test_the_flagship_lights_nothing_off_lattice(flagship_report) -> None:
    """Every blob sits on the extended Shepard lattice, none is off it, and none outshines a trap.

    The extended lattice is whitelisted because a fading ladder is *supposed* to light it
    (``docs/guide.md`` §6.7) — but only up to :attr:`~aodl.check.report.Tolerances.blob_fading`,
    a fifth above a real trap's depth.  The flagship's brightest such node measures well inside
    that, which is what makes the whitelist a bound rather than a blank cheque.
    """
    _, report = flagship_report
    assert not [blob for blob in report.blobs if not blob.on_lattice]
    on_lattice = [blob for blob in report.blobs if blob.on_lattice]
    assert on_lattice  # the extended grid is lit
    brightest = max(blob.rel_intensity for blob in on_lattice)
    assert 0.1 < brightest < report.tolerances.blob_fading  # measured 0.974, gate 1.2
    assert max(report.out_of_band.values()) < 1e-3  # switch_ramp is off, so this is not zero
    notes = " ".join(report.notes)
    assert "edge lines" in notes and "extended lattice is whitelisted up to" in notes


def test_the_flagship_intensity_gates_cover_most_of_the_array(flagship_report) -> None:
    """The edge-line exemption costs the intensity gates the perimeter, and no more.

    A 10x10 fading on both axes leaves 8x8 interior traps, i.e. 64 % of the rows, and the
    report says so rather than leaving a reader to work out what a PASS was actually about
    (``gated_fraction``, WO-24 §2).  ``waist`` gives up the transient frames on top of that.
    """
    _, report = flagship_report
    assert report.gated_fraction["uniformity"] == pytest.approx(0.64)
    assert 0.4 < report.gated_fraction["waist"] <= 0.64
    assert report.gated_fraction["uniformity_median"] == pytest.approx(0.64)
    assert "intensity-gate coverage" in " ".join(report.notes)
    assert "coverage " in report.summary()


# ========================================================== 2. the hurried S19 variant


@pytest.fixture(scope="module")
def hurried_report():
    """The 25/30/25 µs 2x2, checked at tolerances tighter than the flagship's."""
    from aodl.params import default_1030

    plan = plan_motion(HURRIED, default_1030())
    assert plan.report.mode == "s19"
    return plan, plan.check(
        k_subtimes=48,
        tolerances=Tolerances(lateral=0.10, axial=0.06, waist=0.03, uniformity=0.01),
    )


def test_the_hurried_variant_checks_out_and_shows_the_dropped_cubic_term(hurried_report) -> None:
    """Fast enough that Table I's *linear* pupil is visibly not the whole story.

    Eqs. S5-S6 keep the aperture phase to second order; at 25/30/25 µs the aperture — which
    spans ±15.4 µs of drive time, half the traverse — sees a strongly curved chirp, and the
    cubic (coma) and quartic (spherical) terms that are dropped move the real light off Table
    I's prediction by ~0.07 ``w_0`` laterally and ~0.04 ``z_R`` axially.  The astigmatic
    interval, which is a *difference* of the two axes' curvatures, is untouched: 1e-3 ``z_R``,
    two orders below the flagship's.
    """
    _, report = hurried_report
    assert report.passed, report.summary()
    worst = report.worst()
    assert 0.02 < worst["lateral"][0] < 0.10
    assert 0.01 < worst["axial"][0] < 0.06
    assert worst["astigmatism"][0] < 5e-3
    assert worst["waist"][0] < 0.03
    assert worst["uniformity"][0] < 0.01  # a 2-rung ladder barely intermodulates
    assert not [blob for blob in report.blobs if not blob.on_lattice]


def test_the_unhurried_2x2_holds_the_default_tolerances(hurried_report) -> None:
    """Give the same move time and the departure from Table I collapses — at the defaults.

    This is what makes the previous test a statement about the *pace* rather than about the
    checker: the residuals scale with the chirp's curvature, so the 40/60/40 µs version of the
    same 2x2 passes every default gate with a lateral residual at least three times smaller.
    """
    from aodl.params import default_1030

    hurried_plan, hurried = hurried_report
    plan = plan_motion(GENTLE, default_1030())
    assert plan.report.mode == "s19"
    report = plan.check(k_subtimes=48, tolerances=Tolerances(uniformity=0.01, waist=0.01))
    assert report.passed, report.summary()

    gentle = report.worst()
    assert gentle["lateral"][0] < hurried.worst()["lateral"][0] / 3.0
    assert gentle["lateral"][0] < 0.05  # the default gate, comfortably
    assert gentle["axial"][0] < 0.05
    assert gentle["astigmatism"][0] < 1e-3


# ================================== 3. WO-24 §1: can a time median catch a rung fault?

#: The ``Bx`` rung scaled down below.  The array ladder is a rectangle ten rungs wide that
#: slides through thirty as the drive chirps, and this one is live over the first four of the
#: seven check frames — the longest run any single rung gets, which is what gives the time
#: median its best possible chance of seeing the fault.
FAULTED_RUNG = 19

#: Its amplitude.  ``0.80^2 = 0.64``, i.e. a 36 % intensity fault on every trap it feeds — the
#: fault WO-23 found the opened 0.30 uniformity gate passing (finding F-2).
FAULT_AMPLITUDE = 0.80


@pytest.fixture(scope="module")
def rung_fault_report():
    """The flagship with one interior ``Bx`` rung at 80 % amplitude, checked the same way."""
    from aodl.params import default_1030

    plan = plan_motion(FLAGSHIP, default_1030())
    tones = list(plan.wfs.channels["Bx"].tones)
    faulted = tones[FAULTED_RUNG]
    tones[FAULTED_RUNG] = ToneTrack(
        freq=faulted.freq,
        env=replace(faulted.env, amp=FAULT_AMPLITUDE * faulted.env.amp),
        phase0=faulted.phase0,
    )
    broken = MotionPlan(
        spec=plan.spec,
        params=plan.params,
        wfs=replace(plan.wfs, channels={**plan.wfs.channels, "Bx": ChannelWaveform(tuple(tones))}),
        report=plan.report,
        options=plan.options,
    )
    return broken, broken.check(
        k_subtimes=48, n_z=3, tolerances=Tolerances(waist=0.12, uniformity=0.30)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "WO-24 F-2 is NOT closed.  A 36 % single-rung fault survives the flagship's opened "
        "0.30 uniformity gate, and the time median of WO-24 §1 does not close the hole "
        "either: see test_a_shepard_rung_fault_does_not_hold_still for the measurement."
    ),
)
def test_a_scaled_interior_rung_ought_to_fail_the_flagship(rung_fault_report) -> None:
    """One rung at 80 % is a real fault, and *something* ought to say so.  Nothing does."""
    _, report = rung_fault_report
    assert not report.passed, report.summary()


def test_a_shepard_rung_fault_does_not_hold_still(rung_fault_report, flagship_report) -> None:
    """Why the time median cannot close F-2 here: the fault moves, and the physics does not.

    WO-24 §1's premise is that a fault is a *persistent* per-trap offset while the hand-over
    ripple is fade-phase noise, so a per-trap median over frames should keep the first and
    erase the second.  On a fading-Shepard drive it is the other way round, and both halves of
    that are measured here:

    * the **fault moves**.  The B ladder slides through the array — that is what Shepard means
      — so the faulted rung feeds column 0 at the first frame, column 1 at the second, column 4
      at the third, and has left the band by the fifth.  Its −0.3 excursion therefore lands on
      any one trap at *one* frame in seven, and a median over seven erases it exactly: the
      corrupted drive's worst time median is the clean drive's to eight digits (all that
      separates them is the 1e-9 the fault moves the render's global normalization by).
    * the **physics stays put**.  The Eqs. S20–S22 products land on lattice nodes carrying
      those nodes' own optical frequency, so they interfere permanently rather than beating
      (``aodl.check.report._beat_comb``): the clean flagship's 0.137 median is a real, static
      IM3 pattern, not something a longer window or a median would remove.

    So there is no gate on this statistic that separates the two on this drive, at any value —
    which is what WO-24 §1.5 says to report rather than paper over.  The statistic itself is
    sound: ``tests/test_check_verdict.py`` gates it on an Eq. S19 drive, where a tone fault
    *does* hold still, at an 8x separation.
    """
    _, clean = flagship_report
    _, faulted = rung_fault_report

    # The per-frame gate sees the fault - as a 28 % dip at one frame, inside the opened gate.
    assert faulted.worst()["uniformity"][0] == pytest.approx(0.28, abs=0.02)
    assert faulted.worst()["uniformity"][0] < faulted.tolerances.uniformity  # the F-2 hole

    # The time median does not see it at all: identical to the clean drive's.
    clean_median = clean.median_uniformity()[0]
    fault_median = faulted.median_uniformity()[0]
    assert clean_median == pytest.approx(0.137, abs=0.01)
    assert fault_median == pytest.approx(clean_median, abs=1e-6)

    # ... because the fault visits a different column at each frame.  Take the rows it moved
    # by more than 10 % and ask which column they are in, frame by frame.
    table = faulted.table
    moved = np.abs(table["uniformity"] - clean.table["uniformity"]) > 0.10
    assert {int(frame) for frame in table["frame"][moved]} == {0, 1, 2}  # then the rung fades
    columns_hit = {
        int(frame): sorted({int(ix) for ix in table["ix"][moved & (table["frame"] == frame)]})
        for frame in (0, 1, 2)
    }
    assert columns_hit == {0: [0], 1: [0, 1], 2: [4]}  # column 0 is an edge line, ungated

    # No *gated* trap is disturbed at more than one of the seven frames, which is exactly why
    # every one of their medians survives the fault untouched.
    hits = np.zeros((10, 10), dtype=int)
    np.add.at(
        hits,
        (
            table["ix"][moved & (table["gated"] > 0.5)].astype(int),
            table["iy"][moved & (table["gated"] > 0.5)].astype(int),
        ),
        1,
    )
    assert hits.max() == 1
    assert hits.sum() == 16  # two columns of eight interior rows, one frame each


def test_the_median_statistic_is_measured_and_reported_but_not_gated(flagship_report) -> None:
    """Report-only by default: it is in the table, in ``summary()``, and in the notes."""
    _, report = flagship_report
    assert report.tolerances.uniformity_median is None
    assert "uniformity_median" not in report.worst()  # not a gate
    assert "uniformity_median" not in " ".join(report.failures)
    assert "uniformity_median" in report.summary() and "(report-only)" in report.summary()
    assert any("measured but not gated" in note for note in report.notes)
    # every row of a trap carries that trap's median; the edge lines carry nan
    table = report.table
    edge = table["gated"] < 0.5
    assert np.all(np.isnan(table["uniformity_median"][edge]))
    assert np.count_nonzero(np.isfinite(table["uniformity_median"])) == 64 * 7


# ======================================== 4. WO-24 F-11: too few sub-times, said out loud


def test_too_few_subtimes_says_so_and_the_default_does_not(flagship_report) -> None:
    """The flagship's beat comb needs ``k_subtimes >= 46``; ask for 24 and the report says so.

    A uniform schedule over one comb period annihilates every same-node beat *exactly*, but
    only while the fastest of them stays under the schedule's own Nyquist (``b W <= k / 2``).
    Below that the checker falls back to the golden-ratio sub-times, which merely average the
    beats down — verdict noise the caller did not have to accept.  One frame is enough to make
    the point, and a one-frame check is cheap.
    """
    plan, default_report = flagship_report
    assert not any("too small for exact beat cancellation" in note for note in default_report.notes)
    assert any("uniform sub-times" in note for note in default_report.notes)

    starved = plan.check(
        times=plan.check_times()[:1],
        k_subtimes=24,
        n_z=3,
        tolerances=Tolerances(waist=0.12, uniformity=0.30),
    )
    note = next(note for note in starved.notes if "too small for exact beat cancellation" in note)
    assert "k_subtimes=24" in note and "needs >= 46" in note
    assert "golden-ratio fallback" in note and "verdict noise increases" in note
    assert note in starved.summary()
