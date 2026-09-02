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
has a crest factor of 4.6, so the crystal's peak modulation index is 1.4 rad and the
intermodulation of Eqs. S20-S22 spreads the per-trap intensity by about a fifth — measured here
and confirmed to scale as ``C^2`` in ``tests/test_check_bragg.py``.  Mid-hand-over the
fade-speed effect of ``docs/guide.md`` §6.4 (``rho = 0.30`` for this array) widens the spots by
up to 8 % on top of that.  Both are physics; the test pins them from *both* sides so that a
regression in either direction is a failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from aodl.api import plan_motion
from aodl.check.report import Tolerances
from aodl.trajectory.spec import ArraySpec, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us

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

    ``uniformity`` here is the Eqs. S20-S22 intermodulation of a 30-rung ladder driven at a
    crest-boosted modulation index, and ``waist`` is the ``rho = 0.30`` fade-speed apodization
    of ``docs/guide.md`` §6.4.  Neither is a drive error and neither is the checker: they scale
    with ``C^2`` and with the hand-over rate respectively.
    """
    _, report = flagship_report
    worst = report.worst()
    assert 0.08 < worst["uniformity"][0] < 0.30
    assert 0.02 < worst["waist"][0] < 0.12


def test_the_flagship_lights_nothing_off_lattice(flagship_report) -> None:
    """Every blob sits on the extended Shepard lattice; none is off it."""
    _, report = flagship_report
    assert not [blob for blob in report.blobs if not blob.on_lattice]
    assert [blob for blob in report.blobs if blob.on_lattice]  # the extended grid is lit
    assert max(report.out_of_band.values()) < 1e-3  # switch_ramp is off, so this is not zero
    notes = " ".join(report.notes)
    assert "edge lines" in notes and "extended lattice is whitelisted" in notes


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
