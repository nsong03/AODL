r"""Trajectory specification (``trajectory/spec.py``, ``docs/PLAN.md`` §1.4).

The spec layer owns three claims, and every test here checks one of them against a
hand-written closed form rather than against the implementation:

1. **compilation is faithful** — a ``Lift`` moves only ``Z``, a ``Translate`` only ``X``/``Y``,
   a ``Hold`` nothing; all three axes cover the full span ``[0, T]``; every move starts where
   the previous one ended, so the three position laws are continuous (and, for the
   rest-to-rest profiles, have zero velocity at every seam);
2. **the profile name dispatches** to :mod:`aodl.trajectory.ramps` unchanged (Eqs. S14-S17);
3. **pitch and tone spacing are one conversion apart** — ``pitch = deflection_scale * delta_f``
   (Table I), and :meth:`ArraySpec.from_pitch` round-trips it exactly.

Nothing here touches frequencies, waveforms or the device: ``spec.py`` is pure geometry plus
time, and ``tests/test_synthesis_s19.py`` picks it up from there.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aodl.trajectory import ramps
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us

DZ = 5.0 * um
DX, DY = 15.0 * um, 10.0 * um
T_LIFT, T_MOVE, T_HOLD = 60.0 * us, 80.0 * us, 40.0 * us

#: The mini user story of ``docs/workorders/WO-11``: up, across, down.
STORY = (Lift(DZ, T_LIFT), Translate(DX, DY, T_MOVE), Hold(T_HOLD), Lift(-DZ, T_LIFT))


def _spec(*moves, array: ArraySpec | None = None) -> TrajectorySpec:
    return TrajectorySpec(array=array or ArraySpec(), moves=tuple(moves))


# ============================================================================ compilation


def test_compile_spans_every_axis_over_the_whole_trajectory():
    """Three polynomials on ``[0, T]``, one breakpoint set, one segment per move."""
    spec = _spec(*STORY)
    total = T_LIFT + T_MOVE + T_HOLD + T_LIFT
    assert spec.duration == pytest.approx(total, rel=1e-15)

    x, y, z = spec.compile()
    for poly in (x, y, z):
        assert poly.domain == (0.0, spec.duration)
        np.testing.assert_allclose(
            poly.breaks,
            np.cumsum([0.0, T_LIFT, T_MOVE, T_HOLD, T_LIFT]),
            rtol=1e-15,
        )


def test_lift_moves_only_z_and_translate_only_xy():
    """The axis split of ``PLAN`` §1.4: lifts are axial, translates are in-plane."""
    t = np.linspace(0.0, T_LIFT, 25)
    x, y, z = _spec(Lift(DZ, T_LIFT)).compile()
    np.testing.assert_allclose(x(t), 0.0, atol=0.0)
    np.testing.assert_allclose(y(t), 0.0, atol=0.0)
    np.testing.assert_allclose(z(t), ramps.min_jerk(0.0, T_LIFT, 0.0, DZ)(t), rtol=1e-14)

    t = np.linspace(0.0, T_MOVE, 25)
    x, y, z = _spec(Translate(DX, DY, T_MOVE)).compile()
    np.testing.assert_allclose(x(t), ramps.min_jerk(0.0, T_MOVE, 0.0, DX)(t), rtol=1e-14)
    np.testing.assert_allclose(y(t), ramps.min_jerk(0.0, T_MOVE, 0.0, DY)(t), rtol=1e-14)
    np.testing.assert_allclose(z(t), 0.0, atol=0.0)


def test_hold_freezes_all_three_axes_wherever_they_are():
    """A ``Hold`` is a constant segment on every axis at the current position."""
    spec = _spec(Lift(DZ, T_LIFT), Translate(DX, DY, T_MOVE), Hold(T_HOLD))
    x, y, z = spec.compile()
    t = np.linspace(T_LIFT + T_MOVE, spec.duration, 17)
    np.testing.assert_allclose(x(t), DX, rtol=1e-14)
    np.testing.assert_allclose(y(t), DY, rtol=1e-14)
    np.testing.assert_allclose(z(t), DZ, rtol=1e-14)

    # ... and the held segments really are constants, not flat-looking ramps
    for poly, value in ((x, DX), (y, DY), (z, DZ)):
        last = poly.coeffs[-1]
        assert last[0] == pytest.approx(value, rel=1e-14)
        np.testing.assert_allclose(last[1:], 0.0, atol=0.0)


def test_moves_chain_from_the_running_position():
    """Displacements are relative: the story ends where the sum of its moves says."""
    spec = _spec(*STORY)
    x, y, z = spec.compile()
    marks = np.cumsum([0.0, T_LIFT, T_MOVE, T_HOLD, T_LIFT])
    np.testing.assert_allclose([x(m) for m in marks], [0.0, 0.0, DX, DX, DX], atol=1e-18)
    np.testing.assert_allclose([y(m) for m in marks], [0.0, 0.0, DY, DY, DY], atol=1e-18)
    np.testing.assert_allclose([z(m) for m in marks], [0.0, DZ, DZ, DZ, 0.0], atol=1e-18)


def test_compiled_trajectory_is_continuous_and_rest_to_rest_at_every_seam():
    """C^0 across every breakpoint, and (min-jerk) at rest with zero acceleration there.

    The value continuity is what :meth:`TrajectorySpec.compile` asserts internally; the
    velocity/acceleration statement is the reason ``min_jerk`` is the default (Eq. S14), and
    through Eq. S19 it is what makes the drive start and stop with no residual lensing.
    """
    spec = _spec(*STORY)
    polys = spec.compile()
    eps = 1e-15  # far below the microsecond breakpoints, far above float64 round-off
    for poly in polys:
        for seam in poly.breaks[1:-1]:
            assert poly(seam - eps) == pytest.approx(poly(seam + eps), abs=1e-18)
        speed, accel = poly.derivative(), poly.derivative().derivative()
        for seam in poly.breaks:
            assert speed(seam) == pytest.approx(0.0, abs=1e-9 * DX / T_MOVE)
            assert accel(seam) == pytest.approx(0.0, abs=1e-9 * DX / T_MOVE**2)


@pytest.mark.parametrize("profile", sorted(ramps.RAMPS))
def test_profile_names_dispatch_to_the_ramp_module(profile):
    """``Lift``/``Translate`` carry a ramp *name*; compile calls exactly that ramp."""
    t = np.linspace(0.0, T_LIFT, 41)
    expected = ramps.RAMPS[profile](0.0, T_LIFT, 0.0, DZ)

    _, _, z = _spec(Lift(DZ, T_LIFT, profile=profile)).compile()
    np.testing.assert_allclose(z(t), expected(t), rtol=1e-13, atol=1e-18)

    x, y, _ = _spec(Translate(DZ, DZ, T_LIFT, profile=profile)).compile()
    np.testing.assert_allclose(x(t), expected(t), rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(y(t), expected(t), rtol=1e-13, atol=1e-18)


def test_lowering_is_a_lift_with_negative_dz():
    """Documented in ``spec.py``: there is no separate ``Lower`` move, and none is needed."""
    down = _spec(Lift(-DZ, T_LIFT)).compile()[2]
    up = _spec(Lift(DZ, T_LIFT)).compile()[2]
    t = np.linspace(0.0, T_LIFT, 25)
    np.testing.assert_allclose(down(t), -np.asarray(up(t)), rtol=1e-14, atol=1e-18)
    assert down(T_LIFT) == pytest.approx(-DZ, rel=1e-14)


def test_compile_rejects_an_empty_move_list():
    with pytest.raises(ValueError, match="at least one move"):
        TrajectorySpec(array=ArraySpec(), moves=()).compile()
    assert TrajectorySpec(array=ArraySpec(), moves=()).duration == 0.0


# ============================================================================ array geometry


def test_from_pitch_round_trips_through_the_table_i_scale(params1030):
    """``delta_f = pitch / deflection_scale`` and back — the only metres/hertz trade."""
    pitch_x, pitch_y = 10.0 * um, 4.0 * um
    array = ArraySpec.from_pitch(4, 3, pitch_x, pitch_y, params1030)

    scale = params1030.deflection_scale
    assert array.delta_f_x == pytest.approx(pitch_x / scale, rel=1e-15)
    assert array.delta_f_y == pytest.approx(pitch_y / scale, rel=1e-15)
    np.testing.assert_allclose(array.pitch(params1030), (pitch_x, pitch_y), rtol=1e-15)

    # the documented calibration: 1 MHz <-> 10.3 um at the default hardware
    assert ArraySpec(2, 2, 1.0 * MHz, 1.0 * MHz).pitch(params1030)[0] == pytest.approx(
        10.3 * um, rel=1e-12
    )
    assert array.n_traps == 12


def test_detunings_are_the_centred_eq_s18_ladders():
    """``f_0^(n) = (n - (M-1)/2) delta_f``: centred on zero, spacing exactly ``delta_f``."""
    array = ArraySpec(4, 3, 1.0 * MHz, 1.3 * MHz)
    dx, dy = array.detunings()
    np.testing.assert_allclose(dx, np.array([-1.5, -0.5, 0.5, 1.5]) * MHz, rtol=1e-15)
    np.testing.assert_allclose(dy, np.array([-1.0, 0.0, 1.0]) * 1.3 * MHz, rtol=1e-15)
    assert dx.sum() == pytest.approx(0.0, abs=1e-9)
    assert dy.sum() == pytest.approx(0.0, abs=1e-9)

    single = ArraySpec()
    np.testing.assert_allclose(single.detunings()[0], [0.0], atol=0.0)


def test_array_spec_validation():
    """A multi-tone ladder needs a spacing; counts are integers >= 1."""
    with pytest.raises(ValueError, match="delta_f_x must be non-zero"):
        ArraySpec(mx=3, my=1)
    with pytest.raises(ValueError, match="delta_f_y must be non-zero"):
        ArraySpec(mx=1, my=2, delta_f_x=1.0 * MHz)
    with pytest.raises(ValueError, match="mx must be an integer >= 1"):
        ArraySpec(mx=0)
    with pytest.raises(ValueError, match="my must be an integer >= 1"):
        ArraySpec(my=2.5, delta_f_y=1.0 * MHz)
    with pytest.raises(ValueError, match="must be finite"):
        ArraySpec(mx=2, delta_f_x=float("nan"))

    single = ArraySpec(mx=1, my=1)  # a lone tone needs no spacing
    assert (single.delta_f_x, single.delta_f_y) == (0.0, 0.0)


# ============================================================================== hygiene


@pytest.mark.parametrize(
    "move",
    [Lift(DZ, T_LIFT), Translate(DX, DY, T_MOVE), Hold(T_HOLD), ArraySpec(2, 2, 1e6, 1.3e6)],
)
def test_dataclasses_are_frozen_hashable_and_comparable(move):
    """Specs are values: immutable, hashable, and equal when their fields are."""
    assert dataclasses.is_dataclass(move)
    with pytest.raises(dataclasses.FrozenInstanceError):
        move.duration = 1.0  # type: ignore[misc]
    assert move == dataclasses.replace(move)
    assert hash(move) == hash(dataclasses.replace(move))


def test_trajectory_spec_is_frozen_and_normalizes_its_moves():
    spec = TrajectorySpec(array=ArraySpec(), moves=[Hold(T_HOLD)])  # a list is accepted
    assert isinstance(spec.moves, tuple)
    assert hash(spec) == hash(_spec(Hold(T_HOLD)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.moves = ()  # type: ignore[misc]


@pytest.mark.parametrize("duration", [0.0, -1.0 * us, float("inf"), float("nan")])
def test_moves_reject_non_positive_durations(duration):
    for factory in (
        lambda d: Lift(DZ, d),
        lambda d: Translate(DX, DY, d),
        Hold,
    ):
        with pytest.raises(ValueError, match="duration must be finite and positive"):
            factory(duration)


def test_moves_reject_unknown_profiles_and_non_finite_displacements():
    with pytest.raises(ValueError, match="unknown ramp profile"):
        Lift(DZ, T_LIFT, profile="bang_bang")
    with pytest.raises(ValueError, match="unknown ramp profile"):
        Translate(DX, DY, T_MOVE, profile="hold")  # not a rest-to-rest family
    with pytest.raises(ValueError, match="Lift.dz must be finite"):
        Lift(float("inf"), T_LIFT)
    with pytest.raises(ValueError, match="Translate.dy must be finite"):
        Translate(DX, float("nan"), T_MOVE)


def test_trajectory_spec_type_checks_its_fields():
    with pytest.raises(TypeError, match="must be an ArraySpec"):
        TrajectorySpec(array=(2, 2), moves=(Hold(T_HOLD),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"moves\[1\] must be a Lift, Translate or Hold"):
        TrajectorySpec(array=ArraySpec(), moves=(Hold(T_HOLD), "wait"))  # type: ignore[arg-type]
