"""``aodl_webapp.trajectory``: API moves -> ``aodl`` moves (SI, exact), sampled request."""

from __future__ import annotations

import numpy as np
import pytest

from aodl import Hold, Lift, Path, Translate
from aodl.units import MHz, um, us
from aodl_webapp.errors import ApiError
from aodl_webapp.examples import example_request
from aodl_webapp.schemas import (
    ArraySpecModel,
    HoldMove,
    LiftMove,
    PathMove,
    PlanRequest,
    TranslateMove,
)
from aodl_webapp.trajectory import sample_trajectory, seams, to_move, to_spec


def test_to_move_covers_every_kind_exactly():
    lift = to_move(LiftMove(kind="lift", dz_um=5.0, duration_us=40.0, profile="constant_jerk"))
    assert isinstance(lift, Lift)
    assert lift.dz == 5.0 * um and lift.duration == 40.0 * us and lift.profile == "constant_jerk"

    translate = to_move(TranslateMove(kind="translate", dx_um=20.0, dy_um=-3.5, duration_us=60.0))
    assert isinstance(translate, Translate)
    assert (translate.dx, translate.dy, translate.duration) == (20.0 * um, -3.5 * um, 60.0 * us)
    assert translate.profile == "min_jerk"

    hold = to_move(HoldMove(kind="hold", duration_us=100.0))
    assert isinstance(hold, Hold) and hold.duration == 100.0 * us

    path = to_move(
        PathMove(
            kind="path",
            points_um=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            duration_us=200.0,
            closed=True,
        )
    )
    assert isinstance(path, Path)
    assert path.points == (
        (1.0 * um, 2.0 * um, 3.0 * um),
        (4.0 * um, 5.0 * um, 6.0 * um),
        (0.0, 0.0, 0.0),
    )
    assert path.duration == 200.0 * us and path.closed


def test_to_spec_builds_the_array_in_si():
    req = example_request("array_lift_hold")
    spec = to_spec(req)
    assert (spec.array.mx, spec.array.my) == (4, 4)
    assert spec.array.delta_f_x == 1.0 * MHz and spec.array.delta_f_y == 1.3 * MHz
    assert len(spec.moves) == 3
    assert spec.duration == pytest.approx((80.0 + 500.0 + 80.0) * us)


def test_to_spec_refuses_a_degenerate_array_and_path():
    with pytest.raises(ApiError) as info:
        to_spec(
            PlanRequest(
                array=ArraySpecModel(mx=2, my=1, delta_f_x_MHz=0.0),
                moves=[HoldMove(kind="hold", duration_us=10.0)],
            )
        )
    assert info.value.code == "invalid_spec" and "delta_f_x must be non-zero" in info.value.message
    with pytest.raises(ApiError) as info:
        to_spec(
            PlanRequest(
                moves=[PathMove(kind="path", points_um=[(0.0, 0.0, 0.0)], duration_us=10.0)]
            )
        )
    assert info.value.code == "invalid_spec" and "duplicates" in info.value.message


def test_sample_trajectory_ends_at_the_requested_displacement():
    req = example_request("pick_and_place")
    spec = to_spec(req)
    samples = sample_trajectory(spec, n=400)
    assert len(samples.t_us) == 400
    assert samples.t_us[0] == 0.0 and samples.t_us[-1] == pytest.approx(140.0)
    assert samples.x_um[-1] == pytest.approx(20.0, abs=1e-9)
    assert samples.y_um[-1] == pytest.approx(0.0, abs=1e-9)
    assert samples.z_um[-1] == pytest.approx(0.0, abs=1e-9)
    assert max(samples.z_um) == pytest.approx(5.0, abs=1e-9)
    assert samples.seams_us == pytest.approx([0.0, 40.0, 100.0, 140.0])
    assert [s / us for s in seams(spec)] == pytest.approx(samples.seams_us)

    loop = to_spec(example_request("sphere_loop"))
    loop_samples = sample_trajectory(loop)
    assert np.hypot(loop_samples.x_um[-1], loop_samples.y_um[-1]) == pytest.approx(0.0, abs=1e-9)
    assert loop_samples.z_um[-1] == pytest.approx(0.0, abs=1e-9)
    assert max(loop_samples.z_um) == pytest.approx(12.0, abs=0.05)
