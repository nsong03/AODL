"""The seven starter trajectories of API.md §2 (ids used by the landing page).

Geometry in µm and µs as the API speaks it; everything else at the request defaults.
Expected modes at default options (``webapp/docs/reports/REVIEW.md``): ``pick_and_place``,
``square_xy``, ``sphere_loop`` and ``array_arbitrary_path`` plan as plain Eq. S19 (``s19``);
``single_lift_hold``, ``array_lift_hold`` and ``flagship_10x10`` need the fading-Shepard
ladders (``shepard``).  ``tests/test_api_plan.py`` asserts every one of them.
"""

from __future__ import annotations

import math

from .schemas import (
    ArraySpecModel,
    ExampleSpec,
    HoldMove,
    LiftMove,
    PathMove,
    PlanRequest,
    TranslateMove,
)

#: ``{example id: expected PlanReport.mode}`` — what the tests assert.
EXPECTED_MODES: dict[str, str] = {
    "pick_and_place": "s19",
    "square_xy": "s19",
    "sphere_loop": "s19",
    "single_lift_hold": "shepard",
    "array_lift_hold": "shepard",
    "array_arbitrary_path": "s19",
    "flagship_10x10": "shepard",
}


def _lift(dz: float, duration: float) -> LiftMove:
    return LiftMove(kind="lift", dz_um=dz, duration_us=duration)


def _translate(dx: float, dy: float, duration: float) -> TranslateMove:
    return TranslateMove(kind="translate", dx_um=dx, dy_um=dy, duration_us=duration)


def _hold(duration: float) -> HoldMove:
    return HoldMove(kind="hold", duration_us=duration)


def spherical_spiral(n: int = 24, radius: float = 6.0) -> list[tuple[float, float, float]]:
    """``n`` points on a spherical spiral of ``radius`` [µm] centred at ``(0, 0, radius)``.

    The path starts at the origin — the sphere's south pole — and winds up to the north pole:
    polar angle ``theta`` from ``0`` to ``pi`` and azimuth ``phi = 3 theta`` (one and a half
    turns).  Point ``k`` is ``(r sin th cos ph, r sin th sin ph, r (1 - cos th))`` relative to
    the start.  With ``closed=True`` the path then returns to the origin.
    """
    points = []
    for k in range(1, n + 1):
        theta = math.pi * k / n
        phi = 3.0 * theta
        points.append(
            (
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * (1.0 - math.cos(theta)),
            )
        )
    return points


def tilted_circle(
    n: int = 12, radius: float = 8.0, tilt_deg: float = 30.0
) -> list[tuple[float, float, float]]:
    """``n`` points on a circle of ``radius`` [µm] through the path origin, tilted about x.

    The circle lies in the XY plane centred at ``(-radius, 0, 0)`` so that ``phi = 0`` is the
    start, then is rotated by ``tilt_deg`` about the x axis: ``y -> y cos(tilt)``,
    ``z -> y sin(tilt)``.  The last point is exactly the origin, so the loop closes on itself.
    """
    tilt = math.radians(tilt_deg)
    points: list[tuple[float, float, float]] = []
    for k in range(1, n + 1):
        if k == n:
            points.append((0.0, 0.0, 0.0))
            continue
        phi = 2.0 * math.pi * k / n
        x = radius * math.cos(phi) - radius
        y = radius * math.sin(phi)
        points.append((x, y * math.cos(tilt), y * math.sin(tilt)))
    return points


def _request(
    moves: list[LiftMove | TranslateMove | HoldMove | PathMove],
    mx: int = 1,
    my: int = 1,
) -> PlanRequest:
    return PlanRequest(
        array=ArraySpecModel(mx=mx, my=my, delta_f_x_MHz=1.0, delta_f_y_MHz=1.3),
        moves=list(moves),
    )


EXAMPLES: list[ExampleSpec] = [
    ExampleSpec(
        id="pick_and_place",
        title="Pick and place",
        blurb="One tweezer: lift 5 µm, carry 20 µm along x, set down. Plain Eq. S19.",
        tags=["single", "s19", "starter"],
        request=_request([_lift(5.0, 40.0), _translate(20.0, 0.0, 60.0), _lift(-5.0, 40.0)]),
    ),
    ExampleSpec(
        id="square_xy",
        title="Square in the plane",
        blurb="Lift 3 µm, trace a 15 µm square with four translates, lower. Zero focal shift "
        "throughout the lateral moves.",
        tags=["single", "s19"],
        request=_request(
            [
                _lift(3.0, 30.0),
                _translate(15.0, 0.0, 50.0),
                _translate(0.0, 15.0, 50.0),
                _translate(-15.0, 0.0, 50.0),
                _translate(0.0, -15.0, 50.0),
                _lift(-3.0, 30.0),
            ]
        ),
    ),
    ExampleSpec(
        id="sphere_loop",
        title="Spherical spiral",
        blurb="A 24-point Path on a spherical spiral of radius 6 µm, pole to pole and back to "
        "the start, in 300 µs: a plain Eq. S19 drive that spends 87 % of the Eq. 1 axial "
        "budget (at 400 µs it would need the fading ladders).",
        tags=["single", "path", "s19", "3d"],
        request=_request(
            [PathMove(kind="path", points_um=spherical_spiral(), duration_us=300.0, closed=True)]
        ),
    ),
    ExampleSpec(
        id="single_lift_hold",
        title="Lift and hold",
        blurb="Lift 10 µm and hold for 600 µs — three times the Eq. 1 budget. plan_motion "
        "falls back to fading-Shepard ladders; watch the shadow tweezers at each hand-over.",
        tags=["single", "shepard", "hold"],
        request=_request([_lift(10.0, 60.0), _hold(600.0), _lift(-10.0, 60.0)]),
    ),
    ExampleSpec(
        id="array_lift_hold",
        title="4×4 array, lift and hold",
        blurb="A 4×4 array (1.0 / 1.3 MHz) lifted 8 µm and held 500 µs on fading-Shepard "
        "ladders: the array ladder is the Shepard ladder, so the lit grid grows by a column.",
        tags=["array", "shepard", "hold"],
        request=_request([_lift(8.0, 80.0), _hold(500.0), _lift(-8.0, 80.0)], mx=4, my=4),
    ),
    ExampleSpec(
        id="array_arbitrary_path",
        title="3×3 array on a tilted circle",
        blurb="A 3×3 array lifted 4 µm, then a 12-point Path around a circle of radius 8 µm "
        "tilted 30° about x (300 µs, closed), then lowered. Plain Eq. S19.",
        tags=["array", "path", "s19", "3d"],
        request=_request(
            [
                _lift(4.0, 50.0),
                PathMove(kind="path", points_um=tilted_circle(), duration_us=300.0, closed=True),
                _lift(-4.0, 50.0),
            ],
            mx=3,
            my=3,
        ),
    ),
    ExampleSpec(
        id="flagship_10x10",
        title="10×10 flagship traverse",
        blurb="The paper's user story: a 10×10 array lifted 10 µm, carried 40 × 25 µm and set "
        "down — 93 fading-Shepard rungs, 483 frequency groups at mixing order 3.",
        tags=["array", "shepard", "slow"],
        request=_request(
            [_lift(10.0, 150.0), _translate(40.0, 25.0, 250.0), _lift(-10.0, 150.0)],
            mx=10,
            my=10,
        ),
    ),
]

#: ``{id: ExampleSpec}`` for lookups.
EXAMPLES_BY_ID: dict[str, ExampleSpec] = {example.id: example for example in EXAMPLES}


def example_request(example_id: str) -> PlanRequest:
    """The :class:`PlanRequest` of one starter example (a fresh copy)."""
    return EXAMPLES_BY_ID[example_id].request.model_copy(deep=True)


__all__ = [
    "EXAMPLES",
    "EXAMPLES_BY_ID",
    "EXPECTED_MODES",
    "example_request",
    "spherical_spiral",
    "tilted_circle",
]
