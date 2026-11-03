#!/usr/bin/env python
"""Golden fixtures for the frontend's physics tests (W1 work order §2; DESIGN §5).

Writes, under ``webapp/frontend/tests/fixtures/``:

* ``<id>.json`` for ``pick_and_place`` and ``array_lift_hold`` — the waveform IR of the plan,
  the derived constants, 48 evaluation times over the drive span (including times before
  ``tau/2``), the reference evaluation of :func:`aodl_webapp.ir.evaluate_ir` on them (``f_MHz``,
  ``A``, ``phase_rad`` per tone and the carrier-included ``V``), a 2001-point voltage window
  over ``+-0.2 us`` around ``0.4 T``, and two aperture snapshots (``0.6 tau``, ``2 tau``) from
  :func:`aodl_webapp.aperture.aperture_response`;
* ``examples.json`` — the ``/api/presets`` response (hardware presets, profiles, examples).

``NaN`` (the instantaneous frequency where an aperture is empty) is written as ``null``, which
is what the API itself serialises; the TypeScript tests map ``null`` back to ``NaN``.

Run from anywhere: ``python webapp/scripts/gen_frontend_fixtures.py [--out DIR]``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from aodl.units import us
from aodl_webapp.aperture import aperture_response
from aodl_webapp.examples import EXAMPLES, example_request
from aodl_webapp.hardware import HARDWARE_PRESETS, constants, transit_time
from aodl_webapp.ir import evaluate_ir, to_ir
from aodl_webapp.planner import get_plan
from aodl_webapp.schemas import PROFILES, ApertureRequest, PresetsResponse

#: Examples that get an evaluator fixture (one Eq. S19 drive, one fading-Shepard array).
FIXTURE_EXAMPLES = ("pick_and_place", "array_lift_hold")

#: Evaluation times per fixture.
N_TIMES = 48

#: Voltage window: ``+-0.2 us`` around ``0.4 T`` on 2001 points (50 samples per carrier period).
VOLTAGE_HALF_WINDOW_US = 0.2
VOLTAGE_POINTS = 2001

#: Aperture snapshots, in units of ``tau``: one mid-fill, one fully lit.
APERTURE_TIMES_TAU = (0.6, 2.0)
APERTURE_POINTS = 512

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "frontend" / "tests" / "fixtures"


def _clean(obj: Any) -> Any:
    """Recursively turn numpy scalars/arrays into JSON types and NaN/inf into ``None``."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def evaluator_fixture(example_id: str) -> dict[str, Any]:
    """The ``<id>.json`` payload described in the module docstring."""
    req = example_request(example_id)
    plan = get_plan(req)
    ir = to_ir(plan.wfs)
    tau = transit_time(plan.params)
    t0, t1 = plan.wfs.t_span
    duration = plan.spec.duration

    # Times over the whole programmed span, deliberately including a few before tau/2 (where a
    # pair-driven tweezer is still dark but every tone is already defined).
    t_us = np.linspace(t0 / us, t1 / us, N_TIMES)
    evaluation = {name: evaluate_ir(ir, name, t_us) for name in ir.channels}

    centre = 0.4 * duration / us
    window_us = np.linspace(
        centre - VOLTAGE_HALF_WINDOW_US, centre + VOLTAGE_HALF_WINDOW_US, VOLTAGE_POINTS
    )
    voltage: dict[str, Any] = {"t_us": window_us}
    for name in ir.channels:
        voltage[name] = evaluate_ir(ir, name, window_us)["V"]

    aperture = []
    for factor in APERTURE_TIMES_TAU:
        time_us = factor * tau / us
        response = aperture_response(
            ApertureRequest(plan=req, time_us=time_us, n=APERTURE_POINTS, include_voltage=False)
        )
        snapshot: dict[str, Any] = {"time_us": time_us}
        for name, channel in response.channels.items():
            snapshot[name] = {
                "u_mm": channel.u_mm,
                "envelope": channel.envelope,
                "f_inst_MHz": channel.f_inst_MHz,
                "filled": channel.filled,
            }
        aperture.append(snapshot)

    return {
        "example_id": example_id,
        "plan_mode": plan.report.mode,
        "ir": ir.model_dump(mode="json"),
        "constants": constants(plan.params).model_dump(mode="json"),
        "t_us": t_us,
        "eval": evaluation,
        "voltage_window": voltage,
        "aperture": aperture,
    }


def examples_fixture() -> dict[str, Any]:
    """The ``/api/presets`` response."""
    return PresetsResponse(
        hardware_presets=HARDWARE_PRESETS, profiles=list(PROFILES), examples=EXAMPLES
    ).model_dump(mode="json")


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(payload), allow_nan=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(text) / 1e3:.0f} kB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args(argv)
    for example_id in FIXTURE_EXAMPLES:
        write(args.out / f"{example_id}.json", evaluator_fixture(example_id))
    write(args.out / "examples.json", examples_fixture())
    return 0


if __name__ == "__main__":
    sys.exit(main())
