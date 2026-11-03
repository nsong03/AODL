"""``aodl_webapp.codegen``: the generated script runs under plain ``aodl`` and rebuilds the plan."""

from __future__ import annotations

import ast
import re
from typing import Any

import pytest
from conftest import EXAMPLE_IDS

from aodl_webapp.codegen import generate_code
from aodl_webapp.examples import example_request
from aodl_webapp.planner import band_usage_mhz, get_plan


def _strip_slow_lines(code: str) -> str:
    """Drop the file-writing and movie lines, and the whole-drive render/simulate, so the
    reproduction test checks the *plan* and not the deliverables' runtime."""
    kept = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(
            (
                "plan.save(",
                "# plan.movie(",
                "samples = plan.render_samples(",
                "result = plan.simulate(",
            )
        ):
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def _run(code: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 - the point of the test
    return namespace


@pytest.mark.parametrize("example_id", EXAMPLE_IDS)
def test_generated_code_compiles_and_reproduces_the_band_usage(example_id, capsys):
    req = example_request(example_id)
    code = generate_code(req)
    assert "aodl_webapp" not in code, "the script must run with plain aodl"
    ast.parse(code)
    namespace = _run(_strip_slow_lines(code))
    plan = namespace["plan"]
    reference = get_plan(req)
    assert plan.report.mode == reference.report.mode
    assert plan.report.tone_counts == reference.report.tone_counts
    for name, (lo, hi, margin) in reference.report.band_usage.items():
        got = plan.report.band_usage[name]
        assert got[0] == pytest.approx(lo, abs=1e-6)
        assert got[1] == pytest.approx(hi, abs=1e-6)
        assert got[2] == pytest.approx(margin, abs=1e-6)
    assert "AODL motion plan" in capsys.readouterr().out


def test_shepard_plan_emits_the_resolved_config_and_schedule():
    req = example_request("single_lift_hold")
    code = generate_code(req)
    assert "shepard=ShepardConfig(delta_f_x=12.0 * MHz, delta_f_y=12.0 * MHz, eta=0.5)" in code
    assert "# mode: shepard" in code
    assert re.search(r"live rungs per channel: Ax \d+, Bx \d+, Ay \d+, By \d+", code)
    assert "hand-over centres" in code and "shadow tweezers at +-123.60 um" in code
    assert "fading-Shepard ladders hold it" in code
    assert 'phases="schroeder"' in code
    for line in (
        "print(plan.report.summary())",
        'plan.save("aodl_',
        "plan.render_samples(625e6)",
        "plan.simulate()",
        '# plan.movie("aodl_',
    ):
        assert line in code


def test_s19_plan_insists_on_eq_s19():
    code = generate_code(example_request("pick_and_place"))
    assert "shepard=None" in code
    assert "ShepardConfig" not in code
    assert "# mode: s19" in code
    assert "% used" in code


def test_hardware_overrides_only_what_differs():
    req = example_request("pick_and_place")
    base = generate_code(req)
    assert "replace(" not in base
    req.hardware.wavelength_nm = 808.0
    req.hardware.band_MHz = (92.0, 108.0)
    req.hardware.mixing_order = 1
    code = generate_code(req)
    assert "from dataclasses import replace" in code
    assert "replace(params.optics, wavelength=808.0 * nm)" in code
    assert "band=(92.0 * MHz, 108.0 * MHz), mixing_order=1" in code
    assert "focal_length" not in code and "w_in" not in code
    namespace = _run(_strip_slow_lines(code))
    plan = namespace["plan"]
    assert plan.params.optics.wavelength == pytest.approx(808e-9)
    assert plan.params.channels["Ax"].band == pytest.approx((92e6, 108e6))
    assert plan.params.channels["Ax"].mixing_order == 1


def test_path_moves_and_non_default_options_are_emitted():
    req = example_request("array_arbitrary_path")
    req.options.amp = 0.9
    req.options.retard_compensate = True
    req.options.f_z_bias = "auto"
    req.options.t_pad_us = 30.0
    req.moves[0].profile = "constant_accel"
    code = generate_code(req)
    assert "Path(" in code and "closed=True" in code and "points=(" in code
    assert 'profile="constant_accel"' in code
    for option in ("amp=0.9", "retard_compensate=True", 'f_z_bias="auto"', "t_pad=30.0 * us"):
        assert option in code
    namespace = _run(_strip_slow_lines(code))
    reference = get_plan(req)
    assert namespace["plan"].report.mode == reference.report.mode
    for name, usage in band_usage_mhz(reference).items():
        assert namespace["plan"].report.band_usage[name][2] / 1e6 == pytest.approx(
            usage.margin_MHz, abs=1e-12
        )


def test_random_phases_reproduce_exactly():
    """Seeded rng under Eq. S19; explicit B-ladder phases under auto-Shepard."""
    s19 = example_request("pick_and_place")
    s19.options.phases = "random"
    s19.options.phase_seed = 11
    code = generate_code(s19)
    assert "rng=np.random.default_rng(11)" in code and "import numpy as np" in code
    assert (
        _run(_strip_slow_lines(code))["plan"].wfs.channels["Bx"].tones[0].phase0
        == get_plan(s19).wfs.channels["Bx"].tones[0].phase0
    )

    fading = example_request("array_lift_hold")
    fading.options.phases = "random"
    fading.options.phase_seed = 5
    code = generate_code(fading)
    assert "phases={" in code and '"Bx": [' in code
    plan = _run(_strip_slow_lines(code))["plan"]
    reference = get_plan(fading)
    for name in ("Bx", "By"):
        got = [t.phase0 for t in plan.wfs.channels[name].tones]
        want = [t.phase0 for t in reference.wfs.channels[name].tones]
        assert got == pytest.approx(want, abs=1e-15)
