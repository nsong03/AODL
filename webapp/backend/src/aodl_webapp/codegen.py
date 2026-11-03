"""``PlanRequest`` -> a runnable Python script that reproduces the plan with plain ``aodl``.

The script is the "take it to the lab" artefact (DESIGN §3): hardware via
``dataclasses.replace`` for the fields that differ from ``default_1030``, the array, the moves
(``Path`` included), and ``plan_motion`` with the synthesis options **as resolved** — when the
Studio's plan came out as fading-Shepard the script passes the very ``ShepardConfig`` that
``auto_config`` chose (spacings read off the plan's fade envelopes, the requested duty) so the
fade design is explicit and the run is deterministic.  A comment block above the call lists
the mode, the live rung counts, the hand-over schedule with the shadow-tweezer offsets and
the Eq. 1 budget line.  It ends with the five product lines of ``docs/guide.md`` §2.
"""

from __future__ import annotations

import math

from aodl.api import MotionPlan
from aodl.params import CHANNELS
from aodl.units import MHz, um, us
from aodl.waveform.synthesis import max_z_integral

from .hardware import HARDWARE_PRESETS
from .planner import chosen_spacings, get_plan, plan_id
from .schemas import (
    HardwareSpec,
    HoldMove,
    LiftMove,
    PathMove,
    PlanRequest,
    TranslateMove,
)

#: Hardware fields that live on ``OpticsParams``: ``(api field, aodl field, unit constant)``.
_OPTICS_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("wavelength_nm", "wavelength", "nm"),
    ("focal_length_mm", "focal_length", "mm"),
    ("w_in_mm", "w_in", "mm"),
)

#: Hardware fields that live on every ``AODParams``: ``(api field, aodl field, unit or "")``.
_AOD_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("sound_speed_m_s", "sound_speed", ""),
    ("aperture_mm", "aperture", "mm"),
    ("f_center_MHz", "f_center", "MHz"),
    ("band_MHz", "band", "MHz"),
    ("drive_strength", "drive_strength", ""),
    ("mixing_order", "mixing_order", ""),
)


def _num(value: float) -> str:
    """A float literal that round-trips exactly (``repr``), ints spelled as ints."""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    return repr(float(value))


def _quantity(value: float, unit: str) -> str:
    return f"{_num(value)} * {unit}" if unit else _num(value)


def hardware_lines(hw: HardwareSpec) -> tuple[list[str], set[str]]:
    """``params = ...`` lines; only fields differing from ``default_1030`` are overridden.

    Returns the lines and the unit names they use (for the import line).
    """
    default = HARDWARE_PRESETS["default_1030"]
    lines = ["params = default_1030()"]
    units: set[str] = set()
    optics = []
    for api_field, aodl_field, unit in _OPTICS_FIELDS:
        value = getattr(hw, api_field)
        if value != getattr(default, api_field):
            optics.append(f"{aodl_field}={_quantity(value, unit)}")
            units.add(unit)
    if optics:
        lines.append(
            f"params = replace(params, optics=replace(params.optics, {', '.join(optics)}))"
        )
    aods = []
    for api_field, aodl_field, unit in _AOD_FIELDS:
        value = getattr(hw, api_field)
        if value == getattr(default, api_field):
            continue
        if api_field == "band_MHz":
            aods.append(f"{aodl_field}=({_quantity(value[0], unit)}, {_quantity(value[1], unit)})")
        else:
            aods.append(f"{aodl_field}={_quantity(value, unit)}")
        if unit:
            units.add(unit)
    if aods:
        lines.append("params = replace(")
        lines.append("    params,")
        lines.append(
            f"    channels={{name: replace(aod, {', '.join(aods)}) "
            f"for name, aod in params.channels.items()}},"
        )
        lines.append(")")
    return lines, units


def move_lines(req: PlanRequest) -> tuple[list[str], set[str]]:
    """The ``moves = (...)`` tuple, one move per line (a ``Path`` spells out its points)."""
    lines = ["moves = ("]
    kinds: set[str] = set()
    for move in req.moves:
        if isinstance(move, LiftMove):
            kinds.add("Lift")
            profile = "" if move.profile == "min_jerk" else f', profile="{move.profile}"'
            lines.append(
                f"    Lift(dz={_num(move.dz_um)} * um, "
                f"duration={_num(move.duration_us)} * us{profile}),"
            )
        elif isinstance(move, TranslateMove):
            kinds.add("Translate")
            profile = "" if move.profile == "min_jerk" else f', profile="{move.profile}"'
            lines.append(
                f"    Translate(dx={_num(move.dx_um)} * um, dy={_num(move.dy_um)} * um, "
                f"duration={_num(move.duration_us)} * us{profile}),"
            )
        elif isinstance(move, HoldMove):
            kinds.add("Hold")
            lines.append(f"    Hold(duration={_num(move.duration_us)} * us),")
        elif isinstance(move, PathMove):
            kinds.add("Path")
            lines.append("    Path(")
            lines.append("        points=(")
            for x, y, z in move.points_um:
                lines.append(f"            ({_num(x)} * um, {_num(y)} * um, {_num(z)} * um),")
            lines.append("        ),")
            closed = ", closed=True" if move.closed else ""
            lines.append(f"        duration={_num(move.duration_us)} * us{closed},")
            lines.append("    ),")
    lines.append(")")
    return lines, kinds


def _fade_schedule_lines(plan: MotionPlan) -> list[str]:
    """Comment lines: hand-over centres per axis (drive time) and the shadow offsets."""
    events = plan.report.fade_events
    if not events:
        return []
    lines = []
    for axis in ("x", "y"):
        times = [e.time / us for e in events if e.axis == axis]
        if not times:
            continue
        shadow = next(e.shadow for e in events if e.axis == axis) / um
        shown = ", ".join(f"{t:.1f}" for t in times[:8])
        more = f", ... ({len(times)} in all)" if len(times) > 8 else ""
        lines.append(
            f"#   {axis}: hand-over centres at t = {shown}{more} us (drive time; the atom plane "
            f"sees each tau/2 later)"
        )
        lines.append(
            f"#      shadow tweezers at +-{shadow:.2f} um along {axis}, half a trap's power "
            f"mid-fade (Eq. S31)"
        )
    return lines


def synthesis_comment(plan: MotionPlan, req: PlanRequest) -> list[str]:
    """The comment block above ``plan_motion``: mode, rungs, schedule, Eq. 1 budget."""
    report = plan.report
    lines = ["# -- synthesis, as resolved by AODL Studio " + "-" * 36]
    head = report.description.split(" [shepard='auto':")[0]
    lines.append(f"# mode: {report.mode}  ({head})")
    if "[shepard='auto':" in report.description:
        reason = report.description.split("[shepard='auto': ", 1)[1].rstrip("]")
        lines.append(f"# shepard='auto' decided: {reason}")
    counts = ", ".join(f"{name} {report.tone_counts.get(name, 0)}" for name in CHANNELS)
    what = "live rungs" if report.mode == "shepard" else "tones"
    lines.append(f"# {what} per channel: {counts}")
    lines.extend(_fade_schedule_lines(plan))
    requested, ceiling = report.z_budget
    plain = max_z_integral(plan.params)
    if math.isfinite(ceiling):
        lines.append(
            f"# Eq. 1 budget: |int Z dt| = {requested:.3g} m.s of {ceiling:.3g} m.s "
            f"({100.0 * requested / ceiling:.0f} % used)"
        )
    else:
        lines.append(
            f"# Eq. 1 budget: |int Z dt| = {requested:.3g} m.s vs the {plain:.3g} m.s plain "
            f"Eq. S19 ceiling ({100.0 * requested / plain:.0f} %) -> fading-Shepard ladders hold it"
        )
    return lines


def _phase_arrays(plan: MotionPlan) -> list[str]:
    """``phases={...}`` lines with the B ladders' ``phase0`` values spelled out."""
    lines = ["    phases={"]
    for name in ("Bx", "By"):
        cw = plan.wfs.channels[name]
        values = ", ".join(repr(tone.phase0) for tone in cw.tones)
        lines.append(f'        "{name}": [{values}],')
    lines.append("    },")
    return lines


def plan_call_lines(plan: MotionPlan, req: PlanRequest) -> tuple[list[str], bool, bool]:
    """``plan = plan_motion(...)`` with the resolved options.

    Returns the lines, whether ``ShepardConfig`` is used and whether ``numpy`` is needed.
    """
    opts = req.options
    lines = ["plan = plan_motion(", "    spec,", "    params,"]
    uses_config = False
    uses_numpy = False
    explicit_phases = False
    if plan.report.mode == "shepard":
        spacings = chosen_spacings(plan)
        assert spacings is not None
        dfx, dfy = spacings
        uses_config = True
        lines.append(
            f"    shepard=ShepardConfig(delta_f_x={_num(dfx / MHz)} * MHz, "
            f"delta_f_y={_num(dfy / MHz)} * MHz, eta={_num(opts.eta)}),"
        )
        # Under shepard="auto" plan_motion draws the Eq. S19 attempt's random phases first and
        # the ladders' second; an explicit ShepardConfig draws only the latter, so the Studio's
        # B-ladder phases are passed verbatim to reproduce the plan exactly.
        explicit_phases = opts.phases == "random" and opts.shepard == "auto" and opts.eta == 0.5
    else:
        lines.append("    shepard=None,  # plain Eq. S19 fits the band")
    if explicit_phases:
        lines.append(
            f'    # phases="random" (seed {opts.phase_seed}) as drawn through shepard="auto"; '
            f"listed so the script matches the Studio plan"
        )
        lines.extend(_phase_arrays(plan))
    else:
        lines.append(f'    phases="{opts.phases}",')
        if opts.phases == "random":
            uses_numpy = True
            lines.append(f"    rng=np.random.default_rng({opts.phase_seed}),")
    if opts.amp != 1.0:
        lines.append(f"    amp={_num(opts.amp)},")
    if opts.retard_compensate:
        lines.append("    retard_compensate=True,")
    if opts.f_z_bias != "none":
        lines.append('    f_z_bias="auto",')
    if opts.t_pad_us is not None:
        lines.append(f"    t_pad={_num(opts.t_pad_us)} * us,")
    if not opts.check_band:
        lines.append("    check_band=False,")
    lines.append(")")
    return lines, uses_config, uses_numpy


def generate_code(req: PlanRequest) -> str:
    """The script for ``req`` (API.md §4 ``code_python`` / §5.5)."""
    plan = get_plan(req)
    key = plan_id(req)
    hw_lines, units = hardware_lines(req.hardware)
    mv_lines, kinds = move_lines(req)
    call_lines, uses_config, uses_numpy = plan_call_lines(plan, req)
    units |= {"MHz", "um", "us"}

    imports = sorted({"ArraySpec", "TrajectorySpec", "default_1030", "plan_motion", *kinds})
    if uses_config:
        imports.append("ShepardConfig")
    imports = sorted(imports)
    header = [
        '"""AODL Studio export: a 3D tweezer motion plan, reproducible with plain `aodl`.',
        "",
        f"plan_id {key}; {plan.report.mode} drive of a {req.array.mx}x{req.array.my} array, "
        f"{len(req.moves)} move(s), T = {plan.spec.duration / us:.4g} us.",
        "Physics: arXiv:2510.11451 (Eq. S19; fading-Shepard ladders Eqs. S24-S28).",
        '"""',
        "",
    ]
    if any(line.startswith("params = replace") for line in hw_lines):
        header.append("from dataclasses import replace")
        header.append("")
    if uses_numpy:
        header.append("import numpy as np")
        header.append("")
    header.append(f"from aodl import {', '.join(imports)}")
    header.append(f"from aodl.units import {', '.join(sorted(units))}")
    header.append("")

    array_line = (
        f"array = ArraySpec(mx={req.array.mx}, my={req.array.my}, "
        f"delta_f_x={_num(req.array.delta_f_x_MHz)} * MHz, "
        f"delta_f_y={_num(req.array.delta_f_y_MHz)} * MHz)"
    )
    body = [
        "# -- hardware (overrides relative to default_1030 only) " + "-" * 24,
        *hw_lines,
        "",
        "# -- what to build and where to move it (um, us -> SI) " + "-" * 25,
        array_line,
        *mv_lines,
        "spec = TrajectorySpec(array=array, moves=moves)",
        "",
        *synthesis_comment(plan, req),
        *call_lines,
        "",
        "# -- the deliverables (docs/guide.md section 2) " + "-" * 32,
        "print(plan.report.summary())    # mode, band usage, axial budget, fade schedule, caveats",
        f'plan.save("aodl_{key}.npz")  # the parametric NPZ: segment parameters, never samples',
        "samples = plan.render_samples(625e6)  # {channel: float32 samples} for the AWG",
        "result = plan.simulate()        # SpotMetrics per tweezer per frame; result.spot_table()",
        f'# plan.movie("aodl_{key}.mp4")  # simulate + render (needs imageio-ffmpeg)',
    ]
    return "\n".join([*header, *body]) + "\n"


__all__ = [
    "generate_code",
    "hardware_lines",
    "move_lines",
    "plan_call_lines",
    "synthesis_comment",
]
