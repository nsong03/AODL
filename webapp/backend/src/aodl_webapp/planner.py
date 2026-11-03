"""``PlanRequest`` -> :class:`aodl.MotionPlan` (cached) -> ``PlanResponse`` (API.md §4).

The physics authority is :func:`aodl.plan_motion` (DESIGN §1 item 1); this module only maps
the API options onto its keyword arguments, caches the result by :func:`plan_id` (API.md
§0.2) and reads the report back out in display units.  The one piece of synthesis logic that
lives here is the ``eta != 1/2`` branch of ``shepard="auto"`` (:func:`_resolve_shepard`),
because ``plan_motion``'s own ``"auto"`` always sizes the ladders at Table II's ``eta = 1/2``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import numpy as np

from aodl.api import MotionPlan, plan_motion
from aodl.device.aodl import build_terms
from aodl.field.focal import group_terms
from aodl.params import CHANNELS, AODLParams
from aodl.trajectory.spec import TrajectorySpec
from aodl.units import MHz, um, us
from aodl.waveform.shepard import FadeZoneEnvelope, ShepardConfig, auto_config, poly_range
from aodl.waveform.synthesis import max_z_integral

from .cache import LRUCache
from .config import MOVIE_GROUP_BUDGET, PLAN_CACHE_SIZE
from .errors import ApiError, classify
from .hardware import constants, to_params, transit_time
from .ir import to_ir
from .schemas import (
    BandUsage,
    FadeEventModel,
    PlanRequest,
    PlanResponse,
    ZBudget,
)
from .trajectory import sample_trajectory, seams, to_spec

_PLANS: LRUCache[MotionPlan] = LRUCache(PLAN_CACHE_SIZE)
_NOTES: LRUCache[tuple[str, ...]] = LRUCache(PLAN_CACHE_SIZE)
_GROUP_COUNTS: LRUCache[int] = LRUCache(PLAN_CACHE_SIZE)


# ----------------------------------------------------------------------------- identity


def canonical_json(req: PlanRequest) -> str:
    """The canonical serialisation :func:`plan_id` hashes (API.md §0.2)."""
    return json.dumps(req.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def plan_id(req: PlanRequest) -> str:
    """``sha1(canonical JSON)[:16]`` — the stateless API's handle on a plan."""
    return hashlib.sha1(canonical_json(req).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------------------ options


def synth_options(req: PlanRequest) -> dict[str, Any]:
    """The ``plan_motion`` keyword arguments of API.md §3.4, ``shepard`` excluded."""
    opts = req.options
    out: dict[str, Any] = {
        "amp": opts.amp,
        "phases": opts.phases,
        "retard_compensate": opts.retard_compensate,
        "f_z_bias": 0.0 if opts.f_z_bias == "none" else "auto",
        "check_band": opts.check_band,
    }
    if opts.t_pad_us is not None:
        out["t_pad"] = opts.t_pad_us * us
    return out


def _rng(req: PlanRequest) -> np.random.Generator | None:
    """A fresh seeded generator for ``phases="random"`` (API.md §3.4), else ``None``."""
    if req.options.phases != "random":
        return None
    return np.random.default_rng(req.options.phase_seed)


def _is_band_refusal(exc: ValueError) -> bool:
    return "leaves its usable band" in str(exc)


def _auto_config_with_eta(spec: TrajectorySpec, params: AODLParams, eta: float) -> ShepardConfig:
    """:func:`aodl.waveform.shepard.auto_config` with the request's ``eta``.

    The inputs are the ones ``synthesize`` builds for its own ``"auto"`` branch: the lateral
    excursion of each channel, ``max |v P(t) / (2 lambda F)|`` (Eq. S19's half-lateral term,
    ``1 / (2 deflection_scale)`` per metre), and the headroom from the carrier to the nearer
    band edge.
    """
    x, y, _ = spec.compile()
    half = 0.5 / params.deflection_scale
    lateral_span = {
        name: max(abs(v) for v in poly_range((x if name.endswith("x") else y).scale(half)))
        for name in CHANNELS
    }
    headroom = {
        name: min(aod.band[1] - aod.f_center, aod.f_center - aod.band[0])
        for name, aod in params.channels.items()
    }
    return auto_config(spec.array, lateral_span, headroom, eta=eta)


def _resolve_shepard(req: PlanRequest) -> str | ShepardConfig | None:
    """API.md §3.4 ``shepard`` -> the ``plan_motion`` argument.

    ``"never"`` -> ``None``, ``"always"`` -> an explicit :class:`aodl.ShepardConfig`,
    ``"auto"`` -> ``"auto"`` — except at ``eta != 1/2``, where :func:`build_plan` runs the
    two-step auto itself (``plan_motion``'s ``"auto"`` has no ``eta`` knob).
    """
    opts = req.options
    if opts.shepard == "never":
        return None
    if opts.shepard == "always":
        assert opts.shepard_delta_f_MHz is not None  # validated by the schema
        dfx, dfy = opts.shepard_delta_f_MHz
        try:
            return ShepardConfig(delta_f_x=dfx * MHz, delta_f_y=dfy * MHz, eta=opts.eta)
        except ValueError as exc:
            raise classify(exc, "invalid_spec") from exc
    return "auto"


def build_plan(req: PlanRequest) -> tuple[MotionPlan, tuple[str, ...]]:
    """Synthesize the plan; returns it with the web layer's own notes (none from ``aodl``)."""
    spec = to_spec(req)
    params = to_params(req.hardware)
    shepard = _resolve_shepard(req)
    opts = synth_options(req)
    eta = req.options.eta
    notes: list[str] = []
    try:
        if shepard == "auto" and eta != 0.5:
            # plan_motion's "auto" sizes the ladders at Table II's eta = 1/2; honour the
            # request's eta by running the same two steps here: Eq. S19 first, ladders if the
            # band refuses it, with auto_config at the requested duty.
            try:
                plan = plan_motion(spec, params, shepard=None, rng=_rng(req), **opts)
            except ValueError as exc:
                if not _is_band_refusal(exc):
                    raise
                reason = str(exc).split(".  ")[0]
                cfg = _auto_config_with_eta(spec, params, eta)
                plan = plan_motion(spec, params, shepard=cfg, rng=_rng(req), **opts)
                notes.append(
                    f"shepard='auto' at eta = {eta:g}: plain Eq. S19 refused -- {reason}; "
                    f"ladder spacings delta_f = ({cfg.delta_f_x / MHz:.4g}, "
                    f"{cfg.delta_f_y / MHz:.4g}) MHz chosen by auto_config at this duty."
                )
        else:
            plan = plan_motion(spec, params, shepard=shepard, rng=_rng(req), **opts)
    except ValueError as exc:
        raise classify(exc, "invalid_spec") from exc
    except TypeError as exc:
        raise ApiError("invalid_spec", str(exc)) from exc

    if req.options.t_pad_us is not None and req.options.t_pad_us == 0.0:
        notes.append(
            "t_pad_us = 0: the drive ends with the trajectory, so the array's final rest "
            "position is never observable (frames need drive time t - tau/2; the simulate grid "
            "is clamped to t_span[1] - tau/2)."
        )
    if plan.report.mode == "shepard" and req.options.amp != 1.0:
        notes.append(
            f"amp = {req.options.amp:g} on a fading-Shepard drive: valid to simulate, but the "
            f"parametric NPZ has no slot for a FadeZoneEnvelope peak amplitude, so the "
            f"waveform download is refused (unsupported)."
        )
    return plan, tuple(notes)


def get_plan(req: PlanRequest) -> MotionPlan:
    """The cached :class:`aodl.MotionPlan` of ``req``, built on a miss; raises ``ApiError``."""
    key = plan_id(req)
    cached = _PLANS.get(key)
    if cached is not None:
        return cached
    plan, notes = build_plan(req)
    _NOTES.put(key, notes)
    return _PLANS.put(key, plan)


def cached_plan(plan_key: str) -> MotionPlan | None:
    """Look a plan up by its id alone (the NPZ download has no request body)."""
    return _PLANS.get(plan_key)


def web_notes(req: PlanRequest) -> tuple[str, ...]:
    """The web layer's notes for ``req``'s plan (building the plan if needed)."""
    key = plan_id(req)
    notes = _NOTES.get(key)
    if notes is None:
        get_plan(req)
        notes = _NOTES.get(key) or ()
    return notes


# ------------------------------------------------------------------------- derived data


def chosen_spacings(plan: MotionPlan) -> tuple[float, float] | None:
    """``(delta_f_x, delta_f_y)`` [Hz] of a fading-Shepard plan, read off its ``A`` ladders
    (their :class:`aodl.FadeZoneEnvelope`\\ s), or ``None`` for an Eq. S19 plan."""
    spacings: list[float] = []
    for name in ("Ax", "Ay"):
        cw = plan.wfs.channels.get(name)
        if cw is None:
            return None
        found = [tone.env.delta_f for tone in cw.tones if isinstance(tone.env, FadeZoneEnvelope)]
        if not found:
            return None
        spacings.append(found[0])
    return spacings[0], spacings[1]


def shepard_eta(plan: MotionPlan) -> float | None:
    """The fade duty of a fading-Shepard plan (from its first fade envelope), else ``None``."""
    for cw in plan.wfs.channels.values():
        for tone in cw.tones:
            if isinstance(tone.env, FadeZoneEnvelope):
                return tone.env.eta
    return None


def band_usage_mhz(plan: MotionPlan) -> dict[str, BandUsage]:
    """:attr:`aodl.PlanReport.band_usage` in MHz, channels in ``aodl`` order."""
    return {
        name: BandUsage(f_min_MHz=lo / MHz, f_max_MHz=hi / MHz, margin_MHz=margin / MHz)
        for name, (lo, hi, margin) in plan.report.band_usage.items()
    }


def z_budget_model(plan: MotionPlan) -> ZBudget:
    """API.md §4 ``z_budget``: ``ceiling`` null when unbounded, ``fraction`` always against the
    plain Eq. 1 ceiling (the biased one when ``f_z_bias`` is in force on an Eq. S19 plan)."""
    requested, ceiling = plan.report.z_budget
    reference = ceiling if math.isfinite(ceiling) else max_z_integral(plan.params)
    return ZBudget(
        requested_m_s=requested,
        ceiling_m_s=ceiling if math.isfinite(ceiling) else None,
        fraction=requested / reference if reference > 0.0 else 0.0,
    )


def fade_events_model(plan: MotionPlan) -> list[FadeEventModel]:
    return [
        FadeEventModel(time_us=event.time / us, axis=event.axis, shadow_um=event.shadow / um)  # type: ignore[arg-type]
        for event in plan.report.fade_events
    ]


def plan_response(req: PlanRequest) -> PlanResponse:
    """API.md §4 — everything the basic page needs, from one cached plan."""
    from .codegen import generate_code  # local: codegen reads the resolved plan from here

    plan = get_plan(req)
    report = plan.report
    t0, t1 = plan.wfs.t_span
    return PlanResponse(
        plan_id=plan_id(req),
        mode=report.mode,  # type: ignore[arg-type]
        description=report.description,
        summary_text=report.summary(),
        band_usage=band_usage_mhz(plan),
        tone_counts=dict(report.tone_counts),
        z_budget=z_budget_model(plan),
        fade_events=fade_events_model(plan),
        notes=[*report.notes, *web_notes(req)],
        constants=constants(plan.params),
        t_span_us=(t0 / us, t1 / us),
        duration_us=plan.spec.duration / us,
        seams_us=[s / us for s in seams(plan.spec)],
        trajectory=sample_trajectory(plan.spec),
        waveform_ir=to_ir(plan.wfs),
        code_python=generate_code(req),
    )


# ---------------------------------------------------------------------- movie metrics


def probe_times(plan: MotionPlan) -> list[float]:
    """Observation times [s] whose frequency-group count bounds the movie's cost.

    Three probes: the first fully-lit frame (``t = 2 tau``), the middle of the trajectory, and
    — on a fading drive — the first hand-over seen at the atom plane (``fade_events[0].time +
    tau/2``).  A hand-over frame is the busiest a fading drive gets (two ``A`` rungs live, so
    every ``B`` rung feeds two groups), and a budget read off the quiet first frame alone let
    the 4x4 ``array_lift_hold`` movie run at mixing order 3 (152 groups at ``2 tau``, ~220 in a
    hand-over) and take 6 s instead of 2 s.  Times past the simulate coverage limit are clipped.
    """
    tau = transit_time(plan.params)
    latest = plan.wfs.t_span[1] - 0.5 * tau
    times = [2.0 * tau, 0.5 * plan.spec.duration + 0.5 * tau]
    # The first hand-over whose frame is fully lit: ``fade_schedule`` can report a y event at
    # t = 0 (the xi_y = 1/2 ladder starts in a fade zone), which would probe a dark frame.
    lit = [e.time + 0.5 * tau for e in plan.report.fade_events if e.time + 0.5 * tau >= 2.0 * tau]
    if lit:
        times.append(lit[0])
    return sorted({min(max(t, 0.0), latest) for t in times})


def first_frame_groups(req: PlanRequest) -> int:
    """Largest frequency-group count over :func:`probe_times` at the hardware's order.

    The name is historical (it probed only ``t = 2 tau`` at first); the value is what the
    movie-order budget of :func:`resolve_movie_order` compares against.
    """
    key = plan_id(req)
    cached = _GROUP_COUNTS.get(key)
    if cached is not None:
        return cached
    plan = get_plan(req)
    try:
        count = max(len(group_terms(build_terms(plan.wfs, t))) for t in probe_times(plan))
    except ValueError as exc:
        raise classify(exc, "too_many_terms") from exc
    return _GROUP_COUNTS.put(key, count)


def resolve_movie_order(req: PlanRequest, movie_mixing_order: int | None) -> tuple[int, str | None]:
    """DESIGN §1.2: the mixing order the *movie metrics* use, and a note when it differs.

    ``movie_mixing_order`` pins it; otherwise the hardware's order stands unless the busiest
    probed frame (:func:`probe_times`) at that order holds more than :data:`MOVIE_GROUP_BUDGET`
    frequency groups, in which case the metrics drop to order 1 (the plan and report keep the
    requested order).
    """
    requested = req.hardware.mixing_order
    if movie_mixing_order is not None:
        if movie_mixing_order == requested:
            return requested, None
        return movie_mixing_order, (
            f"movie metrics computed at mixing order {movie_mixing_order} as requested "
            f"(the plan and report use the hardware's order {requested})."
        )
    if requested == 1:
        return 1, None
    groups = first_frame_groups(req)
    if groups > MOVIE_GROUP_BUDGET:
        return 1, (
            f"movie metrics fell back to mixing order 1: the busiest probed frame at order "
            f"{requested} holds {groups} frequency groups (budget {MOVIE_GROUP_BUDGET}); pass "
            f"movie_mixing_order={requested} to override."
        )
    return requested, None


def movie_params(req: PlanRequest, movie_mixing_order: int | None) -> AODLParams:
    """The plan's hardware with the movie metrics' mixing order (DESIGN §1.2 rule)."""
    params = get_plan(req).params
    order, _ = resolve_movie_order(req, movie_mixing_order)
    if all(aod.mixing_order == order for aod in params.channels.values()):
        return params
    return replace(
        params,
        channels={name: replace(aod, mixing_order=order) for name, aod in params.channels.items()},
    )


__all__ = [
    "band_usage_mhz",
    "build_plan",
    "cached_plan",
    "canonical_json",
    "chosen_spacings",
    "fade_events_model",
    "first_frame_groups",
    "get_plan",
    "movie_params",
    "plan_id",
    "plan_response",
    "resolve_movie_order",
    "shepard_eta",
    "synth_options",
    "web_notes",
    "z_budget_model",
]
