"""``/api/advanced/*`` — the FFT pupil engine and efficiency model behind the web API (API.md §6).

Owned by the advanced-engine work order (W2).  This module is the only place where
``aodl_advanced`` meets ``aodl_webapp``: it converts the display units of the JSON contract
to SI, resolves each frame's tracked plane and grid centre with the core package's frozen
helpers (:func:`aodl_webapp.simulation.tracked_plane`, :func:`~aodl_webapp.simulation.frame_spots`),
runs the engine, and packs the images.  Routes are plain ``def`` (FastAPI's threadpool); the
aligned reference of ``/simulate`` is cached per request-minus-misalignment.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import APIRouter

from aodl.params import CHANNELS, AODLParams
from aodl.units import MHz, um, us
from aodl.waveform.tones import WaveformSet
from aodl_advanced import efficiency as eff
from aodl_advanced.engine import (
    FrameResult,
    GridSpec,
    ModelSpec,
    XzSpec,
    check_coverage,
    simulate_frames,
)
from aodl_advanced.misalignment import AODMisalignment, Misalignment
from aodl_advanced.pupil import static_offset
from aodl_advanced.schemas import (
    AdvancedSimulateRequest,
    AdvancedSimulateResponse,
    AdvFrame,
    AdvImageXY,
    AdvImageXZ,
    AdvMetricsModel,
    EfficiencyModelModel,
    EfficiencyRequest,
    EfficiencyResponse,
    MisalignmentModel,
    PairMap,
    Ridge,
    SpectrumRequest,
    SpectrumResponse,
    StaticOffset,
    TrajectoryEfficiencyRequest,
    TrajectoryEfficiencyResponse,
)
from aodl_advanced.spectrum import spectrum as rf_spectrum

from ..errors import ApiError, classify
from ..hardware import to_params
from ..planner import get_plan, plan_id
from ..simulation import frame_spots, pack_image, tracked_plane

router = APIRouter(prefix="/advanced", tags=["advanced"])

#: Aligned reference frames kept (one entry per distinct request minus misalignment).
REFERENCE_CACHE_SIZE = 16
#: Points per axis of the pair map (API.md §6.2: ``m = 121``).
PAIR_POINTS = 121

_NS = 1e-9
_MRAD = 1e-3


# ------------------------------------------------------------------------- conversions


def _efficiency_model(params: AODLParams, m: EfficiencyModelModel) -> eff.EfficiencyModel:
    return eff.default_model(
        params,
        interaction_length=m.interaction_length_mm * 1e-3,
        f_tpm=None if m.f_tpm_MHz is None else m.f_tpm_MHz * MHz,
        refractive_index=m.refractive_index,
        peak_efficiency=m.peak_efficiency,
        angular_coupling=m.angular_coupling,
    )


def _misalignment(m: MisalignmentModel) -> Misalignment:
    out: dict[str, AODMisalignment] = {}
    for name in CHANNELS:
        entry = getattr(m, name)
        out[name] = AODMisalignment(
            offset_par=entry.offset_par_um * um,
            offset_perp=entry.offset_perp_um * um,
            rotation=entry.rotation_mrad * _MRAD,
            delay=entry.delay_ns * _NS,
            velocity_scale=entry.velocity_scale,
            tilt_x=entry.tilt_x_mrad * _MRAD,
            tilt_y=entry.tilt_y_mrad * _MRAD,
        )
    return Misalignment(out)


def _list(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


# ------------------------------------------------------------------------- efficiency


@router.post("/efficiency", response_model=EfficiencyResponse)
def efficiency(req: EfficiencyRequest) -> EfficiencyResponse:
    """Single-AOD and pair efficiency curves of the TPM Bragg model (API.md §6.2)."""
    params = to_params(req.hardware)
    model = _efficiency_model(params, req.model)
    aod = params.channels[CHANNELS[0]]
    band = aod.band
    half_band = 0.5 * (band[1] - band[0])
    if req.f_range_MHz is None:
        lo, hi = aod.f_center - 1.5 * half_band, aod.f_center + 1.5 * half_band
    else:
        lo, hi = req.f_range_MHz[0] * MHz, req.f_range_MHz[1] * MHz
    f = np.linspace(lo, hi, int(req.n))
    eta_first = eff.first_aod_efficiency(f, model, params)
    eta_second = eff.second_aod_efficiency(f, model.f_tpm, model, params)
    angle = eff.deflection_angle(f, aod.f_center, params.sound_speed, params.optics.wavelength)
    f_pair = np.linspace(lo, hi, PAIR_POINTS)
    pair = eff.pair_map(f_pair, f_pair, model, params)
    ridge_max, ridge_fb = eff.ridge(f_pair, model, params, band)
    return EfficiencyResponse(
        f_MHz=_list(f / MHz),
        deflection_mrad=_list(angle / _MRAD),
        eta_first=_list(eta_first),
        eta_second_at_center=_list(eta_second),
        bandwidth_3dB_MHz=eff.bandwidth_3db(model, params) / MHz,
        first_aod_usable_MHz=eff.first_aod_usable_width(model, params, band) / MHz,
        band_MHz=(band[0] / MHz, band[1] / MHz),
        f_tpm_MHz=model.f_tpm / MHz,
        pair=PairMap(
            f_A_MHz=_list(f_pair / MHz),
            f_B_MHz=_list(f_pair / MHz),
            eta=[_list(row) for row in pair],
        ),
        ridge=Ridge(
            f_A_MHz=_list(f_pair / MHz),
            eta_max=_list(ridge_max),
            f_B_at_max_MHz=_list(ridge_fb / MHz),
        ),
        assumptions=eff.assumptions_text(model, params),
    )


@router.post("/trajectory_efficiency", response_model=TrajectoryEfficiencyResponse)
def trajectory_efficiency(req: TrajectoryEfficiencyRequest) -> TrajectoryEfficiencyResponse:
    """Efficiency of the four channels along a plan's drive (API.md §6.3).

    ``t_us`` are *observation* times ``t = t_c + tau/2`` spanning the whole programmed drive.
    """
    plan = get_plan(req.plan)
    params = plan.params
    model = _efficiency_model(params, req.model)
    tau = max(aod.transit_time for aod in params.channels.values())
    t0, t1 = plan.wfs.t_span
    times = np.linspace(t0 + 0.5 * tau, t1 + 0.5 * tau, int(req.n))
    table = eff.trajectory_efficiency(plan.wfs, model, times)
    per_channel = {name: _list(table[name]) for name in CHANNELS if name in table}
    product = table["product"]
    return TrajectoryEfficiencyResponse(
        t_us=_list(times / us),
        per_channel=per_channel,
        product=_list(product),
        min_product=float(np.min(product)) if product.size else 0.0,
        assumptions=eff.assumptions_text(model, params)
        + " Per channel: amplitude^2-weighted mean over the live tones at the beam-centre "
        "retarded time t - tau/2; B channels closed over the A channel's live frequency; the "
        "product is the predicted relative trap power.",
    )


# --------------------------------------------------------------------------- simulate


@dataclass
class ReferenceEntry:
    """Cached aligned run of one ``/simulate`` request (minus its misalignment)."""

    planes: list[float]
    centres: list[tuple[float, float]]
    frames: list[FrameResult]


_REFERENCE: OrderedDict[str, ReferenceEntry] = OrderedDict()


def reference_key(req: AdvancedSimulateRequest) -> str:
    """Canonical JSON of the request **without** ``misalignment`` — the reference cache key."""
    payload = req.model_dump(mode="json", exclude={"misalignment"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _engine_specs(
    req: AdvancedSimulateRequest, params: AODLParams
) -> tuple[GridSpec, XzSpec | None, ModelSpec]:
    grid = GridSpec(req.grid.half_x_um * um, req.grid.half_y_um * um, int(req.grid.n))
    xz = None if req.xz is None else XzSpec(int(req.xz.nz), req.xz.z_half_um * um)
    model = ModelSpec(
        pupil_n=int(req.model.pupil_n),
        span_w_in=float(req.model.span_w_in),
        order_model=req.model.order_model,
        comb_dft_points=int(req.model.comb_dft_points),
        max_groups=int(req.model.max_groups),
        efficiency=(
            None
            if req.model.efficiency is None
            else _efficiency_model(params, req.model.efficiency)
        ),
    )
    return grid, xz, model


def _planes_and_centres(
    req: AdvancedSimulateRequest, plan: Any, params: AODLParams, times: Sequence[float]
) -> tuple[list[float], list[tuple[float, float]]]:
    """Tracked plane and grid centre per frame, from the core package's spot metrics."""
    planes: list[float] = []
    centres: list[tuple[float, float]] = []
    for t in times:
        try:
            plane = float(tracked_plane(plan, float(t), params))
            spots = frame_spots(plan, float(t), params) if req.grid.center == "tracked" else []
        except ValueError as exc:
            raise classify(exc, default="coverage") from exc
        planes.append(plane)
        if req.grid.center != "tracked":
            centres.append((req.grid.center.x_um * um, req.grid.center.y_um * um))
            continue
        weights = np.asarray([s.power for s in spots], dtype=np.float64)
        if spots and weights.sum() > 0.0:
            xs = np.asarray([s.x_um for s in spots]) * um
            ys = np.asarray([s.y_um for s in spots]) * um
            centres.append(
                (float(weights @ xs / weights.sum()), float(weights @ ys / weights.sum()))
            )
        else:
            centres.append((0.0, 0.0))
    return planes, centres


def get_reference(
    req: AdvancedSimulateRequest, plan: Any, times: Sequence[float]
) -> ReferenceEntry:
    """The aligned run for ``req`` (cached by :func:`reference_key`, LRU of
    :data:`REFERENCE_CACHE_SIZE`)."""
    key = reference_key(req)
    hit = _REFERENCE.get(key)
    if hit is not None:
        _REFERENCE.move_to_end(key)
        return hit
    params = plan.params
    planes, centres = _planes_and_centres(req, plan, params, times)
    grid, xz, model = _engine_specs(req, params)
    try:
        frames = simulate_frames(
            plan.wfs, params, times, planes, centres, Misalignment.aligned(), grid, xz, model, False
        )
    except ValueError as exc:
        raise ApiError("coverage", str(exc)) from exc
    entry = ReferenceEntry(planes=planes, centres=centres, frames=frames)
    _REFERENCE[key] = entry
    while len(_REFERENCE) > REFERENCE_CACHE_SIZE:
        _REFERENCE.popitem(last=False)
    return entry


def _metrics_model(frame: FrameResult, reference: bool) -> AdvMetricsModel:
    m = frame.metrics
    ref = frame.reference_metrics if reference else None
    rel = None
    strehl = None
    power_rel = None
    if ref is not None:
        strehl = _finite(m.peak / ref.peak) if ref.peak > 0.0 else None
        rel = strehl
        power_rel = _finite(m.power / ref.power) if ref.power > 0.0 else None
    return AdvMetricsModel(
        x_um=m.x / um,
        y_um=m.y / um,
        wx_um=m.wx / um,
        wy_um=m.wy / um,
        z_best_x_um=None if m.z_best_x is None else m.z_best_x / um,
        z_best_y_um=None if m.z_best_y is None else m.z_best_y / um,
        astig_um=None if m.astig is None else m.astig / um,
        peak_rel=rel,
        power_rel=power_rel,
        strehl=strehl,
    )


def _reference_model(frame: FrameResult, ref: FrameResult | None) -> AdvMetricsModel | None:
    if ref is None:
        return None
    m = ref.metrics
    return AdvMetricsModel(
        x_um=m.x / um,
        y_um=m.y / um,
        wx_um=m.wx / um,
        wy_um=m.wy / um,
        z_best_x_um=None if m.z_best_x is None else m.z_best_x / um,
        z_best_y_um=None if m.z_best_y is None else m.z_best_y / um,
        astig_um=None if m.astig is None else m.astig / um,
        peak_rel=1.0,
        power_rel=1.0,
        strehl=1.0,
    )


def _frame_model(frame: FrameResult, ref: FrameResult | None, reference: bool) -> AdvFrame:
    xz = None
    if frame.xz is not None and frame.xz_x is not None and frame.xz_z is not None:
        xz = AdvImageXZ(
            x_um=_list(frame.xz_x / um),
            z_um=_list(frame.xz_z / um),
            row_y_um=frame.row_y / um,
            image=pack_image(frame.xz),
        )
    return AdvFrame(
        time_us=frame.time / us,
        plane_z_um=frame.plane_z / um,
        n_groups=frame.n_groups,
        groups_dropped=frame.groups_dropped,
        xy=AdvImageXY(
            x_um=_list(frame.x / um), y_um=_list(frame.y / um), image=pack_image(frame.xy)
        ),
        xz=xz,
        metrics=_metrics_model(frame, reference),
        reference_metrics=_reference_model(frame, ref) if reference else None,
    )


def _check_times(wfs: WaveformSet, times: Sequence[float]) -> None:
    for t in times:
        if not math.isfinite(t):
            raise ApiError("invalid_spec", "times_us must be finite")
        try:
            check_coverage(wfs, float(t))
        except ValueError as exc:
            raise ApiError("coverage", str(exc)) from exc


@router.post("/simulate", response_model=AdvancedSimulateResponse)
def simulate(req: AdvancedSimulateRequest) -> AdvancedSimulateResponse:
    """FFT-engine frames with per-AOD misalignment and the aligned reference (API.md §6.5)."""
    plan = get_plan(req.plan)
    params = plan.params
    times = [float(t) * us for t in req.times_us]
    _check_times(plan.wfs, times)
    mis = _misalignment(req.misalignment)
    grid, xz, model = _engine_specs(req, params)
    notes: list[str] = []

    if req.reference or mis.is_aligned:
        entry = get_reference(req, plan, times)
        planes, centres = entry.planes, entry.centres
        reference_frames: list[FrameResult] | None = entry.frames
    else:
        planes, centres = _planes_and_centres(req, plan, params, times)
        reference_frames = None

    if mis.is_aligned:
        frames = reference_frames or []
        for frame in frames:
            frame.reference_metrics = frame.metrics
        refs: list[FrameResult | None] = list(frames)
    else:
        try:
            frames = simulate_frames(
                plan.wfs,
                params,
                times,
                planes,
                centres,
                mis,
                grid,
                xz,
                model,
                req.reference,
                reference_frames,
            )
        except ValueError as exc:
            raise ApiError("coverage", str(exc)) from exc
        refs = list(reference_frames) if reference_frames is not None else [None] * len(frames)

    for frame in frames:
        notes.extend(f"t = {frame.time / us:.4g} us: {note}" for note in frame.notes)
    notes.append(
        "metrics describe the brightest spot (intensity moments over +-4 w0 about the peak); "
        "image units are the engine's own — compare only images from this call"
    )
    offsets = static_offset(params, mis)
    return AdvancedSimulateResponse(
        plan_id=plan_id(req.plan),
        frames=[
            _frame_model(frame, ref, req.reference) for frame, ref in zip(frames, refs, strict=True)
        ],
        static_offset_um={
            name: StaticOffset(x=offsets[name][0] / um, y=offsets[name][1] / um)
            for name in CHANNELS
        },
        notes=list(dict.fromkeys(notes)),
    )


# ---------------------------------------------------------------------------- spectrum


@router.post("/spectrum", response_model=SpectrumResponse)
def spectrum(req: SpectrumRequest) -> SpectrumResponse:
    """Hann periodogram, crest factor, PAPR and peak modulation depth (API.md §6.6)."""
    plan = get_plan(req.plan)
    try:
        result = rf_spectrum(
            plan.wfs,
            req.channel,
            req.t_center_us * us,
            req.window_us * us,
            req.sample_rate_MSps * MHz,
        )
    except ValueError as exc:
        code = "coverage" if "outside the programmed drive" in str(exc) else "invalid_spec"
        raise ApiError(code, str(exc)) from exc
    band = plan.params.channels[req.channel].band
    return SpectrumResponse(
        f_MHz=_list(result.f / MHz),
        psd_dB=_list(result.psd_db),
        crest_factor=result.crest_factor,
        papr_dB=result.papr_db,
        peak_depth_rad=result.peak_depth_rad,
        band_MHz=(band[0] / MHz, band[1] / MHz),
        notes=list(result.notes),
    )


__all__ = [
    "PAIR_POINTS",
    "REFERENCE_CACHE_SIZE",
    "ReferenceEntry",
    "get_reference",
    "reference_key",
    "router",
]
