r"""Movie metrics, exact frames and the spot clustering behind them (API.md §5.1-5.2).

**One spot per spatial cluster.**  :func:`aodl.field.measure.measure` reports one
:class:`~aodl.SpotMetrics` per optical-frequency group, as the power-weighted centroid of the
group's pupil terms.  A group can hold terms that are degenerate in frequency yet metres
apart in nothing but position — the Eq. S31 shadow-tweezer pair at ``+- deflection_scale
Delta f`` during a fade, or the two array traps two pitches apart of ``docs/guide.md`` §6.7 —
and the centroid of such a group is a phantom on the main trap carrying the shadows' power
(REVIEW A2).  :func:`cluster_spots` therefore splits every group by the term centres
``theta1 F / k`` (Table I's deflection) into single-linkage clusters within one focal waist and
measures each cluster on its own; spatially separated degenerate terms have negligible pupil
overlap, so adding the clusters in intensity is exact to that overlap (DESIGN §1 item 2).

**Cost.**  :func:`aodl.device.aodl.build_terms` dominates a frame (40 ms for a 4x4 fading
array at mixing order 3, against 5 ms for ``measure``), so :func:`simulate_frames` expands
each frame's terms exactly once, measures them, clusters them, and assembles the
:class:`aodl.SimResult` itself; the coverage rule of :func:`aodl.engine.simulate` is still
enforced by running it on the first and last frame.
"""

from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aodl.api import MotionPlan
from aodl.device.aodl import TermArray, build_terms
from aodl.engine import SimResult, _subset, simulate
from aodl.field.focal import GROUP_TOL, FrameGrid, group_terms, intensity_frame, intensity_slice_xz
from aodl.field.measure import SpotMetrics, measure, track_z
from aodl.params import AODLParams, OpticsParams
from aodl.units import MHz, um, us
from aodl.waveform.tones import WaveformSet

from .cache import LRUCache
from .config import SIM_CACHE_SIZE, SPOT_DIGITS, SPOT_POWER_CUT
from .errors import ApiError, classify
from .hardware import constants, transit_time
from .planner import get_plan, movie_params, plan_id, resolve_movie_order
from .schemas import (
    Bounds,
    CutPoint,
    Cuts,
    FrameRequest,
    FrameResponse,
    ImageXY,
    ImageXZ,
    PackedImage,
    PlaneZ,
    SimulateRequest,
    SimulateResponse,
    Spot,
    SpotFrame,
)
from .trajectory import sample_trajectory

Float = NDArray[np.float64]
Index = NDArray[np.intp]

#: Lateral margin of ``bounds_um`` around the requested trajectory, in focal waists.
BOUNDS_WAISTS = 6.0

#: Axial margin of ``bounds_um``, in Rayleigh ranges.
BOUNDS_RAYLEIGH = 1.0

#: ``z`` extent of the default (``xz=None``) axial cut, in µm about the plane.
DEFAULT_CUT_Z_HALF_UM = 8.0
DEFAULT_CUT_NZ = 96

#: Fields of :class:`Spot`, in order.
SPOT_FIELDS: tuple[str, ...] = (
    "x_um",
    "y_um",
    "z_um",
    "delta_f_um",
    "sigma_astig",
    "wx_um",
    "wy_um",
    "power",
    "power_coherent",
    "df_opt_MHz",
    "group",
    "n_terms",
)

ClusterRow = tuple[SpotMetrics, int, int]


@dataclass
class CachedSim:
    """A simulation run and its clustered spots, one list per frame."""

    result: SimResult
    clusters: list[list[ClusterRow]]
    order: int
    note: str | None


_SIMS: LRUCache[CachedSim] = LRUCache(SIM_CACHE_SIZE)


# ----------------------------------------------------------------------------- clustering


def _single_linkage(x: Float, y: Float, radius: float) -> list[Index]:
    """Single-linkage clusters of the points ``(x, y)``: two points link when closer than
    ``radius``.  Groups are tiny (a handful of degenerate terms), so ``O(n^2)`` is fine."""
    n = x.size
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    close = np.hypot(dx, dy) <= radius
    for i in range(n):
        for j in range(i + 1, n):
            if close[i, j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    return [np.flatnonzero(roots == root) for root in np.unique(roots)]


def _per_term_metrics(
    terms: TermArray, optics: OpticsParams, idx: Index, tol: float
) -> list[SpotMetrics]:
    """:class:`aodl.SpotMetrics` of each term in ``idx`` *on its own*, in one ``measure`` call.

    The subset is re-tagged with distinct optical frequencies (``10 tol`` apart) so that
    :func:`aodl.field.focal.group_terms` puts every term in a group of its own; nothing in
    the metrics but ``df_opt`` depends on the tag, and ``df_opt`` is restored from the term.
    Ordered like ``idx``.  One vectorised call instead of one ``measure`` per singleton: a
    mid-hand-over 10x10 frame splits ~100 groups, which made this the dominant cost.
    """
    subset = _subset(terms, idx)
    tagged = replace(subset, df_opt=np.arange(idx.size, dtype=np.float64) * (10.0 * tol))
    true_tags = np.asarray(terms.df_opt, dtype=np.float64)[idx]
    return [
        replace(metrics, df_opt=float(tag))
        for metrics, tag in zip(measure(tagged, optics, tol), true_tags, strict=True)
    ]


def cluster_spots(
    terms: TermArray, optics: OpticsParams, tol: float = GROUP_TOL
) -> list[ClusterRow]:
    """One ``(SpotMetrics, group, n_terms)`` per spatial cluster of each frequency group.

    For every :func:`aodl.field.focal.group_terms` group the term centres ``theta1 F / k``
    (Table I) are clustered by single linkage within one focal waist ``w0``; a group that is
    one cluster is measured whole, a split group is measured cluster by cluster through
    :func:`aodl.engine._subset` (single-term clusters batched, see :func:`_per_term_metrics`).
    Clusters are ordered by group, then by ``x``.
    """
    if terms.n_terms == 0:
        return []
    groups = group_terms(terms, tol)
    whole = measure(terms, optics, tol)
    k = optics.k
    focal = optics.focal_length
    xc = np.asarray(terms.theta1[0], dtype=np.float64) * focal / k
    yc = np.asarray(terms.theta1[1], dtype=np.float64) * focal / k
    radius = optics.waist0
    out: list[ClusterRow | None] = []
    singles: list[tuple[int, int]] = []  # (output slot, term index) to fill from the batch
    for g, idx in enumerate(groups):
        if idx.size == 1 or (np.ptp(xc[idx]) <= radius and np.ptp(yc[idx]) <= radius):
            out.append((whole[g], g, int(idx.size)))
            continue
        clusters = _single_linkage(xc[idx], yc[idx], radius)
        if len(clusters) == 1:
            out.append((whole[g], g, int(idx.size)))
            continue
        clusters.sort(key=lambda members: float(np.mean(xc[idx[members]])))
        for members in clusters:
            subset = np.asarray(idx[members], dtype=np.intp)
            if subset.size == 1:
                singles.append((len(out), int(subset[0])))
                out.append(None)
            else:
                metrics = measure(_subset(terms, subset), optics, tol)
                out.append((metrics[0], g, int(subset.size)))
    if singles:
        slots, term_idx = zip(*singles, strict=True)
        batch = _per_term_metrics(terms, optics, np.asarray(term_idx, dtype=np.intp), tol)
        # the group of a singleton is the group its term belongs to
        owner = np.empty(terms.n_terms, dtype=np.intp)
        for g, idx in enumerate(groups):
            owner[idx] = g
        for slot, i, single in zip(slots, term_idx, batch, strict=True):
            out[slot] = (single, int(owner[i]), 1)
    return [row for row in out if row is not None]


def spot_model(metrics: SpotMetrics, group: int, n_terms: int) -> Spot:
    """A :class:`Spot` in display units."""
    return Spot(
        x_um=metrics.x / um,
        y_um=metrics.y / um,
        z_um=metrics.z_lab / um,
        delta_f_um=metrics.delta_f / um,
        sigma_astig=metrics.sigma_astig,
        wx_um=metrics.wx / um,
        wy_um=metrics.wy / um,
        power=metrics.power,
        power_coherent=metrics.power_coherent,
        df_opt_MHz=metrics.df_opt / MHz,
        group=group,
        n_terms=n_terms,
    )


def _sig(value: float, digits: int = SPOT_DIGITS) -> float:
    """``value`` rounded to ``digits`` significant digits."""
    if value == 0.0 or not math.isfinite(value):
        return value
    return float(f"{value:.{digits}g}")


def frame_model(rows: list[ClusterRow], cut: float = SPOT_POWER_CUT) -> SpotFrame:
    """Struct-of-arrays :class:`SpotFrame`, rounded, spots below ``cut`` of the max dropped."""
    spots = [spot_model(m, g, n) for m, g, n in rows]
    if spots:
        peak = max(s.power for s in spots)
        spots = [s for s in spots if s.power >= cut * peak]
    columns: dict[str, list[Any]] = {field: [] for field in SPOT_FIELDS}
    for spot in spots:
        for field in SPOT_FIELDS:
            value = getattr(spot, field)
            columns[field].append(value if isinstance(value, int) else _sig(value))
    return SpotFrame(**columns)


# ------------------------------------------------------------------------------ running


def _with_params(wfs: WaveformSet, params: AODLParams) -> WaveformSet:
    return wfs if params is wfs.params else replace(wfs, params=params)


def simulate_frames(
    wfs: WaveformSet, params: AODLParams, times: Float
) -> tuple[SimResult, list[list[ClusterRow]]]:
    """Metrics and clustered spots for every frame, each frame's terms built once.

    ``params`` carries the mixing order the metrics use (DESIGN §1.2).  The first and last
    frame go through :func:`aodl.engine.simulate` so its coverage refusals apply verbatim.
    """
    drive = _with_params(wfs, params)
    edges = np.asarray([np.min(times), np.max(times)], dtype=np.float64)
    try:
        simulate(drive, edges)
    except ValueError as exc:
        raise classify(exc, "coverage") from exc
    names = tuple(drive.channels)
    optics = params.optics
    metrics: list[list[SpotMetrics]] = []
    clusters: list[list[ClusterRow]] = []
    try:
        for t in times:
            terms = build_terms(drive, float(t), names)
            rows = cluster_spots(terms, optics)
            clusters.append(rows)
            # one record per group, in group order, exactly as aodl.engine.simulate stores it
            per_group: dict[int, SpotMetrics] = {}
            for m, g, _ in rows:
                per_group.setdefault(g, m)
            metrics.append([per_group[g] for g in sorted(per_group)])
    except ValueError as exc:
        raise classify(exc, "too_many_terms") from exc
    result = SimResult(times=times, metrics=metrics, wfs=drive, params=params, channels=names)
    return result, clusters


def _times_key(times: Float) -> str:
    return hashlib.sha1(np.ascontiguousarray(times, dtype=np.float64).tobytes()).hexdigest()[:16]


def get_cached_sim(req: Any, times_s: Float, movie_mixing_order: int | None) -> CachedSim:
    """The cached run of ``req``'s plan at ``times_s`` [s] with the movie mixing order."""
    plan_req = req
    plan = get_plan(plan_req)
    order, note = resolve_movie_order(plan_req, movie_mixing_order)
    params = movie_params(plan_req, movie_mixing_order)
    key = f"{plan_id(plan_req)}:{order}:{_times_key(times_s)}"

    def build() -> CachedSim:
        result, clusters = simulate_frames(plan.wfs, params, times_s)
        return CachedSim(result=result, clusters=clusters, order=order, note=note)

    return _SIMS.get_or_build(key, build)


def get_sim(req: Any, times_s: Float, movie_mixing_order: int | None) -> SimResult:
    """The cached :class:`aodl.SimResult` of ``req`` (a ``PlanRequest``) at ``times_s`` [s]."""
    return get_cached_sim(req, np.asarray(times_s, dtype=np.float64), movie_mixing_order).result


def tracked_plane(plan: MotionPlan, t_s: float, params: AODLParams) -> float:
    """The tracked lab plane [m] of a one-frame simulate at ``t_s``: ``track_z`` of its metrics."""
    result, _ = simulate_frames(plan.wfs, params, np.asarray([float(t_s)], dtype=np.float64))
    return track_z(result.metrics[0])


def frame_spots(plan: MotionPlan, t_s: float, params: AODLParams) -> list[Spot]:
    """The clustered spots of one frame at ``t_s`` [s] (what W2 centres its grid on)."""
    _, clusters = simulate_frames(plan.wfs, params, np.asarray([float(t_s)], dtype=np.float64))
    return [spot_model(m, g, n) for m, g, n in clusters[0]]


# --------------------------------------------------------------------------- /simulate


def frame_times(req: SimulateRequest, plan: MotionPlan) -> tuple[Float, list[str]]:
    """API.md §5.1 defaults: ``linspace(0, min(T + tau, t_span[1] - tau/2), n_frames)``."""
    tau = transit_time(plan.params)
    t0, t1 = plan.wfs.t_span
    latest = t1 - 0.5 * tau
    notes: list[str] = []
    start = 0.0 if req.t_start_us is None else req.t_start_us * us
    end = min(plan.spec.duration + tau, latest) if req.t_end_us is None else req.t_end_us * us
    if end > latest + 1e-12:
        notes.append(
            f"t_end_us clamped from {end / us:.4g} to {latest / us:.4g} us: a frame at t needs "
            f"drive time t - tau/2, and the drive is programmed to {t1 / us:.4g} us."
        )
        end = latest
    if start < t0:
        raise ApiError(
            "coverage",
            f"t_start_us = {start / us:.4g} is before the drive starts ({t0 / us:.4g} us): the "
            f"aperture holds no drive at all there (docs/conventions.md §7).",
        )
    if start > end:
        raise ApiError(
            "coverage",
            f"t_start_us = {start / us:.4g} us is past the last observable frame "
            f"{latest / us:.4g} us (= t_span[1] - tau/2).",
        )
    return np.linspace(start, end, req.n_frames), notes


def trajectory_bounds(plan: MotionPlan) -> Bounds:
    """``bounds_um`` from the *requested* trajectory (DESIGN §1 item 2): array half-extent
    plus :data:`BOUNDS_WAISTS` waists laterally, :data:`BOUNDS_RAYLEIGH` Rayleigh ranges
    axially — never from the spot table, whose shadows sit at ``+- 10.3 Delta f`` µm."""
    samples = sample_trajectory(plan.spec, 200)
    optics = plan.params.optics
    pitch_x, pitch_y = plan.spec.array.pitch(plan.params)
    half_x = 0.5 * (plan.spec.array.mx - 1) * abs(pitch_x) / um
    half_y = 0.5 * (plan.spec.array.my - 1) * abs(pitch_y) / um
    lateral = BOUNDS_WAISTS * optics.waist0 / um
    axial = BOUNDS_RAYLEIGH * optics.rayleigh / um
    return Bounds(
        x0=min(samples.x_um) - half_x - lateral,
        x1=max(samples.x_um) + half_x + lateral,
        y0=min(samples.y_um) - half_y - lateral,
        y1=max(samples.y_um) + half_y + lateral,
        z0=min(samples.z_um) - axial,
        z1=max(samples.z_um) + axial,
    )


def brightest_trap_power(frame: SpotFrame, waist0_um: float) -> float:
    """Power of the brightest *trap* in a frame: spots within one waist of each other are the
    same trap and add.

    A fading-Shepard hand-over splits the main trap into two coincident frequency groups of
    half power each (DESIGN §1 item 2), so a per-spot statistic would read the trap at ½ there
    while the Eq. S31 shadows read ½ too; merging coincident spots first restores the trap's
    unit power at every instant, which is what a normalisation reference has to measure.
    """
    power = np.asarray(frame.power, dtype=np.float64)
    if power.size == 0:
        return 0.0
    x = np.asarray(frame.x_um, dtype=np.float64)
    y = np.asarray(frame.y_um, dtype=np.float64)
    order = np.argsort(-power)
    centres: list[tuple[float, float]] = []
    sums: list[float] = []
    for i in order:
        for k, (cx, cy) in enumerate(centres):
            if math.hypot(x[i] - cx, y[i] - cy) <= waist0_um:
                sums[k] += float(power[i])
                break
        else:
            centres.append((float(x[i]), float(y[i])))
            sums.append(float(power[i]))
    return max(sums)


def reference_power(frames: list[SpotFrame], waist0_um: float) -> float:
    """``power_ref`` of API.md §5.1: the median over lit frames of the brightest trap's power.

    Robust to the fill transient (dark frames are skipped), to hand-overs (coincident groups
    are merged by :func:`brightest_trap_power`) and to shadow tweezers (never the brightest).
    A single tweezer or any array therefore reads ``power / power_ref ≈ 1`` on its plateau.
    """
    values = [brightest_trap_power(f, waist0_um) for f in frames]
    lit = np.asarray([v for v in values if v > 0.0], dtype=np.float64)
    return float(np.median(lit)) if lit.size else 0.0


def simulate_response(req: SimulateRequest) -> SimulateResponse:
    """API.md §5.1."""
    plan = get_plan(req.plan)
    times, notes = frame_times(req, plan)
    sim = get_cached_sim(req.plan, times, req.movie_mixing_order)
    if sim.note:
        notes.append(sim.note)
    tau = transit_time(plan.params)
    if times[0] < 0.5 * tau:
        notes.append(
            f"frames before tau/2 = {0.5 * tau / us:.2f} us are dark: a pair-driven tweezer "
            f"needs both counter-propagating wavefronts to reach the beam (docs/conventions.md §7)."
        )
    frames = [frame_model(rows) for rows in sim.clusters]
    power_ref = reference_power(frames, plan.params.optics.waist0 / um)
    return SimulateResponse(
        plan_id=plan_id(req.plan),
        times_us=(times / us).tolist(),
        tracked_z_um=(sim.result.tracked_z() / um).tolist(),
        frames=frames,
        constants=constants(plan.params),
        bounds_um=trajectory_bounds(plan),
        power_ref=power_ref,
        notes=notes,
    )


# ------------------------------------------------------------------------------ /frame


def pack_image(arr: NDArray[Any]) -> PackedImage:
    """API.md §0.3: little-endian ``uint16`` codes of ``arr / peak``, base64, row-major."""
    image = np.asarray(arr, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"pack_image needs a 2-D array, got shape {image.shape}")
    peak = float(np.max(image)) if image.size else 0.0
    if peak > 0.0:
        codes = np.rint(np.clip(image / peak, 0.0, 1.0) * 65535.0).astype("<u2")
    else:
        codes = np.zeros(image.shape, dtype="<u2")
    return PackedImage(
        shape=(int(image.shape[0]), int(image.shape[1])),
        peak=peak,
        data=base64.b64encode(np.ascontiguousarray(codes).tobytes()).decode("ascii"),
    )


def unpack_image(packed: PackedImage) -> Float:
    """Inverse of :func:`pack_image` (to the quantisation step ``peak / 65535``)."""
    codes = np.frombuffer(base64.b64decode(packed.data), dtype="<u2").astype(np.float64)
    return np.asarray(codes.reshape(packed.shape) * (packed.peak / 65535.0), dtype=np.float64)


def _resolve_plane(req: FrameRequest, result: SimResult) -> float:
    if isinstance(req.plane, PlaneZ):
        return req.plane.z_um * um
    if req.plane == "fixed":
        return 0.0
    return track_z(result.metrics[0])


def gaussian_peak(spots: list[Spot], bright: Spot, waist0_um: float) -> float:
    """Astigmatic-Gaussian peak intensity of the trap at ``bright`` [W/m² scale of §0.3].

    ``sum 2 P / (pi wx wy)`` over the spots whose lateral centres lie within ``waist0_um`` of
    ``bright`` (the coincident groups of a hand-over, DESIGN §1 item 2) — the on-axis value
    of the Eq. S11 spot in its Gaussian (unwindowed, single-term) limit, with ``wx, wy`` the
    :class:`aodl.SpotMetrics` 1/e² radii (µm in :class:`Spot`, converted to metres here).
    """
    total = 0.0
    for spot in spots:
        if math.hypot(spot.x_um - bright.x_um, spot.y_um - bright.y_um) > waist0_um:
            continue
        area = math.pi * (spot.wx_um * um) * (spot.wy_um * um)
        if area > 0.0:
            total += 2.0 * spot.power / area
    return total


def frame_response(req: FrameRequest) -> FrameResponse:
    """API.md §5.2: the closed-form intensity frame (Eq. S11) at one instant."""
    plan = get_plan(req.plan)
    params = movie_params(req.plan, req.movie_mixing_order)
    t = req.time_us * us
    result, clusters = simulate_frames(plan.wfs, params, np.asarray([t], dtype=np.float64))
    terms = result.terms(0)
    optics = params.optics
    plane = _resolve_plane(req, result)
    g = req.grid
    grid = FrameGrid(g.x0_um * um, g.x1_um * um, g.nx, g.y0_um * um, g.y1_um * um, g.ny)
    try:
        xy = intensity_frame(terms, optics, grid, plane, result.tol)
    except ValueError as exc:
        raise classify(exc, "too_many_terms") from exc
    spots = [spot_model(m, gr, n) for m, gr, n in clusters[0]]

    xz_model: ImageXZ | None = None
    if req.xz is not None:
        x_axis = np.linspace(grid.x0, grid.x1, req.xz.nx)
        z_axis = np.linspace(
            plane - req.xz.z_half_um * um, plane + req.xz.z_half_um * um, req.xz.nz
        )
        row_y = result.spot_row(0) if req.xz.row_y_um is None else req.xz.row_y_um * um
        slice_xz = intensity_slice_xz(terms, optics, x_axis, z_axis, row_y, result.tol)
        xz_model = ImageXZ(
            x_um=(x_axis / um).tolist(),
            z_um=(z_axis / um).tolist(),
            row_y_um=row_y / um,
            image=pack_image(slice_xz),
        )

    # cuts through the brightest spot: the XY row nearest its y, the XY column nearest its x,
    # and an exact axial cut at its (x, y)
    gauss_peak = 0.0
    if spots:
        bright = max(spots, key=lambda s: s.power)
        through = CutPoint(x_um=bright.x_um, y_um=bright.y_um)
        gauss_peak = gaussian_peak(spots, bright, optics.waist0 / um)
    else:
        through = CutPoint(x_um=0.5 * (g.x0_um + g.x1_um), y_um=0.5 * (g.y0_um + g.y1_um))
    row = int(np.argmin(np.abs(grid.y / um - through.y_um)))
    column = int(np.argmin(np.abs(grid.x / um - through.x_um)))
    if req.xz is not None:
        z_cut = np.linspace(plane - req.xz.z_half_um * um, plane + req.xz.z_half_um * um, req.xz.nz)
    else:
        z_cut = np.linspace(
            plane - DEFAULT_CUT_Z_HALF_UM * um, plane + DEFAULT_CUT_Z_HALF_UM * um, DEFAULT_CUT_NZ
        )
    i_z = intensity_slice_xz(
        terms, optics, np.asarray([through.x_um * um]), z_cut, through.y_um * um, result.tol
    )[:, 0]
    cuts = Cuts(
        x_um=(grid.x / um).tolist(),
        i_x=xy[row].tolist(),
        i_y=xy[:, column].tolist(),
        z_um=(z_cut / um).tolist(),
        i_z=i_z.tolist(),
        through=through,
    )
    return FrameResponse(
        plan_id=plan_id(req.plan),
        time_us=req.time_us,
        plane_z_um=plane / um,
        xy=ImageXY(x_um=(grid.x / um).tolist(), y_um=(grid.y / um).tolist(), image=pack_image(xy)),
        xz=xz_model,
        cuts=cuts,
        spots=spots,
        gauss_peak=gauss_peak,
        exact_peak=float(np.max(xy)) if xy.size else 0.0,
    )


__all__ = [
    "BOUNDS_RAYLEIGH",
    "BOUNDS_WAISTS",
    "SPOT_FIELDS",
    "CachedSim",
    "cluster_spots",
    "frame_model",
    "frame_response",
    "frame_spots",
    "frame_times",
    "gaussian_peak",
    "get_cached_sim",
    "get_sim",
    "pack_image",
    "simulate_frames",
    "simulate_response",
    "spot_model",
    "tracked_plane",
    "trajectory_bounds",
    "unpack_image",
]
