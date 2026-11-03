r"""Frame driver of the FFT pupil engine: lines -> groups -> zoom transforms -> metrics.

Per frame time ``t`` (observation time; the drive seen at the beam centre is the one at
``t_c = t - tau/2``, :func:`aodl.device.conventions.beam_center_time`):

1. every driven channel's transmission is decomposed into optical-frequency lines on the
   pupil grid, through its own misaligned aperture coordinate (:func:`.pupil.channel_lines`);
2. the Cartesian product of the lines is grouped by optical frequency sum and the brightest
   ``max_groups`` groups are kept (:func:`.pupil.pupil_groups`);
3. each group's pupil is zoom-transformed onto the requested focal patch at the requested lab
   plane and the groups add in intensity (:func:`.propagate.group_intensity`) — the
   beat-averaged image, no sub-time averaging;
4. the XZ slice and the ``w_x(Z)`` / ``w_y(Z)`` focus scans are built from the brightest
   ``XZ_MAX_GROUPS`` groups, and the image metrics of :mod:`.metrics` are measured.

The tracked plane and grid centre are **inputs** (the web layer takes them from the ``aodl``
engine's spot metrics), so this package never imports ``aodl.engine``; the aligned reference
is simply the same computation with :meth:`.misalignment.Misalignment.aligned` and is cached
by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from aodl.device.conventions import beam_center_time, geometry
from aodl.params import AODLParams
from aodl.units import us
from aodl.waveform.tones import TIME_TOL, WaveformSet

from .efficiency import EfficiencyModel, channel_relative_amplitude
from .metrics import (
    WINDOW_WAISTS,
    AdvMetrics,
    best_focus,
    image_power,
    refined_peak,
    width_scan,
    windowed_moments,
)
from .misalignment import Misalignment
from .propagate import focal_intensity, xz_slice, yz_slice
from .pupil import (
    Group,
    Line,
    PupilModel,
    channel_coordinate,
    channel_lines,
    pupil_grid,
    pupil_groups,
)

Float = NDArray[np.float64]
Complex = NDArray[np.complex128]

#: Groups the XZ slice and the focus scans include (the brightest ones).
XZ_MAX_GROUPS = 6
#: Smallest ``nz`` for which the focus scans are evaluated (``z_best_*`` otherwise ``None``).
FOCUS_MIN_NZ = 16


@dataclass(frozen=True)
class GridSpec:
    """Focal patch: ``n x n`` samples over ``centre +- half`` per axis [m]."""

    half_x: float
    half_y: float
    n: int
    cx: float = 0.0
    cy: float = 0.0

    def __post_init__(self) -> None:
        if int(self.n) < 2:
            raise ValueError(f"grid needs at least 2 samples per axis, got {self.n!r}")
        if not (self.half_x > 0.0 and self.half_y > 0.0):
            raise ValueError("grid half-widths must be positive")

    @property
    def x(self) -> Float:
        """Lab X sample coordinates [m]."""
        return np.linspace(self.cx - self.half_x, self.cx + self.half_x, int(self.n))

    @property
    def y(self) -> Float:
        """Lab Y sample coordinates [m]."""
        return np.linspace(self.cy - self.half_y, self.cy + self.half_y, int(self.n))

    @property
    def dx(self) -> float:
        """Pixel pitch along X [m]."""
        return 2.0 * self.half_x / (int(self.n) - 1)

    @property
    def dy(self) -> float:
        """Pixel pitch along Y [m]."""
        return 2.0 * self.half_y / (int(self.n) - 1)


@dataclass(frozen=True)
class XzSpec:
    """Axial scan: ``nz`` planes over ``plane +- z_half`` [m]."""

    nz: int
    z_half: float

    def __post_init__(self) -> None:
        if int(self.nz) < 2:
            raise ValueError(f"xz.nz must be at least 2, got {self.nz!r}")
        if not self.z_half > 0.0:
            raise ValueError("xz.z_half must be positive")


@dataclass(frozen=True)
class ModelSpec:
    """Engine settings (API.md §6.5 ``model``)."""

    pupil_n: int = 256
    span_w_in: float = 2.5
    order_model: str = "j1"
    comb_dft_points: int = 16
    max_groups: int = 128
    efficiency: EfficiencyModel | None = None

    @property
    def pupil_model(self) -> PupilModel:
        """The line model this spec asks for."""
        return PupilModel(order_model=self.order_model, comb_dft_points=self.comb_dft_points)


@dataclass
class FrameResult:
    """One rendered frame of the engine (SI units; images in the engine's dropped-prefactor
    scale, which is the ``aodl.field.focal.intensity_frame`` scale)."""

    time: float
    plane_z: float
    n_groups: int
    groups_dropped: int
    x: Float
    y: Float
    xy: Float
    metrics: AdvMetrics
    xz: Float | None = None
    xz_x: Float | None = None
    xz_z: Float | None = None
    row_y: float = 0.0
    reference_metrics: AdvMetrics | None = None
    notes: list[str] = field(default_factory=list)


def check_coverage(wfs: WaveformSet, t: float) -> None:
    """Refuse a frame whose beam-centre drive time runs past the programmed drive.

    Same rule as ``aodl.engine.simulate``: a tone clamp-holds outside its domain and the
    phase stops advancing, so the engine would silently render a dead drive.
    """
    t0, t1 = wfs.t_span
    if t < t0 - TIME_TOL:
        raise ValueError(
            f"frame time t = {t / us:.4g} us is before the drive starts ({t0 / us:.4g} us): the "
            f"aperture holds no drive at all there."
        )
    for name in wfs.channels:
        aod = wfs.params.channels[name]
        t_c = beam_center_time(t, aod)
        if t_c > t1 + TIME_TOL:
            raise ValueError(
                f"frame time t = {t / us:.4g} us needs drive time t_c = t - tau/2 = "
                f"{t_c / us:.4g} us, but the drive is only programmed on "
                f"[{t0 / us:.4g}, {t1 / us:.4g}] us.  Past its end a tone clamp-holds and its "
                f"phase stops advancing, so the simulation would render a dead drive."
            )


def frame_lines(
    wfs: WaveformSet,
    t: float,
    params: AODLParams,
    mis: Misalignment,
    model: ModelSpec,
    x: Float,
    y: Float,
) -> tuple[dict[str, list[Line]], list[str]]:
    """Lines of every driven channel at frame time ``t`` on the pupil grid ``(x, y)``."""
    notes: list[str] = []
    lines: dict[str, list[Line]] = {}
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    for name, cw in wfs.channels.items():
        if cw.n_tones == 0:
            continue
        geom = geometry(name)
        aod = params.channels[name]
        m = mis.channel(name)
        if m.is_rotated:
            xx, yy = np.meshgrid(x, y)
            u = channel_coordinate(xx, yy, geom, m)
            step = dx * dy
        elif geom.axis == 0:
            u = channel_coordinate(x, 0.0, geom, m)
            step = dx
        else:
            u = channel_coordinate(0.0, y, geom, m)
            step = dy
        rel_amp = None
        if model.efficiency is not None:
            rel_amp = channel_relative_amplitude(
                wfs, model.efficiency, name, beam_center_time(t, aod)
            )
        channel, channel_notes = channel_lines(
            cw, t, params, geom, aod, m, model.pupil_model, rel_amp, u, step
        )
        lines[name] = channel
        notes.extend(f"{name}: {note}" for note in channel_notes)
    return lines, notes


def _nyquist_note(lines: dict[str, list[Line]], params: AODLParams, dx: float) -> str | None:
    """Warn when a kept line pair would deflect beyond the pupil grid's alias limit."""
    limit = params.optics.wavelength * params.optics.focal_length / (2.0 * dx)
    worst = 0.0
    for axis in (0, 1):
        names = [n for n in lines if geometry(n).axis == axis]
        if not names:
            continue
        offsets = np.zeros(1)
        for n in names:
            s = geometry(n).sound_sign
            tags = np.asarray([s * line.f_tag for line in lines[n]])
            offsets = (offsets[:, None] + tags[None, :]).ravel()
        worst = max(worst, float(np.max(np.abs(offsets))) * params.deflection_scale)
    if worst > limit:
        return (
            f"a line pair deflects to {worst * 1e6:.1f} um, beyond the pupil grid's alias limit "
            f"of {limit * 1e6:.1f} um (lambda F / 2 dx): that spot folds back onto the image; "
            f"raise pupil_n or lower span_w_in"
        )
    return None


def _tilt(mis: Misalignment) -> tuple[float, float]:
    return (
        float(sum(m.tilt_x for m in mis.values())),
        float(sum(m.tilt_y for m in mis.values())),
    )


def simulate_frame(
    wfs: WaveformSet,
    params: AODLParams,
    t: float,
    plane_z: float,
    centre: tuple[float, float],
    mis: Misalignment,
    grid: GridSpec,
    xz: XzSpec | None,
    model: ModelSpec,
) -> FrameResult:
    """Render one frame at lab plane ``plane_z`` on ``grid`` (re-centred on ``centre``)."""
    check_coverage(wfs, t)
    g = GridSpec(grid.half_x, grid.half_y, grid.n, float(centre[0]), float(centre[1]))
    optics = params.optics
    k, focal = optics.k, optics.focal_length
    px, py = pupil_grid(optics.w_in, model.pupil_n, model.span_w_in)
    lines, notes = frame_lines(wfs, t, params, mis, model, px, py)
    alias = _nyquist_note(lines, params, float(px[1] - px[0]))
    if alias:
        notes.append(alias)
    groups, dropped = pupil_groups(
        lines, params, model.max_groups, px, py, _tilt(mis), model.pupil_model
    )
    if dropped:
        notes.append(
            f"{dropped} faint frequency group(s) beyond max_groups = {model.max_groups} were "
            f"not rendered"
        )
    X, Y = g.x, g.y
    xy = np.zeros((g.n, g.n), dtype=np.float64)
    kept: list[Complex] = []
    for i, group in enumerate(groups):
        pupil = group.pupil()
        xy += focal_intensity(pupil, px, py, X, Y, k, focal, plane_z)
        if xz is not None and i < XZ_MAX_GROUPS:
            kept.append(pupil)
    if xz is not None and len(groups) > XZ_MAX_GROUPS:
        notes.append(
            f"the XZ slice and focus scans cover the brightest {XZ_MAX_GROUPS} of "
            f"{len(groups)} groups"
        )

    half_window = WINDOW_WAISTS * optics.waist0
    xc, yc, wx, wy = windowed_moments(xy, X, Y, half_window)
    peak = refined_peak(xy) if xy.size else 0.0
    power = image_power(xy, g.dx, g.dy)
    result = FrameResult(
        time=float(t),
        plane_z=float(plane_z),
        n_groups=len(groups),
        groups_dropped=dropped,
        x=X,
        y=Y,
        xy=xy,
        metrics=AdvMetrics(x=xc, y=yc, wx=wx, wy=wy, peak=peak, power=power),
        notes=notes,
    )
    if xz is None:
        return result
    z = plane_z + np.linspace(-xz.z_half, xz.z_half, int(xz.nz))
    xz_img = np.zeros((int(xz.nz), g.n), dtype=np.float64)
    for pupil in kept:
        xz_img += xz_slice(pupil, px, py, X, z, yc, k, focal)
    result.xz, result.xz_x, result.xz_z, result.row_y = xz_img, X, z, yc
    if int(xz.nz) >= FOCUS_MIN_NZ and kept and peak > 0.0:
        yz_img = np.zeros((int(xz.nz), g.n), dtype=np.float64)
        for pupil in kept:
            yz_img += yz_slice(pupil, px, py, Y, z, xc, k, focal)
        z_best_x = best_focus(z, width_scan(xz_img, X))
        z_best_y = best_focus(z, width_scan(yz_img, Y))
        result.metrics = AdvMetrics(
            x=xc,
            y=yc,
            wx=wx,
            wy=wy,
            peak=peak,
            power=power,
            z_best_x=z_best_x,
            z_best_y=z_best_y,
            astig=z_best_x - z_best_y,
        )
    return result


def simulate_frames(
    wfs: WaveformSet,
    params: AODLParams,
    times: Sequence[float],
    planes: Sequence[float],
    centres: Sequence[tuple[float, float]],
    mis: Misalignment,
    grid: GridSpec,
    xz: XzSpec | None,
    model: ModelSpec,
    reference: bool = True,
    reference_frames: Sequence[FrameResult] | None = None,
) -> list[FrameResult]:
    """Render the frames at ``times`` with the misalignment ``mis``.

    ``planes`` (lab Z of each frame's evaluation plane) and ``centres`` (grid centre per frame)
    are supplied by the caller.  With ``reference=True`` each frame also carries the metrics
    of the aligned device at the same time/plane/grid — taken from ``reference_frames`` when
    the caller has them cached, computed here otherwise (and the frames *are* the reference
    when ``mis`` is aligned).
    """
    if not (len(times) == len(planes) == len(centres)):
        raise ValueError("times, planes and centres must have the same length")
    frames = [
        simulate_frame(wfs, params, float(t), float(z), c, mis, grid, xz, model)
        for t, z, c in zip(times, planes, centres, strict=True)
    ]
    if not reference:
        return frames
    if mis.is_aligned:
        for frame in frames:
            frame.reference_metrics = frame.metrics
        return frames
    if reference_frames is None:
        reference_frames = simulate_frames(
            wfs, params, times, planes, centres, Misalignment.aligned(), grid, xz, model, False
        )
    for frame, ref in zip(frames, reference_frames, strict=True):
        frame.reference_metrics = ref.metrics
    return frames


def group_pupils(
    wfs: WaveformSet,
    params: AODLParams,
    t: float,
    mis: Misalignment,
    model: ModelSpec,
) -> tuple[list[Group], Float, Float, int, list[str]]:
    """The frame's groups (brightest first) and the pupil grid — the engine's building blocks,
    exposed for tests and diagnostics."""
    check_coverage(wfs, t)
    px, py = pupil_grid(params.optics.w_in, model.pupil_n, model.span_w_in)
    lines, notes = frame_lines(wfs, t, params, mis, model, px, py)
    groups, dropped = pupil_groups(
        lines, params, model.max_groups, px, py, _tilt(mis), model.pupil_model
    )
    return groups, px, py, dropped, notes


__all__ = [
    "FOCUS_MIN_NZ",
    "XZ_MAX_GROUPS",
    "FrameResult",
    "GridSpec",
    "ModelSpec",
    "XzSpec",
    "check_coverage",
    "frame_lines",
    "group_pupils",
    "simulate_frame",
    "simulate_frames",
]
