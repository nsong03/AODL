r"""Simulation driver: a :class:`~aodl.waveform.tones.WaveformSet` and a list of frame times
in, a :class:`SimResult` out (``docs/ARCHITECTURE.md`` §1, layer L5).

Per frame the engine does exactly two things — expand the drive into pupil terms
(:func:`aodl.device.aodl.build_terms`, Eq. S7) and reduce them to closed-form spot metrics
(:func:`aodl.field.measure.measure`, Table I).  Both are cheap, so the metric table for a whole
movie is built eagerly.  *Fields* are not: an intensity frame costs a grid, so
:class:`SimResult` keeps the waveform set and hands out frames lazily
(:meth:`SimResult.frame`, :meth:`SimResult.slice_xz`) by rebuilding that frame's terms on
demand.  Nothing is sampled and no FFT is used anywhere (``CLAUDE.md``).

**Retarded time is the thing to remember.**  A frame at observation time ``t`` is driven by
the waveform at ``t_c = t - tau/2`` (``docs/conventions.md`` §7): the acoustic sample
illuminating the beam centre left the transducer half an aperture transit earlier.  Every
prediction the caller compares against must use ``t_c``, and :func:`simulate` refuses to run
frames whose ``t_c`` falls past the end of the programmed drive — see :func:`simulate`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .device.aodl import TermArray, build_terms
from .field.focal import (
    GROUP_TOL,
    FrameGrid,
    group_terms,
    intensity_frame,
    intensity_slice_xz,
)
from .field.measure import SpotMetrics, measure, track_z
from .params import AODLParams, OpticsParams
from .units import us
from .waveform.tones import TIME_TOL, WaveformSet

Float = NDArray[np.float64]

#: Columns of :meth:`SimResult.spot_table`, in order.  ``frame``/``group`` are integer
#: indices, ``time`` is the frame's observation time [s] and the rest mirror
#: :class:`~aodl.field.measure.SpotMetrics` (SI units).
SPOT_TABLE_KEYS: tuple[str, ...] = (
    "frame",
    "group",
    "time",
    "x",
    "y",
    "z_lab",
    "delta_f",
    "sigma_astig",
    "wx",
    "wy",
    "power",
    "power_coherent",
    "df_opt",
)


def _subset(terms: TermArray, idx: NDArray[np.intp]) -> TermArray:
    """The sub-array of ``terms`` selected by ``idx`` (same fill edges — a frame property)."""
    return TermArray(
        c=terms.c[idx],
        theta1=terms.theta1[:, idx],
        theta2=terms.theta2[:, idx],
        alpha=terms.alpha[:, :, idx],
        df_opt=terms.df_opt[idx],
        edge=terms.edge,
    )


def _coverage_error(
    name: str, tone: int, span: tuple[float, float], t: float, t_c: float, needed: float
) -> str:
    """Message for a frame time whose retarded drive time runs past the programmed drive."""
    t0, t1 = span
    return (
        f"frame time t = {t / us:.4g} us needs drive time t_c = t - tau/2 = {t_c / us:.4g} us, "
        f"but channel {name!r} tone {tone} is only programmed on "
        f"[{t0 / us:.4g}, {t1 / us:.4g}] us.  Past the end of its frequency law the tone "
        f"clamp-holds: the frequency freezes at its terminal value and the phase stops "
        f"advancing altogether, so the simulation would silently render a dead, incoherent "
        f"drive.  Extend the waveform first, e.g. "
        f"wfs = wfs.with_hold_until({needed / us:.6g} * us)  (any later time works too)."
    )


def _validate_coverage(wfs: WaveformSet, times: Float, names: Sequence[str]) -> None:
    """Check that every tone's frequency law covers the retarded times the frames need.

    The upper bound is the load-bearing one: ``ToneTrack`` clamp-holds outside its domain, so
    a frame past the end would silently freeze the phase instead of failing
    (``docs/workorders/WO-02-waveform.md`` §2).  The lower bound is *not* symmetric — during
    the fill transient ``t_c = t - tau/2`` is legitimately earlier than the drive start (the
    beam centre is not yet illuminated, and :func:`aodl.device.aod.fill_edge` windows that
    part of the aperture away), so only frames before the drive itself are rejected.
    """
    t_min, t_max = float(np.min(times)), float(np.max(times))
    for name in names:
        aod = wfs.params.channels[name]
        half_transit = 0.5 * aod.transit_time
        for i, tone in enumerate(wfs.channels[name].tones):
            t0, t1 = tone.t_span
            if t_max - half_transit > t1 + TIME_TOL:
                raise ValueError(
                    _coverage_error(name, i, (t0, t1), t_max, t_max - half_transit, t_max)
                )
            if t_min < t0 - TIME_TOL:
                raise ValueError(
                    f"frame time t = {t_min / us:.4g} us is before channel {name!r} tone {i} "
                    f"starts ({t0 / us:.4g} us): the aperture holds no drive at all there.  "
                    f"The drive is defined to start at t = 0 (docs/conventions.md §7), so "
                    f"simulate from the start of the waveform onwards."
                )


@dataclass
class SimResult:
    """Metrics for every frame, plus lazy access to the fields behind them.

    Attributes
    ----------
    times:
        Observation times of the frames [s], as passed to :func:`simulate`.
    metrics:
        ``metrics[i][g]`` — the :class:`~aodl.field.measure.SpotMetrics` of frequency group
        ``g`` in frame ``i``.  Groups are ordered by optical frequency and match the order of
        :meth:`group_frames`, so ``metrics[i][g].z_lab`` is the colour of ``group_frames[i][g]``.
    wfs, params, channels, tol:
        The drive, hardware, participating channels and grouping tolerance the run used —
        retained so frames can be rebuilt on demand (WO-05's ``_wfs`` / ``_params``).

    Frames are *not* stored: :meth:`frame`, :meth:`group_frames` and :meth:`slice_xz` rebuild
    the frame's pupil terms (one small cache slot keeps a movie's repeated calls on the same
    frame cheap) and evaluate the closed forms of :mod:`aodl.field.focal` on the grid asked for.
    """

    times: Float
    metrics: list[list[SpotMetrics]]
    wfs: WaveformSet
    params: AODLParams
    channels: tuple[str, ...]
    tol: float = GROUP_TOL
    _cache: tuple[int, TermArray] | None = field(default=None, repr=False, compare=False)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.times.size)

    def __len__(self) -> int:
        return self.n_frames

    @property
    def optics(self) -> OpticsParams:
        """Shorthand for ``self.params.optics``."""
        return self.params.optics

    # -- lazy term expansion

    def terms(self, i: int) -> TermArray:
        """Pupil terms of frame ``i`` (rebuilt on demand, one-slot cached)."""
        index = range(self.n_frames)[i]  # normalizes negatives, raises IndexError otherwise
        if self._cache is not None and self._cache[0] == index:
            return self._cache[1]
        terms = build_terms(self.wfs, float(self.times[index]), self.channels)
        self._cache = (index, terms)
        return terms

    def plane(self, i: int, z_lab: float | None = None) -> float:
        """Resolve a frame's evaluation plane: ``z_lab`` if given, else the tracked plane."""
        if z_lab is not None:
            return float(z_lab)
        return track_z(self.metrics[range(self.n_frames)[i]])

    # -- fields

    def frame(self, i: int, grid: FrameGrid, z_lab: float | None = None) -> Float:
        """Intensity frame ``(grid.ny, grid.nx)`` of frame ``i``.

        ``z_lab = None`` evaluates at the *tracked* plane — the power-weighted mean best-focus
        lab Z of that frame (:func:`aodl.field.measure.track_z`), which is what the movie's
        default ``mode="tracked"`` shows; pass ``0.0`` for the static lab focal plane.
        """
        return intensity_frame(self.terms(i), self.optics, grid, self.plane(i, z_lab), self.tol)

    def group_frames(self, i: int, grid: FrameGrid, z_lab: float | None = None) -> list[Float]:
        """Per-frequency-group intensity frames of frame ``i``, aligned with ``metrics[i]``.

        Terms interfere only inside a group, so the groups are exactly the layers a renderer
        may tint separately (by ``metrics[i][g].z_lab``) and add up; their sum reproduces
        :meth:`frame`.
        """
        terms = self.terms(i)
        plane = self.plane(i, z_lab)
        return [
            intensity_frame(_subset(terms, idx), self.optics, grid, plane, self.tol)
            for idx in group_terms(terms, self.tol)
        ]

    def spot_row(self, i: int) -> float:
        """Lab Y of the tweezer row an XZ slice of frame ``i`` should cut [m].

        The power-weighted *mean* Y is the array's **centre**, which for an even number of
        rows falls in the gap *between* two of them — 6.3 waists from either for the 10x10
        user story.  A slice there sees no trap at all, only their far-field tails, and
        because a defocused spot spreads in y faster than it dims, the tails *off* the focal
        plane outshine the ones in it: the panel then draws the light nowhere near the array's
        own Zbar.  So the mean is snapped onto the nearest Y that actually carries a tweezer,
        ignoring groups below half the frame's peak power so that a faint IM3 ghost cannot
        claim the row.  A single tweezer, and any array with an odd, symmetric row count, is
        unaffected: the mean already sits on a row.
        """
        metrics = self.metrics[range(self.n_frames)[i]]
        if not metrics:
            return 0.0
        centre = _power_weighted(metrics, "y")
        cut = 0.5 * max(m.power for m in metrics)
        bright = [m for m in metrics if m.power >= cut] or list(metrics)
        return min(bright, key=lambda m: abs(m.y - centre)).y

    def slice_xz(
        self,
        i: int,
        x_axis: ArrayLike,
        z_axis_lab: ArrayLike,
        y0: float | None = None,
    ) -> Float:
        """Intensity on the ``(X, Z_lab)`` plane at ``Y = y0``, shape ``(nz, nx)``.

        ``y0 = None`` uses :meth:`spot_row`, i.e. the row the tweezers actually sit on.
        """
        if y0 is None:
            y0 = self.spot_row(i)
        return intensity_slice_xz(
            self.terms(i), self.optics, x_axis, z_axis_lab, float(y0), self.tol
        )

    # -- tables

    def tracked_z(self) -> Float:
        """Tracked plane of every frame [m] — ``track_z`` of each frame's metrics."""
        return np.array([track_z(m) for m in self.metrics], dtype=np.float64)

    def spot_table(self) -> dict[str, Float]:
        """Tidy per-frame, per-group metrics: ``{column: array}`` with :data:`SPOT_TABLE_KEYS`.

        One row per (frame, frequency group) — long format, so a run with a single tweezer
        gives one row per frame and ``table["y"]`` plots straight against ``table["time"]``.

        Both readings of a group's light are columns: the incoherent ``power`` (the default
        weight everywhere in this package) and the exact Gram ``power_coherent``, which is what
        the group's own rendered frame integrates to and the only one that can see a degenerate
        pair interfere (:class:`~aodl.field.measure.SpotMetrics`).  They agree except where
        degenerate terms actually overlap — the Fig. S6 shadow-tweezer situation.
        """
        rows = [
            (
                float(i),
                float(g),
                float(self.times[i]),
                m.x,
                m.y,
                m.z_lab,
                m.delta_f,
                m.sigma_astig,
                m.wx,
                m.wy,
                m.power,
                m.power_coherent,
                m.df_opt,
            )
            for i, frame in enumerate(self.metrics)
            for g, m in enumerate(frame)
        ]
        data = np.array(rows, dtype=np.float64).reshape(len(rows), len(SPOT_TABLE_KEYS))
        return {key: data[:, j] for j, key in enumerate(SPOT_TABLE_KEYS)}


def _power_weighted(metrics: Sequence[SpotMetrics], attribute: str) -> float:
    """Power-weighted mean of one :class:`SpotMetrics` attribute (plain mean if unpowered)."""
    if not metrics:
        return 0.0
    values = np.array([getattr(m, attribute) for m in metrics], dtype=np.float64)
    power = np.array([m.power for m in metrics], dtype=np.float64)
    total = float(power.sum())
    if not total > 0.0:
        return float(np.mean(values))
    return float(power @ values / total)


def simulate(
    wfs: WaveformSet,
    times: ArrayLike,
    channels: Sequence[str] | None = None,
    tol: float = GROUP_TOL,
) -> SimResult:
    """Simulate ``wfs`` at the given frame ``times`` [s].

    For each time the drive is expanded into pupil terms at the beam-centre retarded time
    ``t_c = t - tau/2`` (Eq. S7) and reduced to one :class:`~aodl.field.measure.SpotMetrics`
    per optical-frequency group (Table I).  Intensity frames are left to
    :class:`SimResult`'s lazy evaluators.

    Parameters
    ----------
    wfs:
        The drive.  Its ``params`` supply the hardware for the whole run.
    times:
        Frame observation times [s]; scalar or array-like, kept in the order given.
    channels:
        Which channels to include; ``None`` (default) uses every channel present in ``wfs``.
    tol:
        Optical-frequency tolerance for interference grouping [Hz]; shared by the metrics and
        every frame this result renders, so the two always agree.

    Raises
    ------
    ValueError
        If a frame's retarded time ``t_c`` runs past the end of a tone's frequency law.  Tones
        clamp-hold outside their domain — the phase would stop advancing rather than the run
        failing — so extend the drive with
        :meth:`~aodl.waveform.tones.WaveformSet.with_hold_until` first.  Frames before the
        drive starts are rejected for the same "say it, don't fake it" reason.
    """
    frame_times = np.atleast_1d(np.asarray(times, dtype=np.float64)).ravel()
    if frame_times.size == 0:
        raise ValueError("simulate() needs at least one frame time")
    if not np.all(np.isfinite(frame_times)):
        raise ValueError("frame times must all be finite")

    names = tuple(wfs.channels) if channels is None else tuple(channels)
    missing = [name for name in names if name not in wfs.channels]
    if missing:
        raise KeyError(f"channels {missing} are not present in the waveform set")
    _validate_coverage(wfs, frame_times, names)

    optics = wfs.params.optics
    metrics = [measure(build_terms(wfs, float(t), names), optics, tol) for t in frame_times]
    return SimResult(
        times=frame_times,
        metrics=metrics,
        wfs=wfs,
        params=wfs.params,
        channels=names,
        tol=tol,
    )


__all__ = ["SPOT_TABLE_KEYS", "FrameGrid", "SimResult", "simulate"]
