r"""What the drive was *asked* for: the checker's independent expectation (Table I).

:mod:`aodl.check.record`, :mod:`~aodl.check.demod`, :mod:`~aodl.check.pupil`,
:mod:`~aodl.check.transform` and :mod:`~aodl.check.metrics` measure what the rendered samples
actually do.  This module supplies the other half of a verdict: where the tweezers were
*supposed* to be, computed from the requested trajectory and paper Table I alone.

    ``X(t) = X_request(t_eval)  +  deflection_scale * f_x0^(n)``     (Eq. S18 ladder)
    ``Y(t) = Y_request(t_eval)  +  deflection_scale * f_y0^(m)``
    ``Z(t) = Z_request(t_eval)``,      ``Delta F = 0``               (astigmatism-free)

with ``deflection_scale = lambda F / v`` and ``t_eval`` the drive time whose request the atom
plane shows at frame time ``t`` (:meth:`Expectation.eval_time`).  Nothing here evaluates a
waveform, a pupil or a field: the only inputs are
:class:`~aodl.trajectory.spec.TrajectorySpec`, :class:`~aodl.params.AODLParams` and
:mod:`aodl.device.conventions`, so an expectation cannot inherit a synthesis or simulation bug.

**The one thing samples cannot tell you** is whether the drive was written with
``retard_compensate=True``: a compensated and an uncompensated drive differ only by *which*
instant of the trajectory each acoustic sample encodes, and both are perfectly good drives.
It is therefore an explicit flag on :class:`Expectation`, never inferred —
:meth:`aodl.api.MotionPlan.check` reads it from :attr:`aodl.api.MotionPlan.options`.

Comparison against the analytic simulator's ``SimResult`` is available too, through
:func:`sim_delta`, and is deliberately **structural**: :class:`SimResultLike` is a ``Protocol``
describing the attributes the diff reads, and the concrete class is never imported — not even
under ``TYPE_CHECKING``, because the M6 independence rule is enforced by a source scan that
(rightly) does not care why an import is written.  This mirrors
:class:`aodl.field.focal.TermLike`, which consumes the device layer's term arrays the same way.
The checker never *computes* anything through the engine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..params import AODLParams, AODParams
from ..poly import PiecewisePoly
from ..trajectory.spec import ArraySpec, Hold, TrajectorySpec
from ..units import us

Float = NDArray[np.float64]
Index = NDArray[np.intp]

#: One fade whitelist entry: ``(observation time [s], axis, shadow offset [m])``.  The times
#: are **frame** times, not drive times — :meth:`aodl.api.MotionPlan.check` adds the ``tau/2``
#: retardation to :attr:`aodl.api.PlanReport.fade_events` when it builds the list.
Shadow = tuple[float, str, float]

__all__ = [
    "ExpectedTraps",
    "Expectation",
    "Shadow",
    "SimResultLike",
    "SpotMetricsLike",
    "sim_delta",
]


# ------------------------------------------------------------------- the expected scene


@dataclass(frozen=True)
class ExpectedTraps:
    """Where every requested tweezer should be at one frame time, and how fast it moves.

    Attributes
    ----------
    time:
        Frame (observation) time [s].
    t_eval:
        Drive time whose requested position this frame shows
        (:meth:`Expectation.eval_time`).
    ix, iy:
        Column and row index of each trap, ``0 .. mx-1`` / ``0 .. my-1``.  Rows vary fastest,
        so the tables built from these are grouped by column.
    x, y, z:
        Expected lab position [m] of each trap.
    vx, vy:
        Lateral velocity [m/s] of the array centre at ``t_eval`` — what caps the beat-averaging
        window (a window long enough to smear the array by a fraction of a waist would report
        a wide, dim spot for a perfectly good drive).
    columns, rows:
        The distinct expected column ``x`` and row ``y`` coordinates [m].
    """

    time: float
    t_eval: float
    ix: Index
    iy: Index
    x: Float
    y: Float
    z: float
    vx: float
    vy: float
    columns: Float
    rows: Float

    @property
    def n_traps(self) -> int:
        """Number of requested tweezers, ``mx * my``."""
        return int(self.x.size)

    @property
    def speed(self) -> float:
        """``max(|vx|, |vy|)`` [m/s] — the motion the averaging window must not smear."""
        return max(abs(self.vx), abs(self.vy))


def _hermite_poly(times: Float, values: Float) -> PiecewisePoly:
    """Piecewise-cubic Hermite interpolant through ``(times, values)``, exact at the nodes.

    Catmull-Rom tangents (central differences, one-sided at the ends) written straight into
    :class:`~aodl.poly.PiecewisePoly`'s normalized-time coefficients, so the result supports
    :meth:`~aodl.poly.PiecewisePoly.derivative` exactly like a compiled trajectory.  This is
    what lets a *measured* trajectory table stand in for a
    :class:`~aodl.trajectory.spec.TrajectorySpec` (:meth:`Expectation.from_table`).
    """
    widths = np.diff(times)
    slopes = np.diff(values) / widths
    tangents = np.empty_like(values)
    tangents[0] = slopes[0]
    tangents[-1] = slopes[-1]
    if values.size > 2:
        tangents[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])
    m0 = tangents[:-1] * widths
    m1 = tangents[1:] * widths
    y0, y1 = values[:-1], values[1:]
    coeffs = np.stack(
        [y0, m0, -3.0 * y0 - 2.0 * m0 + 3.0 * y1 - m1, 2.0 * y0 + m0 - 2.0 * y1 + m1], axis=1
    )
    return PiecewisePoly.from_segment_coeffs(times, coeffs)


@dataclass(frozen=True)
class Expectation:
    """The requested scene: a trajectory, an array, and the two facts samples cannot carry.

    Attributes
    ----------
    spec:
        What was asked for.  Its :meth:`~aodl.trajectory.spec.TrajectorySpec.compile` gives the
        array-centre laws ``X(t), Y(t), Z(t)``; its ``array`` gives the ladders.
    params:
        The hardware, for ``deflection_scale``, ``tau`` and the optics.
    retard_compensated:
        ``True`` if the drive was synthesized with ``retard_compensate=True``.  **Not
        discoverable from samples** (module docstring) — pass it explicitly.
    amp:
        The common relative tone amplitude the drive was built with.  Report-only: a uniform
        per-channel gain is optically invisible after the global normalization
        :func:`aodl.waveform.export.render_samples` applies, so nothing is gated on it.
    shadows:
        Fade whitelist, ``(observation time, axis, shadow offset [m])`` per hand-over
        (:class:`aodl.api.FadeEvent` with the ``tau/2`` retardation already added).  A
        non-empty list also marks the drive as *fading*, which is what tells the blob audit
        that the Shepard extended grid is expected light rather than a fault.
    fade_pad:
        Half-width [s] of the fade exclusion around each entry of ``shadows``.  ``0.0``
        (default) excludes nothing; a positive value marks frames within ``fade_pad`` of a
        hand-over as fade frames, which drops them from the waist and uniformity gates and
        whitelists the shadow tweezers there.
    laws:
        Optional ``(X, Y, Z)`` position laws overriding ``spec.compile()`` — how
        :meth:`from_table` carries a tabulated trajectory.  ``None`` (default) compiles the
        spec, which is the normal path.
    """

    spec: TrajectorySpec
    params: AODLParams
    retard_compensated: bool = False
    amp: float = 1.0
    shadows: tuple[Shadow, ...] = ()
    fade_pad: float = 0.0
    laws: tuple[PiecewisePoly, PiecewisePoly, PiecewisePoly] | None = None
    _xyz: tuple[PiecewisePoly, PiecewisePoly, PiecewisePoly] = field(
        init=False, repr=False, compare=False
    )
    _vel: tuple[PiecewisePoly, PiecewisePoly] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TrajectorySpec):
            raise TypeError(f"Expectation.spec must be a TrajectorySpec, got {type(self.spec)!r}")
        if not isinstance(self.params, AODLParams):
            raise TypeError(f"Expectation.params must be an AODLParams, got {type(self.params)!r}")
        pad = float(self.fade_pad)
        if not math.isfinite(pad) or pad < 0.0:
            raise ValueError(f"fade_pad must be finite and non-negative, got {self.fade_pad!r}")
        shadows: list[Shadow] = []
        for entry in self.shadows:
            t, axis, offset = entry
            if axis not in ("x", "y"):
                raise ValueError(f"shadow axis must be 'x' or 'y', got {axis!r}")
            shadows.append((float(t), str(axis), float(offset)))
        xyz = self.spec.compile() if self.laws is None else tuple(self.laws)
        if len(xyz) != 3 or not all(isinstance(p, PiecewisePoly) for p in xyz):
            raise TypeError("Expectation.laws must be three PiecewisePoly position laws")
        object.__setattr__(self, "fade_pad", pad)
        object.__setattr__(self, "shadows", tuple(shadows))
        object.__setattr__(self, "_xyz", (xyz[0], xyz[1], xyz[2]))
        object.__setattr__(self, "_vel", (xyz[0].derivative(), xyz[1].derivative()))

    # -- construction from a measured table

    @classmethod
    def from_table(
        cls,
        times: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        array: ArraySpec,
        params: AODLParams,
        **options: Any,
    ) -> Expectation:
        """Build an expectation from a *tabulated* array-centre trajectory — the lab path.

        A rearrangement schedule that came out of a solver, a trajectory measured off a camera,
        or any other ``(t, X, Y, Z)`` table stands in for a
        :class:`~aodl.trajectory.spec.TrajectorySpec` here: the table is turned into a
        piecewise-cubic Hermite interpolant (exact at the nodes, ``C^1``, differentiable for the
        beat-window cap) and carried in :attr:`laws`.  ``spec`` becomes the array plus a single
        :class:`~aodl.trajectory.spec.Hold` of the table's duration, so ``spec.array`` and
        ``spec.duration`` keep their meanings.

        ``times`` must start at ``0`` — the drive starts there by definition
        (``docs/conventions.md`` §7) — and increase.  ``options`` are forwarded to the
        constructor (``retard_compensated``, ``amp``, ``shadows``, ``fade_pad``).
        """
        t = np.asarray(times, dtype=np.float64).ravel()
        columns = [np.asarray(v, dtype=np.float64).ravel() for v in (x, y, z)]
        if t.size < 2:
            raise ValueError(f"from_table needs at least two samples, got {t.size}")
        if any(v.shape != t.shape for v in columns):
            raise ValueError(
                "from_table needs x, y and z to match times: got "
                f"{[v.shape for v in columns]} against {t.shape}"
            )
        if not np.all(np.isfinite(t)) or not all(np.all(np.isfinite(v)) for v in columns):
            raise ValueError("from_table needs finite times and positions")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("from_table needs strictly increasing times")
        if t[0] != 0.0:
            raise ValueError(
                f"from_table needs times[0] == 0: the drive starts at t = 0 by definition "
                f"(docs/conventions.md §7), got {t[0]!r}.  Shift the table instead of "
                "re-basing the expectation, so the retardation stays meaningful."
            )
        laws = (
            _hermite_poly(t, columns[0]),
            _hermite_poly(t, columns[1]),
            _hermite_poly(t, columns[2]),
        )
        spec = TrajectorySpec(array=array, moves=(Hold(float(t[-1])),))
        return cls(spec=spec, params=params, laws=laws, **options)

    # -- derived properties

    @property
    def duration(self) -> float:
        """Programmed trajectory length ``T`` [s]."""
        return float(self._xyz[0].domain[1] - self._xyz[0].domain[0])

    @property
    def transit_time(self) -> float:
        """The stack's aperture transit ``tau`` [s] — the longest of the four channels."""
        return max(aod.transit_time for aod in self.params.channels.values())

    @property
    def fading(self) -> bool:
        """Was this drive built from fading-Shepard ladders?  (Any hand-over in ``shadows``.)"""
        return bool(self.shadows)

    @property
    def fading_axes(self) -> tuple[str, ...]:
        """The axes that hand over, in ``("x", "y")`` order (empty for an Eq. S19 drive)."""
        seen = {axis for _, axis, _ in self.shadows}
        return tuple(axis for axis in ("x", "y") if axis in seen)

    # -- the expectation itself

    def eval_time(self, t: float, aod: AODParams | None = None) -> float:
        """Drive time whose requested position the atom plane shows at frame time ``t`` [s].

        ``clamp(t - tau/2, 0, T)`` normally — the acoustic sample illuminating the beam centre
        left the transducer half an aperture transit earlier (``docs/conventions.md`` §7) — and
        ``clamp(t, 0, T)`` when the drive was written with ``retard_compensate=True``, which
        reads the trajectory ``tau/2`` ahead precisely so that the atom plane matches the
        request at ``t``.

        The clamp is the trajectory's own behaviour at both ends: before ``t = 0`` nothing has
        been asked for yet, and past ``T`` the drive holds its terminal (rest) state.
        ``aod`` selects the channel whose ``tau`` is used; ``None`` uses the stack's longest.
        """
        lead = (
            0.0
            if self.retard_compensated
            else 0.5 * (self.transit_time if aod is None else aod.transit_time)
        )
        return float(min(max(float(t) - lead, 0.0), self.duration))

    def center(self, t: float) -> tuple[float, float, float]:
        """Requested array-centre position ``(X, Y, Z)`` [m] at frame time ``t``."""
        te = self.eval_time(t)
        return tuple(float(p(te)) for p in self._xyz)  # type: ignore[return-value]

    def traps(self, t: float) -> ExpectedTraps:
        """Every requested tweezer's expected position at frame time ``t``.  Table I + Eq. S18.

        The array centre comes from the compiled trajectory at :meth:`eval_time`; the ladders
        add ``deflection_scale * f_0^(n)`` (:meth:`aodl.trajectory.spec.ArraySpec.detunings`)
        on each axis.  ``Z`` is common to every trap and the astigmatic interval is zero by
        construction, which is the claim being checked.
        """
        te = self.eval_time(t)
        array = self.spec.array
        scale = self.params.deflection_scale
        fx, fy = array.detunings()
        columns = float(self._xyz[0](te)) + scale * fx
        rows = float(self._xyz[1](te)) + scale * fy
        ix, iy = (
            idx.ravel()
            for idx in np.meshgrid(
                np.arange(array.mx, dtype=np.intp),
                np.arange(array.my, dtype=np.intp),
                indexing="ij",
            )
        )
        return ExpectedTraps(
            time=float(t),
            t_eval=te,
            ix=ix,
            iy=iy,
            x=columns[ix],
            y=rows[iy],
            z=float(self._xyz[2](te)),
            vx=float(self._vel[0](te)),
            vy=float(self._vel[1](te)),
            columns=columns,
            rows=rows,
        )

    def lattice(self, t: float, extend: int = 1) -> tuple[Float, Float]:
        """The trap lattice at frame time ``t``, widened by ``extend`` pitches on every side.

        Returns ``(xs, ys)``: the expected column and row coordinates [m] plus ``extend`` more
        at each end.  Two kinds of light legitimately land there rather than on a requested
        trap, which is why the blob audit whitelists it:

        * a **fading-Shepard extended column** — the ladder always has a rung on its way in or
          out, so an array is ``M + 1`` wide at every instant for even ``M`` and ``M + 2``
          during a hand-over for odd ``M`` (``docs/guide.md`` §6.7), and the shadow tweezers of
          Eq. S31 sit one ladder spacing away;
        * a **commensurate intermodulation product** — IM3 lines at ``f_j + f_k - f_i`` are
          sums of ladder frequencies, so they land on the same comb (Eqs. S20-S22).

        A single-tone axis (``m = 1``, spacing ``0``) has no lattice to widen and returns its
        one coordinate.
        """
        step = int(extend)
        if step < 0:
            raise ValueError(f"extend must be non-negative, got {extend!r}")
        traps = self.traps(t)
        array = self.spec.array
        scale = self.params.deflection_scale
        out: list[Float] = []
        for count, spacing, base in (
            (array.mx, array.delta_f_x, traps.columns),
            (array.my, array.delta_f_y, traps.rows),
        ):
            pitch = scale * spacing
            if count == 1 or pitch == 0.0 or step == 0:
                out.append(np.asarray(base, dtype=np.float64))
                continue
            extra = pitch * np.arange(1, step + 1, dtype=np.float64)
            out.append(np.concatenate([base[0] - extra[::-1], base, base[-1] + extra]))
        return out[0], out[1]

    def in_fade(self, t: float) -> bool:
        """Is frame time ``t`` inside a hand-over window (``|t - shadow.time| <= fade_pad``)?"""
        return any(abs(float(t) - time) <= self.fade_pad for time, _, _ in self.shadows)

    def shadow_offsets(self, t: float) -> tuple[tuple[str, float], ...]:
        """``(axis, offset)`` of every shadow tweezer pair active at frame time ``t``.

        Empty unless :meth:`in_fade` — outside a hand-over the Eq. S31 pair carries no power.
        """
        return tuple(
            (axis, offset)
            for time, axis, offset in self.shadows
            if abs(float(t) - time) <= self.fade_pad
        )

    def edge_lines(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Column and row indices whose *depth* a fading ladder does not hold constant.

        The ``p_A + p_B = 1`` identity keeps every **interior** node of a fading array exactly
        flat through a hand-over (measured to 1e-15 across the ``docs/guide.md`` flagship), but
        the ladder itself slides: the rung arriving at one end and the one leaving at the other
        trade their light with the extended grid's extra node, so the array's two edge lines on
        a fading axis breathe between full depth and dark.  That is the ``docs/guide.md`` §6.6
        edge-column caveat, and it is why the intensity gates skip these lines on a fading
        drive.  An Eq. S19 drive has no fading axis and returns two empty tuples.
        """
        array = self.spec.array
        axes = self.fading_axes
        cols = (0, array.mx - 1) if "x" in axes and array.mx > 1 else ()
        rows = (0, array.my - 1) if "y" in axes and array.my > 1 else ()
        return tuple(sorted(set(cols))), tuple(sorted(set(rows)))

    def min_spacing(self) -> float:
        """Smallest live tone spacing [Hz] in the drive, or ``inf`` when nothing can beat.

        The slowest beat note a frame average has to cover: two traps one spacing apart differ
        in optical frequency by exactly that spacing (``docs/conventions.md`` §4), so a window
        of ``2 / delta_f_min`` holds two full cycles of the slowest pair.  A fading-Shepard
        drive contributes its ladder spacing too — recovered from the shadow offsets, which are
        ``deflection_scale * delta_f`` by Eq. S31 — because its live rungs beat against each
        other even when the array itself is a single tweezer.
        """
        array = self.spec.array
        spacings = [
            abs(spacing)
            for count, spacing in ((array.mx, array.delta_f_x), (array.my, array.delta_f_y))
            if count > 1 and spacing != 0.0
        ]
        scale = self.params.deflection_scale
        spacings.extend(abs(offset) / scale for _, _, offset in self.shadows if offset != 0.0)
        return min(spacings) if spacings else math.inf

    def describe(self) -> str:
        """One line naming the array, the trajectory and the retardation convention."""
        array = self.spec.array
        mode = "retard-compensated" if self.retard_compensated else "lagging tau/2"
        fades = f", {len(self.shadows)} hand-over(s)" if self.shadows else ""
        return (
            f"{array.mx}x{array.my} array, T = {self.duration / us:.4g} us, {mode}"
            f"{fades}, amp = {self.amp:g}"
        )


# --------------------------------------------------------------- the simulator comparison


class SpotMetricsLike(Protocol):
    """Structural contract for :class:`aodl.field.measure.SpotMetrics` (report-only)."""

    @property
    def x(self) -> float:
        """Lab X [m]."""

    @property
    def y(self) -> float:
        """Lab Y [m]."""

    @property
    def z_lab(self) -> float:
        """Best-focus lab Z [m]."""

    @property
    def wx(self) -> float:
        """1/e^2 intensity radius along x [m]."""

    @property
    def wy(self) -> float:
        """1/e^2 intensity radius along y [m]."""

    @property
    def power(self) -> float:
        """Group power on the simulator's scale."""

    @property
    def df_opt(self) -> float:
        """Optical-frequency tag [Hz]."""


class SimResultLike(Protocol):
    """Structural contract for :class:`aodl.engine.SimResult`.

    The checker shares *no* code with the simulator, so the comparison is made through a
    protocol and the concrete class is imported only under ``TYPE_CHECKING`` — the
    :class:`aodl.field.focal.TermLike` precedent.  Anything carrying frame times and a
    per-frame list of spot metrics works, including a hand-built stub.
    """

    @property
    def times(self) -> NDArray[Any]:
        """``(n_frames,)`` frame observation times [s]."""

    @property
    def metrics(self) -> Sequence[Sequence[SpotMetricsLike]]:
        """``metrics[i][g]`` — one record per optical-frequency group per frame."""


def sim_delta(rows: Mapping[str, Float], sim: SimResultLike, tol_match: float) -> dict[str, float]:
    """Nearest-neighbour differences between the checker's table and a simulator run.

    **Report-only, by construction.**  The two paths model different physics — the checker
    rebuilds the literal ``exp(i C V)`` crystal from samples, the simulator expands it to
    ``mixing_order`` — so a disagreement here is a *model gap* to read, not a verdict.  Nothing
    in :class:`aodl.check.report.CheckReport.passed` depends on it.

    Parameters
    ----------
    rows:
        The checker's long table (:attr:`aodl.check.report.CheckReport.table`); it needs the
        ``time``, ``x``, ``y``, ``z_lab``, ``wx``, ``wy`` and ``power`` columns.
    sim:
        Anything satisfying :class:`SimResultLike`.
    tol_match:
        Largest lateral distance [m] at which a checker row and a simulator group are taken to
        be the same tweezer.  Half a tweezer pitch is the natural value.

    Returns
    -------
    ``{"n_rows", "n_matched", "max_dx", "max_dy", "max_dz", "max_dw", "max_dpower", "rms_dxy"}``
    — positions and radii in metres, ``dpower`` relative to each frame's own total (the two
    paths carry different absolute scales, so only the *pattern* is comparable).
    """
    times = np.asarray(rows["time"], dtype=np.float64)
    sim_times = np.asarray(sim.times, dtype=np.float64)
    out = {
        "n_rows": float(times.size),
        "n_matched": 0.0,
        "max_dx": 0.0,
        "max_dy": 0.0,
        "max_dz": 0.0,
        "max_dw": 0.0,
        "max_dpower": 0.0,
        "rms_dxy": 0.0,
    }
    if times.size == 0 or sim_times.size == 0:
        return out

    check_power = np.asarray(rows["power"], dtype=np.float64)
    residuals: list[float] = []
    for t in np.unique(times):
        frame = int(np.argmin(np.abs(sim_times - t)))
        metrics = list(sim.metrics[frame])
        if not metrics:
            continue
        sim_xy = np.array([[m.x, m.y] for m in metrics], dtype=np.float64)
        sim_total = float(sum(m.power for m in metrics)) or 1.0
        here = np.nonzero(times == t)[0]
        check_total = float(check_power[here].sum()) or 1.0
        for i in here:
            dx = sim_xy[:, 0] - float(rows["x"][i])
            dy = sim_xy[:, 1] - float(rows["y"][i])
            distance = np.hypot(dx, dy)
            j = int(np.argmin(distance))
            if distance[j] > tol_match:
                continue
            m = metrics[j]
            out["n_matched"] += 1.0
            out["max_dx"] = max(out["max_dx"], abs(float(dx[j])))
            out["max_dy"] = max(out["max_dy"], abs(float(dy[j])))
            out["max_dz"] = max(out["max_dz"], abs(m.z_lab - float(rows["z_lab"][i])))
            out["max_dw"] = max(
                out["max_dw"],
                abs(m.wx - float(rows["wx"][i])),
                abs(m.wy - float(rows["wy"][i])),
            )
            out["max_dpower"] = max(
                out["max_dpower"],
                abs(m.power / sim_total - float(check_power[i]) / check_total),
            )
            residuals.append(float(distance[j]) ** 2)
    if residuals:
        out["rms_dxy"] = float(math.sqrt(sum(residuals) / len(residuals)))
    return out
