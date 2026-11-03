r"""What to build and where to move it: array geometry + a list of moves (``PLAN.md`` §1.4).

This is layer L2, the *user intent* end of the pipeline: an :class:`ArraySpec` says how many
tweezers to make and how far apart, a tuple of moves says how the array's centre travels, and
:meth:`TrajectorySpec.compile` turns the moves into the three position laws

.. math::

    X(t),\; Y(t),\; Z(t) \qquad [\text{m}],\ t \in [0, T]

that Eq. S19 synthesis (:func:`aodl.waveform.synthesis.synthesize`) maps onto the four RF
channels.  Nothing here knows about frequencies except :meth:`ArraySpec.from_pitch`, which is
the one place a *pitch* in metres is traded for a *tone spacing* in hertz through Table I's
``deflection_scale = lambda F / v``.

**The four moves.**  A :class:`Lift` changes only ``Z``, a :class:`Translate` only ``X``/``Y``
(both axes at once, sharing one profile and duration), a :class:`Hold` freezes all three, and
a :class:`Path` threads a clamped cubic spline through arbitrary 3D waypoints.  There is
deliberately **no** ``Lower`` class: lowering is ``Lift(dz=-...)``, because the profile, the
continuity rules and the bandwidth cost are identical either way.  Every move is
rest-to-rest, so the array is stationary at every seam and any sequence is continuous by
construction (asserted in :meth:`TrajectorySpec.compile`).

The trajectory always starts at the origin ``(0, 0, 0)`` — the static focal spot of the
undriven AODL — and every displacement is *relative* to where the previous move ended.

Time profiles come from :mod:`aodl.trajectory.ramps` (Eqs. S14-S17) by name, defaulting to
``"min_jerk"``: it starts and ends with zero velocity *and* zero acceleration, which through
Eq. S19 means the drive starts and stops with zero chirp, i.e. no residual lensing (Table I).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from ..params import AODLParams
from ..poly import PiecewisePoly
from .ramps import RAMPS, hold

#: Relative tolerance of the endpoint-continuity check run by :meth:`TrajectorySpec.compile`.
#: Segments are built from a running position, so the only difference across a seam is
#: floating-point round-off (~1e-16 relative); anything larger is a construction bug.
CONTINUITY_RTOL = 1e-9

#: Distance [m] below which a :class:`Path` waypoint counts as *the same point* as another:
#: ``closed=True`` does not append a return to the start when the last point already is one,
#: and a chord shorter than this is a zero-length chord.  A picometre is far below anything
#: a tweezer resolves (the focal waist is a micron) and far above float64 round-off on
#: micron-scale coordinates.
PATH_POINT_TOL = 1e-12


def _positive_duration(duration: float, what: str) -> float:
    value = float(duration)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{what} duration must be finite and positive, got {duration!r}")
    return value


def _known_profile(profile: str) -> str:
    name = str(profile)
    if name not in RAMPS:
        raise ValueError(f"unknown ramp profile {profile!r}; choose one of {sorted(RAMPS)}")
    return name


def _finite(value: float, what: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return out


# --------------------------------------------------------------------------- array geometry


@dataclass(frozen=True)
class ArraySpec:
    """The tone ladders that make the array: ``mx x my`` tweezers at spacings ``delta_f``.

    Attributes
    ----------
    mx, my:
        Number of tones on ``Bx`` / ``By``, i.e. array columns and rows (``>= 1``).
    delta_f_x, delta_f_y:
        Tone spacings [Hz] of those ladders (Eq. S18).  Through Table I the tweezer pitch is
        ``deflection_scale * delta_f = lambda F Delta f / v`` — 10.3 µm per MHz at the default
        hardware; use :meth:`from_pitch` to say it in metres instead.  A single-tone axis
        (``m = 1``) has no spacing, so its ``delta_f`` is ignored and defaults to ``0``; a
        multi-tone axis with ``delta_f = 0`` is rejected (it would stack every tone of the
        ladder on one exactly degenerate trap).

    Note
    ----
    Give the two axes *different* spacings whenever the traps must stay distinguishable:
    with ``delta_f_x == delta_f_y`` every anti-diagonal of the array shares one optical
    frequency ``f_x + f_y``, so those traps become mutually coherent and
    :func:`aodl.field.focal.group_terms` reports one group per anti-diagonal instead of one
    per trap (``docs/conventions.md`` §4).
    """

    mx: int = 1
    my: int = 1
    delta_f_x: float = 0.0
    delta_f_y: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mx", "my"):
            count = getattr(self, name)
            if int(count) != count or int(count) < 1:
                raise ValueError(f"ArraySpec.{name} must be an integer >= 1, got {count!r}")
            object.__setattr__(self, name, int(count))
        for name, count in (("delta_f_x", self.mx), ("delta_f_y", self.my)):
            spacing = _finite(getattr(self, name), f"ArraySpec.{name}")
            if count > 1 and spacing == 0.0:
                axis = name[-1]
                raise ValueError(
                    f"ArraySpec.{name} must be non-zero for a ladder of m{axis} = {count} "
                    f"tones: a zero spacing would stack them all on one degenerate trap.  "
                    f"Use ArraySpec.from_pitch(...) to set it from a tweezer pitch in metres."
                )
            object.__setattr__(self, name, spacing)

    @classmethod
    def from_pitch(
        cls,
        mx: int,
        my: int,
        pitch_x: float,
        pitch_y: float,
        params: AODLParams,
    ) -> ArraySpec:
        """Build from tweezer *pitches* [m] instead of tone spacings (Table I).

        ``delta_f = pitch / deflection_scale`` with ``deflection_scale = lambda F / v``, the
        inverse of the mapping :meth:`pitch` applies — the two round-trip exactly.
        """
        scale = params.deflection_scale
        return cls(
            mx=mx,
            my=my,
            delta_f_x=_finite(pitch_x, "pitch_x") / scale,
            delta_f_y=_finite(pitch_y, "pitch_y") / scale,
        )

    def pitch(self, params: AODLParams) -> tuple[float, float]:
        """Tweezer pitches ``(pitch_x, pitch_y)`` [m] of this array.  Table I.

        ``pitch = deflection_scale * delta_f = lambda F Delta f / v``.
        """
        scale = params.deflection_scale
        return scale * self.delta_f_x, scale * self.delta_f_y

    @property
    def n_traps(self) -> int:
        """Number of tweezers ``mx * my``."""
        return self.mx * self.my

    def detunings(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The two ladders ``f_0^{(n)} = (n - (M-1)/2) delta_f`` [Hz], centred on zero.

        Eq. S18/S19's ``f_x0^{(n)}`` and ``f_y0^{(m)}`` — the static part of the ``Bx``/``By``
        drives, and (times ``deflection_scale``) the trap offsets from the array centre.
        """
        return (
            (np.arange(self.mx, dtype=np.float64) - 0.5 * (self.mx - 1)) * self.delta_f_x,
            (np.arange(self.my, dtype=np.float64) - 0.5 * (self.my - 1)) * self.delta_f_y,
        )


# ---------------------------------------------------------------------------------- moves


@dataclass(frozen=True)
class Lift:
    """Move the array ``dz`` [m] out of the focal plane in ``duration`` [s].

    Lab ``Z`` (``docs/conventions.md`` §6), so ``dz > 0`` lifts and **``dz < 0`` lowers** —
    there is no separate ``Lower`` move.  ``X`` and ``Y`` are held.

    Costs bandwidth: Eq. S19 buys the offset with a co-chirp on all four channels, so a
    *sustained* ``Z`` walks every channel's frequency at ``Z / (2 lens_scale)``
    (:func:`aodl.waveform.synthesis.max_z_integral`).
    """

    dz: float
    duration: float
    profile: str = "min_jerk"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dz", _finite(self.dz, "Lift.dz"))
        object.__setattr__(self, "duration", _positive_duration(self.duration, "Lift"))
        object.__setattr__(self, "profile", _known_profile(self.profile))


@dataclass(frozen=True)
class Translate:
    """Move the array ``(dx, dy)`` [m] in the focal plane in ``duration`` [s].

    Both axes ramp *simultaneously* with the same profile and duration (a diagonal move);
    ``Z`` is held.  Counter-chirping within each pair means the lateral motion carries no
    focal shift at all — the paper's key result, and why this move leaves ``Z`` alone.
    """

    dx: float
    dy: float
    duration: float
    profile: str = "min_jerk"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dx", _finite(self.dx, "Translate.dx"))
        object.__setattr__(self, "dy", _finite(self.dy, "Translate.dy"))
        object.__setattr__(self, "duration", _positive_duration(self.duration, "Translate"))
        object.__setattr__(self, "profile", _known_profile(self.profile))


@dataclass(frozen=True)
class Hold:
    """Keep the array where it is for ``duration`` [s] (all three axes frozen)."""

    duration: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", _positive_duration(self.duration, "Hold"))


@dataclass(frozen=True)
class Path:
    """Smooth 3D waypoint path: a clamped cubic spline through relative waypoints.

    The array leaves the position where the path starts, visits every waypoint in order and
    comes to rest at the last one (or back where it started, with ``closed=True``).  Knot
    times are allocated in proportion to the chord lengths between consecutive waypoints, so
    the array moves at a roughly uniform speed along the polyline, and the position law
    through the knots is the **clamped cubic spline** of :class:`scipy.interpolate.CubicSpline`
    (``bc_type="clamped"``): zero velocity at both ends — rest-to-rest like every other move,
    so it chains with a continuous frequency law — ``C^2`` at the interior knots and ``C^1``
    at its two ends, where the acceleration steps exactly as the ``constant_jerk`` profile's
    does (Eq. S15).  A single waypoint *is* that profile: the clamped cubic between two rest
    points is ``3 tau^2 - 2 tau^3``.

    Attributes
    ----------
    points:
        Waypoints ``(dx, dy, dz)`` [m], **relative to the position where the path starts**
        (the implicit first waypoint is ``(0, 0, 0)``).  At least one; consecutive waypoints
        must differ (a zero-length chord has no direction to interpolate along).
    duration:
        Total time [s] of the move, strictly positive.
    closed:
        ``True`` appends a return to the start, ``(0, 0, 0)``, unless the last point already
        is one (within :data:`PATH_POINT_TOL`).

    Everything downstream — Eq. S19 synthesis, the band checks, ``shepard="auto"`` — works
    unchanged, because it only ever asks for the compiled piecewise polynomials
    (:meth:`TrajectorySpec.compile`); each spline piece is one degree-3 segment per axis in
    normalized local time.
    """

    points: tuple[tuple[float, float, float], ...]
    duration: float
    closed: bool = False

    def __post_init__(self) -> None:
        raw = tuple(self.points)
        if not raw:
            raise ValueError("Path needs at least one waypoint")
        normalized: list[tuple[float, float, float]] = []
        for i, point in enumerate(raw):
            values = tuple(point)
            if len(values) != 3:
                raise ValueError(f"Path.points[{i}] must be a (dx, dy, dz) triple, got {point!r}")
            normalized.append(
                (
                    _finite(values[0], f"Path.points[{i}][0]"),
                    _finite(values[1], f"Path.points[{i}][1]"),
                    _finite(values[2], f"Path.points[{i}][2]"),
                )
            )
        closed = bool(self.closed)
        if closed and any(abs(v) > PATH_POINT_TOL for v in normalized[-1]):
            normalized.append((0.0, 0.0, 0.0))
        object.__setattr__(self, "points", tuple(normalized))
        object.__setattr__(self, "duration", _positive_duration(self.duration, "Path"))
        object.__setattr__(self, "closed", closed)

    @property
    def end(self) -> tuple[float, float, float]:
        """Where the path ends, relative to where it started [m]."""
        return self.points[-1]

    def compile(
        self, t0: float, start: tuple[float, float, float]
    ) -> tuple[PiecewisePoly, PiecewisePoly, PiecewisePoly]:
        """Position laws ``(X, Y, Z)`` [m] on ``[t0, t0 + duration]`` from absolute ``start``.

        Waypoints are ``start + points[i]``, knot times ``t0 + duration * cumlen_i /
        cumlen_n`` from the chord lengths, and the spline is clamped (zero velocity at both
        ends).  ``CubicSpline.c[j, k, axis]`` is the coefficient of ``(t - knot_k)^(3 - j)``,
        so in the normalized local time of segment ``k`` (width ``T_k``) the coefficient of
        ``tau^m`` is ``c[3 - m, k, axis] * T_k^m``.
        """
        relative = np.array([(0.0, 0.0, 0.0), *self.points], dtype=np.float64)
        chords = np.linalg.norm(np.diff(relative, axis=0), axis=1)
        short = np.nonzero(chords <= PATH_POINT_TOL)[0]
        if short.size:
            i = int(short[0])
            raise ValueError(
                f"Path waypoint {i} duplicates waypoint {i - 1} (chord length "
                f"{chords[i]:.3g} m): waypoints are relative to the path start, consecutive "
                f"ones must differ, and a zero-length chord has no direction to interpolate "
                f"along.  Drop the duplicate, or use Hold to pause."
            )
        cumulative = np.concatenate([[0.0], np.cumsum(chords)])
        knots = float(t0) + self.duration * cumulative / cumulative[-1]
        absolute = relative + np.asarray(start, dtype=np.float64)[None, :]
        spline = CubicSpline(knots, absolute, axis=0, bc_type="clamped")
        widths = np.diff(knots)
        powers = widths[:, None] ** np.arange(4, dtype=np.float64)[None, :]  # (K, 4)
        out = []
        for axis in range(3):
            # c[3 - m, k] is the coefficient of (t - knot_k)^m; normalize to tau^m.
            coeffs = spline.c[::-1, :, axis].T * powers  # (K, 4), column m <-> tau^m
            out.append(PiecewisePoly(knots, coeffs))
        return out[0], out[1], out[2]


#: Anything accepted in :attr:`TrajectorySpec.moves`.
Move = Lift | Translate | Hold | Path


# ------------------------------------------------------------------------- the trajectory


def _check_continuity(poly: PiecewisePoly, axis: str) -> None:
    """Raise unless every segment ends where the next one starts.

    Cheap and exact: in normalized local time a segment's value at ``tau = 1`` is the sum of
    its coefficients and at ``tau = 0`` its constant term, so no evaluation is needed.
    """
    if poly.n_segments < 2:
        return
    left = poly.coeffs[:-1].sum(axis=1)
    right = poly.coeffs[1:, 0]
    scale = float(np.max(np.abs(poly.coeffs)))
    gap = np.abs(left - right)
    tol = CONTINUITY_RTOL * max(scale, 1e-30)
    bad = np.nonzero(gap > tol)[0]
    if bad.size:
        k = int(bad[0])
        raise ValueError(
            f"{axis}(t) is discontinuous at t = {poly.breaks[k + 1]!r} s: segment {k} ends at "
            f"{left[k]!r} m but segment {k + 1} starts at {right[k]!r} m (gap {gap[k]:.3e} m).  "
            f"Moves are rest-to-rest and chained from a running position, so this is a bug."
        )


@dataclass(frozen=True)
class TrajectorySpec:
    """An array plus the moves it makes: the whole input to Eq. S19 synthesis.

    Attributes
    ----------
    array:
        The :class:`ArraySpec` — how many tweezers, at what spacing.
    moves:
        Moves in order, applied from the origin.  At least one is required; a *static* array
        is ``(Hold(duration),)``.

    Example
    -------
    ::

        spec = TrajectorySpec(
            array=ArraySpec(2, 2, 1.0 * MHz, 1.3 * MHz),
            moves=(
                Lift(5 * um, 60 * us),                 # up
                Translate(15 * um, 10 * um, 80 * us),  # across
                Lift(-5 * um, 60 * us),                # and back down
            ),
        )
        wfs = synthesize(spec, default_1030())
    """

    array: ArraySpec
    moves: tuple[Move, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.array, ArraySpec):
            raise TypeError(f"TrajectorySpec.array must be an ArraySpec, got {type(self.array)!r}")
        moves = tuple(self.moves)
        for i, move in enumerate(moves):
            if not isinstance(move, Lift | Translate | Hold | Path):
                raise TypeError(
                    f"TrajectorySpec.moves[{i}] must be a Lift, Translate or Hold (or a Path), "
                    f"got {move!r}"
                )
        object.__setattr__(self, "moves", moves)

    @property
    def duration(self) -> float:
        """Total programmed time ``T`` [s] — the sum of the move durations."""
        return float(sum(move.duration for move in self.moves))

    def compile(self) -> tuple[PiecewisePoly, PiecewisePoly, PiecewisePoly]:
        """The array-centre position laws ``(X(t), Y(t), Z(t))`` [m], ``t`` from 0.

        One segment set per move, concatenated: **all three axes cover the full span**
        ``[0, T]`` (an axis a move does not touch is held at its current value), so the three
        polynomials share their domain and can be combined term by term in Eq. S19 synthesis.

        Continuity of the value at every seam is checked here (:func:`_check_continuity`);
        velocity continuity comes for free from the rest-to-rest profiles of
        :mod:`aodl.trajectory.ramps` — every profile except ``"linear"`` also starts and ends
        at rest, and ``"min_jerk"`` at zero acceleration too — and from the clamped ends of a
        :class:`Path` spline.
        """
        if not self.moves:
            raise ValueError(
                "a TrajectorySpec needs at least one move; use moves=(Hold(duration),) for a "
                "static array"
            )
        segments: tuple[list[PiecewisePoly], ...] = ([], [], [])
        position = [0.0, 0.0, 0.0]
        t0 = 0.0
        for move in self.moves:
            span = move.duration
            if isinstance(move, Path):
                start = (position[0], position[1], position[2])
                for axis, poly in enumerate(move.compile(t0, start)):
                    segments[axis].append(poly)
                position = [start[axis] + move.end[axis] for axis in range(3)]
                t0 += span
                continue
            if isinstance(move, Lift):
                target = [position[0], position[1], position[2] + move.dz]
                ramped = (False, False, True)
                profile = move.profile
            elif isinstance(move, Translate):
                target = [position[0] + move.dx, position[1] + move.dy, position[2]]
                ramped = (True, True, False)
                profile = move.profile
            else:  # Hold
                target = list(position)
                ramped = (False, False, False)
                profile = "min_jerk"  # unused: nothing ramps
            for axis in range(3):
                if ramped[axis]:
                    segments[axis].append(RAMPS[profile](t0, span, position[axis], target[axis]))
                else:
                    segments[axis].append(hold(t0, span, position[axis]))
            position = target
            t0 += span

        out = tuple(PiecewisePoly.concat(segs) for segs in segments)
        for label, poly in zip("XYZ", out, strict=True):
            _check_continuity(poly, label)
        return out[0], out[1], out[2]


__all__ = [
    "CONTINUITY_RTOL",
    "PATH_POINT_TOL",
    "ArraySpec",
    "Hold",
    "Lift",
    "Move",
    "Path",
    "TrajectorySpec",
    "Translate",
]
