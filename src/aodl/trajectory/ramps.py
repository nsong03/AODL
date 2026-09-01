"""Rest-to-rest time profiles (Eqs. S14-S17).

Every ramp returns a :class:`~aodl.poly.PiecewisePoly` on ``[t0, t0 + T]`` that runs from
``y_i`` at ``t0`` to ``y_f`` at ``t0 + T``.  The ordinate is deliberately unnamed: the very
same profiles drive **positions** [m] in :mod:`aodl.trajectory` and, after the synthesis
mapping of Eq. S19, **frequency detunings** [Hz] in :mod:`aodl.waveform` — polynomial
closure (``docs/ARCHITECTURE.md`` §0.2) means a min-jerk position segment becomes a
min-jerk frequency segment becomes an exact polynomial phase.

With ``d = y_f - y_i`` and ``tau`` the *normalized local time* of each segment
(``tau = (t - t_seg0) / T_seg``, see :mod:`aodl.poly`):

============================ ======== ========= ================================
profile                      segments continuity peak |acceleration|
============================ ======== ========= ================================
:func:`min_jerk`             1        C^2        ``5.7735 |d| / T^2``
:func:`constant_jerk`        1        C^1        ``6 |d| / T^2``
:func:`constant_accel`       2        C^1        ``4 |d| / T^2``
:func:`switching_constant_jerk` 3     C^2        ``8 |d| / T^2``
:func:`linear`               1        C^0        ``0`` (velocity step at the ends)
============================ ======== ========= ================================

``constant_accel`` is the time-optimal profile at a given acceleration limit; ``min_jerk``
is the default because it starts and ends with zero velocity *and* zero acceleration,
which for a frequency law means the tweezer starts and stops with no residual lensing
(``Zbar`` and ``Delta F`` follow ``fdot``, Table I).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..poly import PiecewisePoly


def _duration(T: float) -> float:
    T = float(T)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"ramp duration T must be finite and positive, got {T!r}")
    return T


def _one_segment(t0: float, T: float, coeffs: NDArray[np.float64]) -> PiecewisePoly:
    t0 = float(t0)
    T = _duration(T)
    return PiecewisePoly(np.array([t0, t0 + T], dtype=np.float64), coeffs[None, :])


def min_jerk(t0: float, T: float, y_i: float, y_f: float) -> PiecewisePoly:
    """Minimum-jerk quintic, ``y_i + d (10 tau^3 - 15 tau^4 + 6 tau^5)``.  Eq. S14.

    Starts and ends at rest with zero acceleration (``ydot = yddot = 0`` at both ends),
    which is what makes it the default profile for tweezer motion.
    """
    d = float(y_f) - float(y_i)
    coeffs = np.array([float(y_i), 0.0, 0.0, 10.0 * d, -15.0 * d, 6.0 * d])
    return _one_segment(t0, T, coeffs)


def constant_jerk(t0: float, T: float, y_i: float, y_f: float) -> PiecewisePoly:
    """Constant-jerk cubic, ``y_i + d (3 tau^2 - 2 tau^3)``.  Eq. S15.

    Zero velocity at both ends; the acceleration steps from ``+6 d / T^2`` to
    ``-6 d / T^2`` (jerk ``-12 d / T^3`` is constant across the whole move).
    """
    d = float(y_f) - float(y_i)
    coeffs = np.array([float(y_i), 0.0, 3.0 * d, -2.0 * d])
    return _one_segment(t0, T, coeffs)


def constant_accel(t0: float, T: float, y_i: float, y_f: float) -> PiecewisePoly:
    """Bang-bang constant-acceleration profile, two parabolic halves.  Eq. S16.

    In the *global* normalized time ``tau = (t - t0) / T`` the profile is

    * ``tau in [0, 1/2]``:  ``y_i + d * 2 tau^2``
    * ``tau in [1/2, 1]``:  ``y_i + d * (1 - 2 (1 - tau)^2)``

    each re-expressed here in its own segment-local normalized time ``sigma`` (so
    ``sigma = 2 tau`` on the first half and ``sigma = 2 tau - 1`` on the second):
    ``[y_i, 0, d/2]`` and ``[y_i + d/2, d, -d/2]``.  The acceleration is
    ``+4 d / T^2`` then ``-4 d / T^2``; velocity is continuous and vanishes at both ends.
    """
    t0 = float(t0)
    T = _duration(T)
    y_i = float(y_i)
    d = float(y_f) - y_i
    breaks = np.array([t0, t0 + 0.5 * T, t0 + T], dtype=np.float64)
    coeffs = np.array(
        [
            [y_i, 0.0, 0.5 * d],
            [y_i + 0.5 * d, d, -0.5 * d],
        ]
    )
    return PiecewisePoly(breaks, coeffs)


def switching_constant_jerk(t0: float, T: float, y_i: float, y_f: float) -> PiecewisePoly:
    """Switching-constant-jerk profile: three cubics, jerk ``+J, -J, +J``.  Eq. S17.

    Jerk switches at ``T/4`` and ``3T/4`` with ``J = 32 d / T^3``, giving the
    acceleration triangle ``0 -> +8d/T^2 -> -8d/T^2 -> 0``.  Integrating from rest:

    * ``[t0, t0 + T/4]``     ``[y_i,          0,   0,    d/12]``
    * ``[t0+T/4, t0+3T/4]``  ``[y_i + d/12,   d/2, d,   -2d/3]``
    * ``[t0+3T/4, t0+T]``    ``[y_i + 11d/12, d/4, -d/4, d/12]``

    (segment-local normalized time).  The result is C^2: ``y``, ``ydot`` and ``yddot``
    are continuous at both switches, and ``ydot = yddot = 0`` at both ends — like
    min-jerk, but with bounded (piecewise-constant) jerk instead of a smooth quintic.
    """
    t0 = float(t0)
    T = _duration(T)
    y_i = float(y_i)
    d = float(y_f) - y_i
    breaks = np.array([t0, t0 + 0.25 * T, t0 + 0.75 * T, t0 + T], dtype=np.float64)
    coeffs = np.array(
        [
            [y_i, 0.0, 0.0, d / 12.0],
            [y_i + d / 12.0, 0.5 * d, d, -2.0 * d / 3.0],
            [y_i + 11.0 * d / 12.0, 0.25 * d, -0.25 * d, d / 12.0],
        ]
    )
    return PiecewisePoly(breaks, coeffs)


def linear(t0: float, T: float, y_i: float, y_f: float) -> PiecewisePoly:
    """Linear ramp ``y_i + d tau`` — constant velocity ``d / T``, discontinuous at the ends.

    For a frequency law this is the constant-chirp segment: ``fdot = d / T`` is constant,
    so the tweezer sits at a fixed axial offset (Table I) for the whole segment.
    """
    d = float(y_f) - float(y_i)
    return _one_segment(t0, T, np.array([float(y_i), d]))


def hold(t0: float, T: float, y: float) -> PiecewisePoly:
    """Constant segment ``y`` on ``[t0, t0 + T]`` (zero velocity, zero chirp)."""
    return PiecewisePoly.constant(float(y), float(t0), float(t0) + _duration(T))


#: Ramp family name -> constructor ``(t0, T, y_i, y_f) -> PiecewisePoly``.  ``hold`` is
#: excluded: it takes a single value rather than a start/end pair.
RAMPS = {
    "min_jerk": min_jerk,
    "constant_jerk": constant_jerk,
    "constant_accel": constant_accel,
    "switching_constant_jerk": switching_constant_jerk,
    "linear": linear,
}


__all__ = [
    "RAMPS",
    "constant_accel",
    "constant_jerk",
    "hold",
    "linear",
    "min_jerk",
    "switching_constant_jerk",
]
