"""``PiecewisePoly`` — the polynomial workhorse of the pipeline.

Trajectory profiles, tone frequency laws and tone phases are all piecewise polynomials
(``docs/ARCHITECTURE.md`` §0.2): a min-jerk position segment maps through synthesis
(Eq. S19) to a polynomial frequency law, whose antiderivative (phase) and derivative
(chirp) are again polynomials.  Everything here is exact and vectorized; there is no
numerical differentiation anywhere in the pipeline.

**Normalized local time.**  Segment ``k`` covers ``[breaks[k], breaks[k+1]]`` and is
evaluated as ``sum_j coeffs[k, j] * tau**j`` with

    tau = (t - breaks[k]) / (breaks[k+1] - breaks[k])   in [0, 1]

so coefficients stay O(1) even for microsecond-scale segments carrying megahertz-scale
frequency laws.  Derivative/antiderivative carry the corresponding ``1/T`` / ``T``
Jacobians.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Maximum polynomial degree accepted on construction.  Fixes the width of the
#: coefficient block in the parametric NPZ format (padded serialization).
MAX_DEGREE = 9

#: Relative tolerance used when matching breakpoints/domains (scaled by the domain span).
_REL_TOL = 1e-12


def _as_float_array(x: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    arr = np.array(x, dtype=np.float64, copy=True)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    arr.flags.writeable = False
    return arr


def _rebase_row(c: NDArray[np.float64], tau0: float, delta: float) -> NDArray[np.float64]:
    """Re-express ``sum_j c[j] tau**j`` in a sub-interval's own normalized time.

    With ``tau = tau0 + delta * sigma`` the new coefficients are
    ``c'[m] = delta**m * sum_{j>=m} C(j, m) tau0**(j-m) c[j]``.
    """
    n = c.size
    out = np.zeros(n, dtype=np.float64)
    for m in range(n):
        acc = 0.0
        power = 1.0  # tau0 ** (j - m)
        for j in range(m, n):
            acc += math.comb(j, m) * power * c[j]
            power *= tau0
        out[m] = acc * delta**m
    return out


def _pad_coeffs(coeffs: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    if coeffs.shape[1] == width:
        return coeffs
    out = np.zeros((coeffs.shape[0], width), dtype=np.float64)
    out[:, : coeffs.shape[1]] = coeffs
    return out


@dataclass(frozen=True, eq=False)
class PiecewisePoly:
    """A piecewise polynomial in normalized local time.

    Parameters
    ----------
    breaks:
        ``(K+1,)`` strictly increasing breakpoints [s].
    coeffs:
        ``(K, D+1)`` coefficients; segment ``k`` evaluates as
        ``sum_j coeffs[k, j] * tau**j`` with ``tau`` the normalized local time.

    Instances are immutable (the stored arrays are read-only copies).  Equality is
    identity-based on purpose — compare ``breaks``/``coeffs`` with ``numpy.testing``.
    """

    breaks: NDArray[np.float64]
    coeffs: NDArray[np.float64]

    def __post_init__(self) -> None:
        breaks = _as_float_array(self.breaks, "breaks", 1)
        coeffs = _as_float_array(self.coeffs, "coeffs", 2)
        if breaks.size < 2:
            raise ValueError("breaks needs at least 2 entries (one segment)")
        if np.any(np.diff(breaks) <= 0.0):
            raise ValueError("breaks must be strictly increasing")
        if coeffs.shape[0] != breaks.size - 1:
            raise ValueError(
                f"coeffs must have one row per segment: got {coeffs.shape[0]} rows "
                f"for {breaks.size - 1} segments"
            )
        if coeffs.shape[1] < 1:
            raise ValueError("coeffs must have at least one column (the constant term)")
        if coeffs.shape[1] - 1 > MAX_DEGREE:
            raise ValueError(
                f"degree {coeffs.shape[1] - 1} exceeds MAX_DEGREE={MAX_DEGREE} "
                "(serialization padding width)"
            )
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "coeffs", coeffs)

    # ------------------------------------------------------------------ basics

    @property
    def degree(self) -> int:
        """Polynomial degree ``D`` (uniform across segments; low-order rows are padded)."""
        return int(self.coeffs.shape[1] - 1)

    @property
    def n_segments(self) -> int:
        """Number of segments ``K``."""
        return int(self.coeffs.shape[0])

    @property
    def domain(self) -> tuple[float, float]:
        """``(t_start, t_end)`` [s]."""
        return float(self.breaks[0]), float(self.breaks[-1])

    @property
    def _span(self) -> float:
        return float(self.breaks[-1] - self.breaks[0])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        t0, t1 = self.domain
        return f"PiecewisePoly(K={self.n_segments}, D={self.degree}, domain=({t0:g}, {t1:g}))"

    # ------------------------------------------------------------- evaluation

    def __call__(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Evaluate at ``t`` (vectorized).

        **Clamp-hold outside the domain**: ``t < t_start`` returns the value at
        ``t_start`` and ``t > t_end`` returns the value at ``t_end``.  Waveforms are
        therefore defined for all time — a tone simply holds its terminal frequency
        before/after its programmed segments, which is what the device does when the
        aperture is still filled with the last commanded drive.

        Returns a float for scalar input, otherwise an array shaped like ``t``.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        tc = np.clip(t_arr, self.breaks[0], self.breaks[-1])
        idx = np.searchsorted(self.breaks, tc, side="right") - 1
        idx = np.clip(idx, 0, self.n_segments - 1)
        widths = np.diff(self.breaks)[idx]
        tau = (tc - self.breaks[idx]) / widths
        c = self.coeffs[idx]  # (..., D+1)
        out = np.array(c[..., -1], dtype=np.float64, copy=True)
        for j in range(self.degree - 1, -1, -1):
            out = out * tau + c[..., j]
        if t_arr.ndim == 0:
            return float(out)
        return out

    # -------------------------------------------------------------- calculus

    def derivative(self) -> PiecewisePoly:
        """Exact derivative ``dp/dt`` (carries the ``1/T`` Jacobian of normalized time)."""
        widths = np.diff(self.breaks)
        if self.degree == 0:
            return PiecewisePoly(self.breaks, np.zeros((self.n_segments, 1)))
        j = np.arange(1, self.degree + 1, dtype=np.float64)
        new = self.coeffs[:, 1:] * j[None, :] / widths[:, None]
        return PiecewisePoly(self.breaks, new)

    def antiderivative(self, c0: float = 0.0) -> PiecewisePoly:
        """Exact antiderivative with ``P(t_start) = c0``.

        Carries the ``T`` Jacobian of normalized time and accumulates the per-segment
        integrals, so the result is continuous across every breakpoint even when the
        integrand is discontinuous there (this is what makes tone phases
        phase-continuous by construction).
        """
        widths = np.diff(self.breaks)
        j = np.arange(1, self.degree + 2, dtype=np.float64)
        new = np.zeros((self.n_segments, self.degree + 2), dtype=np.float64)
        new[:, 1:] = self.coeffs * widths[:, None] / j[None, :]
        seg_integral = new[:, 1:].sum(axis=1)
        starts = np.empty(self.n_segments, dtype=np.float64)
        starts[0] = c0
        if self.n_segments > 1:
            starts[1:] = c0 + np.cumsum(seg_integral)[:-1]
        new[:, 0] = starts
        return PiecewisePoly(self.breaks, new)

    # ---------------------------------------------------------- construction

    @classmethod
    def constant(cls, value: float, t0: float, t1: float) -> PiecewisePoly:
        """The constant ``value`` on ``[t0, t1]``."""
        return cls(np.array([t0, t1], dtype=np.float64), np.array([[float(value)]]))

    @classmethod
    def from_segment_coeffs(cls, breaks: ArrayLike, coeffs: ArrayLike) -> PiecewisePoly:
        """Build from explicit breakpoints and normalized-time segment coefficients."""
        return cls(np.asarray(breaks), np.asarray(coeffs))

    @staticmethod
    def concat(polys: Sequence[PiecewisePoly]) -> PiecewisePoly:
        """Join polynomials with contiguous domains into one."""
        polys = list(polys)
        if not polys:
            raise ValueError("concat needs at least one polynomial")
        span = polys[-1].domain[1] - polys[0].domain[0]
        tol = _REL_TOL * max(abs(span), abs(polys[0].domain[0]), 1.0)
        for left, right in zip(polys[:-1], polys[1:], strict=True):
            if abs(left.domain[1] - right.domain[0]) > tol:
                raise ValueError(
                    f"concat requires contiguous domains: {left.domain} then {right.domain}"
                )
        width = max(p.coeffs.shape[1] for p in polys)
        breaks = np.concatenate([polys[0].breaks] + [p.breaks[1:] for p in polys[1:]])
        coeffs = np.concatenate([_pad_coeffs(p.coeffs, width) for p in polys], axis=0)
        return PiecewisePoly(breaks, coeffs)

    # ------------------------------------------------------------ transforms

    def shift(self, dt: float) -> PiecewisePoly:
        """Translate in time: ``q(t) = p(t - dt)``."""
        return PiecewisePoly(self.breaks + float(dt), self.coeffs)

    def scale(self, s: float) -> PiecewisePoly:
        """Scale values: ``q(t) = s * p(t)``."""
        return PiecewisePoly(self.breaks, self.coeffs * float(s))

    def offset(self, c: float) -> PiecewisePoly:
        """Offset values: ``q(t) = p(t) + c``."""
        new = self.coeffs.copy()
        new[:, 0] += float(c)
        return PiecewisePoly(self.breaks, new)

    def _refined(self, breaks: NDArray[np.float64]) -> PiecewisePoly:
        """Re-express ``self`` on a finer breakpoint set spanning the same domain."""
        new_breaks = np.asarray(breaks, dtype=np.float64)
        widths = np.diff(self.breaks)
        coeffs = np.zeros((new_breaks.size - 1, self.coeffs.shape[1]), dtype=np.float64)
        for i in range(new_breaks.size - 1):
            s, e = new_breaks[i], new_breaks[i + 1]
            k = int(np.searchsorted(self.breaks, 0.5 * (s + e), side="right") - 1)
            k = min(max(k, 0), self.n_segments - 1)
            tau0 = (s - self.breaks[k]) / widths[k]
            delta = (e - s) / widths[k]
            coeffs[i] = _rebase_row(self.coeffs[k], tau0, delta)
        return PiecewisePoly(new_breaks, coeffs)

    def __add__(self, other: PiecewisePoly | float) -> PiecewisePoly:
        """Sum two polynomials on the *same* overall domain (breaks refined to the union).

        Adding a plain number is shorthand for :meth:`offset`.
        """
        if isinstance(other, (int, float, np.floating, np.integer)):
            return self.offset(float(other))
        if not isinstance(other, PiecewisePoly):
            return NotImplemented
        tol = _REL_TOL * max(self._span, other._span, 1.0)
        same_start = abs(self.breaks[0] - other.breaks[0]) <= tol
        same_end = abs(self.breaks[-1] - other.breaks[-1]) <= tol
        if not (same_start and same_end):
            raise ValueError(
                f"addition requires identical domains, got {self.domain} and {other.domain}"
            )
        merged = np.concatenate([self.breaks, other.breaks])
        merged.sort()
        keep = [merged[0]]
        for value in merged[1:]:
            if value - keep[-1] > tol:
                keep.append(value)
        union = np.asarray(keep, dtype=np.float64)
        union[0] = self.breaks[0]
        union[-1] = self.breaks[-1]
        left = self._refined(union)
        right = other._refined(union)
        width = max(left.coeffs.shape[1], right.coeffs.shape[1])
        total = _pad_coeffs(left.coeffs, width) + _pad_coeffs(right.coeffs, width)
        return PiecewisePoly(union, total)

    __radd__ = __add__

    def __neg__(self) -> PiecewisePoly:
        return self.scale(-1.0)

    def __sub__(self, other: PiecewisePoly | float) -> PiecewisePoly:
        if isinstance(other, (int, float, np.floating, np.integer)):
            return self.offset(-float(other))
        if not isinstance(other, PiecewisePoly):
            return NotImplemented
        return self.__add__(other.scale(-1.0))


__all__ = ["MAX_DEGREE", "PiecewisePoly"]
