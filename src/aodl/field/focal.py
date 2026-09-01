r"""Closed-form focal field per pupil term, frequency grouping, patch accumulation.

Eq. S11 (``docs/PLAN.md`` §1.3) gives the field behind the objective at defocus ``Z``::

    U(X, Y, Z) ~ int int U_in(x, y) P(x, y) e^{-i k (x X + y Y)/F} e^{-i k Z (x^2+y^2)/(2F^2)}

Every pupil term produced by the device layer is separable, ``P = P_x(x) P_y(y)``, with the
per-axis factor ``(alpha0 + alpha1 u + alpha2 u^2) exp(i theta2 u^2 + i theta1 u)`` (Eq. S5–S8:
``theta1`` = deflection, ``theta2`` = chirp lens, ``alpha`` = amplitude Taylor terms, i.e.
intensity/tilt/acoustic irising).  Multiplied by the input Gaussian ``exp(-u^2/w_in^2)`` and by
the Eq. S11 kernel, each axis integral is exactly the closed form of :mod:`aodl.field.gaussian`
with the mapping pinned by ``tests/test_focal_geometry.py``::

    a = 1/w_in^2 - i (theta2 - k Z_S11 / (2 F^2))     # beam radius, chirp lens, defocus
    b = i (theta1 - k X / F)                          # deflection, image coordinate

    field_axis = alpha0 I0(a, b) + alpha1 I1(a, b) + alpha2 I2(a, b)
    U_term(X, Y) = c * field_x(X) * field_y(Y)

A term whose aperture is still filling carries a fill edge on that axis; the full-line moments
``I_n`` are then replaced by the edge moments ``E_n`` / ``F_n`` of the same module — or, when
the axis' counter-propagating pair is filling from *both* sides at once, by the two-sided
window moments ``W_n``.

**Dropped prefactors (intensity-safe).**  ``1/(i lambda F)``, the propagation phase
``e^{i k F}`` and the common image curvature ``e^{i k (X^2+Y^2)/(2F)}`` are omitted, exactly as
in :mod:`aodl.field.reference`.  They are term-independent and of unit modulus, so every
intensity and every term-to-term interference is unaffected; only an overall (constant) scale
is.  :func:`aodl.field.measure.measure` reports powers on this same scale, so
``sum(power) ~ integral of intensity_frame``.

**Interference.**  Each term carries an optical frequency offset ``df_opt``.  Atoms and cameras
average over MHz beat notes, so terms are grouped by ``df_opt`` (default tolerance 1 kHz, and no
group wider than that — see :func:`group_terms`): degenerate terms are summed *coherently* and
the groups add in *intensity*,
``I = sum_g |sum_{k in g} U_k|^2`` (``docs/PLAN.md`` §1.3, the Supplement's interlaced-fading
logic).  Each group is evaluated only on a bounding patch around its spots.

**Lab Z.**  Every ``z_lab`` argument of this module is a *lab* coordinate (paper Table I sign);
it is converted internally to the Eq. S11 defocus with ``Z_S11 = Z_LAB_SIGN * z_lab``
(:mod:`aodl.device.conventions` is the sign authority — the one device import ``field/`` makes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..device.conventions import Z_LAB_SIGN
from ..params import OpticsParams
from ..units import kHz
from .gaussian import (
    gauss_moments,
    gauss_moments_lower,
    gauss_moments_upper,
    gauss_moments_window,
)

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]
Index = NDArray[np.intp]

#: Default optical-frequency tolerance for interference grouping (``docs/ARCHITECTURE.md`` §3).
#: See :func:`group_terms` for why 1 kHz — and why the tolerance is also a hard cap on a
#: group's *width*, not only on the gap between neighbours.
GROUP_TOL: float = 1.0 * kHz

#: Smallest patch side, in pixels, that :func:`intensity_frame` will evaluate.
MIN_PATCH_PX: int = 3

#: Patch half-width, in intensity 1/e^2 radii, around each spot centre.
PATCH_WAISTS: float = 4.0

#: Number of transverse axes (0 = x, 1 = y).
N_AXES: int = 2

# Internal fill-edge side codes (``_WINDOW`` = bounded on both sides; see `_axis_edges`).
_NO_EDGE, _LOWER, _UPPER, _WINDOW = 0, 1, -1, 2
_SIDE_WORDS = {"lower": _LOWER, "low": _LOWER, "upper": _UPPER, "up": _UPPER, "none": _NO_EDGE}


class TermLike(Protocol):
    """Structural contract for ``device.aodl.TermArray`` (WO-03 §3).

    ``field/`` never imports the device term builder (dependencies run one way only), so terms
    are consumed *structurally*: any object carrying these attributes works, which is also what
    keeps the tests independent of the device layer.

    The members are declared **read-only** (properties, not settable variables) because that
    is what this module actually requires: nothing here ever writes a term array, and a
    mutable protocol member would exclude every immutable implementation — including
    ``device.aodl.TermArray``, a frozen dataclass, whose fields are read-only to a type
    checker.  A plain (settable) attribute satisfies a read-only property, so the frozen
    dataclass, an ordinary mutable object and a namespace-style stub all conform.
    """

    @property
    def c(self) -> NDArray[Any]:
        """``(N,)`` complex amplitude of each pupil term."""

    @property
    def theta1(self) -> NDArray[Any]:
        """``(2, N)`` linear pupil phase [rad/m] per axis (0 = x, 1 = y) — deflection."""

    @property
    def theta2(self) -> NDArray[Any]:
        """``(2, N)`` quadratic pupil phase [rad/m^2] per axis — chirp lens."""

    @property
    def alpha(self) -> NDArray[Any]:
        """``(2, 3, N)`` amplitude-polynomial coefficients ``(a0, a1, a2)`` per axis."""

    @property
    def df_opt(self) -> NDArray[Any]:
        """``(N,)`` optical frequency offset [Hz]; the interference-grouping tag."""

    @property
    def edge(self) -> Any:
        """Optional per-axis aperture fill state, ``(edge_x, edge_y)``; see
        :func:`_axis_bounds`."""


@dataclass(frozen=True)
class FrameGrid:
    """Rectangular image-plane sampling grid, in **meters** (lab X/Y).

    ``x = linspace(x0, x1, nx)``, ``y = linspace(y0, y1, ny)``; frames are returned with shape
    ``(ny, nx)`` — row index selects ``y``, column index selects ``x`` (imshow convention).
    """

    x0: float
    x1: float
    nx: int
    y0: float
    y1: float
    ny: int

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError(f"grid needs at least 2 samples per axis, got {self.nx}x{self.ny}")
        if not self.x1 > self.x0 or not self.y1 > self.y0:
            raise ValueError("grid bounds must satisfy x0 < x1 and y0 < y1")

    @property
    def x(self) -> Float:
        """Sample coordinates along x [m]."""
        return np.linspace(self.x0, self.x1, self.nx)

    @property
    def y(self) -> Float:
        """Sample coordinates along y [m]."""
        return np.linspace(self.y0, self.y1, self.ny)

    @property
    def dx(self) -> float:
        """Pixel pitch along x [m]."""
        return (self.x1 - self.x0) / (self.nx - 1)

    @property
    def dy(self) -> float:
        """Pixel pitch along y [m]."""
        return (self.y1 - self.y0) / (self.ny - 1)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x0, x1, y0, y1)`` for ``matplotlib.pyplot.imshow(..., extent=...)``."""
        return (self.x0, self.x1, self.y0, self.y1)


@dataclass(frozen=True)
class _AxisTerms:
    """One axis of a term array, unpacked into plain arrays.

    ``lo`` / ``hi`` are the aperture fill bounds of :func:`_axis_bounds`: ``-inf`` / ``+inf``
    where the acoustic column has already passed, so a fully filled axis is
    ``(-inf, +inf)``, a one-sided fill has exactly one finite bound and a
    counter-propagating pair mid-fill has two.
    """

    theta1: Float
    theta2: Float
    alpha: Complex
    lo: Float
    hi: Float

    @property
    def windowed(self) -> NDArray[np.bool_]:
        """Per term: is this axis' pupil truncated by an acoustic wavefront?"""
        return np.asarray(np.isfinite(self.lo) | np.isfinite(self.hi))

    def take(self, idx: Index) -> _AxisTerms:
        """Subset of the terms (used per frequency group)."""
        return _AxisTerms(
            theta1=self.theta1[idx],
            theta2=self.theta2[idx],
            alpha=self.alpha[:, idx],
            lo=self.lo[idx],
            hi=self.hi[idx],
        )


def _n_terms(terms: TermLike) -> int:
    return int(np.asarray(terms.df_opt).size)


def _side_codes(raw: Any, n: int) -> NDArray[np.int8] | None:
    """``+1`` (content at ``u >= u0``) / ``-1`` (``u <= u0``) / ``0``, or ``None`` if ``raw``
    is not a side specification at all."""
    if raw is None:
        return None
    arr = np.asarray(raw)
    if arr.dtype.kind in "SU":
        try:
            codes = [_SIDE_WORDS[str(word).lower()] for word in arr.ravel()]
        except KeyError:
            return None
        out = np.asarray(codes, dtype=np.int8)
    elif arr.dtype.kind in "iuf":
        flat = arr.ravel()
        if not np.all(np.isin(flat, (-1, 0, 1))):
            return None
        out = flat.astype(np.int8)
    else:
        return None
    if out.size not in (1, n):
        return None
    return np.broadcast_to(out, (n,)).astype(np.int8)


def _edge_pair(item: Any, n: int) -> tuple[Float, NDArray[np.int8]] | None:
    """Read a single ``(u0, side)`` fill-edge record, broadcast over ``n`` terms."""
    if hasattr(item, "side") and (hasattr(item, "u_edge") or hasattr(item, "u0")):
        raw_u0 = getattr(item, "u_edge", None)
        if raw_u0 is None:
            raw_u0 = item.u0
        raw_side: Any = item.side
    else:
        try:
            raw_u0, raw_side = item
        except (TypeError, ValueError):
            return None
    side = _side_codes(raw_side, n)
    if side is None:
        return None
    if raw_u0 is None:
        return np.full(n, np.nan), np.zeros(n, dtype=np.int8)
    u0 = np.broadcast_to(np.asarray(raw_u0, dtype=np.float64).ravel(), (n,)).astype(np.float64)
    side = np.where(np.isnan(u0), np.int8(_NO_EDGE), side).astype(np.int8)
    return u0, side


def _window_pair(item: Any, n: int) -> tuple[Float, Float] | None:
    """Read a two-sided ``(lo, hi)`` fill window, broadcast over ``n`` terms.

    Recognized either by field name — ``device.aodl``'s ``FillWindow(lo, hi)``, which is how
    the device layer always sends one — or as a plain pair whose second entry is *not* a side
    specification.  That is what keeps the two 2-tuple forms apart: a side is one of
    ``-1, 0, +1, "lower", "upper"``, while an aperture coordinate is a length in meters and
    is never any of those (``+-1 m`` would be a hundred apertures out).
    """
    if hasattr(item, "lo") and hasattr(item, "hi"):
        raw_lo, raw_hi = item.lo, item.hi
    else:
        try:
            raw_lo, raw_hi = item
        except (TypeError, ValueError):
            return None
        if raw_lo is None or raw_hi is None or _side_codes(raw_hi, n) is not None:
            return None
    try:
        lo = np.asarray(raw_lo, dtype=np.float64).ravel()
        hi = np.asarray(raw_hi, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    return (
        np.broadcast_to(lo, (n,)).astype(np.float64),
        np.broadcast_to(hi, (n,)).astype(np.float64),
    )


def _entry_bounds(item: Any, n: int) -> tuple[Float, Float] | None:
    """One fill record — half-line ``(u0, side)`` or window ``(lo, hi)`` — as ``(lo, hi)``."""
    window = _window_pair(item, n)
    if window is not None:
        return window
    pair = _edge_pair(item, n)
    if pair is None:
        return None
    u0, side = pair
    lo = np.where(side == _LOWER, u0, -np.inf)
    hi = np.where(side == _UPPER, u0, np.inf)
    return lo.astype(np.float64), hi.astype(np.float64)


def _axis_bounds(edge: Any, axis: int, n: int) -> tuple[Float, Float]:
    """Normalize a term array's ``edge`` field into per-term fill bounds ``(lo, hi)``.

    The aperture holds acoustic content on ``lo <= u <= hi``, with ``-inf`` / ``+inf`` for a
    side the sound has already crossed — so ``(-inf, +inf)`` is a fully filled axis,
    ``(u0, +inf)`` the ``sound_sign = -1`` half-line (lower-edge moments ``E_n``),
    ``(-inf, u1)`` the ``+1`` half-line (upper-edge moments ``F_n``) and a pair of finite
    bounds the two-sided window of a counter-propagating pair (window moments ``W_n``).
    This is the ``s u <= v t - D/2`` convention of ``device/aod.py``, intersected per axis by
    ``device/aodl.py``.

    WO-03 §3 froze this field only as "per-axis fill-edge info ``(u0 or None, side)``", and
    WO-10 added the window, so several shapes are accepted:

    * ``None`` or a missing attribute — no edge on either axis;
    * ``edge[axis] is None`` — that axis is fully filled;
    * ``edge[axis] = (u0, side)`` — ``u0`` float (``None``/NaN meaning "no edge"), ``side`` one
      of ``+1, -1, "lower", "upper"``; either may be a per-term array.  ``device.conventions``'
      ``FillEdge(u_edge, side)`` named tuple lands here, by unpacking or by attribute;
    * ``edge[axis] = (lo, hi)`` — a two-sided window; ``device.aodl``'s ``FillWindow(lo, hi)``
      lands here by field name, a plain pair by :func:`_window_pair`'s rule;
    * ``edge[axis] = [entry, ...]`` — one entry per term, each ``None`` or one of the above.
    """
    unbounded = (np.full(n, -np.inf), np.full(n, np.inf))
    if edge is None:
        return unbounded
    try:
        item = edge[axis]
    except (TypeError, KeyError, IndexError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"terms.edge must be indexable per axis, got {edge!r}") from exc
    if item is None:
        return unbounded

    bounds = _entry_bounds(item, n)
    if bounds is not None:
        return bounds

    entries = list(item)
    if len(entries) != n:
        raise ValueError(f"per-term edge list for axis {axis} has {len(entries)} != {n} entries")
    lo, hi = unbounded
    for i, entry in enumerate(entries):
        if entry is None:
            continue
        single = _entry_bounds(entry, 1)
        if single is None:
            raise ValueError(f"unrecognized fill-edge entry {entry!r} for axis {axis}")
        lo[i], hi[i] = single[0][0], single[1][0]
    return lo, hi


def _axis_edges(edge: Any, axis: int, n: int) -> tuple[Float, NDArray[np.int8]]:
    """Half-line view of :func:`_axis_bounds`: ``(u0, side)`` arrays of length ``n``.

    ``side`` is ``+1`` when the aperture holds content at ``u >= u0`` (lower-edge moments),
    ``-1`` for ``u <= u0`` (upper-edge moments) and ``0`` when the axis is fully filled
    (``u0`` is then NaN) — the WO-03 contract, preserved bit for bit.

    A **two-sided window cannot be expressed in this view**: it is reported as
    ``side = _WINDOW`` with ``u0`` its *lower* bound.  A consumer that only knows half-lines
    therefore integrates over ``u >= lo`` and over-counts the light a window passes; use
    :func:`_axis_bounds` instead, as the field integrals here do.  (The one such consumer is
    :func:`aodl.field.measure._pupil_power_axis`, whose per-term ``power`` is consequently an
    over-estimate while a counter-propagating pair is mid-fill, ``tau/2 <= t < tau``.)
    """
    lo, hi = _axis_bounds(edge, axis, n)
    bounded_lo = np.isfinite(lo)
    bounded_hi = np.isfinite(hi)
    side = np.select(
        [bounded_lo & bounded_hi, bounded_lo, bounded_hi],
        [np.int8(_WINDOW), np.int8(_LOWER), np.int8(_UPPER)],
        default=np.int8(_NO_EDGE),
    ).astype(np.int8)
    u0 = np.where(bounded_lo, lo, np.where(bounded_hi, hi, np.nan))
    return np.asarray(u0, dtype=np.float64), side


def _axis_terms(terms: TermLike, axis: int) -> _AxisTerms:
    """Unpack one axis of a term array, validating the frozen shapes."""
    n = _n_terms(terms)
    theta1 = np.asarray(terms.theta1, dtype=np.float64)
    theta2 = np.asarray(terms.theta2, dtype=np.float64)
    alpha = np.asarray(terms.alpha, dtype=np.complex128)
    if theta1.shape != (N_AXES, n) or theta2.shape != (N_AXES, n):
        raise ValueError(f"theta1/theta2 must have shape (2, {n}), got {theta1.shape}")
    if alpha.shape != (N_AXES, 3, n):
        raise ValueError(f"alpha must have shape (2, 3, {n}), got {alpha.shape}")
    lo, hi = _axis_bounds(getattr(terms, "edge", None), axis, n)
    return _AxisTerms(theta1[axis], theta2[axis], alpha[axis], lo, hi)


def _axis_field(
    at: _AxisTerms, optics: OpticsParams, coord: ArrayLike, z_s11: ArrayLike
) -> Complex:
    """Closed-form axis factor, shape ``(n_terms, *broadcast(coord, z_s11).shape)``.

    ``coord`` is the image coordinate on this axis [m] and ``z_s11`` the **Eq. S11** defocus
    [m] (not lab Z); the two are broadcast against each other, so the same routine serves an
    XY frame (scalar Z) and an XZ slice (Z varying along one axis).
    """
    k = optics.k
    focal = optics.focal_length
    coord_b, z_b = np.broadcast_arrays(
        np.asarray(coord, dtype=np.float64), np.asarray(z_s11, dtype=np.float64)
    )
    shape = coord_b.shape
    flat_coord = coord_b.reshape(1, -1)
    flat_z = z_b.reshape(1, -1)

    a = 1.0 / optics.w_in**2 - 1j * (at.theta2[:, None] - k * flat_z / (2.0 * focal**2))
    b = 1j * (at.theta1[:, None] - k * flat_coord / focal)
    a = np.broadcast_to(a, b.shape)
    i0, i1, i2 = (np.array(m, dtype=np.complex128, copy=True) for m in gauss_moments(a, b))

    bounded_lo = np.isfinite(at.lo)
    bounded_hi = np.isfinite(at.hi)
    lower = bounded_lo & ~bounded_hi
    if np.any(lower):
        e0, e1, e2 = gauss_moments_lower(a[lower], b[lower], at.lo[lower][:, None])
        i0[lower], i1[lower], i2[lower] = e0, e1, e2
    upper = bounded_hi & ~bounded_lo
    if np.any(upper):
        f0, f1, f2 = gauss_moments_upper(a[upper], b[upper], at.hi[upper][:, None])
        i0[upper], i1[upper], i2[upper] = f0, f1, f2
    window = bounded_lo & bounded_hi
    if np.any(window):
        w0, w1, w2 = gauss_moments_window(
            a[window], b[window], at.lo[window][:, None], at.hi[window][:, None]
        )
        i0[window], i1[window], i2[window] = w0, w1, w2

    field = at.alpha[0][:, None] * i0 + at.alpha[1][:, None] * i1 + at.alpha[2][:, None] * i2
    return field.reshape((at.theta1.size, *shape))


def term_field(
    terms: TermLike, optics: OpticsParams, X: ArrayLike, Y: ArrayLike, z_lab: float
) -> Complex:
    """Per-term focal field ``U_term(X, Y)`` at lab plane ``z_lab`` (Eq. S11, closed form).

    Parameters
    ----------
    terms:
        Term array (see :class:`TermLike`).
    optics:
        Wavelength / focal length / input radius.
    X, Y:
        Image-plane coordinates [m], broadcast against each other.
    z_lab:
        Scalar **lab** defocus [m]; converted with ``Z_S11 = Z_LAB_SIGN * z_lab``.

    Returns
    -------
    ``complex128`` array of shape ``(n_terms, *broadcast(X, Y).shape)`` — per term, not summed,
    because terms only interfere within a frequency group (see :func:`group_terms`).  Constant
    prefactors are dropped (module docstring).
    """
    n = _n_terms(terms)
    xb, yb = np.broadcast_arrays(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64))
    z_s11 = Z_LAB_SIGN * float(z_lab)
    fx = _axis_field(_axis_terms(terms, 0), optics, xb.ravel(), z_s11)
    fy = _axis_field(_axis_terms(terms, 1), optics, yb.ravel(), z_s11)
    c = np.asarray(terms.c, dtype=np.complex128).ravel()
    return (c[:, None] * fx * fy).reshape((n, *xb.shape))


def _cap_diameter(chunk: Index, df: Float, tol: float) -> list[Index]:
    """Split one neighbour-chained cluster until every piece spans at most ``tol``.

    ``chunk`` is in ascending ``df`` order.  An oversized piece is cut at its largest
    internal gap and both halves are re-examined; ``np.argmax`` takes the first maximum, so
    a tie is broken at the lowest frequency and the result is fully deterministic.  Iterative
    rather than recursive: a long chain of ``tol/2`` steps would otherwise nest one level per
    term.
    """
    out: list[Index] = []
    stack = [chunk]
    while stack:
        piece = stack.pop()
        values = df[piece]
        if piece.size < 2 or values[-1] - values[0] <= tol:
            out.append(piece)
            continue
        cut = int(np.argmax(np.diff(values))) + 1
        stack.append(piece[cut:])  # right first, so the left half pops (and lands) first
        stack.append(piece[:cut])
    return out


def group_terms(terms: TermLike, tol: float = GROUP_TOL) -> list[Index]:
    """Cluster terms by optical frequency ``df_opt`` (``docs/PLAN.md`` §1.3).

    Terms in one group are summed *coherently*; separate groups add in intensity, because
    their beat note averages away over any atomic or camera integration time.  Two rules
    decide the split, and a group must satisfy both:

    1. **neighbour chaining** — consecutive ``df_opt`` more than ``tol`` apart start a new
       group;
    2. **diameter cap** — no group may span more than ``tol`` end to end.  Any chained
       cluster that is wider is cut at its largest internal gap, largest gap first, until
       every piece fits (:func:`_cap_diameter`).

    Rule 2 exists because single-linkage chaining alone is transitive and a tone *ladder* is
    exactly the pathological input: 40 array terms spaced 9 kHz would chain into one
    40-member "group" under a 10 kHz tolerance, i.e. one 360 kHz-wide blob of terms declared
    mutually coherent, when in reality every pair beats at ≥ 9 kHz.

    The default ``tol`` = :data:`GROUP_TOL` = 1 kHz sits in an empty gap of the physics
    (``docs/PLAN.md`` §1.2-1.3): the coherent cases are *exact* degeneracies — an IM3 product
    landing on a fundamental (Eqs. S20-S22), or the shadow-tweezer pairs of Fig. S6 — which
    agree to a few ULP, i.e. 0 Hz; the incoherent cases are separated by tone spacings, which
    are ≥ 100 kHz for any array a 10 MHz band can hold.  Nothing legitimate lives in between,
    so the tolerance only has to be small enough not to glue a ladder and large enough to
    absorb round-off.

    Returns a list of index arrays (ascending within each group, groups ordered by frequency).
    """
    df = np.asarray(terms.df_opt, dtype=np.float64).ravel()
    if df.size == 0:
        return []
    order = np.argsort(df, kind="stable")
    splits = np.flatnonzero(np.diff(df[order]) > tol) + 1
    clusters = np.split(order, splits)
    return [np.sort(chunk) for cluster in clusters for chunk in _cap_diameter(cluster, df, tol)]


def spot_params(
    terms: TermLike, optics: OpticsParams, z_lab: float
) -> tuple[Float, Float, Float, Float]:
    r"""Per-term spot centre and intensity 1/e^2 radii at the lab plane ``z_lab``.

    This is the single derivation shared by the patch policy of :func:`intensity_frame` and by
    :mod:`aodl.field.measure`.

    **Centre.** ``b = i(theta1 - k X / F) = -i (k/F) (X - X_c)`` vanishes at
    ``X_c = theta1 F / k`` — the Table I deflection — independently of ``a``, so the centre does
    not move with defocus.

    **Radius.** With ``alpha = (1, 0, 0)`` the axis factor is ``I0 = sqrt(pi/a) e^{b^2/(4a)}``, so

        |field|^2 = (pi/|a|) exp(2 Re(b^2/4a))
                  = (pi/|a|) exp(-(k/F)^2 (X - X_c)^2 Re(1/a) / 2).

    Matching ``exp(-2 (X - X_c)^2 / w^2)`` (1/e^2 **intensity** radius) gives

        w = (2 F / k) / sqrt(Re(1/a)),     Re(1/a) = Re(a)/|a|^2 = (1/w_in^2)/|a|^2,

    i.e. ``w = (2F/k) |a| w_in``.  At ``a = 1/w_in^2`` that is ``2F/(k w_in) = lambda F/(pi w_in)
    = optics.waist0``; writing ``a = 1/w_in^2 - i k (Z - Z_focus)/(2F^2)`` with the per-axis
    focus ``Z_focus = 2 F^2 theta2 / k`` and ``z_R = 2F^2/(k w_in^2) = optics.rayleigh`` it
    reduces to the textbook ``w(Z) = waist0 sqrt(1 + ((Z - Z_focus)/z_R)^2)``.

    The amplitude polynomial (``alpha1``, ``alpha2``: tilt and acoustic irising) perturbs the
    profile but not this Gaussian envelope, and is second-order for patch sizing at
    ``PATCH_WAISTS = 4`` radii.  A fill edge is *not*: it truncates the pupil, and the
    resulting hard-edge tails decay only as ``1/dX^2``, so :func:`intensity_frame` stops
    patching an axis that carries one.  The exact profile is always given by
    :func:`term_field`.

    Returns
    -------
    ``(Xc, Yc, wx, wy)``, each of shape ``(n_terms,)`` [m].
    """
    k = optics.k
    focal = optics.focal_length
    theta1 = np.asarray(terms.theta1, dtype=np.float64)
    theta2 = np.asarray(terms.theta2, dtype=np.float64)
    z_s11 = Z_LAB_SIGN * float(z_lab)
    centres = theta1 * focal / k
    a = 1.0 / optics.w_in**2 - 1j * (theta2 - k * z_s11 / (2.0 * focal**2))
    w = (2.0 * focal / k) / np.sqrt(np.real(1.0 / a))
    return centres[0], centres[1], w[0], w[1]


def _index_span(lo: float, hi: float, start: float, step: float, n: int) -> tuple[int, int] | None:
    """Half-open index range covering ``[lo, hi]``, clipped to the grid, at least 3 px wide.

    ``None`` when the interval misses the grid entirely (the spot is off canvas).
    """
    if n <= MIN_PATCH_PX:
        return 0, n
    i0 = int(np.floor((lo - start) / step))
    i1 = int(np.ceil((hi - start) / step)) + 1
    if i1 <= 0 or i0 >= n:
        return None
    i0 = max(i0, 0)
    i1 = min(i1, n)
    if i1 - i0 < MIN_PATCH_PX:
        i1 = min(n, i0 + MIN_PATCH_PX)
        i0 = max(0, i1 - MIN_PATCH_PX)
    return i0, i1


def intensity_frame(
    terms: TermLike,
    optics: OpticsParams,
    grid: FrameGrid,
    z_lab: float,
    tol: float = GROUP_TOL,
) -> Float:
    """Intensity frame ``I(X, Y)`` at the lab plane ``z_lab``, shape ``(grid.ny, grid.nx)``.

    Terms are grouped by optical frequency (:func:`group_terms`); each group's complex field is
    accumulated on the union bounding patch of its spots (centre +- ``PATCH_WAISTS`` times the
    larger of the two 1/e^2 radii, clipped to the grid, at least ``MIN_PATCH_PX`` on a side),
    squared, and added into the canvas.  Cross-group beat notes average out, so groups add in
    intensity (``docs/PLAN.md`` §1.3).

    Prefactors are dropped as in :func:`term_field`; the scale matches
    :attr:`aodl.field.measure.SpotMetrics.power`.
    """
    canvas = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    if _n_terms(terms) == 0:
        return canvas

    atx = _axis_terms(terms, 0)
    aty = _axis_terms(terms, 1)
    c = np.asarray(terms.c, dtype=np.complex128).ravel()
    xc, yc, wx, wy = spot_params(terms, optics, z_lab)
    margin = PATCH_WAISTS * np.maximum(wx, wy)
    x, y = grid.x, grid.y
    z_s11 = Z_LAB_SIGN * float(z_lab)

    for idx in group_terms(terms, tol):
        # A fill edge truncates the pupil, so that axis' far field grows hard-edge tails that
        # decay only as 1/dX^2: no bounded multiple of the Gaussian radius contains them (at
        # PATCH_WAISTS = 4 up to ~15% of the light is dropped, and the cut is bright enough to
        # show as a square in a rendered frame).  Patch that axis only once it is fully filled.
        xs = (
            (0, grid.nx)
            if np.any(atx.windowed[idx])
            else _index_span(
                float(np.min(xc[idx] - margin[idx])),
                float(np.max(xc[idx] + margin[idx])),
                grid.x0,
                grid.dx,
                grid.nx,
            )
        )
        ys = (
            (0, grid.ny)
            if np.any(aty.windowed[idx])
            else _index_span(
                float(np.min(yc[idx] - margin[idx])),
                float(np.max(yc[idx] + margin[idx])),
                grid.y0,
                grid.dy,
                grid.ny,
            )
        )
        if xs is None or ys is None:
            continue
        fx = _axis_field(atx.take(idx), optics, x[xs[0] : xs[1]], z_s11)
        fy = _axis_field(aty.take(idx), optics, y[ys[0] : ys[1]], z_s11)
        patch = np.einsum("n,ny,nx->yx", c[idx], fy, fx)
        canvas[ys[0] : ys[1], xs[0] : xs[1]] += np.abs(patch) ** 2
    return canvas


def intensity_slice_xz(
    terms: TermLike,
    optics: OpticsParams,
    x_axis: ArrayLike,
    z_axis_lab: ArrayLike,
    y0: float,
    tol: float = GROUP_TOL,
) -> Float:
    """Intensity on an ``(X, Z_lab)`` slice at fixed ``Y = y0`` — the movie's XZ panel.

    Eq. S11 again, now scanned in its defocus variable instead of at one plane: the axial
    coordinate enters only through ``a``'s ``- k Z_S11 / (2 F^2)`` term (module docstring), so
    a Z scan costs no more than an X scan and needs no propagation step.  Lab Z is converted
    on the way in (``Z_S11 = Z_LAB_SIGN * z_lab``), which is why the panel's vertical axis
    reads as the paper's Table I ``Zbar``.  This is the side panel of ``docs/ARCHITECTURE.md``
    §3 decision 3.

    Same frequency grouping as :func:`intensity_frame` (``docs/PLAN.md`` §1.3).  Returns shape
    ``(nz, nx)`` (row index selects ``z_axis_lab``, column index selects ``x_axis``).

    No patching here: a spot sweeps through focus along the Z axis of this panel, so a
    Z-independent bounding box would be wrong, and panel grids are small.
    """
    x = np.asarray(x_axis, dtype=np.float64).ravel()
    z_lab = np.asarray(z_axis_lab, dtype=np.float64).ravel()
    out = np.zeros((z_lab.size, x.size), dtype=np.float64)
    if _n_terms(terms) == 0:
        return out

    atx = _axis_terms(terms, 0)
    aty = _axis_terms(terms, 1)
    c = np.asarray(terms.c, dtype=np.complex128).ravel()
    z_s11 = Z_LAB_SIGN * z_lab

    for idx in group_terms(terms, tol):
        fx = _axis_field(atx.take(idx), optics, x[None, :], z_s11[:, None])
        fy = _axis_field(aty.take(idx), optics, y0, z_s11)
        out += np.abs(np.einsum("n,nz,nzx->zx", c[idx], fy, fx)) ** 2
    return out


__all__ = [
    "GROUP_TOL",
    "MIN_PATCH_PX",
    "PATCH_WAISTS",
    "FrameGrid",
    "TermLike",
    "Z_LAB_SIGN",
    "group_terms",
    "intensity_frame",
    "intensity_slice_xz",
    "spot_params",
    "term_field",
]
