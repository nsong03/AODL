r"""Averaged intensities -> numbers: profile fits, best focus, blob audit, accumulation.

The checker never compares *fields* to anything; it measures the rendered intensity the way
an experiment would — fit a spot, find the plane where it is smallest, look for light that
should not be there — so that a disagreement with the simulator is a statement about physics
rather than about a phase convention.

Three things here deserve their reasons written down.

**Fitting by weighted log-parabola.**  ``I(x) = I_0 exp(-2 (x - x_0)^2 / w^2)`` is a parabola
in ``log I``, so a *linear* least-squares fit of ``log I`` recovers ``(x_0, w, I_0)`` exactly
— no initial guess, no iteration, no ``scipy.optimize``, and bit-for-bit reproducible.  The
``I^2`` weights undo the logarithm's distortion of the residuals (Guo's refinement of
Caruana's method) so that the fit is the one a linear fit of the *intensities* would give,
and they make the tails, where a neighbouring trap or a ghost lives, count for almost
nothing.  Sample selection does the rest: only the contiguous run above ``e^{-2}`` of the
peak is used, which for an array is well inside one pitch.

**Best focus from ``w^2``, not from the on-axis peak.**  Through focus a Gaussian obeys
``w^2(Z) = w_0^2 (1 + ((Z - Z_f)/z_R)^2)`` — a parabola whose vertex is the waist — and each
transverse axis has its own.  The on-axis intensity of an *astigmatic* beam peaks somewhere
else entirely: at the circle of least confusion, which is midway between the line foci while
they are closer than ``2 z_R`` and splits into a pair of maxima at
``(Z_x + Z_y)/2 -+ sqrt(((Z_x - Z_y)/2)^2 - z_R^2)`` once they are not.  Either way it is one
number where there are two, and it says nothing about their separation.  Since
``Delta F = Z_x - Z_y`` is a verdict-bearing quantity for the astigmatism-free claim
(Table I), the per-axis parabola is the only fit that can be used.

**Outer-product accumulation.**  The field is separable, ``U(X, Y) = U_x(X) U_y(Y)``, so the
time-averaged intensity is ``<|U_x(X)|^2 |U_y(Y)|^2>``, averaged as a *product* per sub-time.
Averaging the two factors separately and multiplying afterwards is wrong whenever the two
axes beat at related frequencies — which is exactly the case for an atom array, whose x and y
tone ladders are commensurate by construction, and for the Fig. S6 shadow tweezers during a
fade.  :func:`accumulate_intensity` therefore forms the outer product first and averages
second; :func:`accumulate_marginals` gives the same canvas's two marginals in ``O(n_x + n_y)``
instead of ``O(n_x n_y)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]

#: Relative level bounding the contiguous run of samples a profile fit uses: the 1/e^2
#: intensity radius itself, i.e. ``|x - x_0| <= w``.
FIT_LEVEL = math.exp(-2.0)

#: Fallback level when the profile is too coarsely sampled to give three points at
#: :data:`FIT_LEVEL` (``|x - x_0| <= sqrt(2) w``).
FIT_LEVEL_WIDE = math.exp(-4.0)

__all__ = [
    "FIT_LEVEL",
    "FIT_LEVEL_WIDE",
    "Blob",
    "TrapFit",
    "accumulate_intensity",
    "accumulate_marginals",
    "best_focus",
    "find_blobs",
    "fit_gaussian_1d",
    "profile_moments",
]


@dataclass(frozen=True)
class TrapFit:
    """Everything the checker measures about one trap in one frame, SI units.

    Attributes
    ----------
    x, y:
        Fitted lateral centre [m].
    z_lab:
        Best-focus **lab** Z [m], the mean of the two per-axis parabola vertices
        (Table I's ``Zbar``).
    delta_f:
        Astigmatic interval ``Z_x - Z_y`` [m] (Table I's ``Delta F``).
    sigma_astig:
        ``delta_f / z_R``, the paper's dimensionless astigmatism.
    wx, wy:
        Fitted 1/e^2 intensity radii [m] at ``z_lab``.
    peak:
        Fitted peak intensity, on the checker's own (prefactor-dropped) scale.
    power:
        Intensity integrated over the trap's patch, same scale.
    beat_std:
        Standard deviation of the per-sub-time peak divided by its mean — how much of the
        trap's brightness is beating inside the averaging window.  Report-only: a large value
        means the window is too short for the beats present, not that the drive is wrong.
    """

    x: float
    y: float
    z_lab: float
    delta_f: float
    sigma_astig: float
    wx: float
    wy: float
    peak: float
    power: float
    beat_std: float


@dataclass(frozen=True)
class Blob:
    """A local intensity maximum found on the full-field canvas.

    Attributes
    ----------
    time:
        Frame time [s].
    x, y:
        Sub-pixel position [m].
    rel_intensity:
        Peak intensity relative to the reference the audit was run against (normally the
        median trap peak of the same frame).
    on_lattice:
        Whether the blob sits on an expected array node — a commensurate intermodulation
        product or a Shepard extended column lands there and is not a fault; light *off* the
        lattice is.
    """

    time: float
    x: float
    y: float
    rel_intensity: float
    on_lattice: bool


def _as_profile(coords: ArrayLike, profile: ArrayLike) -> tuple[Float, Float]:
    x = np.asarray(coords, dtype=np.float64)
    intensity = np.asarray(profile, dtype=np.float64)
    if x.ndim != 1 or intensity.ndim != 1 or x.shape != intensity.shape:
        raise ValueError(
            f"coords and profile must be matching 1-D arrays, got {x.shape} and {intensity.shape}"
        )
    if x.size < 3:
        raise ValueError(f"a profile needs at least 3 samples, got {x.size}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(intensity)):
        raise ValueError("coords and profile must be finite")
    return x, intensity


def profile_moments(coords: ArrayLike, profile: ArrayLike) -> tuple[float, float]:
    """Raw intensity-weighted centroid and RMS width of a profile — no model assumed.

    Reported alongside the fits because they are model-free: for a clean Gaussian
    ``rms = w / 2``, and a large disagreement between the two is itself the signal that the
    spot is not one Gaussian (a ghost, a shadow tweezer, a partially filled aperture).
    Negative samples are clipped to zero first.
    """
    x, intensity = _as_profile(coords, profile)
    weight = np.clip(intensity, 0.0, None)
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("profile carries no power, so it has no moments")
    center = float((weight * x).sum() / total)
    variance = float((weight * (x - center) ** 2).sum() / total)
    return center, math.sqrt(max(variance, 0.0))


def _fit_window(intensity: Float, level: float) -> tuple[int, int]:
    """Contiguous run around the brightest sample that stays above ``level * peak``."""
    peak_index = int(np.argmax(intensity))
    threshold = level * float(intensity[peak_index])
    lo = peak_index
    while lo > 0 and intensity[lo - 1] > threshold:
        lo -= 1
    hi = peak_index
    while hi < intensity.size - 1 and intensity[hi + 1] > threshold:
        hi += 1
    return lo, hi + 1


def fit_gaussian_1d(coords: ArrayLike, profile: ArrayLike) -> tuple[float, float, float]:
    """``(center, radius_1e2, peak)`` of a Gaussian intensity profile.  Exact on a Gaussian.

    A weighted linear least-squares fit of ``log I`` to a parabola, weights ``I^2``, over the
    contiguous run of samples above ``e^{-2}`` of the peak (widened to ``e^{-4}`` and then to
    the whole profile if that leaves fewer than three points).  ``radius_1e2`` is the 1/e^2
    **intensity** radius ``w``, the same convention as
    :attr:`aodl.params.OpticsParams.waist0`.

    Raises ``ValueError`` if the selected samples do not curve downward — i.e. if what was
    handed in is not a peak.  See :func:`profile_moments` for the model-free companion.
    """
    x, intensity = _as_profile(coords, profile)
    if float(intensity.max()) <= 0.0:
        raise ValueError("profile has no positive samples to fit")

    keep = np.zeros(intensity.size, dtype=bool)
    for level in (FIT_LEVEL, FIT_LEVEL_WIDE, 0.0):
        lo, hi = _fit_window(intensity, level)
        keep = np.zeros(intensity.size, dtype=bool)
        keep[lo:hi] = True
        keep &= intensity > 0.0
        if int(keep.sum()) >= 3:
            break
    else:
        raise ValueError(
            f"only {int(keep.sum())} positive samples around the peak — a log-parabola fit "
            "needs 3; sample the profile more finely"
        )

    xs = x[keep]
    ys = intensity[keep]
    origin = float(x[int(np.argmax(intensity))])
    scale = float(np.max(np.abs(xs - origin)))
    if scale == 0.0:
        raise ValueError("profile samples around the peak all share one coordinate")
    s = (xs - origin) / scale

    weights = ys**2
    basis = np.stack([np.ones_like(s), s, s * s], axis=1)
    lhs = basis.T @ (weights[:, None] * basis)
    rhs = basis.T @ (weights * np.log(ys))
    a0, a1, a2 = np.linalg.solve(lhs, rhs)
    if a2 >= 0.0:
        raise ValueError(
            "the weighted log-parabola opens upward: these samples are not a peak "
            f"(curvature {a2:+.3g} in normalized units)"
        )
    center_s = -0.5 * a1 / a2
    peak = math.exp(float(a0 + a1 * center_s + a2 * center_s * center_s))
    radius = scale * math.sqrt(-2.0 / float(a2))
    return origin + scale * float(center_s), radius, peak


def best_focus(z_planes: ArrayLike, w2: ArrayLike) -> float:
    """Vertex of the through-focus parabola ``w^2(Z) = w_0^2 (1 + ((Z - Z_f)/z_R)^2)``.

    ``z_planes`` are lab Z [m] and ``w2`` the squared 1/e^2 radii measured there; the return
    value is the lab Z of the waist.  Least squares over all planes (exact with three), so
    three planes straddling the focus are enough and more only average noise down.

    A vertex *outside* the sampled range is returned, not refused: the parabola is the exact
    law for an ideal beam, so extrapolating it a little is meaningful (and the caller centres
    the stack on its expectation anyway, so a large extrapolation is itself the finding).
    What is refused is a stack with no upward curvature at all — there is no waist anywhere
    on that curve, and inventing one would be worse than failing.
    """
    z = np.asarray(z_planes, dtype=np.float64)
    y = np.asarray(w2, dtype=np.float64)
    if z.ndim != 1 or y.shape != z.shape:
        raise ValueError(f"z_planes and w2 must be matching 1-D arrays, got {z.shape}, {y.shape}")
    if z.size < 3:
        raise ValueError(f"best_focus needs at least 3 planes, got {z.size}")
    scale = float(np.max(np.abs(z - z.mean())))
    if scale == 0.0:
        raise ValueError("best_focus needs planes at distinct Z")
    s = (z - z.mean()) / scale
    coeffs = np.polynomial.polynomial.polyfit(s, y, 2)
    # A straight line fits with a curvature of round-off, not zero, so the test is relative.
    if coeffs[2] <= 1e-9 * float(np.max(np.abs(y))):
        raise ValueError(
            "w^2(Z) does not curve upward over this stack, so there is no waist on it "
            f"(curvature {coeffs[2]:+.3g} against a w^2 scale of {float(np.max(np.abs(y))):.3g}); "
            "widen the Z range or centre it on the expectation"
        )
    return float(z.mean() + scale * (-0.5 * coeffs[1] / coeffs[2]))


def find_blobs(
    canvas: ArrayLike,
    xs: ArrayLike,
    ys: ArrayLike,
    floor: float,
    *,
    merge_radius: float,
    reference: float | None = None,
) -> list[tuple[float, float, float]]:
    """Local maxima of an intensity canvas above ``floor``, merged within ``merge_radius``.

    Parameters
    ----------
    canvas:
        ``(len(xs), len(ys))`` intensities, indexed ``[ix, iy]``.
    xs, ys:
        Uniform image coordinates [m] of the two axes.
    floor:
        **Absolute** intensity threshold — a maximum below it is not reported.
    merge_radius:
        Two maxima closer than this [m] are one blob; the brighter survives.  One waist is
        the right value: anything closer is the same spot resolved twice.
    reference:
        Divisor for the reported intensity (default: the canvas maximum).  Pass the frame's
        median trap peak to get :attr:`Blob.rel_intensity` directly.

    Returns
    -------
    ``[(x, y, rel_intensity), ...]``, brightest first, with sub-pixel positions from a
    parabolic vertex through each maximum's three neighbours along each axis.
    """
    values = np.asarray(canvas, dtype=np.float64)
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if values.ndim != 2 or values.shape != (x.size, y.size):
        raise ValueError(
            f"canvas must be shaped (len(xs), len(ys)) = {(x.size, y.size)}, got {values.shape}"
        )
    radius = float(merge_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"merge_radius must be positive and finite, got {merge_radius!r}")
    peak = float(values.max()) if values.size else 0.0
    divisor = peak if reference is None else float(reference)
    if divisor <= 0.0:
        return []

    nx, ny = values.shape
    padded = np.full((nx + 2, ny + 2), -np.inf)
    padded[1:-1, 1:-1] = values
    is_max = values > float(floor)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            if dx == 1 and dy == 1:
                continue
            is_max &= values >= padded[dx : dx + nx, dy : dy + ny]

    found: list[tuple[float, float, float]] = []
    for ix, iy in zip(*np.nonzero(is_max), strict=True):
        found.append(
            (
                _vertex(x, values[:, iy], int(ix)),
                _vertex(y, values[ix, :], int(iy)),
                float(values[ix, iy]),
            )
        )
    found.sort(key=lambda blob: blob[2], reverse=True)

    kept: list[tuple[float, float, float]] = []
    for bx, by, value in found:
        if any(math.hypot(bx - kx, by - ky) < radius for kx, ky, _ in kept):
            continue
        kept.append((bx, by, value))
    return [(bx, by, value / divisor) for bx, by, value in kept]


def _vertex(coords: Float, profile: Float, index: int) -> float:
    """Sub-sample peak position: parabola through the three samples around ``index``."""
    if index == 0 or index == coords.size - 1:
        return float(coords[index])
    left, middle, right = profile[index - 1 : index + 2]
    denominator = left - 2.0 * middle + right
    if denominator >= 0.0:
        return float(coords[index])
    step = float(coords[index + 1] - coords[index])
    return float(coords[index] + 0.5 * step * (left - right) / denominator)


def _intensities(ux: ArrayLike, uy: ArrayLike) -> tuple[Float, Float]:
    ax = np.abs(np.asarray(ux, dtype=np.complex128)) ** 2
    ay = np.abs(np.asarray(uy, dtype=np.complex128)) ** 2
    if ax.ndim < 1 or ay.ndim < 1:
        raise ValueError("ux and uy must have at least one axis (the sub-time axis)")
    if ax.shape[:-1] != ay.shape[:-1]:
        raise ValueError(
            "ux and uy must agree on every axis but the last (the image coordinate): "
            f"got {ax.shape} and {ay.shape}"
        )
    if ax.shape[0] == 0:
        raise ValueError("no sub-times to accumulate")
    return ax, ay


def accumulate_intensity(ux: ArrayLike, uy: ArrayLike) -> Float:
    """Sub-time average of the separable intensity ``|U_x|^2 (x) |U_y|^2``.

    ``ux`` and ``uy`` are shaped ``(k, ..., n_x)`` and ``(k, ..., n_y)``: axis 0 is the
    sub-time (averaged over), any middle axes are carried (Z planes, traps), the last is the
    image coordinate.  Returns ``(..., n_x, n_y)``.

    The outer product is formed *per sub-time* and averaged afterwards — see the module
    docstring for why the other order is wrong.
    """
    ax, ay = _intensities(ux, uy)
    return np.asarray(np.einsum("k...i,k...j->...ij", ax, ay) / ax.shape[0], dtype=np.float64)


def accumulate_marginals(ux: ArrayLike, uy: ArrayLike) -> tuple[Float, Float]:
    """The two axis marginals of :func:`accumulate_intensity`, without building the canvas.

    Returns ``(Ix, Iy)`` of shapes ``(..., n_x)`` and ``(..., n_y)``, equal to
    ``accumulate_intensity(ux, uy).sum(axis=-1)`` and ``.sum(axis=-2)`` exactly — each
    sub-time's profile weighted by that sub-time's power on the *other* axis, which is what
    keeps the two axes' beats correlated.
    """
    ax, ay = _intensities(ux, uy)
    k = ax.shape[0]
    return (
        np.asarray(np.einsum("k...i,k...->...i", ax, ay.sum(axis=-1)) / k, dtype=np.float64),
        np.asarray(np.einsum("k...j,k...->...j", ay, ax.sum(axis=-1)) / k, dtype=np.float64),
    )
