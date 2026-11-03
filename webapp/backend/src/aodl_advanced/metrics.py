r"""Image metrics of the FFT engine: centroid, second-moment radii, peak, power, best focus.

Everything is measured on rendered intensity images (there is no closed form to read off
here).  Radii are the intensity second-moment radii ``2 sigma`` — equal to the ``1/e^2``
radius ``w`` for a Gaussian spot (``sigma^2 = w^2 / 4``).  Moments are taken over a window of
half-width ``WINDOW_WAISTS`` input-beam waists about the brightest pixel, so for an array the
metrics describe the brightest trap rather than the array's extent.

**Best focus.**  For a Gaussian beam ``w^2(Z) = w_0^2 (1 + ((Z - Z_0)/z_R)^2)`` is exactly a
parabola in ``Z``, so the minimum of ``w^2`` sampled on a uniform Z grid is refined with the
three-point parabolic formula and is exact for an unaberrated spot (:func:`best_focus`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float64]

#: Half-width of the moment window about the brightest pixel, in units of ``w_0``.
WINDOW_WAISTS = 4.0


@dataclass(frozen=True)
class AdvMetrics:
    """Spot metrics of one frame of the FFT engine, SI units.

    ``z_best_x`` / ``z_best_y`` / ``astig`` are ``None`` when no focus scan was requested.
    """

    x: float
    y: float
    wx: float
    wy: float
    peak: float
    power: float
    z_best_x: float | None = None
    z_best_y: float | None = None
    astig: float | None = None


def image_power(img: Float, dx: float, dy: float) -> float:
    """``sum I dx dy``."""
    return float(np.sum(img) * dx * dy)


def peak_pixel(img: Float) -> tuple[int, int]:
    """``(row, col)`` of the brightest pixel."""
    i = int(np.argmax(img))
    return divmod(i, img.shape[1])


def refined_peak(img: Float) -> float:
    """Peak intensity to sub-pixel precision — a quadratic fit of ``ln I`` on the 3x3 block
    about the brightest pixel, which is exact for a Gaussian spot.

    The raw pixel maximum depends on where the spot centre falls between pixel centres (up to
    ``1 - exp(-2 (dx/2)^2 / w^2)`` ~ 1.6 % for a 1.07 µm waist on 0.19 µm pixels), which would
    make a Strehl ratio between a shifted and an unshifted spot meaningless at the percent
    level.  Falls back to the pixel value at the image border or when the block is not a
    concave bump.
    """
    row, col = peak_pixel(img)
    ny, nx = img.shape
    pixel = float(img[row, col])
    if row == 0 or col == 0 or row == ny - 1 or col == nx - 1:
        return pixel
    block = img[row - 1 : row + 2, col - 1 : col + 2]
    if np.any(block <= 0.0):
        return pixel
    u, v = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    design = np.column_stack(
        [np.ones(9), u.ravel(), v.ravel(), u.ravel() ** 2, v.ravel() ** 2, (u * v).ravel()]
    )
    coef, *_ = np.linalg.lstsq(design, np.log(block).ravel(), rcond=None)
    a, b, c, d, e, f = (float(value) for value in coef)
    hessian = np.array([[2.0 * d, f], [f, 2.0 * e]])
    if d >= 0.0 or e >= 0.0 or np.linalg.det(hessian) <= 0.0:
        return pixel
    u0, v0 = np.linalg.solve(hessian, [-b, -c])
    if abs(u0) > 1.0 or abs(v0) > 1.0:
        return pixel
    return float(math.exp(a + b * u0 + c * v0 + d * u0 * u0 + e * v0 * v0 + f * u0 * v0))


def moments_1d(profile: Float, coord: Float) -> tuple[float, float]:
    """Centroid and ``2 sigma`` radius of a non-negative 1-D profile on ``coord``."""
    total = float(np.sum(profile))
    if not total > 0.0:
        return float(coord[coord.size // 2]), 0.0
    mean = float(np.sum(profile * coord) / total)
    var = float(np.sum(profile * (coord - mean) ** 2) / total)
    return mean, 2.0 * math.sqrt(max(var, 0.0))


def windowed_moments(
    img: Float, x: Float, y: Float, half_width: float
) -> tuple[float, float, float, float]:
    """``(x_c, y_c, w_x, w_y)`` from second moments over ``+-half_width`` about the peak pixel.

    The window is clipped to the image; an all-zero image returns the grid centre and zero
    radii.
    """
    if not np.any(img > 0.0):
        return float(x[x.size // 2]), float(y[y.size // 2]), 0.0, 0.0
    row, col = peak_pixel(img)
    xs = (x >= x[col] - half_width) & (x <= x[col] + half_width)
    ys = (y >= y[row] - half_width) & (y <= y[row] + half_width)
    sub = img[np.ix_(ys, xs)]
    xc, wx = moments_1d(sub.sum(axis=0), x[xs])
    yc, wy = moments_1d(sub.sum(axis=1), y[ys])
    return xc, yc, wx, wy


def width_scan(slice_img: Float, coord: Float) -> Float:
    """``2 sigma`` radius of every row of an ``(nz, n)`` slice along ``coord``."""
    return np.asarray([moments_1d(row, coord)[1] for row in slice_img], dtype=np.float64)


def best_focus(z: Float, w: Float) -> float:
    """``Z`` of the minimum of ``w(Z)``, refined parabolically on ``w^2`` (exact for a Gaussian).

    At an end of the scan the end value is returned (no extrapolation).
    """
    zs = np.asarray(z, dtype=np.float64).ravel()
    w2 = np.asarray(w, dtype=np.float64).ravel() ** 2
    if zs.size != w2.size or zs.size == 0:
        raise ValueError("best_focus needs equally long z and w arrays")
    if zs.size < 3:
        return float(zs[int(np.argmin(w2))])
    i = int(np.argmin(w2))
    if i == 0 or i == zs.size - 1:
        return float(zs[i])
    h = zs[i + 1] - zs[i]
    denominator = w2[i + 1] - 2.0 * w2[i] + w2[i - 1]
    if denominator <= 0.0:
        return float(zs[i])
    return float(zs[i] - 0.5 * h * (w2[i + 1] - w2[i - 1]) / denominator)


__all__ = [
    "WINDOW_WAISTS",
    "AdvMetrics",
    "best_focus",
    "image_power",
    "moments_1d",
    "peak_pixel",
    "refined_peak",
    "width_scan",
    "windowed_moments",
]
