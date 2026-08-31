"""Direct-quadrature evaluation of Eq. S11 — **tests only**.

This is the deliberately dumb backend: it sums the pupil integral on a dense grid with no
Taylor truncation, no closed forms and no FFTs.  It exists to bound the error of the
analytic path (``field/gaussian.py`` + ``field/focal.py``) and is far too slow for movies.

Eq. S11 (``docs/PLAN.md`` §1.3), at defocus ``Z``::

    U(X, Y, Z) ~ (1 / (i lambda F)) * int int U_in(x, y) P(x, y)
                 exp(-i k (x X + y Y) / F) exp(-i k Z (x^2 + y^2) / (2 F^2)) dx dy

Two constant-modulus prefactors are **omitted**: the propagation phase ``exp(i k F)`` and
the common image-plane curvature ``exp[i k (X^2 + Y^2) / (2 F)]``.  They are pure phases
with unit modulus, identical for every pupil term, so intensities and all term-to-term
interference are unaffected; the analytic path drops the same factors.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..params import OpticsParams

Complex = NDArray[np.complex128]

#: Rough ceiling on the number of complex entries held per chunk (~32 MB at complex128).
_CHUNK_ELEMENTS = 2_000_000


def _trapz_weights(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Trapezoid weights for a uniform grid (kept local: no numpy 1.x/2.x API split)."""
    du = float(u[1] - u[0])
    w = np.full(u.size, du, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _aperture_grid(optics: OpticsParams, n: int, span: float) -> NDArray[np.float64]:
    return np.linspace(-span * optics.w_in, span * optics.w_in, int(n))


def reference_field_separable(
    pupil_x: Callable[[NDArray[np.float64]], ArrayLike],
    pupil_y: Callable[[NDArray[np.float64]], ArrayLike],
    optics: OpticsParams,
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    n: int = 8001,
    span: float = 6.0,
) -> Complex:
    """Brute-force Eq. S11 for a pupil that factorizes as ``P(x, y) = px(x) py(y)``.

    Parameters
    ----------
    pupil_x, pupil_y:
        ``u -> complex`` 1D pupil factors, **including** the input-beam Gaussian
        ``exp(-u^2 / w_in^2)``.
    optics:
        Wavelength / focal length / input radius.
    X, Y, Z:
        Image coordinates [m], broadcast against each other (``Z`` is the Eq. S11 defocus
        variable, not a lab coordinate — sign conventions live in ``device/conventions.py``).
    n:
        Number of trapezoid samples per axis.
    span:
        Half-width of the integration interval in units of ``w_in``.

    Returns
    -------
    Complex field ``(1 / (i lambda F)) * Ix(X, Z) * Iy(Y, Z)``, shaped like the broadcast
    of ``X, Y, Z`` (see the module docstring for the omitted constant phases).
    """
    k = optics.k
    focal = optics.focal_length
    u = _aperture_grid(optics, n, span)
    weights = _trapz_weights(u)
    px = np.asarray(pupil_x(u), dtype=np.complex128) * weights
    py = np.asarray(pupil_y(u), dtype=np.complex128) * weights
    if px.shape != u.shape or py.shape != u.shape:
        raise ValueError("pupil_x/pupil_y must be vectorized over the aperture grid")

    xb, yb, zb = np.broadcast_arrays(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
    )
    shape = xb.shape
    xf, yf, zf = xb.ravel(), yb.ravel(), zb.ravel()
    out = np.empty(xf.size, dtype=np.complex128)

    u2 = u * u
    chunk = max(1, _CHUNK_ELEMENTS // u.size)
    for start in range(0, xf.size, chunk):
        sl = slice(start, start + chunk)
        quad = np.exp(-1j * (k / (2.0 * focal * focal)) * zf[sl, None] * u2[None, :])
        lin_x = np.exp(-1j * (k / focal) * xf[sl, None] * u[None, :])
        lin_y = np.exp(-1j * (k / focal) * yf[sl, None] * u[None, :])
        ix = np.sum(px[None, :] * quad * lin_x, axis=1)
        iy = np.sum(py[None, :] * quad * lin_y, axis=1)
        out[sl] = ix * iy

    prefactor = 1.0 / (1j * optics.wavelength * focal)
    return (prefactor * out).reshape(shape)


def reference_field_2d(
    pupil: Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike],
    optics: OpticsParams,
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    n: int = 801,
    span: float = 6.0,
) -> Complex:
    """Brute-force Eq. S11 for a general (non-separable) pupil ``P(x, y)``.

    ``pupil`` is called once with a broadcast ``(n, n)`` meshgrid and must return the
    complex pupil **including** the input-beam Gaussian.  The double sum is evaluated as
    ``ex @ P_w @ ey`` per image point (the Eq. S11 kernel is separable even when the pupil
    is not), chunked over image points.
    """
    k = optics.k
    focal = optics.focal_length
    u = _aperture_grid(optics, n, span)
    weights = _trapz_weights(u)
    p = np.asarray(pupil(u[:, None], u[None, :]), dtype=np.complex128)
    if p.shape != (u.size, u.size):
        raise ValueError(f"pupil must broadcast to shape {(u.size, u.size)}, got {p.shape}")
    pw = p * weights[:, None] * weights[None, :]

    xb, yb, zb = np.broadcast_arrays(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
    )
    shape = xb.shape
    xf, yf, zf = xb.ravel(), yb.ravel(), zb.ravel()
    out = np.empty(xf.size, dtype=np.complex128)

    u2 = u * u
    chunk = max(1, _CHUNK_ELEMENTS // (2 * u.size))
    for start in range(0, xf.size, chunk):
        sl = slice(start, start + chunk)
        quad = np.exp(-1j * (k / (2.0 * focal * focal)) * zf[sl, None] * u2[None, :])
        ex = quad * np.exp(-1j * (k / focal) * xf[sl, None] * u[None, :])
        ey = quad * np.exp(-1j * (k / focal) * yf[sl, None] * u[None, :])
        out[sl] = np.einsum("mi,ij,mj->m", ex, pw, ey, optimize=True)

    prefactor = 1.0 / (1j * optics.wavelength * focal)
    return (prefactor * out).reshape(shape)


__all__ = ["reference_field_2d", "reference_field_separable"]
