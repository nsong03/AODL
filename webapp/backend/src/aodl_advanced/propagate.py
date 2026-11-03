r"""Pupil -> focal plane by a 2-D zoom (chirp-z) transform; defocus; XZ / YZ slices.

Eq. S11 (``docs/PLAN.md`` §1.3) gives the field behind the objective at defocus ``Z``::

    U(X, Y, Z) ~ int int P(x, y) e^{-i k Z_S11 (x^2+y^2)/(2F^2)} e^{-i k (x X + y Y)/F} dx dy

The pupil ``P`` lives on a uniform, *centred* grid ``x_n = (n - (N_x-1)/2) dx``,
``y_m = (m - (N_y-1)/2) dy`` (:func:`aodl_advanced.pupil.pupil_grid`) and the integral is
evaluated as a Riemann sum on an arbitrary uniform output grid with ``scipy.signal.czt`` along
each axis.  As in ``aodl.check.transform``, the Riemann sum is exact up to *aliasing*: the
sampled pupil reproduces the true image plus copies displaced by ``lambda F / dx`` (171 µm at
the product defaults with ``pupil_n = 256``, ``span_w_in = 2.5``), so a spot deflected beyond
``lambda F / (2 dx)`` folds back — the engine warns about that case.

The same prefactors ``aodl.field.focal`` drops are dropped here (``1/(i lambda F)``, the
propagation phase and the common image curvature), and the Riemann weight ``dx dy`` is kept,
so intensities land on the **same absolute scale** as ``aodl.field.focal.intensity_frame`` —
the cross-engine test exploits that.

The axial sign is read from :func:`aodl.device.conventions.z_s11_from_lab`; nothing here
chooses one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import czt

from aodl.device.conventions import z_s11_from_lab

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]


def _uniform_step(coords: Float, name: str) -> float:
    """Spacing of a uniform grid (``0`` for a single point), refusing a non-uniform one."""
    if coords.ndim != 1 or coords.size < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {coords.shape}")
    if coords.size == 1:
        return 0.0
    step = (coords[-1] - coords[0]) / (coords.size - 1)
    if step == 0.0:
        raise ValueError(f"{name} must be strictly monotonic")
    if np.max(np.abs(np.diff(coords) - step)) > 1e-9 * abs(step):
        raise ValueError(f"{name} must be uniformly spaced for the chirp-z zoom transform")
    return float(step)


def _axis_czt(p: Complex, d: float, out: Float, kappa: float, axis: int) -> Complex:
    """``sum_n p[..., n, ...] exp(-i kappa x_n X_j)`` along ``axis`` on the *centred* grid
    ``x_n = (n - (N-1)/2) d``, for every output coordinate ``X_j`` in ``out``.

    With ``x_n = x_0 + n d`` the sum is ``e^{-i kappa x_0 X_j} sum_n p_n a^{-n} w^{n j}``,
    ``a = e^{i kappa d X_0}``, ``w = e^{-i kappa d dX}`` — a chirp-z transform.
    """
    n = p.shape[axis]
    x0 = -0.5 * (n - 1) * d
    step = _uniform_step(out, "output coordinates")
    moved = np.moveaxis(p, axis, -1)
    if out.size == 1:
        phase = np.exp(-1j * kappa * (x0 + d * np.arange(n)) * float(out[0]))
        transformed = np.sum(moved * phase, axis=-1)[..., None]
    else:
        a = np.exp(1j * kappa * d * float(out[0]))
        w = np.exp(-1j * kappa * d * step)
        transformed = czt(moved, out.size, w, a, axis=-1)
        transformed = transformed * np.exp(-1j * kappa * x0 * out)
    return np.asarray(np.moveaxis(transformed, -1, axis), dtype=np.complex128)


def zoom_dft(
    P: ArrayLike,
    dx: float,
    dy: float,
    X_out: ArrayLike,
    Y_out: ArrayLike,
    k_over_F: float,
) -> Complex:
    """Eq. S11 kernel ``sum_{m,n} P[m, n] exp(-i (k/F)(x_n X + y_m Y)) dx dy`` on a uniform
    output grid, returned with shape ``(len(Y_out), len(X_out))`` (rows = Y, imshow order).

    ``P`` is ``(N_y, N_x)`` on the centred pupil grid (module docstring); leading batch axes
    are carried through.  The defocus factor is *not* included — multiply by
    :func:`defocus_phase` first.
    """
    p = np.asarray(P, dtype=np.complex128)
    x_out = np.asarray(X_out, dtype=np.float64).ravel()
    y_out = np.asarray(Y_out, dtype=np.float64).ravel()
    along_x = _axis_czt(p, float(dx), x_out, float(k_over_F), axis=-1)
    along_y = _axis_czt(along_x, float(dy), y_out, float(k_over_F), axis=-2)
    return np.asarray(along_y * (float(dx) * float(dy)), dtype=np.complex128)


def defocus_phase(x: ArrayLike, y: ArrayLike, z_lab: float, k: float, F: float) -> Complex:
    """``exp(-i k Z_S11 (x^2 + y^2) / (2 F^2))`` with ``Z_S11 = conventions.z_s11_from_lab(z_lab)``.

    ``x`` is a row vector (shape ``(N_x,)``) and ``y`` a column (``(N_y,)``); the result is
    ``(N_y, N_x)``.  Eq. S11's defocus factor, signed by the lab-Z convention of
    :mod:`aodl.device.conventions`.
    """
    z_s11 = float(z_s11_from_lab(float(z_lab)))
    xs = np.asarray(x, dtype=np.float64).ravel()[None, :]
    ys = np.asarray(y, dtype=np.float64).ravel()[:, None]
    return np.asarray(np.exp(-1j * (k * z_s11 / (2.0 * F * F)) * (xs * xs + ys * ys)))


def focal_intensity(
    P: ArrayLike,
    x: Float,
    y: Float,
    X_out: Float,
    Y_out: Float,
    k: float,
    F: float,
    z_lab: float,
) -> Float:
    """``|zoom_dft(P defocus)|^2`` on the output grid at lab plane ``z_lab``, shape ``(ny, nx)``."""
    dx = _uniform_step(np.asarray(x, dtype=np.float64), "pupil x")
    dy = _uniform_step(np.asarray(y, dtype=np.float64), "pupil y")
    pz = np.asarray(P, dtype=np.complex128) * defocus_phase(x, y, z_lab, k, F)
    field = zoom_dft(pz, dx, dy, X_out, Y_out, k / F)
    return np.asarray(np.abs(field) ** 2, dtype=np.float64)


def group_intensity(
    pupils: Sequence[ArrayLike],
    x: Float,
    y: Float,
    X_out: Float,
    Y_out: Float,
    k: float,
    F: float,
    z_lab: float,
) -> Float:
    """``sum_g |zoom_dft(P_g defocus)|^2``: the beat-averaged intensity of DESIGN §1.7.

    Each pupil is one optical-frequency group; groups add in intensity because their beat
    notes average out over any camera or atomic timescale (no sub-time averaging anywhere).
    """
    out = np.zeros((np.size(Y_out), np.size(X_out)), dtype=np.float64)
    for pupil in pupils:
        out += focal_intensity(pupil, x, y, X_out, Y_out, k, F, z_lab)
    return out


#: Defocused pupil planes transformed per chirp-z call (memory cap on the ``(nz, N, N)`` batch).
Z_BLOCK = 8


def _defocus_stack(x: Float, y: Float, zs: Float, k: float, F: float) -> Complex:
    """``(nz, N_y, N_x)`` stack of :func:`defocus_phase` factors, one per lab plane.

    The quadratic phase is separable, ``e^{-i a (x^2 + y^2)} = e^{-i a x^2} e^{-i a y^2}``, so
    the stack is a rank-1 product per plane: ``2 nz N`` complex exponentials instead of
    ``nz N^2`` (the exponential, not the chirp-z transform, was the cost of a focus scan).
    """
    z_s11 = np.asarray(z_s11_from_lab(zs), dtype=np.float64)
    a = (k / (2.0 * F * F)) * z_s11
    ex = np.exp(-1j * a[:, None] * (x * x)[None, :])
    ey = np.exp(-1j * a[:, None] * (y * y)[None, :])
    return np.asarray(ey[:, :, None] * ex[:, None, :], dtype=np.complex128)


def _axial_scan(
    P: ArrayLike,
    x: Float,
    y: Float,
    out_coords: Float,
    z_out: Float,
    fixed: float,
    k: float,
    F: float,
    scan_axis: int,
) -> Float:
    """Shared body of :func:`xz_slice` (``scan_axis = 0``) and :func:`yz_slice` (``1``).

    The planes are processed in blocks of :data:`Z_BLOCK`: each block is one batched chirp-z
    call along the scanned axis for every pupil row/column, then a one-point transform along
    the fixed axis — a defocus scan costs little more than the XY frame itself.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    dx = _uniform_step(xs, "pupil x")
    dy = _uniform_step(ys, "pupil y")
    p = np.asarray(P, dtype=np.complex128)
    zs = np.asarray(z_out, dtype=np.float64).ravel()
    out = np.empty((zs.size, np.size(out_coords)), dtype=np.float64)
    point = np.asarray([float(fixed)])
    for start in range(0, zs.size, Z_BLOCK):
        stop = min(start + Z_BLOCK, zs.size)
        pz = p[None, :, :] * _defocus_stack(xs, ys, zs[start:stop], k, F)
        if scan_axis == 0:
            field = zoom_dft(pz, dx, dy, out_coords, point, k / F)[:, 0, :]
        else:
            field = zoom_dft(pz, dx, dy, point, out_coords, k / F)[:, :, 0]
        out[start:stop] = np.abs(field) ** 2
    return out


def xz_slice(
    P: ArrayLike,
    x: Float,
    y: Float,
    x_out: Float,
    z_out: Float,
    y_row: float,
    k: float,
    F: float,
) -> Float:
    """Intensity on the ``(X, Z_lab)`` plane at ``Y = y_row``, shape ``(nz, nx)``.

    Per ``Z``: apply :func:`defocus_phase`, chirp-z along x for every pupil row, then a
    one-point transform along y at ``y_row`` (batched over Z, :func:`_axial_scan`).
    """
    return _axial_scan(P, x, y, np.asarray(x_out, dtype=np.float64), z_out, y_row, k, F, 0)


def yz_slice(
    P: ArrayLike,
    x: Float,
    y: Float,
    y_out: Float,
    z_out: Float,
    x_col: float,
    k: float,
    F: float,
) -> Float:
    """Intensity on the ``(Y, Z_lab)`` plane at ``X = x_col``, shape ``(nz, ny)`` (the twin of
    :func:`xz_slice`, used for the ``w_y(Z)`` focus scan)."""
    return _axial_scan(P, x, y, np.asarray(y_out, dtype=np.float64), z_out, x_col, k, F, 1)


__all__ = [
    "defocus_phase",
    "focal_intensity",
    "group_intensity",
    "xz_slice",
    "yz_slice",
    "zoom_dft",
]
