r"""Pupil -> focal field by zoom (chirp-z) transform, plus the sub-time schedule.

Eq. S11 (``docs/PLAN.md`` §1.3) says the objective Fourier-transforms the pupil, with defocus
entering as a quadratic aperture phase::

    U(X, Z) ~ int P(u) exp(-i k Z u^2 / (2 F^2)) exp(-i k u X / F) du

The checker evaluates that integral as a **Riemann sum over the aperture grid**, on whatever
image coordinates are asked for, using ``scipy.signal.czt``: a zoom transform reaches an
arbitrary uniform ``X`` grid — a 5 um patch around one trap, say — without the padding an FFT
would need to land the same sampling.  The same two constant-modulus prefactors the simulator
drops (``exp(i k F)`` and the common image-plane curvature) are dropped here, so intensities
and every term-to-term interference match.

**The Riemann sum is not an approximation.**  By Poisson summation, sampling the pupil at
spacing ``du`` and summing gives the exact transform *plus copies displaced by*
``lambda F / du`` in the image plane — 8.2 mm on the ``bragg_band`` grid and 1.4 mm on the
``weak`` grid, against a field of view of tens of microns.  Nothing is being expanded in
``du``; there is no trapezoid error term.  What ``du`` must satisfy is a *spectral*
condition, not an accuracy one: every order that matters must fit inside ``±1/(2 du)``,
which is exactly the ``Lambda / 8`` argument of :mod:`aodl.check.pupil`.

What is left is round-off.  ``scipy.signal.czt`` evaluates the sum through a Bluestein
convolution that raises ``w`` to ``N^2 / 2``, a phase of 2e5 rad on the 24576-cell grid, so
the transform carries a relative error growing with ``N``: measured 7.7e-14 at 4096 cells and
3.8e-13 at 24576 against an extended-precision evaluation of the same sum
(``tests/test_check_transform.py``).  Five orders of magnitude below anything M6 asks of it.

The axial sign is not decided here.  ``z_lab`` is a **lab** coordinate and is converted with
:func:`aodl.device.conventions.z_s11_from_lab`, the single authority for
``Z_S11 = Z_LAB_SIGN Z_lab`` (``docs/conventions.md`` §6).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import czt

from ..device import conventions
from ..params import OpticsParams
from .pupil import ApertureGrid

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]

#: Golden ratio, the low-discrepancy step of :func:`subtimes`.
GOLDEN_RATIO = 0.5 * (1.0 + math.sqrt(5.0))

__all__ = ["GOLDEN_RATIO", "subtimes", "zoom_field"]


def _uniform_step(coords: Float, name: str) -> float:
    """Spacing of a uniform grid, refusing a non-uniform one (the CZT would lie about it)."""
    if coords.ndim != 1 or coords.size < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {coords.shape}")
    if coords.size == 1:
        return 0.0
    step = (coords[-1] - coords[0]) / (coords.size - 1)
    if step == 0.0:
        raise ValueError(f"{name} must be strictly monotonic, got a zero step")
    drift = np.max(np.abs(np.diff(coords) - step))
    if drift > 1e-9 * abs(step):
        raise ValueError(
            f"{name} must be uniformly spaced — the zoom transform evaluates a geometric "
            "progression of image points and would silently return the wrong ones otherwise"
        )
    return float(step)


def zoom_field(
    pupil: ArrayLike,
    grid: ApertureGrid,
    optics: OpticsParams,
    coords: ArrayLike,
    z_lab: float,
) -> Complex:
    """Eq. S11 on a uniform image grid, at lab defocus ``z_lab``.

    Parameters
    ----------
    pupil:
        Complex pupil sampled on ``grid.u``, shape ``(..., grid.n)``.  Leading axes are
        carried through untouched, so a whole batch of sub-times (or of axis pupils) can be
        transformed in one call — which also amortizes the CZT's plan over the batch.
    grid:
        The aperture sampling the pupil lives on.
    optics:
        Wavelength / focal length (``k``, ``F``).
    coords:
        Uniform 1-D image coordinates [m] along this axis.
    z_lab:
        Lab-frame defocus [m], converted to the Eq. S11 variable by
        :func:`aodl.device.conventions.z_s11_from_lab`.

    Returns
    -------
    Complex field of shape ``(..., len(coords))``, with the same dropped prefactors as
    :mod:`aodl.field.reference` (see the module docstring), i.e. one axis factor of

    ``du * sum_n P(u_n) exp(-i k Z_S11 u_n^2 / (2 F^2)) exp(-i k u_n X / F)``.
    """
    p = np.asarray(pupil, dtype=np.complex128)
    if p.shape[-1] != grid.n:
        raise ValueError(
            f"pupil's last axis must match the aperture grid: got {p.shape[-1]}, expected {grid.n}"
        )
    x = np.asarray(coords, dtype=np.float64)
    step = _uniform_step(x, "coords")
    z_s11 = float(conventions.z_s11_from_lab(float(z_lab)))

    focal = optics.focal_length
    kappa = optics.k / focal
    u = grid.u
    p = p * np.exp(-1j * (optics.k * z_s11 / (2.0 * focal * focal)) * u * u)

    a = np.exp(1j * kappa * grid.du * x[0])
    if x.size == 1:
        transformed = np.sum(p * a ** (-np.arange(grid.n)), axis=-1)[..., None]
    else:
        w = np.exp(-1j * kappa * grid.du * step)
        transformed = czt(p, x.size, w, a, axis=-1)
    return np.asarray(grid.du * transformed * np.exp(-1j * kappa * u[0] * x), dtype=np.complex128)


def subtimes(t: float, window: float, k: int) -> Float:
    """``k`` golden-ratio low-discrepancy instants inside ``[t - W/2, t + W/2]``.

    ``t_j = t + W (frac(j phi) - 1/2)``, ``phi`` the golden ratio.  Atoms and cameras see the
    intensity averaged over the MHz beat notes between pupil terms (``docs/PLAN.md`` §1.3), so
    the checker averages over a window instead of trusting a single instant.

    **Why not a uniform grid.**  A uniform set of ``k`` instants spaced ``W/k`` annihilates
    every beat *exactly* — except those commensurate with it, at multiples of ``k/W``, which
    it samples at one single phase and passes through at **full** amplitude.  Which beats
    those are is a property of the drive, and an atom array's comb is regular by construction
    (its beats are integer combinations ``n df_x + m df_y`` of two spacings), so the failure
    set is systematically reachable rather than a coincidence.

    The golden-ratio schedule spreads the residual instead.  It annihilates nothing exactly,
    but for beats up to about ``k/2`` cycles per window it leaves at most ~0.2 of the
    amplitude (measured 0.21 at ``k = 64``), and at the uniform schedule's own failure
    frequencies — the multiples of ``k/W`` — it leaves under 0.05.  It has weak spots of its
    own, at the *Fibonacci* multiples of ``1/W`` (55, 89, 144, 233, ... cycles per window),
    which is the price of ``phi`` being the worst-approximable irrational; those are sparse
    and a regular comb hits them only by accident.  All of this is deterministic, so a
    checker run is exactly reproducible.  Both behaviours are pinned by
    ``tests/test_check_transform.py``.
    """
    count = int(k)
    if count < 1:
        raise ValueError(f"subtimes needs at least one sub-time, got k = {k!r}")
    width = float(window)
    if not math.isfinite(width) or width < 0.0:
        raise ValueError(f"subtimes window must be finite and non-negative, got {window!r}")
    j = np.arange(count, dtype=np.float64)
    frac = np.mod(j * GOLDEN_RATIO, 1.0)
    return np.asarray(float(t) + width * (frac - 0.5), dtype=np.float64)
