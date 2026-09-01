r"""Closed-form spot metrics per frequency group (paper Table I quantities).

One :class:`SpotMetrics` record per interference group of :func:`aodl.field.focal.group_terms`
— position, best-focus lab Z, astigmatic interval, 1/e^2 radii and power — all evaluated
analytically from the pupil terms.  Nothing here fits a rendered frame: every number follows
from the ``(a, b)`` mapping of :mod:`aodl.field.focal` and from Table I,

    X = deflection_scale (f_Bx - f_Ax),   Zbar = 1/2 lens_scale sum fdot,
    Delta F = lens_scale (fdot_Ax + fdot_Bx - fdot_Ay - fdot_By),

which in pupil variables read ``X_c = theta1_x F / k`` and
``Z_axis,lab = Z_LAB_SIGN 2 F^2 theta2_axis / k`` (the per-axis focus), so that
``Zbar = (Z_x + Z_y)/2`` and ``Delta F = Z_x - Z_y``.

Group aggregation uses power weights ``|c|^2`` (the intensity centroid a camera would report);
for the single-term case, which is what the tests pin, weighting is irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..params import OpticsParams
from .focal import (
    GROUP_TOL,
    Z_LAB_SIGN,
    TermLike,
    _axis_bounds,
    _n_terms,
    group_terms,
)
from .gaussian import gauss_moments_lower

Float = NDArray[np.float64]


@dataclass(frozen=True)
class SpotMetrics:
    """Analytic metrics of one frequency group ("one tweezer"), SI units.

    Attributes
    ----------
    x, y:
        Lab position of the spot [m], power-weighted over the group's terms.
    z_lab:
        Best-focus **lab** Z [m]: the mean of the two per-axis foci, i.e. Table I's ``Zbar``.
    delta_f:
        Astigmatic interval ``Z_x,lab - Z_y,lab`` [m] — Table I's ``Delta F``; zero for the
        paper's astigmatism-free 3D control.
    sigma_astig:
        ``delta_f / z_R``, the paper's dimensionless astigmatism ``sigma_astig``.
    wx, wy:
        Intensity 1/e^2 radii [m] at ``z_lab`` (:func:`aodl.field.focal.spot_params`).
    power:
        ``sum_n |c_n|^2 int int |U_n|^2 dX dY`` over the group, on the same (prefactor-dropped)
        scale as :func:`aodl.field.focal.intensity_frame`.  Interference cross-terms inside a
        group redistribute this power spatially but are not included here.
    df_opt:
        Optical frequency tag of the group [Hz] (power-weighted mean of its members).
    """

    x: float
    y: float
    z_lab: float
    delta_f: float
    sigma_astig: float
    wx: float
    wy: float
    power: float
    df_opt: float


def _tail_moments(lam: float, edge: Float) -> Float:
    r"""Real tail moments ``T_m = int_edge^{+inf} u^m e^{-lam u^2} du``, ``m = 0..4``, ``(5, n)``.

    The two infinite bounds are the two fill states that need no wavefront: ``edge = -inf``
    returns the full-line moments (the odd ones vanish by symmetry) and ``edge = +inf``
    returns zero, so *any* window — full aperture, one-sided or two-sided — is the difference
    :func:`_window_moments` takes.

    The finite branch reuses the stable ``erfcx`` form of
    :func:`aodl.field.gaussian.gauss_moments_lower` (``b = 0``, so the result is real) and the
    integration-by-parts recursion

        T_m = u0^{m-1} e^{-lam u0^2} / (2 lam) + (m - 1)/(2 lam) T_{m-2}

    for the two moments beyond its ``E_2``.
    """
    n = edge.size
    full0 = np.sqrt(np.pi / lam)
    out = np.zeros((5, n), dtype=np.float64)
    unbounded = np.isneginf(edge)
    out[0] = np.where(unbounded, full0, 0.0)
    out[2] = np.where(unbounded, full0 / (2.0 * lam), 0.0)
    out[4] = np.where(unbounded, 3.0 * full0 / (4.0 * lam**2), 0.0)

    finite = np.isfinite(edge)
    if not np.any(finite):
        return out
    u0 = edge[finite]
    e0, e1, e2 = (np.real(m) for m in gauss_moments_lower(lam, 0.0, u0))
    g0 = np.exp(-lam * u0 * u0)
    e3 = u0**2 * g0 / (2.0 * lam) + (2.0 / (2.0 * lam)) * e1
    e4 = u0**3 * g0 / (2.0 * lam) + (3.0 / (2.0 * lam)) * e2
    out[0, finite], out[1, finite], out[2, finite] = e0, e1, e2
    out[3, finite], out[4, finite] = e3, e4
    return out


def _window_moments(lam: float, lo: Float, hi: Float) -> Float:
    r"""Real moments ``M_m = int_lo^hi u^m e^{-lam u^2} du``, ``m = 0..4``, shape ``(5, n)``.

    ``M_m = T_m(lo) - T_m(hi)`` (:func:`_tail_moments`) — the ``b = 0``, five-moment analogue
    of :func:`aodl.field.gaussian.gauss_moments_window`, and it inherits that function's
    cancellation caveat: when both bounds sit far out on the same tail the difference loses
    relative digits, but such a window passes essentially no light and the absolute error
    stays at roundoff on the full-aperture scale the term is compared against.

    A fully filled axis is ``(-inf, +inf)`` and a one-sided fill has one infinite bound, so
    the same expression covers every fill state (``docs/conventions.md`` §7).
    """
    return _tail_moments(lam, lo) - _tail_moments(lam, hi)


def _pupil_power_axis(alpha: NDArray[np.complex128], lo: Float, hi: Float, w_in: float) -> Float:
    r"""``S = int |p(u)|^2 du`` per term for one axis, over the *filled* aperture ``[lo, hi]``.

    ``p(u) = (a0 + a1 u + a2 u^2) exp(-u^2/w_in^2)`` (the pupil phases drop out of ``|p|^2``), so
    ``|p|^2 = (r0 + r1 u + r2 u^2 + r3 u^3 + r4 u^4) e^{-2u^2/w_in^2}`` with real ``r`` from the
    Hermitian square of the amplitude polynomial, and ``S = sum_m r_m M_m`` with the windowed
    moments of :func:`_window_moments`.

    The bounds come from :func:`aodl.field.focal._axis_bounds`, so the **two-sided** window a
    counter-propagating pair leaves while both crystals are filling (``tau/2 <= t < tau``,
    ``docs/conventions.md`` §7) is integrated over as such.  Integrating that case as the
    half-line ``u >= lo`` instead would count light the second wavefront has not delivered yet
    — a factor 2.2 on one windowed axis at ``t = 0.55 tau`` (4.9 with both axes windowed).

    Together with Parseval this gives the exact term power (see :func:`measure`).
    """
    lam = 2.0 / w_in**2
    a0, a1, a2 = alpha[0], alpha[1], alpha[2]
    r = np.stack(
        [
            np.abs(a0) ** 2,
            2.0 * np.real(a0 * np.conj(a1)),
            np.abs(a1) ** 2 + 2.0 * np.real(a0 * np.conj(a2)),
            2.0 * np.real(a1 * np.conj(a2)),
            np.abs(a2) ** 2,
        ]
    )
    return np.einsum("mn,mn->n", r, _window_moments(lam, lo, hi))


def _term_power(terms: TermLike, optics: OpticsParams) -> Float:
    r"""Per-term ``int int |U|^2 dX dY``, shape ``(n_terms,)``.

    Parseval for the Eq. S11 kernel: with ``field(X) = int p(u) e^{-i k u X / F} du`` (the
    quadratic defocus phase is unimodular and drops out),

        int |field(X)|^2 dX = (2 pi F / k) int |p(u)|^2 du = lambda F int |p|^2 du,

    hence ``int int |U|^2 = |c|^2 (lambda F)^2 S_x S_y`` — exact, including the amplitude
    polynomial (acoustic irising) and a partially filled aperture (one-sided or the two-sided
    window of a filling counter-propagating pair), and independent of Z as power conservation
    requires.  For an unwindowed ``alpha = (1, 0, 0)`` term this equals the familiar
    ``peak * wx * wy * pi/2``.
    """
    n = _n_terms(terms)
    alpha = np.asarray(terms.alpha, dtype=np.complex128)
    c = np.asarray(terms.c, dtype=np.complex128).ravel()
    edge = getattr(terms, "edge", None)
    s = np.ones(n, dtype=np.float64)
    for axis in range(2):
        lo, hi = _axis_bounds(edge, axis, n)
        s *= _pupil_power_axis(alpha[axis], lo, hi, optics.w_in)
    return np.abs(c) ** 2 * (optics.wavelength * optics.focal_length) ** 2 * s


def measure(terms: TermLike, optics: OpticsParams, tol: float = GROUP_TOL) -> list[SpotMetrics]:
    """Analytic metrics, one :class:`SpotMetrics` per frequency group (Table I quantities).

    Groups come from :func:`aodl.field.focal.group_terms` (default 1 kHz tolerance), so the
    list matches the groups :func:`aodl.field.focal.intensity_frame` renders coherently.
    """
    n = _n_terms(terms)
    if n == 0:
        return []
    k = optics.k
    focal = optics.focal_length
    theta1 = np.asarray(terms.theta1, dtype=np.float64)
    theta2 = np.asarray(terms.theta2, dtype=np.float64)
    df_opt = np.asarray(terms.df_opt, dtype=np.float64).ravel()
    weight = np.abs(np.asarray(terms.c, dtype=np.complex128).ravel()) ** 2
    power = _term_power(terms, optics)

    # Per-term, per-axis best-focus lab Z, hoisted out of the group loop.  Each group needs
    # its members' 1/e^2 radii *at its own* best-focus plane, and `focal.spot_params` derives
    # that dependence in closed form: `w(Z) = waist0 sqrt(1 + ((Z - Z_focus)/z_R)^2)` about
    # `Z_focus = Z_LAB_SIGN 2 F^2 theta2 / k`.  Evaluating it here costs one pass over the
    # terms instead of one per group (`spot_params` re-derives *every* term's radius each
    # time it is called, so the loop was O(groups x terms) for an O(terms) quantity).
    z_focus = Z_LAB_SIGN * 2.0 * focal**2 * theta2 / k
    waist0, rayleigh = optics.waist0, optics.rayleigh

    out: list[SpotMetrics] = []
    for idx in group_terms(terms, tol):
        w = weight[idx]
        total = float(w.sum())
        w = w / total if total > 0.0 else np.full(idx.size, 1.0 / idx.size)

        th1 = theta1[:, idx] @ w
        th2 = theta2[:, idx] @ w
        z_axis_lab = Z_LAB_SIGN * 2.0 * focal**2 * th2 / k
        z_lab = float(0.5 * (z_axis_lab[0] + z_axis_lab[1]))
        delta_f = float(z_axis_lab[0] - z_axis_lab[1])
        defocus = (z_lab - z_focus[:, idx]) / rayleigh
        radii = waist0 * np.sqrt(1.0 + defocus**2)

        out.append(
            SpotMetrics(
                x=float(th1[0] * focal / k),
                y=float(th1[1] * focal / k),
                z_lab=z_lab,
                delta_f=delta_f,
                sigma_astig=delta_f / rayleigh,
                wx=float(radii[0] @ w),
                wy=float(radii[1] @ w),
                power=float(power[idx].sum()),
                df_opt=float(df_opt[idx] @ w),
            )
        )
    return out


def track_z(metrics: list[SpotMetrics]) -> float:
    """Power-weighted mean best-focus lab Z [m] — the movie's auto-tracked plane.

    Zero for an empty list or a scene with no power (nothing to track).
    """
    if not metrics:
        return 0.0
    power = np.array([m.power for m in metrics], dtype=np.float64)
    z_lab = np.array([m.z_lab for m in metrics], dtype=np.float64)
    total = float(power.sum())
    if not total > 0.0:
        return float(np.mean(z_lab))
    return float(power @ z_lab / total)


__all__ = ["SpotMetrics", "measure", "track_z"]
