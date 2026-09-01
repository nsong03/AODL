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

**Two powers, two questions.**  :attr:`SpotMetrics.power` adds the group's terms in
*intensity* and :attr:`SpotMetrics.power_coherent` adds them in *amplitude* first (the exact
Gram form of :func:`_coherent_power`).  Terms in one group are degenerate, so the coherent
number is the physical one; the two differ only when degenerate terms actually overlap in the
image plane, which is exactly the Fig. S6 shadow-tweezer situation the fading-Shepard scheme
is built to avoid (``docs/PLAN.md`` §1.3).  ``power`` stays the default weight everywhere —
it is the per-spot bookkeeping quantity, insensitive to where a group's members sit — and
``power_coherent`` is what to ask when the question is "how much light is really there".
"""

from __future__ import annotations

from collections.abc import Sequence
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
from .gaussian import (
    gauss_moments,
    gauss_moments_lower,
    gauss_moments_upper,
    gauss_moments_window,
)

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]
Index = NDArray[np.intp]


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
        group redistribute this power spatially but are not included here — this is the
        *incoherent* sum, and it is what every weighting in this package uses.
    power_coherent:
        ``int int |sum_n U_n|^2 dX dY`` over the same group: the terms added as *amplitudes*
        first, cross-terms included (:func:`_coherent_power`).  Equal to ``power`` for a
        one-term group, and to it within round-off whenever the group's members are further
        apart than their pupil overlap allows; smaller (down to zero) when co-located
        degenerate terms interfere destructively, larger when they add.  This is the number
        the rendered frame integrates to, and the one the Fig. S6 static Mach-Zehnder pair of
        a simultaneously fading AODL makes phase-dependent.
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
    power_coherent: float
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


def _pair_indices(groups: Sequence[Index]) -> tuple[Index, Index, Index]:
    """Every ``(j, k)`` pair inside every group, plus the group each pair belongs to.

    The Gram sum is over ordered pairs (both ``(j, k)`` and ``(k, j)``), which is what makes
    the accumulated result real by construction rather than by taking a real part of half of
    it.  Groups are small — a group is a set of *exactly* degenerate terms — so the ``m^2``
    is paid on a handful of members, while flattening every group's pairs into one array is
    what keeps the closed forms below a single vectorized call per axis.
    """
    if not groups:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty, empty
    j = np.concatenate([np.repeat(idx, idx.size) for idx in groups])
    k = np.concatenate([np.tile(idx, idx.size) for idx in groups])
    owner = np.concatenate(
        [np.full(idx.size * idx.size, g, dtype=np.intp) for g, idx in enumerate(groups)]
    )
    return j.astype(np.intp), k.astype(np.intp), owner


def _edge_value(power: int, u: Float, a: Complex, b: Complex) -> Complex:
    """``u^power exp(-a u^2 + b u)`` at a finite bound, ``0`` where the bound is infinite.

    The boundary term of the integration-by-parts recursion in :func:`_cross_moments`; an
    infinite bound contributes nothing because ``Re(a) > 0`` kills the integrand there.
    """
    finite = np.isfinite(u)
    safe = np.where(finite, u, 0.0)
    value = safe**power * np.exp(-a * safe * safe + b * safe)
    return np.asarray(np.where(finite, value, 0.0), dtype=np.complex128)


def _cross_moments(a: Complex, b: Complex, lo: Float, hi: Float) -> Complex:
    r"""Moments ``M_m = int_lo^hi u^m e^{-a u^2 + b u} du``, ``m = 0..4``, shape ``(5, n)``.

    The complex, two-sided generalization of :func:`_window_moments`: ``b`` no longer
    vanishes, because a *cross* term between two pupils carries their deflection difference
    (:func:`_pupil_overlap_axis`).  ``m = 0, 1, 2`` come from :mod:`aodl.field.gaussian` —
    full line, one edge, or a window, selected exactly as :func:`aodl.field.focal._axis_field`
    selects them — and the two beyond its ``I_2`` follow from integrating
    ``d/du [u^{m-1} e^{-a u^2 + b u}]`` over the same interval:

        ``2 a M_m = [lo^{m-1} g(lo) - hi^{m-1} g(hi)] + (m - 1) M_{m-2} + b M_{m-1}``,

    with ``g(u) = e^{-a u^2 + b u}`` and the boundary terms dropped at an infinite bound
    (:func:`_edge_value`).  One formula therefore covers all four fill states, and it
    reproduces the ``E_1``/``E_2`` recursion of :func:`aodl.field.gaussian.gauss_moments_lower`
    at ``m = 1, 2``.
    """
    size = np.broadcast(a, b, lo, hi).size
    m0 = np.zeros(size, dtype=np.complex128)
    m1 = np.zeros(size, dtype=np.complex128)
    m2 = np.zeros(size, dtype=np.complex128)
    bounded_lo, bounded_hi = np.isfinite(lo), np.isfinite(hi)
    branches = (
        (~bounded_lo & ~bounded_hi, lambda m: gauss_moments(a[m], b[m])),
        (bounded_lo & ~bounded_hi, lambda m: gauss_moments_lower(a[m], b[m], lo[m])),
        (~bounded_lo & bounded_hi, lambda m: gauss_moments_upper(a[m], b[m], hi[m])),
        (bounded_lo & bounded_hi, lambda m: gauss_moments_window(a[m], b[m], lo[m], hi[m])),
    )
    for mask, evaluate in branches:
        if np.any(mask):
            m0[mask], m1[mask], m2[mask] = evaluate(mask)

    two_a = 2.0 * a
    m3 = (_edge_value(2, lo, a, b) - _edge_value(2, hi, a, b) + 2.0 * m1 + b * m2) / two_a
    m4 = (_edge_value(3, lo, a, b) - _edge_value(3, hi, a, b) + 3.0 * m2 + b * m3) / two_a
    return np.stack([m0, m1, m2, m3, m4])


def _pupil_overlap_axis(
    alpha: Complex,
    theta1: Float,
    theta2: Float,
    lo: Float,
    hi: Float,
    j: Index,
    k: Index,
    w_in: float,
) -> Complex:
    r"""One axis' pupil overlap ``O(j, k) = int p_j(u) conj(p_k(u)) du``, per pair.

    With the pupil of :mod:`aodl.field.focal` (its ``a``/``b`` mapping at image coordinate
    ``0``), ``p_j(u) = (alpha0 + alpha1 u + alpha2 u^2)_j exp(-a_j u^2 + i theta1_j u)``, so

        ``p_j conj(p_k) = [poly_j conj(poly_k)](u) exp(-(a_j + conj(a_k)) u^2 + b_jk u)``,
        ``a_j + conj(a_k) = 2 / w_in^2 - i (theta2_j - theta2_k)``,
        ``b_jk = i (theta1_j - theta1_k)``.

    The defocus ``Z`` sits inside ``a_j`` with a *common* sign for every term, so it cancels
    in ``a_j + conj(a_k)``: the overlap — and hence the coherent power — is independent of the
    plane it is evaluated at, as power conservation requires.  ``poly_j conj(poly_k)`` is kept
    to its full degree 4 (:func:`_hermitian_cross`), which is what the field path's
    ``|alpha0 I0 + alpha1 I1 + alpha2 I2|^2`` amounts to and what makes ``O(j, j)`` the
    incoherent :func:`_pupil_power_axis` exactly.

    The integration window is the *intersection* of the two terms' filled apertures; a pair
    whose windows do not overlap contributes nothing.
    """
    a = np.asarray(2.0 / w_in**2 - 1j * (theta2[j] - theta2[k]), dtype=np.complex128)
    b = np.asarray(1j * (theta1[j] - theta1[k]), dtype=np.complex128)
    window_lo = np.maximum(lo[j], lo[k])
    window_hi = np.minimum(hi[j], hi[k])
    empty = window_lo >= window_hi
    if np.any(empty):  # pragma: no cover - build_terms drops such a frame entirely
        window_lo = np.where(empty, -np.inf, window_lo)
        window_hi = np.where(empty, np.inf, window_hi)
    r = _hermitian_cross(alpha[:, j], alpha[:, k])
    overlap = (r * _cross_moments(a, b, window_lo, window_hi)).sum(axis=0)
    return np.asarray(np.where(empty, 0.0, overlap), dtype=np.complex128)


def _hermitian_cross(alpha_j: Complex, alpha_k: Complex) -> Complex:
    """Coefficients of ``poly_j(u) conj(poly_k(u))``, degree 4, shape ``(5, n)``.

    The cross-term generalization of the Hermitian square in :func:`_pupil_power_axis`, to
    which it reduces (coefficient by coefficient) at ``j = k``.
    """
    a0, a1, a2 = alpha_j
    b0, b1, b2 = np.conj(alpha_k)
    return np.stack(
        [
            a0 * b0,
            a0 * b1 + a1 * b0,
            a0 * b2 + a1 * b1 + a2 * b0,
            a1 * b2 + a2 * b1,
            a2 * b2,
        ]
    )


def _coherent_power(terms: TermLike, optics: OpticsParams, groups: Sequence[Index]) -> Float:
    r"""Per-group ``int int |sum_n U_n|^2 dX dY`` — the exact Gram form, shape ``(n_groups,)``.

    The same Parseval identity :func:`_term_power` uses, applied to a *pair* of terms: with
    ``field_j(X) = int p_j(u) e^{-i k u X / F} du``,

        ``int field_j conj(field_k) dX = lambda F int p_j conj(p_k) du``,

    so a group's coherent power is the Gram sum

    .. math::

        P = (\lambda F)^2 \sum_{j,k} c_j c_k^* \, O_x(j,k)\, O_y(j,k)

    over its members, with the per-axis pupil overlaps of :func:`_pupil_overlap_axis`.  The
    diagonal is exactly :func:`_term_power`, so the result is the incoherent
    :attr:`SpotMetrics.power` plus the cross-terms.  The sum is real because the ordered pairs
    come in conjugate couples (:func:`_pair_indices`); the imaginary part is discarded at
    round-off level.

    Two terms interfere only where their pupils overlap, and ``O`` carries that: the overlap
    of two spots ``Delta X`` apart is suppressed by ``exp(-(k w_in Delta X / (2 F))^2 / 2)``,
    i.e. by their far-field separation in waists.  So co-located degenerate terms (the static
    Mach-Zehnder pair of Fig. S6, an IM3 product landing on its own fundamental) interfere
    fully, while the ``+- deflection_scale Delta f`` shadow tweezers — degenerate with each
    other but tens of microns apart — do not, and their coherent power is their incoherent one.
    """
    n_groups = len(groups)
    if n_groups == 0:
        return np.zeros(0, dtype=np.float64)
    n = _n_terms(terms)
    j, k, owner = _pair_indices(groups)
    theta1 = np.asarray(terms.theta1, dtype=np.float64)
    theta2 = np.asarray(terms.theta2, dtype=np.float64)
    alpha = np.asarray(terms.alpha, dtype=np.complex128)
    c = np.asarray(terms.c, dtype=np.complex128).ravel()
    edge = getattr(terms, "edge", None)

    # On the diagonal the Gram weight *is* ``|c|^2``: writing it that way rather than as
    # ``c conj(c)`` costs nothing, removes a cancellation, and — the reason it matters — makes
    # the result independent of the term's overall phase, which a static drive advances from
    # one frame to the next while nothing physical changes.
    same = j == k
    weight = np.asarray(
        np.where(same, np.abs(c[j]) ** 2 + 0.0j, c[j] * np.conj(c[k])), dtype=np.complex128
    )
    gram = weight * (optics.wavelength * optics.focal_length) ** 2
    for axis in range(2):
        lo, hi = _axis_bounds(edge, axis, n)
        gram = gram * _pupil_overlap_axis(
            alpha[axis], theta1[axis], theta2[axis], lo, hi, j, k, optics.w_in
        )
    contribution = np.asarray(np.real(gram), dtype=np.float64)
    return np.asarray(
        np.bincount(owner, weights=contribution, minlength=n_groups), dtype=np.float64
    )


def measure(terms: TermLike, optics: OpticsParams, tol: float = GROUP_TOL) -> list[SpotMetrics]:
    """Analytic metrics, one :class:`SpotMetrics` per frequency group (Table I quantities).

    Groups come from :func:`aodl.field.focal.group_terms` (default 1 kHz tolerance), so the
    list matches the groups :func:`aodl.field.focal.intensity_frame` renders coherently — and
    each record carries both readings of that group's light: the incoherent
    :attr:`SpotMetrics.power` and the exact Gram :attr:`SpotMetrics.power_coherent`
    (:func:`_coherent_power`), which is the integral of the group's own rendered frame.
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

    groups = group_terms(terms, tol)
    coherent = _coherent_power(terms, optics, groups)

    out: list[SpotMetrics] = []
    for group, idx in enumerate(groups):
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
                power_coherent=float(coherent[group]),
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
