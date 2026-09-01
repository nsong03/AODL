r"""Closed-form 1D integrals ``\int u^n exp(-a u^2 + b u) du`` (``docs/PLAN.md`` §1.3).

Every pupil term of the AODL stack is (polynomial) x (Gaussian) x exp(linear + quadratic
phase) and separable in x and y, so each focal-field axis factor is a combination of the
moments implemented here (Eq. S11 evaluated in closed form — no FFTs anywhere).

``a`` packs the input-beam radius, the chirp lens and the defocus Z; ``b`` packs the
deflection and the image coordinate.  Both are complex with ``Re(a) > 0``; every function
is vectorized over numpy broadcasting and returns ``complex128``.

Four families:

* full line     ``I_n(a, b)        = \int_{-inf}^{+inf} u^n e^{-a u^2 + b u} du``
* lower edge    ``E_n(a, b, u0)    = \int_{u0}^{+inf}   u^n e^{-a u^2 + b u} du``
* upper edge    ``F_n(a, b, u1)    = \int_{-inf}^{u1}   u^n e^{-a u^2 + b u} du``
* window        ``W_n(a, b, u0, u1) = \int_{u0}^{u1}    u^n e^{-a u^2 + b u} du``

The edge families model a partially filled aperture (the acoustic wavefront still
entering the crystal), and are computed through the scaled complementary error function
``erfcx`` so that they stay finite where the naive ``exp(b^2/4a) * erfc(w)`` product
overflows or underflows to garbage.

The edge itself is not an optical aperture: it is the acoustic fill boundary
``s u <= v t - D/2`` of Eq. S4's retarded time (``docs/conventions.md`` §7), which windows
the Eq. S11 pupil integral to the half-line the sound has reached.  The input beam is
modelled as an uncropped Gaussian (``docs/PLAN.md`` §1.5, decision 2), so this is the only
window the field integrals ever see.

The two-sided ``W_n`` is what a **counter-propagating pair** needs: while both crystals of
one axis are still filling, each contributes its own half-line and the light that gets
through sees their intersection ``[D/2 - v t, v t - D/2]`` — empty until ``t = tau/2``, the
whole aperture at ``t = tau``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import wofz

Complex = NDArray[np.complex128]


def _as_complex(x: ArrayLike) -> Complex:
    return np.asarray(x, dtype=np.complex128)


def _check_a(a: Complex) -> None:
    if not np.all(np.real(a) > 0.0):
        raise ValueError("Re(a) > 0 is required for the Gaussian moments to converge")


def _squeeze(x: Complex) -> Complex:
    """Return a numpy scalar for 0-d results, the array itself otherwise."""
    return x[()]


def erfcx_complex(z: ArrayLike) -> Complex:
    """Scaled complementary error function ``erfcx(z) = exp(z^2) erfc(z)`` for complex ``z``.

    Implemented as ``wofz(1j z)`` (Faddeeva function): ``wofz(x) = exp(-x^2) erfc(-i x)``,
    hence ``wofz(1j z) = exp(z^2) erfc(z)``.  For ``Re(z) >= 0`` this decays like
    ``1/(z sqrt(pi))`` instead of overflowing, which is the whole point.
    """
    return _squeeze(np.asarray(wofz(1j * _as_complex(z)), dtype=np.complex128))


def gauss_moments(a: ArrayLike, b: ArrayLike) -> tuple[Complex, Complex, Complex]:
    r"""Full-line moments ``I_0, I_1, I_2`` of ``u^n exp(-a u^2 + b u)`` (``docs/PLAN.md`` §1.3).

    ``I0 = sqrt(pi/a) exp(b^2/4a)``, ``I1 = (b/2a) I0``,
    ``I2 = (1/2a + b^2/4a^2) I0`` — the ``c0, c1, c2`` prefactors of a polynomial pupil
    amplitude multiply these directly.
    """
    a = _as_complex(a)
    b = _as_complex(b)
    _check_a(a)
    i0 = np.sqrt(np.pi / a) * np.exp(b * b / (4.0 * a))
    i1 = (b / (2.0 * a)) * i0
    i2 = (1.0 / (2.0 * a) + b * b / (4.0 * a * a)) * i0
    return _squeeze(i0), _squeeze(i1), _squeeze(i2)


def gauss_moments_lower(
    a: ArrayLike, b: ArrayLike, u0: ArrayLike
) -> tuple[Complex, Complex, Complex]:
    r"""Lower-edge moments ``E_n(a, b, u0) = \int_{u0}^{+inf} u^n e^{-a u^2 + b u} du``.

    The Eq. S11 axis integral over an aperture filled only for ``u >= u0`` — the
    ``sound_sign = -1`` fill window of Eq. S4 / ``docs/conventions.md`` §7.

    Completing the square with ``w = sqrt(a) u0 - b/(2 sqrt(a))`` (principal square root)
    and ``g0 = exp(-a u0^2 + b u0)`` (the integrand at the edge, bounded in all physical
    cases) gives, using the identity ``b^2/4a - w^2 = -a u0^2 + b u0``,

        Re(w) >= 0:  E0 = 0.5 sqrt(pi/a) erfcx(w) g0
        Re(w) <  0:  E0 = I0(a, b) - 0.5 sqrt(pi/a) erfcx(-w) g0      [erfc(w) = 2 - erfc(-w)]

    The branch is selected with ``np.where`` on ``Re(w)`` and ``erfcx`` is only ever
    evaluated in its decaying half-plane, so the naive ``exp(b^2/4a) * erfc(w)`` product
    (which overflows/underflows for the physically relevant ``|b^2/4a| ~ 10^5``) is never
    formed.  Integration by parts then gives

        E1 = g0/(2a) + (b/2a) E0
        E2 = u0 g0/(2a) + E0/(2a) + (b/2a) E1
    """
    a = _as_complex(a)
    b = _as_complex(b)
    u0 = _as_complex(u0)
    _check_a(a)
    sqrt_a = np.sqrt(a)
    w = sqrt_a * u0 - b / (2.0 * sqrt_a)
    g0 = np.exp(-a * u0 * u0 + b * u0)
    pref = 0.5 * np.sqrt(np.pi / a)
    positive = np.real(w) >= 0.0
    # Always evaluate erfcx in the right half-plane, where it is bounded and decaying.
    tail = pref * erfcx_complex(np.where(positive, w, -w)) * g0
    i0 = np.sqrt(np.pi / a) * np.exp(b * b / (4.0 * a))
    e0 = np.where(positive, tail, i0 - tail)
    e1 = g0 / (2.0 * a) + (b / (2.0 * a)) * e0
    e2 = u0 * g0 / (2.0 * a) + e0 / (2.0 * a) + (b / (2.0 * a)) * e1
    return _squeeze(e0), _squeeze(e1), _squeeze(e2)


def gauss_moments_upper(
    a: ArrayLike, b: ArrayLike, u1: ArrayLike
) -> tuple[Complex, Complex, Complex]:
    r"""Upper-edge moments ``F_n(a, b, u1) = \int_{-inf}^{u1} u^n e^{-a u^2 + b u} du``.

    The same Eq. S11 axis integral for the ``sound_sign = +1`` fill window (content at
    ``u <= u1``, ``docs/conventions.md`` §7).

    By the reflection ``u -> -u``: ``F_n(a, b, u1) = (-1)^n E_n(a, -b, -u1)``.
    """
    a = _as_complex(a)
    b = _as_complex(b)
    u1 = _as_complex(u1)
    f0, f1, f2 = gauss_moments_lower(a, -b, -u1)
    return _squeeze(np.asarray(f0)), _squeeze(-np.asarray(f1)), _squeeze(np.asarray(f2))


def gauss_moments_window(
    a: ArrayLike, b: ArrayLike, u0: ArrayLike, u1: ArrayLike
) -> tuple[Complex, Complex, Complex]:
    r"""Window moments ``W_n(a, b, u0, u1) = \int_{u0}^{u1} u^n e^{-a u^2 + b u} du``.

    The Eq. S11 axis integral over an aperture filled **only between two wavefronts** — the
    window a counter-propagating pair leaves while both acoustic columns are still entering
    their crystals (``docs/conventions.md`` §7).  Channel ``sound_sign = -1`` fills
    ``u >= D/2 - v t`` and channel ``+1`` fills ``u <= v t - D/2``, so the light crossing
    both sees ``[D/2 - v t, v t - D/2]``: empty for ``t < tau/2`` (a point only transmits
    once *both* waves have reached it), the whole aperture at ``t = tau``.

    Written as a difference of the lower-edge family,

        W_n(a, b, u0, u1) = E_n(a, b, u0) - E_n(a, b, u1),

    so it inherits the ``erfcx`` scaling — and hence the range safety — of
    :func:`gauss_moments_lower`.  ``u0 < u1`` is required (compared on the real parts, the
    bounds being physical aperture coordinates): an empty or reversed window has no moments,
    and its term is dark, which :func:`aodl.device.aodl.build_terms` handles by dropping the
    term rather than by calling this function.

    Note
    ----
    **Cancellation caveat.**  The difference is benign when the window contains, or is
    bounded by, the integrand's peak.  When *both* edges sit far out on the same tail the
    two ``E_n`` are nearly equal — each close to the full-line ``I_n`` on the left tail, or
    each exponentially small on the right — and the subtraction loses digits: the *absolute*
    error stays at roundoff times the larger operand (at worst ``~eps |I_n|``), so the
    *relative* error of ``W_n`` grows as ``|I_n| / |W_n|``.

    This is harmless in the optical model, because such a window passes essentially no
    light: the term it belongs to is negligible next to any fully filled one, and the error
    it carries is bounded by ``eps`` times the full-aperture term it is being compared
    against — never by ``eps`` times something big and visible.  ``tests/test_window.py``
    pins the crossover: agreement with quadrature holds to 1e-9 relative wherever
    ``|W0| > 1e-12 |I0|``, and degrades only below that, where the absolute error stays
    ``< 1e-13 |I0|``.
    """
    a = _as_complex(a)
    b = _as_complex(b)
    lo = _as_complex(u0)
    hi = _as_complex(u1)
    _check_a(a)
    if not np.all(np.real(lo) < np.real(hi)):
        raise ValueError(
            "gauss_moments_window needs u0 < u1: an empty or reversed aperture window has "
            "no moments (its term is dark and should be dropped instead)"
        )
    e0, e1, e2 = gauss_moments_lower(a, b, lo)
    f0, f1, f2 = gauss_moments_lower(a, b, hi)
    w0 = np.asarray(e0) - np.asarray(f0)
    w1 = np.asarray(e1) - np.asarray(f1)
    w2 = np.asarray(e2) - np.asarray(f2)
    return _squeeze(w0), _squeeze(w1), _squeeze(w2)


__all__ = [
    "erfcx_complex",
    "gauss_moments",
    "gauss_moments_lower",
    "gauss_moments_upper",
    "gauss_moments_window",
]
