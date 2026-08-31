r"""Closed-form Gaussian moments vs numerical quadrature, plus numerical-stability pins.

The closed forms in ``aodl.field.gaussian`` are exact; ``scipy.integrate.quad`` is not.
For strongly oscillatory draws the integrand can be many orders of magnitude larger than
its integral (``|b^2/4a|`` up to ~140 inside the work order's draw box), and no
double-precision quadrature can resolve that.  The random sweep therefore keeps the work
order's distributions but **rejects draws whose cancellation ratio**

    cond = (\int |integrand| du) / |I0| = sqrt(pi/Re a) exp(Re(b)^2 / 4 Re a) / |I0|

exceeds ``COND_MAX`` (73% of draws survive), which keeps quad's own error a factor ~25
below the required 1e-9.  The badly conditioned corner is covered instead by the explicit
stability cases below, where the closed form is checked against an exponentially rescaled
quadrature that has no cancellation problem.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad
from scipy.special import erfc

from aodl.field.gaussian import (
    erfcx_complex,
    gauss_moments,
    gauss_moments_lower,
    gauss_moments_upper,
)
from aodl.units import mm

N_DRAWS = 200
COND_MAX = 1e4
RTOL = 1e-9


def _quad_moment(a, b, n, lo, hi, scale=1.0):
    """``\\int_lo^hi u^n exp(-a u^2 + b u) du`` by quadrature (real and imag parts apart).

    ``scale`` divides the integrand (and multiplies the result back) so that extreme
    exponents can be kept inside double range.  Oscillatory integrands routinely trip
    QUADPACK's roundoff/subdivision warnings while still converging well past the
    tolerances asserted here, so those warnings are silenced deliberately.
    """

    def integrand(u, part):
        value = u**n * np.exp(-a * u * u + b * u) / scale
        return value.real if part == 0 else value.imag

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        re, _ = quad(integrand, lo, hi, args=(0,), limit=2000, epsabs=0.0, epsrel=1e-12)
        im, _ = quad(integrand, lo, hi, args=(1,), limit=2000, epsabs=0.0, epsrel=1e-12)
    return (re + 1j * im) * scale


def _draw(rng):
    """One work-order draw: a = 10^U(-2,2) e^{i theta}, |b| <= 10 sqrt|a|, u0 in [-3,3]/sqrt|a|."""
    a = 10 ** rng.uniform(-2.0, 2.0) * np.exp(1j * np.deg2rad(rng.uniform(-80.0, 80.0)))
    b = 10.0 * np.sqrt(abs(a)) * rng.uniform(0.0, 1.0) * np.exp(1j * rng.uniform(0.0, 2 * np.pi))
    u0 = rng.uniform(-3.0, 3.0) / np.sqrt(abs(a))
    return a, b, u0


def test_erfcx_matches_definition(rng):
    """erfcx(z) = exp(z^2) erfc(z) wherever the naive product is representable."""
    z = rng.uniform(-2.0, 2.0, size=64) + 1j * rng.uniform(-2.0, 2.0, size=64)
    naive = np.exp(z * z) * erfc(z)
    np.testing.assert_allclose(erfcx_complex(z), naive, rtol=1e-11, atol=0.0)
    # ... and stays bounded far out in the right half-plane, where the naive form dies.
    far = erfcx_complex(np.array([40.0, 200.0 - 300.0j]))
    assert np.all(np.isfinite(far))
    assert far[0] == pytest.approx(1.0 / (40.0 * np.sqrt(np.pi)), rel=1e-3)


def test_moments_match_quadrature(rng):
    """I_n, E_n and F_n against quad over >= 200 well-conditioned random draws."""
    accepted = 0
    attempts = 0
    while accepted < N_DRAWS:
        attempts += 1
        assert attempts < 20 * N_DRAWS, "draw rejection is out of control"
        a, b, u0 = _draw(rng)
        moments_i = gauss_moments(a, b)
        l1 = np.sqrt(np.pi / a.real) * np.exp(b.real**2 / (4.0 * a.real))
        if l1 / abs(moments_i[0]) > COND_MAX:
            continue
        accepted += 1

        width = 1.0 / np.sqrt(a.real)
        peak = b.real / (2.0 * a.real)
        lo = min(peak - 14.0 * width, u0 - 14.0 * width)
        hi = max(peak + 14.0 * width, u0 + 14.0 * width)
        moments_e = gauss_moments_lower(a, b, u0)
        moments_f = gauss_moments_upper(a, b, u0)
        for n in range(3):
            for got, ref in (
                (moments_i[n], _quad_moment(a, b, n, lo, hi)),
                (moments_e[n], _quad_moment(a, b, n, u0, hi)),
                (moments_f[n], _quad_moment(a, b, n, lo, u0)),
            ):
                assert got == pytest.approx(ref, rel=RTOL)
    assert accepted == N_DRAWS


def test_edge_moments_split_the_line(rng):
    """E_n(a, b, u) + F_n(a, b, u) = I_n(a, b) — exact, and independent of the erfc branch."""
    for _ in range(50):
        a, b, u0 = _draw(rng)
        moments_i = gauss_moments(a, b)
        moments_e = gauss_moments_lower(a, b, u0)
        moments_f = gauss_moments_upper(a, b, u0)
        for n in range(3):
            scale = max(abs(moments_i[n]), abs(moments_e[n]), abs(moments_f[n]))
            assert abs(moments_e[n] + moments_f[n] - moments_i[n]) < 1e-13 * scale


def test_stability_physical_scale():
    """Physical AODL scale: the naive exp(b^2/4a) erfc(w) product is nan; erfcx is exact.

    ``a ~ 2.5e5 - 3.7e5j m^-2`` is (1/w_in^2, chirp + defocus) for a 2 mm beam and
    ``b = i 1e6 m^-1`` is a few-waist deflection; the edge sits at the aperture rim.
    Here ``b^2/4a ~ -3.1e5 - 4.6e5j``, so ``exp(b^2/4a)`` underflows to 0 while
    ``erfc(w)`` overflows: their product is nan, and the moments would be lost entirely.
    """
    a = 2.5e5 - 3.7e5j
    b = 1e6j
    for u0 in (2.0 * mm, -2.0 * mm):
        w = np.sqrt(a) * u0 - b / (2.0 * np.sqrt(a))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            naive = 0.5 * np.sqrt(np.pi / a) * np.exp(b * b / (4.0 * a)) * erfc(w)
        assert not np.isfinite(naive), "naive form is expected to fail here"

        moments_e = gauss_moments_lower(a, b, u0)
        moments_f = gauss_moments_upper(a, b, u0)
        assert np.all(np.isfinite(np.array(moments_e)))
        assert np.all(np.isfinite(np.array(moments_f)))

        width = 1.0 / np.sqrt(a.real)
        hi = max(u0, 0.0) + 14.0 * width
        lo = min(u0, 0.0) - 14.0 * width
        for n in range(3):
            assert moments_e[n] == pytest.approx(_quad_moment(a, b, n, u0, hi), rel=1e-8)
            assert moments_f[n] == pytest.approx(_quad_moment(a, b, n, lo, u0), rel=1e-8)


def test_stability_large_exponent():
    """|b^2/4a| ~ 700: everything stays finite and E_n + F_n = I_n still holds exactly.

    The quadrature reference is rescaled by the edge value ``g0`` so that the *reference*
    (not the closed form) is the thing kept in range.
    """
    a, b, u0 = 1.0, 53.0, 40.0
    assert abs(b * b / (4.0 * a)) == pytest.approx(702.25)

    moments_i = gauss_moments(a, b)
    moments_e = gauss_moments_lower(a, b, u0)
    moments_f = gauss_moments_upper(a, b, u0)
    assert np.all(np.isfinite(np.array(moments_i)))
    assert np.all(np.isfinite(np.array(moments_e)))
    assert np.all(np.isfinite(np.array(moments_f)))

    g0 = np.exp(-a * u0 * u0 + b * u0)
    for n in range(3):
        assert moments_e[n] + moments_f[n] == pytest.approx(moments_i[n], rel=1e-12)
        ref = _quad_moment(a, b, n, u0, u0 + 40.0, scale=g0)
        assert moments_e[n] == pytest.approx(ref, rel=1e-10)


def test_broadcasting_and_domain_checks():
    a = np.array([[1.0 + 0.5j], [2.0 - 1.0j]])
    b = np.array([0.3j, -0.7 + 0.2j, 1.1])
    i0, i1, i2 = gauss_moments(a, b)
    assert i0.shape == i1.shape == i2.shape == (2, 3)
    e0, e1, e2 = gauss_moments_lower(a, b, np.array([[0.2], [-0.4]]))
    assert e0.shape == e1.shape == e2.shape == (2, 3)
    # scalar in -> numpy scalar out
    assert np.ndim(gauss_moments(1.0, 0.5)[0]) == 0

    for bad in (-1.0 + 0.0j, 0.0 + 1.0j):
        with pytest.raises(ValueError, match="Re\\(a\\)"):
            gauss_moments(bad, 0.0)
        with pytest.raises(ValueError, match="Re\\(a\\)"):
            gauss_moments_lower(bad, 0.0, 0.0)
        with pytest.raises(ValueError, match="Re\\(a\\)"):
            gauss_moments_upper(bad, 0.0, 0.0)
