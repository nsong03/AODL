r"""Fading-Shepard tone ladders: unbounded axial holds in a bounded RF band (Eqs. S24-S28).

Eq. S19 buys a sustained axial offset with a co-chirp on all four channels, so every
channel's frequency walks at ``fdot_Z = Z / (2 lens_scale)`` for as long as the array stays
off the focal plane.  That walk is what Eq. 1 budgets, and it runs out fast: at the default
hardware ``Z = 10 µm`` exhausts a one-sided 10 MHz headroom in **206 µs**
(:func:`aodl.waveform.synthesis.max_z_integral`).  The hold is not physically expensive —
only the *bookkeeping* is, because a single tone has to carry the whole integral.

The fading-Shepard scheme (paper §IV, Eqs. S24-S28) replaces each channel's single tone by
an infinite **ladder** of tones spaced ``Delta f``, all co-chirping together, and hands the
job over from one rung to the next as the ladder slides:

.. math::

    f_\mu^{(n)}(t) = f_{\text{lat},\mu}(t) + f_Z(t) + (n + \xi_\mu)\,\Delta f,
    \qquad n \in \mathbb{Z}

with ``f_lat = -(v/2 lambda F) X`` on ``Ax`` and ``+(v/2 lambda F) X`` on ``Bx`` (``Y`` for
the ``Ay``/``By`` pair) and the usual ``f_Z(t) = (v^2/2 lambda F^2) int_0^t Z dt'`` — now
*unbounded*, which is the entire point.  What stays bounded is the set of rungs that are
**switched on**: each tone carries the fade window of Eqs. S26/S27 evaluated at its own

.. math::

    g_n(t) = f_Z(t) + (n + \xi_\mu)\,\Delta f            \qquad\text{(the S25 } f_{\mu,Z}^{(n)})

— the frequency *minus* the lateral term — so a rung is live only while ``|g_n|`` sits
inside the window and is silent everywhere else.  As ``f_Z`` grows past ``Delta f`` the
whole pattern of live rungs simply repeats, one index lower: the drive is periodic in
``f_Z`` even though ``f_Z`` is not periodic in anything.  That is Shepard's endlessly
rising tone, and it is why :func:`aodl.waveform.synthesis.synthesize` can hold ``Z``
indefinitely inside a fixed band (:class:`FadeZoneEnvelope`, :func:`shepard_ladder`).

**The fade windows.**  With duty ``eta`` (default ``1/2``), exponent ``p`` and width ``M``,
Eq. S26 (``A`` channels, ``M = 1``) and Eq. S27 (``B`` channels, ``M = M_x`` or ``M_y``) are
the same function of ``|g|``:

.. math::

    A^{(n)} = \begin{cases}
      1 & |g| \le (M-\eta)\Delta f/2\\
      \cos^{p}\theta,\quad
        \theta = \dfrac{\pi}{2\eta}\Big(\dfrac{|g|}{\Delta f} - \dfrac{M}{2}\Big) + \dfrac{\pi}{4}
        & \text{in between}\\
      0 & |g| \ge (M+\eta)\Delta f/2
    \end{cases}

``theta`` runs from ``0`` at the inner boundary to ``pi/2`` at the outer one, so the window
is continuous, equals ``1`` on the plateau, ``2^{-p/2}`` at the fade centre ``|g| = M
Delta f / 2`` and ``0`` outside.  Only the two boundaries and ``p`` matter to the shape:
``k = pi / (2 eta Delta f)`` and ``theta = k(|g| - M Delta f/2) + pi/4``.

**Why ``p_A + p_B = 1`` keeps the light constant.**  Take the x pair and let
``s = g_a in [0, Delta f/2]`` be the fade coordinate of rung ``a``.  Its neighbour ``a-1``
has ``g_{a-1} = s - Delta f``, i.e. ``|g_{a-1}| = Delta f - s``, and the two windows are
mirror images about the fade centre:

    ``theta(s) = k(s - M Delta f/2) + pi/4``   and   ``theta(Delta f - s) = pi/2 - theta(s)``

so with ``theta = theta(s)`` the *dying* rung has ``cos^p theta`` and the *rising* one
``sin^p theta`` on every channel.  A tweezer is a product of one ``A`` line and one ``B``
line (the pupil is a product, Eq. S7), and its position depends only on the index
*difference*, so the two **co-located** combinations are ``(a, a)`` and ``(a-1, a-1)``:

    ``I_total ~ (cos^{p_A}\theta cos^{p_B}\theta)^2 + (sin^{p_A}\theta sin^{p_B}\theta)^2
      = cos^{2(p_A + p_B)}\theta + sin^{2(p_A + p_B)}\theta``

which is ``cos^2 + sin^2 = 1`` — *constant through the hand-over* — exactly when
``p_A + p_B = 1``.  The two co-located pairs are **not** frequency-degenerate (their
``df_opt`` differ by ``2 Delta f``), so they add in intensity, which is what makes the
identity a statement about power rather than about amplitudes.  Table II's two settings
split that unit exponent differently: ``(1/2, 1/2)`` for a single tweezer, ``(1, 0)`` for an
array, where the ``B`` ladder *is* the array ladder and must not be shaped at all.

**Shadow tweezers (Eq. S31).**  The same four lines make two *cross* combinations,
``(A_a, B_{a-1})`` and ``(A_{a-1}, B_a)``, whose index difference is ``-+1``: two extra
traps at ``+- deflection_scale * Delta f``, with relative intensities
``cos^{2 p_A}\theta sin^{2 p_B}\theta`` and ``sin^{2 p_A}\theta cos^{2 p_B}\theta``.  At the
fade centre (``theta = pi/4``) every one of the four combinations carries ``2^{-(p_A+p_B)} =
1/2`` of the main trap's total, so **each shadow peaks at half the main tweezer** — not at a
quarter of it: the shadow amplitude is the *geometric mean* ``(cos sin)^{1/2}`` of the two
co-located products, not their product.  The two shadows are exactly frequency-degenerate
with each other (both tag ``2 f_Z + (2a-1)\Delta f``), which is the static Mach-Zehnder pair
of Fig. S6; they are metres apart in nothing but ``X``, so they land in one
:func:`aodl.field.focal.group_terms` group with two spots.

**Interlacing.**  ``xi`` offsets a channel's ladder by a fraction of ``Delta f``.  Table II
uses ``xi = 0`` on the x pair and ``xi = 1/2`` on the y pair, so with ``eta = 1/2`` and equal
spacings the x fade zones tile exactly the gaps between the y ones: at any instant one axis
is handing over and the other is on its plateau, and the shadow tweezers are therefore
always a single-axis ``+-Delta f`` pair rather than the sixteen-ray grid that simultaneous
fading would produce (``docs/PLAN.md`` §1.3).  Shadows never vanish entirely — with
``eta = 1/2`` some axis is always fading — they simply never appear on both axes at once.

All frequencies here are **detunings** from ``f_center`` (Eq. S2 rotating frame,
``docs/conventions.md`` §1), and every quantity is SI.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import polynomial as npoly
from numpy.typing import ArrayLike, NDArray

from ..params import CHANNELS
from ..poly import PiecewisePoly
from ..trajectory.spec import ArraySpec
from .tones import ChannelWaveform, ToneTrack

#: Amplitude floor of the irising log-derivative clamp (:class:`FadeZoneEnvelope`).  The
#: window's *value* is never clamped; only ``dA``/``d2A`` are evaluated as if the envelope
#: had bottomed out here.  See :class:`FadeZoneEnvelope` for why this is not enough on its
#: own, and :data:`SLOPE_CLAMP` for the bound that does the work at ``p < 1``.
A_FLOOR = 1e-3

#: Slope ceiling of the same clamp, as ``tan theta`` — i.e. the envelope's log-derivative in
#: units of its own fade rate, ``|A'/A| <= p * SLOPE_CLAMP * dtheta/dt``.  ``3`` keeps the
#: shoulder exact out to ``A = 0.56`` (at ``p = 1/2``) and freezes the *shape* — never the
#: amplitude — beyond it.  See :class:`FadeZoneEnvelope`.
SLOPE_CLAMP = 3.0

#: Default fade duty ``eta`` (paper Table II).  ``1/2`` is the interlacing value: the fade
#: zones of a ``xi = 0`` ladder exactly tile the plateaus of a ``xi = 1/2`` one.
ETA_DEFAULT = 0.5

#: Phase conventions accepted by :func:`shepard_ladder` (mirrors
#: :data:`aodl.waveform.synthesis.PHASE_MODES`).
PHASE_MODES: tuple[str, ...] = ("schroeder", "zero", "random")

#: Fraction of the remaining band headroom that :func:`auto_config` gives to the fade
#: window of a free (single-tweezer) axis.  The rest is slack for round-off and for the
#: lateral term's own extrema.
AUTO_FILL = 0.9

#: Slack on the ladder-index bounds of :func:`active_indices`, in index units.  A rung
#: whose ``|g|`` only ever grazes the outer boundary this closely is identically silent.
INDEX_EPS = 1e-9

#: Relative cut for trimming numerically-zero high-order coefficients before root-finding.
_ROOT_TRIM = 1e-12

_TWO_PI = 2.0 * math.pi
_PI_2 = 0.5 * math.pi
_PI_4 = 0.25 * math.pi


def _shaped(values: ArrayLike, t: NDArray[np.float64]) -> NDArray[np.float64] | float:
    """Match :meth:`PiecewisePoly.__call__`: float for scalar input, array otherwise."""
    if t.ndim == 0:
        return float(np.asarray(values))
    return np.asarray(values, dtype=np.float64)


# ============================================================== piecewise-polynomial tools


def _segment_roots(coeffs: NDArray[np.float64], level: float) -> NDArray[np.float64]:
    """Real roots of ``sum_j c_j tau^j = level`` inside ``[0, 1]`` (normalized local time).

    Exact rather than sampled, and deliberately conservative about conditioning: leading
    coefficients that are numerically zero relative to the segment's own scale are trimmed
    before ``polyroots``, which is ill-conditioned about a vanishing leading term.
    """
    c = np.array(coeffs, dtype=np.float64, copy=True)
    c[0] -= float(level)
    scale = float(np.max(np.abs(c), initial=0.0))
    if scale == 0.0:  # identically at the level: no isolated crossing
        return np.zeros(0, dtype=np.float64)
    keep = np.nonzero(np.abs(c) > _ROOT_TRIM * scale)[0]
    if keep[-1] < 1:  # a nonzero constant: never reaches the level
        return np.zeros(0, dtype=np.float64)
    roots = np.asarray(npoly.polyroots(c[: keep[-1] + 1]), dtype=np.complex128)
    real = roots.real[np.abs(roots.imag) <= 1e-9 * (1.0 + np.abs(roots.real))]
    return np.sort(real[(real >= 0.0) & (real <= 1.0)])


def poly_crossings(p: PiecewisePoly, level: float) -> NDArray[np.float64]:
    """Times [s] inside ``p``'s domain where ``p(t) = level`` (sorted, deduplicated).

    Segment-exact: each segment's crossings are the real roots of its own polynomial in
    normalized local time (:func:`_segment_roots`), mapped back to seconds.
    """
    found: list[NDArray[np.float64]] = []
    for k in range(p.n_segments):
        tau = _segment_roots(p.coeffs[k], level)
        if tau.size:
            found.append(p.breaks[k] + tau * (p.breaks[k + 1] - p.breaks[k]))
    if not found:
        return np.zeros(0, dtype=np.float64)
    return np.unique(np.concatenate(found))


def poly_range(p: PiecewisePoly) -> tuple[float, float]:
    """``(min, max)`` of a piecewise polynomial over its own domain — exact, not sampled.

    A polynomial's extrema on a segment are at its endpoints or at a root of its
    derivative, so the candidate set is the breakpoints plus those roots.  Because ``p`` is
    continuous, the returned pair is also exactly the *range* of ``p``, which is what
    :func:`active_indices` needs.
    """
    candidates = [p.breaks]
    for k in range(p.n_segments):
        coeffs = p.coeffs[k]
        deriv = coeffs[1:] * np.arange(1, coeffs.size, dtype=np.float64)
        if deriv.size == 0:
            continue
        tau = _segment_roots(deriv, 0.0)
        if tau.size:
            candidates.append(p.breaks[k] + tau * (p.breaks[k + 1] - p.breaks[k]))
    times = np.concatenate(candidates)
    values = np.asarray(p(times), dtype=np.float64)
    return float(values.min()), float(values.max())


# ========================================================================= the fade window


def fade_window(
    g: ArrayLike,
    delta_f: float,
    eta: float = ETA_DEFAULT,
    p: float = 0.5,
    m: float = 1.0,
    amp: float = 1.0,
) -> NDArray[np.float64] | float:
    r"""The Eq. S26/S27 fade window ``A(g)`` — plateau, ``cos^p`` shoulder, zero.

    ``A = amp`` for ``|g| <= (m - eta) delta_f / 2``, ``0`` for ``|g| >= (m + eta) delta_f /
    2`` and ``amp cos^p(theta)`` in between, with ``theta = (pi / 2 eta)(|g| / delta_f - m /
    2) + pi / 4`` running from ``0`` to ``pi / 2`` across the shoulder.  ``m = 1`` is Eq. S26
    (the ``A`` channels), ``m = M`` Eq. S27 (a ``B`` ladder ``M`` tones wide).

    This is the static window as a function of the fade coordinate; :class:`FadeZoneEnvelope`
    is the same function composed with ``g(t)``.
    """
    values, _, _ = _window_terms(np.abs(np.asarray(g, dtype=np.float64)), delta_f, eta, p, m, amp)
    return _shaped(values, np.asarray(g, dtype=np.float64))


def clamp_floor(p: float) -> float:
    """Smallest ``cos theta`` the irising log-derivatives are evaluated at (see
    :class:`FadeZoneEnvelope`): the stricter of the :data:`A_FLOOR` amplitude rule and the
    :data:`SLOPE_CLAMP` slope rule.  ``p = 0`` has no shoulder shape and returns ``1``."""
    if p <= 0.0:
        return 1.0
    return max(A_FLOOR ** (1.0 / p), 1.0 / math.sqrt(1.0 + SLOPE_CLAMP**2))


def _window_terms(
    u: NDArray[np.float64], delta_f: float, eta: float, p: float, m: float, amp: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""``(A, dA/du, d2A/du2)`` of the fade window at ``u = |g|``, with the log clamp.

    With ``k = pi / (2 eta delta_f)``, ``c = cos theta`` and ``s = sin theta``:

        ``A      = amp c^p``
        ``A'(u)  = -p k s A / c``
        ``A''(u) = p k^2 A [(p - 1) s^2 / c^2 - 1]``

    written so that the only divisions are by ``c``.  Those divisions are what diverge at
    the outer boundary (``c -> 0``): the *log*-derivatives ``A'/A`` and ``A''/A`` — the only
    form :mod:`aodl.device.aodl` ever uses (``docs/conventions.md`` §3) — blow up for every
    ``p > 0``.  ``c`` is therefore floored at

        ``c_floor = max(A_FLOOR^{1/p}, cos(arctan(SLOPE_CLAMP)))``,

    which freezes both log-derivatives at the value they take there, leaving ``A`` itself
    exact and both derivatives continuous and vanishing with it.  The first bound is the
    amplitude rule; the second — the load-bearing one for ``p < 1`` — is the slope rule,
    and :class:`FadeZoneEnvelope` explains why the amplitude rule alone does not converge.

    Outside the shoulder both derivatives are exactly zero (the branch is selected, not
    computed): the plateau is flat and the tone is off.  ``A`` is continuous everywhere;
    ``dA`` has a corner at the outer boundary for ``p <= 1`` and ``d2A`` jumps at both, as it
    does for :class:`~aodl.waveform.tones.SmoothOnOff`.
    """
    k = math.pi / (2.0 * eta * delta_f)
    theta = np.clip(k * (u - 0.5 * m * delta_f) + _PI_4, 0.0, _PI_2)
    c = np.cos(theta)
    s = np.sin(theta)
    inner = 0.5 * (m - eta) * delta_f
    outer = 0.5 * (m + eta) * delta_f
    shoulder = (u > inner) & (u < outer)
    value = np.where(u <= inner, amp, np.where(shoulder, amp * c**p, 0.0))
    if p == 0.0:  # rectangular window: no shoulder shape at all, hence no slope
        zero = np.zeros(value.shape, dtype=np.float64)
        return value, zero, zero
    c_clamped = np.maximum(c, clamp_floor(p))
    first = np.where(shoulder, -p * k * s * value / c_clamped, 0.0)
    second = np.where(
        shoulder, p * k * k * value * ((p - 1.0) * s * s / (c_clamped * c_clamped) - 1.0), 0.0
    )
    return value, first, second


@dataclass(frozen=True)
class FadeZoneEnvelope:
    """The Eq. S26/S27 window evaluated along a tone's fade coordinate ``g(t)``.

    An :class:`~aodl.waveform.tones.Envelope`: ``A(t) = fade_window(g(t))``, with ``dA`` and
    ``d2A`` by the chain rule through ``u = |g|``,

        ``dA  = A'(u) sgn(g) gdot``,
        ``d2A = A''(u) gdot^2 + A'(u) sgn(g) gddot``

    — exact, never numerically differentiated, because ``g`` is a
    :class:`~aodl.poly.PiecewisePoly` and hands out its own derivatives.  The window is
    branch-wise smooth in ``|g|`` and ``|g|`` is branch-wise smooth in ``t``, so the pieces
    are exactly those separated by :attr:`zone_times`; nothing is evaluated across a
    branch (the branch is selected by mask, so a vectorized call over a whole run is one
    pass and still exact).

    Parameters
    ----------
    g:
        Fade coordinate ``f_Z(t) + (n + xi) delta_f`` [Hz] — the tone's frequency *minus*
        its lateral term (Eq. S25).  ``xi`` and the rung index live in its constant term.
    delta_f:
        Ladder spacing [Hz], strictly positive.
    eta:
        Fade duty in ``(0, m]``; ``1/2`` by default (Table II).
    p:
        Window exponent ``>= 0``.  ``p = 0`` is a rectangle (the array ``B`` ladder).
    m:
        Window width in units of ``delta_f``: ``1`` for Eq. S26, ``M`` for Eq. S27.
    amp:
        Peak relative amplitude in ``[0, 1]``.

    Note
    ----
    **The clamp, and why an amplitude floor alone is not enough.**  For ``p < 1`` the
    shoulder's slope ``d cos^p ~ cos^{p-1}`` diverges at the outer edge, and the
    log-derivatives that feed acoustic irising (``alpha1 = -s (A'/A)/v``,
    ``alpha2 = (A''/A)/(2v^2)``, ``docs/conventions.md`` §3) diverge there for *any*
    ``p > 0``.  Both are therefore evaluated at a floored ``cos theta``
    (:func:`clamp_floor`); ``A`` itself is never clamped.

    Flooring the *amplitude* at :data:`A_FLOOR` — the obvious rule, on the argument that a
    line that faint cannot matter — does **not** control the error, because the Eq. S5
    aperture polynomial is *normalized*: a line's weight in the frame is
    ``|c|^2 (1 + (alpha1 w_in)^2/4 + ...) ~ A^2 (1 + (p tan(theta) k gdot w_in / v)^2/4)``
    per axis, and with ``A = cos^p theta`` and ``tan theta ~ A^{-1/p}`` that product scales
    as ``A^{2 - 2/p}`` — *divergent* for ``p < 1``.  At the single-tweezer ``p = 1/2`` and a
    10 µm hold, a line clamped at ``A = 1e-3`` would contribute some ``10^2`` **times** a
    full line's power instead of ``10^-6`` of it.  So the floor that does the work is the
    slope rule :data:`SLOPE_CLAMP`: ``|A'/A| <= p * SLOPE_CLAMP * dtheta/dt`` caps the
    log-derivative at a few times the fade's own rate, which bounds the residual weight at

        ``~ SLOPE_CLAMP * (p (pi/2) rho)^2 / 4``,
        ``rho = (w_in / v) / T_fade``,  ``T_fade = eta * delta_f / |gdot|``

    — the fraction of the fade that happens while the beam crosses one input radius.  That
    is the *physical* validity condition of the whole degree-2 aperture expansion, not an
    artefact of the clamp: a fade much faster than the beam transit apodizes the pupil
    beyond anything a quadratic can describe.  Keep ``rho`` small (a wide ``delta_f``,
    which is also the cheap choice — see :func:`auto_config`) and the clamp is invisible:
    ``rho = 0.05`` bounds it at ``1e-3`` of a line.  Where it does bite it *freezes* the
    apodization shape rather than letting a polynomial extrapolate an envelope that has
    already fallen by a factor ``1/A`` across the beam, which no ``1 + a1 u + a2 u^2``
    can represent.
    """

    g: PiecewisePoly
    delta_f: float
    eta: float = ETA_DEFAULT
    p: float = 0.5
    m: float = 1.0
    amp: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.g, PiecewisePoly):
            raise TypeError(f"FadeZoneEnvelope.g must be a PiecewisePoly, got {type(self.g)!r}")
        values = {}
        for name in ("delta_f", "eta", "p", "m", "amp"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"FadeZoneEnvelope.{name} must be finite, got {value!r}")
            values[name] = value
            object.__setattr__(self, name, value)
        if values["delta_f"] <= 0.0:
            raise ValueError(f"FadeZoneEnvelope.delta_f must be positive, got {self.delta_f!r}")
        if not 0.0 < values["eta"] <= values["m"]:
            raise ValueError(
                f"FadeZoneEnvelope needs 0 < eta <= m (the inner boundary (m - eta) delta_f / 2 "
                f"must not be negative), got eta={self.eta!r}, m={self.m!r}"
            )
        if values["p"] < 0.0:
            raise ValueError(f"FadeZoneEnvelope.p must be non-negative, got {self.p!r}")
        if not 0.0 <= values["amp"] <= 1.0:
            raise ValueError(f"FadeZoneEnvelope.amp must lie in [0, 1], got {self.amp!r}")

    # -- geometry of the window

    @property
    def g_inner(self) -> float:
        """Plateau half-width ``(m - eta) delta_f / 2`` [Hz]: ``A = amp`` inside."""
        return 0.5 * (self.m - self.eta) * self.delta_f

    @property
    def g_centre(self) -> float:
        """Fade centre ``m delta_f / 2`` [Hz], where ``A = amp 2^{-p/2}``."""
        return 0.5 * self.m * self.delta_f

    @property
    def g_outer(self) -> float:
        """Support half-width ``(m + eta) delta_f / 2`` [Hz]: ``A = 0`` outside."""
        return 0.5 * (self.m + self.eta) * self.delta_f

    # -- cached derived polynomials (frozen dataclasses keep a __dict__)

    @functools.cached_property
    def _gdot(self) -> PiecewisePoly:
        return self.g.derivative()

    @functools.cached_property
    def _gddot(self) -> PiecewisePoly:
        return self._gdot.derivative()

    @functools.cached_property
    def g_range(self) -> tuple[float, float]:
        """``(min, max)`` of ``g`` over its domain — exact (:func:`poly_range`)."""
        return poly_range(self.g)

    @property
    def is_active(self) -> bool:
        """Is this tone ever audible, i.e. does ``|g|`` ever enter the outer boundary?"""
        lo, hi = self.g_range
        return lo < self.g_outer and hi > -self.g_outer

    def crossing_times(self, level: float) -> NDArray[np.float64]:
        """Drive times [s] where ``|g(t)| = level`` (sorted; empty when it never does).

        ``level = g_centre`` gives the instants of maximal hand-over (the fade centres),
        ``0`` the plateau centres, and the two boundaries the edges of the shoulder.
        """
        value = abs(float(level))
        found = [poly_crossings(self.g, value)]
        if value > 0.0:
            found.append(poly_crossings(self.g, -value))
        return np.unique(np.concatenate(found))

    @functools.cached_property
    def zone_times(self) -> NDArray[np.float64]:
        """Sorted drive times where ``A(t)`` changes branch — the segmentation of the run.

        The union of the ``|g| = 0``, ``|g| = g_inner`` and ``|g| = g_outer`` crossings: on
        each interval between consecutive entries the envelope is a single smooth formula
        (plateau, one signed shoulder, or off).  Cached, because the root-finding is the
        only non-trivial cost in the envelope and callers (probes, spectrograms, the
        notebook) ask for it repeatedly.
        """
        levels = (0.0, self.g_inner, self.g_outer)
        return np.unique(np.concatenate([self.crossing_times(level) for level in levels]))

    # -- the Envelope protocol

    def _terms(
        self, t: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        g = np.asarray(self.g(t), dtype=np.float64)
        value, first, second = _window_terms(
            np.abs(g), self.delta_f, self.eta, self.p, self.m, self.amp
        )
        return value, first * np.sign(g), second

    def A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Envelope value ``fade_window(g(t))`` (Eqs. S26/S27)."""
        t_arr = np.asarray(t, dtype=np.float64)
        value, _, _ = self._terms(t_arr)
        return _shaped(value, t_arr)

    def dA(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """First derivative ``A'(|g|) sgn(g) gdot`` [1/s] (clamped shoulder, see the class)."""
        t_arr = np.asarray(t, dtype=np.float64)
        _, first, _ = self._terms(t_arr)
        return _shaped(first * np.asarray(self._gdot(t_arr), dtype=np.float64), t_arr)

    def d2A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Second derivative ``A''(|g|) gdot^2 + A'(|g|) sgn(g) gddot`` [1/s^2]."""
        t_arr = np.asarray(t, dtype=np.float64)
        _, first, second = self._terms(t_arr)
        gdot = np.asarray(self._gdot(t_arr), dtype=np.float64)
        gddot = np.asarray(self._gddot(t_arr), dtype=np.float64)
        return _shaped(second * gdot * gdot + first * gddot, t_arr)


# ================================================================== Table II configuration


@dataclass(frozen=True)
class ChannelFade:
    """One channel's fade parameters: window width ``m``, exponent ``p``, offset ``xi``.

    Paper Table II, with ``eta`` carried once on the :class:`ShepardConfig`:

    ============== ============ ============ ============ ============
    configuration  ``Ax``       ``Ay``       ``Bx``       ``By``
    ============== ============ ============ ============ ============
    single tweezer (1, 1/2, 0)  (1, 1/2, ½)  (1, 1/2, 0)  (1, 1/2, ½)
    ``Mx x My``    (1, 1, 0)    (1, 1, ½)    (Mx, 0, 0)   (My, 0, ½)
    ============== ============ ============ ============ ============

    Both rows obey ``p_A + p_B = 1`` per axis (the constant-power identity of the module
    docstring) and interlace the y pair by ``xi = 1/2``.
    """

    m: int
    p: float
    xi: float

    def __post_init__(self) -> None:
        if int(self.m) != self.m or int(self.m) < 1:
            raise ValueError(f"ChannelFade.m must be an integer >= 1, got {self.m!r}")
        object.__setattr__(self, "m", int(self.m))
        for name in ("p", "xi"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"ChannelFade.{name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.p < 0.0:
            raise ValueError(f"ChannelFade.p must be non-negative, got {self.p!r}")


def table_ii(array: ArraySpec) -> dict[str, ChannelFade]:
    """Paper Table II for ``array``: ``{channel: ChannelFade}``, decided per axis.

    An axis carrying a single tweezer (``m = 1``) takes the single-tweezer row — the fade is
    split evenly, ``p_A = p_B = 1/2`` — and an axis carrying a ladder takes the array row,
    where the ``B`` channel *is* the array ladder (``m = M``, unshaped ``p = 0``) and the
    ``A`` channel does all the fading (``p = 1``).  A mixed spec (``1 x My``, say) therefore
    gets the single-tweezer row on x and the array row on y, which is the only reading that
    reduces to Table II on both pure cases.
    """
    fades: dict[str, ChannelFade] = {}
    for a_name, b_name, count, xi in (
        ("Ax", "Bx", array.mx, 0.0),
        ("Ay", "By", array.my, 0.5),
    ):
        if count == 1:
            fades[a_name] = ChannelFade(m=1, p=0.5, xi=xi)
            fades[b_name] = ChannelFade(m=1, p=0.5, xi=xi)
        else:
            fades[a_name] = ChannelFade(m=1, p=1.0, xi=xi)
            fades[b_name] = ChannelFade(m=count, p=0.0, xi=xi)
    return fades


@dataclass(frozen=True)
class ShepardConfig:
    """Ladder spacings, duty and per-channel Table II rows for a fading-Shepard synthesis.

    Parameters
    ----------
    delta_f_x, delta_f_y:
        Ladder spacing [Hz] of the x and y pairs, strictly positive.  **For an array axis
        this is the array's own spacing**: the ``B`` ladder and the Shepard ladder are the
        same object (Eq. S27), so :meth:`resolve` refuses a value that disagrees with the
        :class:`~aodl.trajectory.spec.ArraySpec`.
    eta:
        Fade duty shared by all four channels; ``1/2`` (:data:`ETA_DEFAULT`) interlaces.
    config:
        ``"auto"`` takes :func:`table_ii` from the spec; a mapping ``{channel: ChannelFade}``
        overrides individual channels on top of it (that is how the "simultaneous fading"
        comparison sets ``xi_y = 0``).
    """

    delta_f_x: float
    delta_f_y: float
    eta: float = ETA_DEFAULT
    config: str | Mapping[str, ChannelFade] = "auto"

    def __post_init__(self) -> None:
        for name in ("delta_f_x", "delta_f_y", "eta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"ShepardConfig.{name} must be finite and positive, got {getattr(self, name)!r}"
                )
            object.__setattr__(self, name, value)
        if isinstance(self.config, str):
            if self.config != "auto":
                raise ValueError(
                    f"ShepardConfig.config must be 'auto' or a {{channel: ChannelFade}} mapping, "
                    f"got {self.config!r}"
                )
        elif isinstance(self.config, Mapping):
            unknown = [name for name in self.config if name not in CHANNELS]
            if unknown:
                raise ValueError(
                    f"ShepardConfig.config has unknown channel name(s) {unknown}; "
                    f"valid names are {list(CHANNELS)}"
                )
            for name, fade in self.config.items():
                if not isinstance(fade, ChannelFade):
                    raise TypeError(f"config[{name!r}] must be a ChannelFade, got {fade!r}")
        else:
            raise TypeError(
                f"ShepardConfig.config must be 'auto' or a mapping, got {type(self.config)!r}"
            )

    def spacing(self, channel: str) -> float:
        """Ladder spacing [Hz] of the axis ``channel`` belongs to."""
        if channel not in CHANNELS:
            raise KeyError(f"unknown channel {channel!r}; valid names are {list(CHANNELS)}")
        return self.delta_f_x if channel.endswith("x") else self.delta_f_y

    def resolve(self, array: ArraySpec) -> dict[str, ChannelFade]:
        """The per-channel Table II rows for ``array``, with any overrides applied.

        Raises ``ValueError`` when a ladder axis' spacing disagrees with the array's own:
        the ``B`` ladder of Eq. S27 *is* the array ladder, so one number describes both and
        two different numbers describe nothing.
        """
        for axis, count, array_spacing, spacing in (
            ("x", array.mx, array.delta_f_x, self.delta_f_x),
            ("y", array.my, array.delta_f_y, self.delta_f_y),
        ):
            if count > 1 and not math.isclose(spacing, abs(array_spacing), rel_tol=1e-12):
                raise ValueError(
                    f"ShepardConfig.delta_f_{axis} = {spacing!r} Hz conflicts with the array's "
                    f"own delta_f_{axis} = {array_spacing!r} Hz: for an m{axis} = {count} ladder "
                    f"the Shepard ladder and the array ladder are the same tones (Eq. S27), so "
                    f"they must share one spacing.  Pass delta_f_{axis}={abs(array_spacing)!r} "
                    f"or change the ArraySpec."
                )
        fades = table_ii(array)
        if isinstance(self.config, Mapping):
            fades.update(self.config)
        return fades


def auto_config(
    array: ArraySpec,
    lateral_span: Mapping[str, float],
    headroom: Mapping[str, float],
    eta: float = ETA_DEFAULT,
) -> ShepardConfig:
    """Pick ladder spacings that fit the band, given each channel's lateral excursion.

    A tone is only ever driven where its window is open, so a channel's whole occupancy is

        ``|f| <= max|f_lat| + (m + eta) delta_f / 2``

    (:func:`shepard_band_bound`).  For an *array* axis ``delta_f`` is already fixed by the
    :class:`~aodl.trajectory.spec.ArraySpec`; for a free axis this hands the fade window
    :data:`AUTO_FILL` of whatever headroom the lateral term leaves, which is the widest
    ladder that still fits — and the widest ladder is the cheapest one, since the rung count
    goes as ``(f_Z span) / delta_f``.

    Parameters
    ----------
    array:
        Geometry, for the ladder spacings that are already decided.
    lateral_span:
        ``{channel: max |f_lat|}`` [Hz].
    headroom:
        ``{channel: distance from f_center to the nearer band edge}`` [Hz].
    eta:
        Fade duty.
    """
    fades = table_ii(array)
    spacing: dict[str, float] = {}
    for axis, a_name, b_name, count, array_spacing in (
        ("x", "Ax", "Bx", array.mx, array.delta_f_x),
        ("y", "Ay", "By", array.my, array.delta_f_y),
    ):
        if count > 1:
            spacing[axis] = abs(float(array_spacing))
            continue
        budgets = [
            AUTO_FILL * (headroom[name] - lateral_span[name]) / (0.5 * (fades[name].m + eta))
            for name in (a_name, b_name)
        ]
        best = min(budgets)
        if not best > 0.0:
            raise ValueError(
                f"no room for a Shepard ladder on the {axis} axis: the lateral term alone "
                f"already spends the band headroom "
                f"({max(lateral_span[a_name], lateral_span[b_name]):.4g} Hz of "
                f"{min(headroom[a_name], headroom[b_name]):.4g} Hz).  Reduce the lateral "
                f"excursion or widen the band."
            )
        spacing[axis] = best
    return ShepardConfig(delta_f_x=spacing["x"], delta_f_y=spacing["y"], eta=eta)


def shepard_band_bound(fade: ChannelFade, delta_f: float, eta: float, lateral_span: float) -> float:
    r"""Largest ``|f|`` a channel's *live* tones ever reach [Hz] — the Shepard claim.

    A rung is driven only while ``|g| < (m + eta) delta_f / 2``, and its frequency is
    ``f = f_lat + g``, so

    .. math::

        \max_{\text{live}} |f_\mu^{(n)}| \le \frac{(M+\eta)\Delta f}{2} + \max|f_{\text{lat}}|

    **however large** ``int Z dt`` — hence ``f_Z`` — grows.  That is the whole claim of the
    scheme, and it is a property of the window, not of the trajectory: the ladder slides, the
    live window does not.  (The frequency *laws* of individual rungs are unbounded; they are
    simply multiplied by zero outside the window, and never launched by the transducer.)
    """
    return 0.5 * (fade.m + eta) * delta_f + abs(lateral_span)


# =============================================================================== the ladder


def ladder_phases(indices: ArrayLike, m: int) -> NDArray[np.float64]:
    r"""Schroeder phases of a tone ladder, by rung index.  **Eq. S28.**

    .. math::

        \varphi^{(n)} = \operatorname{mod}\!\Big(\frac{2\pi\,n(n-1)}{2M},\ 2\pi\Big)

    The generalization of Eq. S23 to a ladder that is *indexed* rather than counted: ``n`` is
    the rung's own integer index (negative rungs included, as a sliding Shepard ladder
    requires), and ``M`` is the ladder's width, so neighbouring rungs keep the quadratic
    progression that spreads the drive's crest and scatters the IM3 contributions
    (:mod:`aodl.waveform.synthesis`).  ``M = 1`` gives all-zero phases, because ``n(n-1)/2``
    is an integer.

    The reduction is applied to ``n(n-1)/2`` — exactly an integer for an integer rung index —
    *before* scaling by ``2 pi / M``, which is algebraically the formula above but keeps a
    whole number of turns landing on exactly ``0``.  Reducing afterwards rounds such a rung to
    ``2 pi - 7e-15`` instead (``M = 1``, ``n = 6``): the same phase physically, but it would
    make the ``M = 1`` claim above false and put the result outside ``[0, 2 pi)``.
    """
    if int(m) < 1:
        raise ValueError(f"ladder width m must be a positive integer, got {m!r}")
    width = float(int(m))
    n = np.asarray(indices, dtype=np.float64)
    return _TWO_PI * np.mod(0.5 * n * (n - 1.0), width) / width


def active_indices(
    f_z: PiecewisePoly, fade: ChannelFade, delta_f: float, eta: float = ETA_DEFAULT
) -> NDArray[np.int64]:
    r"""The rungs that are audible at some instant of the run — the *finite* ladder.

    Rung ``n`` is live iff ``|f_Z(t) + (n + xi) delta_f| < (m + eta) delta_f / 2`` for some
    ``t``.  ``f_Z`` is continuous, so its range is exactly ``[min, max]``
    (:func:`poly_range`) and the condition is the open index interval

        ``(-g_out - max f_Z) / delta_f - xi  <  n  <  (g_out - min f_Z) / delta_f - xi``,

    whose width is ``(f_Z span) / delta_f + (m + eta)``: the ladder grows one rung per
    ``delta_f`` of axial integral, plus a constant for the window itself.  Everything
    outside is identically silent and is never built.
    """
    lo, hi = poly_range(f_z)
    g_out = 0.5 * (fade.m + eta) * delta_f
    lower = (-g_out - hi) / delta_f - fade.xi
    upper = (g_out - lo) / delta_f - fade.xi
    candidates = np.arange(math.floor(lower) - 1, math.ceil(upper) + 2, dtype=np.int64)
    keep = (candidates > lower + INDEX_EPS) & (candidates < upper - INDEX_EPS)
    indices = candidates[keep]
    if indices.size == 0:  # degenerate window grazing a boundary: keep the nearest rung
        indices = np.asarray([int(round(0.5 * (lower + upper)))], dtype=np.int64)
    return indices


def _resolve_ladder_phases(
    phases: str | ArrayLike,
    indices: NDArray[np.int64],
    m: int,
    rng: np.random.Generator | None,
) -> NDArray[np.float64]:
    """Turn the ``phases`` argument of :func:`shepard_ladder` into one phase per rung [rad]."""
    if isinstance(phases, str):
        if phases == "schroeder":
            return ladder_phases(indices, m)
        if phases == "zero":
            return np.zeros(indices.size, dtype=np.float64)
        if phases == "random":
            generator = np.random.default_rng() if rng is None else rng
            return np.asarray(generator.uniform(0.0, _TWO_PI, size=indices.size), dtype=np.float64)
        raise ValueError(f"phases must be one of {PHASE_MODES} or an array, got {phases!r}")
    values = np.asarray(phases, dtype=np.float64).ravel()
    if values.size != indices.size:
        raise ValueError(
            f"explicit phases must have one entry per active rung: {values.size} != {indices.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("explicit phases must all be finite")
    return values


def shepard_ladder(
    fade: ChannelFade,
    delta_f: float,
    lateral: PiecewisePoly,
    f_z: PiecewisePoly,
    *,
    eta: float = ETA_DEFAULT,
    amp: float = 1.0,
    phases: str | ArrayLike = "schroeder",
    rng: np.random.Generator | None = None,
) -> ChannelWaveform:
    r"""One channel's fading-Shepard ladder (Eqs. S24-S28).

    Every live rung (:func:`active_indices`) becomes a
    :class:`~aodl.waveform.tones.ToneTrack` whose frequency law is

        ``f^{(n)}(t) = f_lat(t) + f_Z(t) + (n + xi) delta_f``

    and whose envelope is the :class:`FadeZoneEnvelope` on ``g^{(n)} = f_Z + (n + xi)
    delta_f`` — the same polynomial minus the lateral term, which is why a lateral move does
    *not* shift the fade schedule.

    Parameters
    ----------
    fade:
        This channel's Table II row.
    delta_f:
        Ladder spacing [Hz] of the channel's axis.
    lateral, f_z:
        Eq. S19's ``-+ v P(t) / (2 lambda F)`` and ``f_Z(t)`` [Hz], on one common domain
        (already including whatever hold tail the caller wants: the envelopes read ``f_z``
        directly, so extending it here is what keeps ``A(t)`` defined over the tail).
    eta, amp, phases, rng:
        Fade duty, peak amplitude, and the ladder's phase convention — ``"schroeder"``
        (Eq. S28 by rung index), ``"zero"``, ``"random"`` or an explicit per-rung array.

    Returns
    -------
    A :class:`~aodl.waveform.tones.ChannelWaveform` with one tone per live rung, ordered by
    rung index.
    """
    if lateral.domain != f_z.domain:
        raise ValueError(
            f"the lateral term and f_Z must share one domain, got {lateral.domain} and {f_z.domain}"
        )
    indices = active_indices(f_z, fade, delta_f, eta)
    phi = _resolve_ladder_phases(phases, indices, fade.m, rng)
    tones = []
    for n, phase0 in zip(indices, phi, strict=True):
        g = f_z.offset((float(n) + fade.xi) * delta_f)
        tones.append(
            ToneTrack(
                freq=lateral + g,
                env=FadeZoneEnvelope(
                    g=g, delta_f=delta_f, eta=eta, p=fade.p, m=fade.m, amp=float(amp)
                ),
                phase0=float(phase0),
            )
        )
    return ChannelWaveform(tuple(tones))


__all__ = [
    "AUTO_FILL",
    "A_FLOOR",
    "ETA_DEFAULT",
    "INDEX_EPS",
    "PHASE_MODES",
    "SLOPE_CLAMP",
    "ChannelFade",
    "FadeZoneEnvelope",
    "ShepardConfig",
    "active_indices",
    "auto_config",
    "clamp_floor",
    "fade_window",
    "ladder_phases",
    "poly_crossings",
    "poly_range",
    "shepard_band_bound",
    "shepard_ladder",
    "table_ii",
]
