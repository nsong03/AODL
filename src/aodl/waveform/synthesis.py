r"""Waveform synthesis: array ladders, Schroeder phases, and the Eq. S19 solver.

:func:`synthesize` is the front door — a :class:`~aodl.trajectory.spec.TrajectorySpec` in,
a band-checked four-channel :class:`~aodl.waveform.tones.WaveformSet` out — and it is built
from two reusable pieces that also stand on their own:

* **the ladder** (:func:`array_tones`) — ``M`` equally spaced constant tones
  ``f_n = center + (n - (M-1)/2) delta_f``, which Table I maps to a row of tweezers at pitch
  ``deflection_scale * delta_f`` (``lambda F Delta f / v``, 10.3 µm per MHz at the default
  hardware).  Crossing such a ladder on ``Ax`` with one on ``Ay`` gives the ``Mx x My``
  array of ``docs/PLAN.md`` §3 (M2), because the pupil is a *product* (Eq. S7);
* **the common ramp** (:func:`add_common_ramp`) — the same frequency law added to every tone
  of a channel, which translates the whole ladder rigidly (Eq. S19's ``+ v X(t) / (2 lambda
  F)`` term) without changing its internal spacing, and carries the ``f_Z`` co-chirp.

**Why the phases matter.**  All ``M`` tones share one crystal, so the drive is their sum and
the transmission ``exp(i C V)`` mixes them: third-order intermodulation puts ghost lines at
``f_j + f_k - f_i`` (:mod:`aodl.device.mixing`, Eqs. S20-S22).  How much light ends up in
those ghosts depends on the tone phases, because IM3 amplitudes carry
``exp(-i(phi_j + phi_k - phi_i))`` and neighbouring contributions to one ghost either add or
cancel.  Setting every phase to zero is the worst case — all tones crest together, the drive
has an ``M``-fold peak, and the ghost contributions add in phase.  The **Schroeder phases**
of :func:`schroeder_phases` (Eq. S23, generalized in Eq. S28) spread the crest into a
chirp-like waveform and scatter the ghost contributions, which is what makes them the
package default for array ladders.

**Time zero and the retardation lag.**  A synthesized waveform's time axis starts at ``t = 0``
— the instant the first sample enters the transducer.  The acoustic sample that illuminates
the beam centre left the transducer ``tau/2`` earlier, so the tweezers reproduce the requested
trajectory *delayed by half an aperture transit*: the atom-plane response at observation time
``t`` is what the spec asked for at ``t - tau/2`` (``docs/conventions.md`` §7, 5.8 µs at the
default hardware).  **That lag is the default**: the drive is written exactly as Eq. S19
states it, and every comparison — tests, notebooks, :func:`aodl.engine.simulate` — evaluates
the requested profile at ``t - tau/2``.  ``synthesize(..., retard_compensate=True)`` reads the
trajectory ``tau/2`` ahead instead (:func:`_advanced`), so the atom plane matches the request
at ``t``; it does not, and cannot, remove the fill transient the first ``tau`` of any run
genuinely has, and it starts the array half a transit into its first move rather than at rest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.polynomial import polynomial as npoly
from numpy.typing import ArrayLike, NDArray

from ..params import CHANNELS, AODLParams
from ..poly import PiecewisePoly
from ..trajectory.spec import TrajectorySpec
from ..units import MHz, ms, um, us
from .shepard import (
    ChannelFade,
    ShepardConfig,
    auto_config,
    ladder_phases,
    poly_range,
    shepard_band_bound,
    shepard_ladder,
)
from .tones import TIME_TOL, ChannelWaveform, ConstantEnvelope, ToneTrack, WaveformSet

#: Default programmed span of :func:`array_tones` [s] when no ``t1`` is given.  A static
#: ladder has nothing to schedule, so the span exists only to satisfy the
#: :class:`~aodl.waveform.tones.WaveformSet` "all tones cover the same span" rule and the
#: :func:`aodl.engine.simulate` coverage check; 1 ms is ~87 aperture transits, long enough
#: for any single-frame or short-move M2 scene.  Pass ``t1`` (or extend afterwards with
#: :meth:`~aodl.waveform.tones.ChannelWaveform.with_hold_until`) for a longer run.
DEFAULT_SPAN = 1.0 * ms

#: Phase conventions accepted by :func:`array_tones`.
PHASE_MODES: tuple[str, ...] = ("schroeder", "zero", "random")

#: Default tail of :func:`synthesize`, in acoustic transit times ``tau``: the drive is held
#: at its terminal frequency for this long past the end of the trajectory, so a simulation
#: can still probe the last requested instant (which needs drive time ``t - tau/2``) and see
#: the array come to rest.
T_PAD_TRANSITS = 2.0

#: Slack [Hz] on the band comparison of :func:`synthesize`.  Sub-hertz round-off on a
#: megahertz-scale extremum is not a band violation.
BAND_TOL = 1.0

#: Relative cut for trimming numerically-zero high-order coefficients before root-finding
#: (:func:`_poly_extrema`); ``polyroots`` is ill-conditioned about a vanishing leading term.
_ROOT_TRIM = 1e-12

_TWO_PI = 2.0 * math.pi


def schroeder_phases(n_tones: int) -> NDArray[np.float64]:
    r"""Schroeder phases of an ``M``-tone equal-amplitude ladder [rad].  Eq. S23/S28.

    .. math::

        \varphi_n = \operatorname{mod}\!\Big(\frac{2\pi\, n (n-1)}{2M},\; 2\pi\Big),
        \qquad n = 0 \ldots M-1.

    The quadratic phase progression makes the sum of the tones sweep in frequency instead of
    cresting all at once: the drive's peak amplitude grows like ``sqrt(M)`` rather than
    ``M``, so at a fixed peak RF power each tone can be driven harder, and — the reason this
    module cares — the IM3 contributions to any one ghost line arrive with scattered phases
    instead of adding coherently (``docs/PLAN.md`` §1.2).

    This is :func:`aodl.waveform.shepard.ladder_phases` — the one implementation of Eq. S28
    in the package — evaluated on the ``M`` rungs ``n = 0 … M-1`` of a *counted* ladder; the
    Shepard ladders of M4 call it on their own (signed) rung indices instead.

    Returns
    -------
    ``(M,)`` array of phases in ``[0, 2 pi)``; ``M = 0`` gives an empty array.
    """
    count = int(n_tones)
    if count < 0:
        raise ValueError(f"n_tones must be non-negative, got {n_tones!r}")
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    return ladder_phases(np.arange(count, dtype=np.int64), count)


def _resolve_phases(
    phases: str | ArrayLike,
    n_tones: int,
    rng: np.random.Generator | None,
) -> NDArray[np.float64]:
    """Turn the ``phases`` argument of :func:`array_tones` into an ``(M,)`` array [rad]."""
    if isinstance(phases, str):
        if phases == "schroeder":
            return schroeder_phases(n_tones)
        if phases == "zero":
            return np.zeros(n_tones, dtype=np.float64)
        if phases == "random":
            generator = np.random.default_rng() if rng is None else rng
            return np.asarray(generator.uniform(0.0, _TWO_PI, size=n_tones), dtype=np.float64)
        raise ValueError(f"phases must be one of {PHASE_MODES} or an array, got {phases!r}")
    values = np.asarray(phases, dtype=np.float64).ravel()
    if values.size != n_tones:
        raise ValueError(
            f"explicit phases must have one entry per tone: {values.size} != {n_tones}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("explicit phases must all be finite")
    return values


def array_tones(
    n_tones: int,
    delta_f: float,
    center: float = 0.0,
    amp: float = 1.0,
    phases: str | ArrayLike = "schroeder",
    t0: float = 0.0,
    t1: float | None = None,
    rng: np.random.Generator | None = None,
) -> ChannelWaveform:
    r"""One channel's static tone ladder: ``M`` equal-amplitude tones spaced ``delta_f``.

    .. math::

        f_n = \text{center} + \Big(n - \frac{M-1}{2}\Big)\,\Delta f,
        \qquad n = 0 \ldots M-1

    — the array ladder of Eq. S18/S19, centred on ``center`` so the row of tweezers is
    symmetric about the deflection ``center`` maps to.  Through Table I the tweezers land at
    pitch ``deflection_scale * delta_f`` (``lambda F Delta f / v``).

    Parameters
    ----------
    n_tones:
        Number of tones ``M >= 1``.
    delta_f:
        Tone spacing [Hz] (a detuning difference, Eq. S2 rotating frame).  ``M = 1`` ignores
        it.  Negative values simply mirror the ladder.
    center:
        Detuning [Hz] of the ladder's centre.
    amp:
        Common tone amplitude in ``[0, 1]`` (a :class:`~aodl.waveform.tones.ConstantEnvelope`
        on every tone).  The physical drive strength ``C`` lives in
        :attr:`aodl.params.AODParams.drive_strength`; this is the *relative* amplitude.
    phases:
        ``"schroeder"`` (default, :func:`schroeder_phases`), ``"zero"``, ``"random"``
        (uniform on ``[0, 2 pi)``, drawn from ``rng``), or an explicit ``(M,)`` array of
        phases [rad].  This is the knob the M2 notebook sweeps: Schroeder beats zero and
        random on per-trap intensity spread, because it scatters the IM3 contributions
        (module docstring).
    t0, t1:
        Programmed span [s] of the constant frequency laws.  ``t1 = None`` uses
        ``t0 + `` :data:`DEFAULT_SPAN`.
    rng:
        Generator for ``phases="random"``; ``None`` draws a fresh unseeded one.  Pass
        ``np.random.default_rng(seed)`` when the comparison must be reproducible.

    Returns
    -------
    A :class:`~aodl.waveform.tones.ChannelWaveform` with ``M`` constant-frequency tones.
    """
    count = int(n_tones)
    if count < 1:
        raise ValueError(f"an array ladder needs at least one tone, got {n_tones!r}")
    spacing, middle = float(delta_f), float(center)
    if not (math.isfinite(spacing) and math.isfinite(middle)):
        raise ValueError("delta_f and center must be finite")
    start = float(t0)
    end = start + DEFAULT_SPAN if t1 is None else float(t1)
    if not end > start:
        raise ValueError(f"array_tones needs t1 > t0, got t0={start!r}, t1={end!r}")

    offsets = np.arange(count, dtype=np.float64) - 0.5 * (count - 1.0)
    frequencies = middle + offsets * spacing
    phi = _resolve_phases(phases, count, rng)
    env = ConstantEnvelope(amp=float(amp))
    return ChannelWaveform(
        tuple(
            ToneTrack(
                freq=PiecewisePoly.constant(float(f), start, end),
                env=env,
                phase0=float(p),
            )
            for f, p in zip(frequencies, phi, strict=True)
        )
    )


def add_common_ramp(cw: ChannelWaveform, ramp: PiecewisePoly) -> ChannelWaveform:
    """Add the same frequency law to every tone of ``cw`` (rigid ladder translation).

    A term the whole channel shares — Eq. S19's lateral term ``+ v X(t) / (2 lambda F)``, or
    the axial ``f_Z(t) = v^2 / (2 lambda F^2) \\int Z dt'`` once M3 needs it — moves every
    tweezer of the ladder by the same amount and leaves the spacing alone.  Because
    :class:`~aodl.poly.PiecewisePoly` addition refines to the union of the breakpoints, the
    result is still exact: the phase stays the exact antiderivative, and the chirp rate
    ``fdot`` (hence the Table I lensing) picks up the ramp's derivative exactly.

    ``ramp`` must span the same domain as the tones — build the ladder with matching
    ``t0``/``t1``, or extend one side first (:meth:`~aodl.waveform.tones.ToneTrack.
    with_hold_until`).

    Returns a new :class:`~aodl.waveform.tones.ChannelWaveform`; envelopes and ``phase0``
    are untouched.
    """
    if not isinstance(ramp, PiecewisePoly):
        raise TypeError(f"add_common_ramp needs a PiecewisePoly ramp, got {type(ramp)!r}")
    tones = []
    for i, tone in enumerate(cw.tones):
        try:
            freq = tone.freq + ramp
        except ValueError as exc:
            raise ValueError(
                f"the common ramp must cover the same span as the tones: tone {i} spans "
                f"{tone.t_span} but the ramp spans {ramp.domain}.  Build the ladder with "
                f"matching t0/t1 (array_tones(..., t0=..., t1=...)) or extend the shorter "
                f"side with with_hold_until()."
            ) from exc
        tones.append(ToneTrack(freq=freq, env=tone.env, phase0=tone.phase0))
    return ChannelWaveform(tuple(tones))


# =========================================================== Eq. S19 trajectory synthesis


def f_z_ramp(z: PiecewisePoly, params: AODLParams) -> PiecewisePoly:
    r"""The axial co-chirp ``f_Z`` of Eq. S19 from a lab-``Z`` trajectory.

    .. math::

        f_Z(t) = \frac{v^2}{2\lambda F^2}\int_0^t Z(t')\,dt'
               = \frac{1}{2\,\text{lens\_scale}}\int_0^t Z(t')\,dt'

    Added to **all four** channels, it is what lifts the array out of the focal plane: the
    four equal chirps make a spherical lens whose Table I focus is ``Zbar = 2 lens_scale
    fdot_Z = Z(t)`` — round, astigmatism-free, and laterally static (Eq. S8).

    Exact by construction: :meth:`~aodl.poly.PiecewisePoly.antiderivative` integrates the
    trajectory polynomial in closed form, so ``fdot_Z`` recovers ``Z`` to round-off and the
    tone phase stays the exact antiderivative of the frequency law.
    """
    return z.antiderivative().scale(0.5 / params.lens_scale)


def max_z_integral(params: AODLParams, *, biased: bool = False) -> float:
    r"""Largest ``|int Z dt|`` [m·s] the RF band can buy on the tightest channel.  Eq. 1.

    *The factor-of-2 bookkeeping.*  Table I reads
    ``Zbar = (1/2) lens_scale (fdot_Ax + fdot_Bx + fdot_Ay + fdot_By)``.  Eq. S19 puts the
    **same** ``f_Z`` on all four channels and splits the lateral term antisymmetrically inside
    each counter-propagating pair (``-v X / 2 lambda F`` on A, ``+v X / 2 lambda F`` on B), so
    the lateral parts cancel in that four-channel sum and

        ``Zbar = (1/2) lens_scale * 4 fdot_Z = 2 lens_scale fdot_Z``.

    One channel's chirp therefore buys *twice* the ``Zbar`` a lone ``lens_scale * fdot`` would
    suggest — four channels contributing, halved by Table I.  Integrating in time,

        ``int_0^t Z dt' = 2 lens_scale f_Z(t)``,

    and Eq. S19 starts the drive at the carrier (``f_Z(0) = 0``), so the reachable excursion
    is the **headroom on one side** of ``f_center``, not the whole band:

        ``|int Z dt| <= 2 lens_scale min(f_hi - f_center, f_center - f_lo)``
                     ``= lens_scale (f_hi - f_lo)``   for a centred carrier.

    At the default hardware (``f_center = 100 MHz``, band ``+/- 10 MHz``,
    ``lens_scale = 1.03e-16 m.s``) that is ``2.06e-9 m.s`` — ``Z = 10 µm`` sustained for
    206 µs, which is why long holds off the focal plane need the fading-Shepard waveforms of
    M4 (Eqs. S24-S28).

    Parameters
    ----------
    biased:
        ``True`` returns the ceiling with the ``f_z_bias`` pre-compensation of
        :func:`synthesize` in force: starting the drive half an excursion below ``f_center``
        centres the ``f_Z`` walk in the band, so it may span ``2 x headroom`` instead of
        ``headroom`` and the reachable ``|int Z dt|`` **doubles** — ``docs/PLAN.md`` §1.5's
        412 µs at ``Z = 10 µm``.  The lattice is untouched by the bias (it is common to all
        four channels, so it cancels in every Table I difference and carries no chirp).

    Either number assumes the whole headroom goes to ``f_Z``: the array ladder and the lateral
    term take their own share of it, so a real trajectory gets less.
    """
    headroom = min(
        min(aod.band[1] - aod.f_center, aod.f_center - aod.band[0])
        for aod in params.channels.values()
    )
    return (4.0 if biased else 2.0) * params.lens_scale * headroom


def _poly_extrema(p: PiecewisePoly) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((min, t_min), (max, t_max))`` of a piecewise polynomial over its own domain.

    Exact rather than sampled: a polynomial's extrema on a segment are at its endpoints or at
    a root of its derivative, so the candidate set is the breakpoints plus the real roots of
    each segment's derivative inside ``[0, 1]`` of normalized local time.
    """
    candidates = [p.breaks]
    for k in range(p.n_segments):
        coeffs = p.coeffs[k]
        deriv = coeffs[1:] * np.arange(1, coeffs.size, dtype=np.float64)
        scale = float(np.max(np.abs(deriv), initial=0.0))
        if scale == 0.0:
            continue
        keep = np.nonzero(np.abs(deriv) > _ROOT_TRIM * scale)[0]
        if keep[-1] < 1:  # derivative is constant: no interior extremum
            continue
        roots = np.asarray(npoly.polyroots(deriv[: keep[-1] + 1]), dtype=np.complex128)
        real = roots.real[np.abs(roots.imag) <= 1e-9 * (1.0 + np.abs(roots.real))]
        tau = real[(real >= 0.0) & (real <= 1.0)]
        if tau.size:
            width = p.breaks[k + 1] - p.breaks[k]
            candidates.append(p.breaks[k] + tau * width)
    times = np.concatenate(candidates)
    values = np.asarray(p(times), dtype=np.float64)
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    return (float(values[lo]), float(times[lo])), (float(values[hi]), float(times[hi]))


def _bias_clause(params: AODLParams, biased: bool) -> str:
    """What the ``f_z_bias`` option is worth here — quoted in the band-violation report."""
    ceiling = max_z_integral(params, biased=True)
    hold_10um = ceiling / (10.0 * um)
    if biased:
        return (
            f"  That ceiling is already the pre-biased one (f_z_bias centres the f_Z walk in "
            f"the band, doubling Eq. 1's one-sided budget to {ceiling:.4g} m.s, "
            f"Z = 10 um for {hold_10um / us:.4g} us)."
        )
    return (
        f"  Pre-biasing the start frequency (f_z_bias='auto') centres the f_Z walk in the band "
        f"and doubles that ceiling to {ceiling:.4g} m.s "
        f"(Z = 10 um for {hold_10um / us:.4g} us)."
    )


def _band_message(
    name: str,
    tone: int,
    aod_band: tuple[float, float],
    f_center: float,
    low: tuple[float, float],
    high: tuple[float, float],
    params: AODLParams,
    requested: float,
    biased: bool = False,
) -> str:
    """The band-violation report: what went out, by how much, and what would fit."""
    lo, hi = aod_band
    (f_min, t_min), (f_max, t_max) = low, high
    under, over = lo - (f_center + f_min), (f_center + f_max) - hi
    excess, when = (over, t_max) if over > under else (under, t_min)
    ceiling = max_z_integral(params, biased=biased)
    hold_10um = ceiling / (10.0 * um)
    # The budget is one-sided (f_Z starts at the carrier), so what buys it is the distance
    # from f_center to the *nearer* band edge -- not the full band width, which would
    # overstate the ceiling by (hi - lo) / (2 headroom) on an off-centre band.
    headroom = 0.5 * ceiling / params.lens_scale
    return (
        f"channel {name!r} tone {tone} leaves its usable band: the drive spans "
        f"[{(f_center + f_min) / MHz:.4f}, {(f_center + f_max) / MHz:.4f}] MHz "
        f"(f_center {f_center / MHz:.4f} MHz + detuning "
        f"[{f_min / MHz:+.4f}, {f_max / MHz:+.4f}] MHz), outside the limit "
        f"[{lo / MHz:.4f}, {hi / MHz:.4f}] MHz by {excess / MHz:.4f} MHz at "
        f"t = {when / us:.4g} us.  Sustained axial offset is what costs bandwidth (Eq. 1): "
        f"every channel carries f_Z = int Z dt / (2 lens_scale), and starting from f_center "
        f"the {headroom / MHz:.4g} MHz to the nearer band edge buys at most "
        f"|int Z dt| = {ceiling:.4g} m.s "
        f"(Z = 10 um held for {hold_10um / us:.4g} us) with nothing else using it — this "
        f"trajectory asks for {requested:.4g} m.s of that, and the array ladder and the "
        f"lateral term take their own share.  Shorten the hold, reduce Z, narrow the array, "
        f"split the move, or switch to the fading-Shepard tones of M4 (shepard='auto', "
        f"Eqs. S24-S28); pass check_band=False to synthesize it anyway for plotting."
        + _bias_clause(params, biased)
    )


def _check_bands(wfs: WaveformSet, requested: float, biased: bool = False) -> None:
    """Raise unless every tone stays inside its channel's absolute RF band (Eq. 1).

    The waveform IR carries *detunings*, so the quantity that must fit is
    ``f_center + f(t)`` (``docs/conventions.md`` §1) — checked on the exact extrema of each
    tone's frequency law over the whole programmed span, hold included.
    """
    for name, cw in wfs.channels.items():
        aod = wfs.params.channels[name]
        lo, hi = aod.band
        for i, tone in enumerate(cw.tones):
            low, high = _poly_extrema(tone.freq)
            if aod.f_center + low[0] < lo - BAND_TOL or aod.f_center + high[0] > hi + BAND_TOL:
                raise ValueError(
                    _band_message(
                        name,
                        i,
                        (lo, hi),
                        aod.f_center,
                        low,
                        high,
                        wfs.params,
                        requested,
                        biased,
                    )
                )


def _with_switch_ramp(cfg: ShepardConfig, switch_ramp: float | None) -> ShepardConfig:
    """``cfg`` with :attr:`ShepardConfig.switch_ramp` overridden when one was passed."""
    if switch_ramp is None:
        return cfg
    return replace(cfg, switch_ramp=float(switch_ramp))


def _phase_argument(
    phases: str | ArrayLike | Mapping[str, str | ArrayLike], name: str
) -> str | ArrayLike:
    """The :func:`array_tones` ``phases`` argument for channel ``name``."""
    if isinstance(phases, Mapping):
        return phases.get(name, "schroeder")
    return phases


def _padded(p: PiecewisePoly, t_end: float) -> PiecewisePoly:
    """``p`` held at its terminal value out to ``t_end`` — the ``t_pad`` tail as a polynomial.

    The M3 path gets its tail from :meth:`~aodl.waveform.tones.ChannelWaveform.
    with_hold_until`, which appends a constant *frequency* segment per tone.  A Shepard tone
    also carries an envelope built on ``f_Z``, and an envelope must be defined wherever its
    tone is, so the tail is applied to the underlying polynomials instead — before either the
    frequency law or the fade coordinate is derived from them.  Same semantics: the drive
    freezes, so the array sits at rest with ``Zbar = 0`` through the tail.
    """
    _, t1 = p.domain
    if t_end <= t1 + TIME_TOL:
        return p
    return PiecewisePoly.concat([p, PiecewisePoly.constant(float(p(t1)), t1, float(t_end))])


def _advanced(p: PiecewisePoly, lead: float) -> PiecewisePoly:
    r"""``q(t) = p(min(t + lead, T))`` on ``p``'s own domain — the §2.2 pre-compensation.

    Writing the drive from a trajectory read ``lead = tau/2`` *ahead* of drive time is what
    cancels the retardation lag: the atom plane responds to the acoustic sample that left the
    transducer half a transit earlier (``docs/conventions.md`` §7), so a drive built from
    ``X(t + tau/2)`` reproduces ``X`` at observation time ``t`` rather than at ``t - tau/2``.

    Exact, not resampled: the shifted polynomial is re-expressed on a breakpoint set that
    includes the domain start (:meth:`~aodl.poly.PiecewisePoly._refined` rebases a segment onto
    a sub-interval in closed form), the segments before it are dropped, and the ``lead``-long
    tail is a constant at the trajectory's terminal value — which is exact too, because every
    move is rest-to-rest, so ``p`` is already flat there.
    """
    t0, t1 = p.domain
    shift = float(lead)
    if shift <= 0.0:
        return p
    if shift >= t1 - t0:
        return PiecewisePoly.constant(float(p(t1)), t0, t1)
    shifted = p.shift(-shift)  # domain (t0 - lead, t1 - lead), value p(t + lead)
    keep = shifted.breaks[shifted.breaks > t0 + TIME_TOL]
    refined = shifted._refined(np.concatenate([[t0 - shift, t0], keep]))
    head = PiecewisePoly(refined.breaks[1:], refined.coeffs[1:])
    return PiecewisePoly.concat([head, PiecewisePoly.constant(float(p(t1)), t1 - shift, t1)])


def _lateral_terms(x: PiecewisePoly, y: PiecewisePoly, params: AODLParams) -> dict[str, float]:
    """Eq. S19's ``-+ v P(t) / (2 lambda F)`` scale — half the lateral drive per pair member."""
    optics = params.optics
    half = 0.5 * params.sound_speed / (optics.wavelength * optics.focal_length)
    return {"Ax": -half, "Bx": half, "Ay": -half, "By": half}


def _shepard_band_message(
    name: str,
    aod_band: tuple[float, float],
    f_center: float,
    bound: float,
    window: float,
    lateral: float,
    delta_f: float,
    fade: ChannelFade,
    eta: float,
    comb: float,
) -> str:
    """The band-violation report of the *bounded* fading-Shepard excursion."""
    lo, hi = aod_band
    excess = max(lo - (f_center - bound), (f_center + bound) - hi)
    comb_clause = (
        ""
        if comb == 0.0
        else (
            f" and the {abs(comb) / MHz:.4f} MHz comb offset that puts an even-M ladder on "
            f"Eq. S19's lattice"
        )
    )
    return (
        f"channel {name!r} leaves its usable band even with fading-Shepard tones: its live "
        f"rungs span [{(f_center - bound) / MHz:.4f}, {(f_center + bound) / MHz:.4f}] MHz, "
        f"outside the limit [{lo / MHz:.4f}, {hi / MHz:.4f}] MHz by {excess / MHz:.4f} MHz.  "
        f"Unlike Eq. S19 this bound does not grow with the hold (Eqs. S24-S28: the ladder "
        f"slides, the live window does not) — it is the fade window "
        f"(M + eta) delta_f / 2 = ({fade.m} + {eta:g}) x {delta_f / MHz:.4f} MHz / 2 = "
        f"{window / MHz:.4f} MHz plus the lateral term's {lateral / MHz:.4f} MHz"
        f"{comb_clause}.  Narrow the "
        f"ladder (delta_f), shrink the array (M), reduce the lateral excursion, or widen the "
        f"band; pass check_band=False to synthesize it anyway for plotting."
    )


def _check_shepard_bands(
    cfg: ShepardConfig,
    fades: Mapping[str, ChannelFade],
    lateral: Mapping[str, PiecewisePoly],
    combs: Mapping[str, float],
    params: AODLParams,
) -> None:
    """Raise unless every channel's *live* tones stay in band (the Shepard claim, S24-S28).

    Eq. 1 does not apply here: ``f_Z`` is deliberately unbounded and the tones that carry it
    out of band are switched off before they get there.  What must fit is
    :func:`aodl.waveform.shepard.shepard_band_bound`, which the whole run obeys however long
    the axial hold lasts.
    """
    for name in CHANNELS:
        aod = params.channels[name]
        lo, hi = aod.band
        span = max(abs(value) for value in poly_range(lateral[name]))
        delta_f = cfg.spacing(name)
        window = 0.5 * (fades[name].m + cfg.eta) * delta_f
        comb = combs[name]
        bound = shepard_band_bound(fades[name], delta_f, cfg.eta, span, comb)
        if aod.f_center - bound < lo - BAND_TOL or aod.f_center + bound > hi + BAND_TOL:
            raise ValueError(
                _shepard_band_message(
                    name,
                    (lo, hi),
                    aod.f_center,
                    bound,
                    window,
                    span,
                    delta_f,
                    fades[name],
                    cfg.eta,
                    comb,
                )
            )


def _synthesize_shepard(
    spec: TrajectorySpec,
    params: AODLParams,
    cfg: ShepardConfig,
    xyz: tuple[PiecewisePoly, PiecewisePoly, PiecewisePoly],
    t_end: float,
    *,
    amp: float,
    phases: str | ArrayLike | Mapping[str, str | ArrayLike],
    rng: np.random.Generator | None,
    check_band: bool,
    note: str,
) -> WaveformSet:
    """Build the four fading-Shepard ladders (Eqs. S24-S28).  See :func:`synthesize`."""
    x, y, z = xyz
    scales = _lateral_terms(x, y, params)
    f_z = _padded(f_z_ramp(z, params), t_end)
    lateral = {
        name: _padded((x if name.endswith("x") else y).scale(scale), t_end)
        for name, scale in scales.items()
    }
    fades = cfg.resolve(spec.array)
    combs = cfg.comb_offsets(fades)
    if check_band:
        _check_shepard_bands(cfg, fades, lateral, combs, params)

    channels: dict[str, ChannelWaveform] = {}
    for name in CHANNELS:
        # Eq. S28 puts the Schroeder progression on the B ladders; the A ladders take zero.
        tone_phases = _phase_argument(phases, name) if name.startswith("B") else "zero"
        channels[name] = shepard_ladder(
            fades[name],
            cfg.spacing(name),
            lateral[name],
            f_z,
            eta=cfg.eta,
            amp=amp,
            phases=tone_phases,
            rng=rng,
            comb_offset=combs[name],
            switch_ramp=cfg.switch_ramp,
        )
    counts = ", ".join(f"{name}:{channels[name].n_tones}" for name in CHANNELS)
    array = spec.array
    lattice = ", ".join(f"{name}:{combs[name] / MHz:+.4g}" for name in ("Bx", "By"))
    extras = f"; comb offset (MHz) {lattice}" if any(combs.values()) else ""
    if cfg.switch_ramp > 0.0:
        extras += f"; switch_ramp = {cfg.switch_ramp / us:.4g} us"
    return WaveformSet(
        channels=channels,
        params=params,
        description=(
            f"Fading-Shepard synthesis (Eqs. S24-S28): {array.mx}x{array.my} array, "
            f"{len(spec.moves)} move(s), T = {spec.duration / us:.4g} us + "
            f"{(t_end - spec.duration) / us:.4g} us hold; "
            f"delta_f = ({cfg.delta_f_x / MHz:.4g}, {cfg.delta_f_y / MHz:.4g}) MHz, "
            f"eta = {cfg.eta:g}; rungs {counts}{extras}{note}"
        ),
    )


def _synthesize_s19(
    spec: TrajectorySpec,
    params: AODLParams,
    xyz: tuple[PiecewisePoly, PiecewisePoly, PiecewisePoly],
    t_end: float,
    *,
    amp: float,
    phases: str | ArrayLike | Mapping[str, str | ArrayLike],
    rng: np.random.Generator | None,
    check_band: bool,
    f_z_bias: float | str = 0.0,
    note: str = "",
) -> WaveformSet:
    """Build the plain Eq. S19 drive (one tone per A channel, the array ladder on B)."""
    x, y, z = xyz
    duration = spec.duration
    scales = _lateral_terms(x, y, params)
    f_z = f_z_ramp(z, params)
    bias = resolve_f_z_bias(f_z_bias, f_z, params)
    array = spec.array
    ladders = {"Bx": (array.mx, array.delta_f_x), "By": (array.my, array.delta_f_y)}
    common = {
        name: ((x if name.endswith("x") else y).scale(scale) + f_z).offset(bias)
        for name, scale in scales.items()
    }

    channels: dict[str, ChannelWaveform] = {}
    for name in CHANNELS:
        n_tones, delta_f = ladders.get(name, (1, 0.0))
        # A single-tone channel's phase is a global factor over every Eq. S7 term, so the A
        # channels take 0 rather than consuming the caller's ladder phases.
        tone_phases = _phase_argument(phases, name) if name in ladders else (0.0,)
        ladder = array_tones(
            n_tones, delta_f, amp=amp, phases=tone_phases, t0=0.0, t1=duration, rng=rng
        )
        channels[name] = add_common_ramp(ladder, common[name])

    bias_note = "" if bias == 0.0 else f"; f_Z pre-bias {bias / MHz:+.4g} MHz"
    wfs = WaveformSet(
        channels=channels,
        params=params,
        description=(
            f"Eq. S19 synthesis: {array.mx}x{array.my} array, {len(spec.moves)} move(s), "
            f"T = {duration / us:.4g} us + {(t_end - duration) / us:.4g} us hold"
            f"{bias_note}{note}"
        ),
    )
    if t_end > duration:
        wfs = wfs.with_hold_until(t_end)
    if check_band:
        _check_bands(wfs, _requested_z_integral(f_z, params), bias != 0.0)
    return wfs


def _requested_z_integral(f_z: PiecewisePoly, params: AODLParams) -> float:
    """``max |int Z dt|`` [m.s] the trajectory asks for — the Eq. 1 quantity."""
    lo, hi = poly_range(f_z)
    return 2.0 * params.lens_scale * max(abs(lo), abs(hi))


def requested_z_integral(spec: TrajectorySpec, params: AODLParams) -> float:
    """``max_t |int_0^t Z dt'|`` [m.s] of ``spec`` — what Eq. 1 budgets (:func:`max_z_integral`).

    The quantity a band refusal quotes and :class:`aodl.api.PlanReport` reports: the axial
    integral is what a plain Eq. S19 drive has to carry on every channel at once.
    """
    _, _, z = spec.compile()
    return _requested_z_integral(f_z_ramp(z, params), params)


def resolve_f_z_bias(bias: float | str, f_z: PiecewisePoly, params: AODLParams) -> float:
    r"""The constant [Hz] the ``f_z_bias`` option adds to all four channels.  §2.3.

    Eq. S19 starts every channel at its carrier (``f_Z(0) = 0``) and then walks it one way, so
    only the headroom on *one side* of ``f_center`` is usable and the Eq. 1 budget is halved
    for no physical reason (:func:`max_z_integral`).  Offsetting the start frequency by

    .. math::

        c = -\tfrac{1}{2}\big(\min_t f_Z + \max_t f_Z\big)
          \;\;\xrightarrow{\;Z \ge 0\;}\;\; -\tfrac{1}{2}\,\frac{v^2}{2\lambda F^2}\!\int Z\,dt

    centres the walk in the band, and the reachable ``|int Z dt|`` doubles.  ``"auto"`` is that
    value, clamped to ``+-`` the tightest channel's headroom so the bias alone can never be
    what leaves the band; a float is taken verbatim.

    The bias is **common to all four channels and constant in time**, so by Table I it moves
    nothing: it cancels in both lateral differences (``X``, ``Y``) and carries no chirp, hence
    no ``Zbar`` and no ``Delta F``.  All it changes is where in the band the drive sits.
    """
    headroom = min(
        min(aod.band[1] - aod.f_center, aod.f_center - aod.band[0])
        for aod in params.channels.values()
    )
    if isinstance(bias, str):
        if bias != "auto":
            raise ValueError(f"f_z_bias must be a number or 'auto', got {bias!r}")
        lo, hi = poly_range(f_z)
        return float(np.clip(-0.5 * (lo + hi), -headroom, headroom))
    value = float(bias)
    if not math.isfinite(value):
        raise ValueError(f"f_z_bias must be finite, got {bias!r}")
    return value


def synthesize(
    spec: TrajectorySpec,
    params: AODLParams,
    *,
    amp: float = 1.0,
    phases: str | ArrayLike | Mapping[str, str | ArrayLike] = "schroeder",
    t_pad: float | None = None,
    rng: np.random.Generator | None = None,
    check_band: bool = True,
    shepard: str | ShepardConfig | None = None,
    retard_compensate: bool = False,
    f_z_bias: float | str = 0.0,
    switch_ramp: float | None = None,
) -> WaveformSet:
    r"""Compile a trajectory into the four AODL channel drives.  **Eq. S19.**

    With ``(X, Y, Z)`` from :meth:`~aodl.trajectory.spec.TrajectorySpec.compile`, the common
    sound speed ``v`` and the optics ``lambda, F``:

    .. math::

        f_Z(t) &= \frac{v^2}{2\lambda F^2}\int_0^t Z\,dt' \\
        f_{Ax}(t) &= -\frac{v}{2\lambda F}X(t) + f_Z(t),
        \qquad f_{Bx}^{(n)}(t) = f_{x0}^{(n)} + \frac{v}{2\lambda F}X(t) + f_Z(t) \\
        f_{Ay}(t) &= -\frac{v}{2\lambda F}Y(t) + f_Z(t),
        \qquad f_{By}^{(m)}(t) = f_{y0}^{(m)} + \frac{v}{2\lambda F}Y(t) + f_Z(t)

    with the array ladders ``f_x0^(n) = (n - (Mx-1)/2) delta_f_x`` (Eq. S18) carrying
    Schroeder phases.  All frequencies are **detunings** from each channel's ``f_center``
    (Eq. S2 rotating frame).

    Why this is the astigmatism-free solution (Table I): the lateral terms enter the ``A``
    and ``B`` members of a pair with opposite signs, so they *differ* — giving
    ``X = deflection_scale (f_Bx - f_Ax)`` — while cancelling in the chirp sums; the ``f_Z``
    term is common to all four, so it *adds* — giving ``Zbar = 2 lens_scale fdot_Z = Z(t)``
    with ``Delta F = 0`` identically.  Four channels, three degrees of freedom, no
    astigmatism.

    Parameters
    ----------
    spec:
        The trajectory: array geometry plus the moves.
    params:
        Hardware.  All four channels must share one sound speed (``params.sound_speed``
        raises otherwise) — the Table I scales assume it.
    amp:
        Common relative tone amplitude in ``[0, 1]``
        (a :class:`~aodl.waveform.tones.ConstantEnvelope` on every tone).
    phases:
        Phase convention for the ``Bx``/``By`` ladders: ``"schroeder"`` (default), ``"zero"``,
        ``"random"``, an explicit per-tone array, or a ``{channel: convention}`` mapping when
        the two ladders need different ones.  The single-tone ``A`` channels always get
        ``phase0 = 0``: a whole channel's common phase multiplies *every* pupil term of
        Eq. S7 alike, so it is a global factor with no observable effect.
    t_pad:
        Extra time [s] the drive is held at its terminal frequency past the end of the
        trajectory; ``None`` uses ``2 tau``.  Simulating the last requested instant needs
        drive time ``t - tau/2``, and :func:`aodl.engine.simulate` refuses to clamp-hold past
        a tone's domain, so this tail is what lets a run probe the array at rest.
    rng:
        Generator for ``phases="random"``.
    check_band:
        Verify that ``f_center + f(t)`` stays inside every channel's band (Eq. 1) and raise
        :class:`ValueError` naming the excursion, the limit and the feasible ``|int Z dt|``
        otherwise.  ``False`` skips the check — **for plotting infeasible drives only**; the
        hardware would simply not diffract there.  Under ``shepard`` the quantity checked is
        the *bounded* live-tone excursion instead (:func:`_check_shepard_bands`).
    shepard:
        Fading-Shepard tone ladders (Eqs. S24-S28, :mod:`aodl.waveform.shepard`), which trade
        Eq. 1's one-sided budget for a bounded excursion and hold ``Z`` indefinitely:

        * ``None`` (default) — plain Eq. S19, refused by the band check when the hold is too
          long;
        * ``"auto"`` — run the Eq. 1 band check first and switch to fading-Shepard *only* if
          it fails, choosing ladder spacings with
          :func:`aodl.waveform.shepard.auto_config`.  Which path was taken is stated in
          :attr:`~aodl.waveform.tones.WaveformSet.description` (the decision is made even
          when ``check_band=False``, which then only skips the final verification);
        * a :class:`~aodl.waveform.shepard.ShepardConfig` — always fading-Shepard, with the
          caller's spacings, duty and per-channel Table II overrides.
    retard_compensate:
        Evaluate the trajectory at ``min(t + tau/2, T)`` instead of ``t`` (:func:`_advanced`),
        so the *atom plane* reproduces the request at observation time ``t`` rather than at
        ``t - tau/2``.  ``False`` (default) is Eq. S19 verbatim and reproduces v1 bit for bit;
        ``True`` costs the array its rest position at ``t = 0`` (the drive starts half a
        transit into the move) and does not remove the fill transient, which is a property of
        the aperture, not of the schedule.
    f_z_bias:
        Constant [Hz] added to all four channels (:func:`resolve_f_z_bias`), **plain Eq. S19
        only**.  ``0.0`` (default) starts the drive at the carrier as Eq. S19 states; ``"auto"``
        centres the ``f_Z`` walk in the band, which doubles Eq. 1's budget
        (:func:`max_z_integral` with ``biased=True``).  Being common and constant it cancels in
        every Table I quantity, so the array's positions are untouched.  Rejected together with
        an explicit :class:`~aodl.waveform.shepard.ShepardConfig` (the ladders are already
        band-bounded); with ``shepard="auto"`` it applies to the Eq. S19 attempt, and is simply
        not needed if that attempt fails and the ladders take over.
    switch_ramp:
        Overrides :attr:`~aodl.waveform.shepard.ShepardConfig.switch_ramp` [s] when not
        ``None`` — the raised-cosine on/off ramp of the ``p = 0`` array rungs
        (:class:`~aodl.waveform.shepard.SwitchRamped`).  Only meaningful under ``shepard``.

    Returns
    -------
    A :class:`~aodl.waveform.tones.WaveformSet` on ``[0, T + t_pad]`` with all four channels
    driven: one tone on ``Ax``/``Ay``, ``Mx``/``My`` on ``Bx``/``By`` — or, under
    ``shepard``, one tone per live ladder rung on every channel.

    Note
    ----
    The waveform's time axis starts at 0 and the atom-plane response lags it by ``tau/2``
    unless ``retard_compensate=True`` (module docstring).  With the default, compare
    measurements at ``t`` against the requested profile at ``t - tau/2``.
    """
    if not isinstance(spec, TrajectorySpec):
        raise TypeError(f"synthesize() needs a TrajectorySpec, got {type(spec)!r}")
    if not isinstance(params, AODLParams):
        raise TypeError(f"synthesize() needs an AODLParams, got {type(params)!r}")
    x, y, z = spec.compile()
    duration = spec.duration
    if t_pad is None:
        t_pad = T_PAD_TRANSITS * max(aod.transit_time for aod in params.channels.values())
    t_pad = float(t_pad)
    if not math.isfinite(t_pad) or t_pad < 0.0:
        raise ValueError(f"t_pad must be finite and non-negative, got {t_pad!r}")

    # Eq. S19's lateral coefficient v / (2 lambda F) = 1 / (2 deflection_scale): half the
    # lateral drive goes on each member of a counter-propagating pair, so their *difference*
    # is the full Table I deflection while their chirps cancel.  Reading `sound_speed` (in
    # `_lateral_terms`) also asserts that the four channels agree on v, which every scale
    # assumes.
    if retard_compensate:
        lead = 0.5 * max(aod.transit_time for aod in params.channels.values())
        x, y, z = (_advanced(p, lead) for p in (x, y, z))
    xyz = (x, y, z)
    t_end = duration + t_pad
    biased = not (isinstance(f_z_bias, float | int) and float(f_z_bias) == 0.0)

    if shepard is None:
        return _synthesize_s19(
            spec,
            params,
            xyz,
            t_end,
            amp=amp,
            phases=phases,
            rng=rng,
            check_band=check_band,
            f_z_bias=f_z_bias,
        )

    if isinstance(shepard, ShepardConfig):
        if biased:
            raise ValueError(
                "f_z_bias applies to plain Eq. S19 only: a fading-Shepard drive never lets "
                "f_Z leave the fade window in the first place (Eqs. S24-S28), so biasing the "
                "start frequency buys nothing and only eats band.  Drop f_z_bias, or drop the "
                "ShepardConfig and let shepard='auto' choose."
            )
        return _synthesize_shepard(
            spec,
            params,
            _with_switch_ramp(shepard, switch_ramp),
            xyz,
            t_end,
            amp=amp,
            phases=phases,
            rng=rng,
            check_band=check_band,
            note="",
        )
    if not isinstance(shepard, str):
        raise TypeError(f"shepard must be None, 'auto' or a ShepardConfig, got {type(shepard)!r}")
    if shepard != "auto":
        raise ValueError(f"shepard must be None, 'auto' or a ShepardConfig, got {shepard!r}")

    # "auto": let Eq. 1 decide.  The plain drive is built unchecked and then measured against
    # the band, so the *reason* it does not fit can be quoted in the Shepard set's description.
    plain = _synthesize_s19(
        spec,
        params,
        xyz,
        t_end,
        amp=amp,
        phases=phases,
        rng=rng,
        check_band=False,
        f_z_bias=f_z_bias,
    )
    try:
        _check_bands(plain, _requested_z_integral(f_z_ramp(z, params), params), biased)
    except ValueError as exc:
        reason = str(exc).split(".  ")[0]
    else:
        return replace(
            plain,
            description=plain.description + " [shepard='auto': plain Eq. S19 fits the band]",
        )

    scales = _lateral_terms(x, y, params)
    lateral_span = {
        name: max(abs(v) for v in poly_range((x if name.endswith("x") else y).scale(scale)))
        for name, scale in scales.items()
    }
    headroom = {
        name: min(aod.band[1] - aod.f_center, aod.f_center - aod.band[0])
        for name, aod in params.channels.items()
    }
    cfg = _with_switch_ramp(auto_config(spec.array, lateral_span, headroom), switch_ramp)
    dropped = " (f_z_bias dropped: the ladders are band-bounded already)" if biased else ""
    return _synthesize_shepard(
        spec,
        params,
        cfg,
        xyz,
        t_end,
        amp=amp,
        phases=phases,
        rng=rng,
        check_band=check_band,
        note=f" [shepard='auto': plain Eq. S19 refused -- {reason}{dropped}]",
    )


__all__ = [
    "BAND_TOL",
    "DEFAULT_SPAN",
    "PHASE_MODES",
    "T_PAD_TRANSITS",
    "add_common_ramp",
    "array_tones",
    "f_z_ramp",
    "max_z_integral",
    "requested_z_integral",
    "resolve_f_z_bias",
    "schroeder_phases",
    "synthesize",
]
