r"""Array ladders, Schroeder phases and common frequency ramps (Eqs. S18/S19, S23/S28).

The M2 half of waveform synthesis: everything needed to put a *static* tone ladder on a
channel and to move the whole ladder together.  The full Eq. S19 trajectory solver — the
one that turns an (X, Y, Z) waypoint list into four band-checked channel drives — is M3;
this module deliberately stops at the two pieces M2 needs:

* **the ladder** (:func:`array_tones`) — ``M`` equally spaced constant tones
  ``f_n = center + (n - (M-1)/2) delta_f``, which Table I maps to a row of tweezers at pitch
  ``deflection_scale * delta_f`` (``lambda F Delta f / v``, 10.3 µm per MHz at the default
  hardware).  Crossing such a ladder on ``Ax`` with one on ``Ay`` gives the ``Mx x My``
  array of ``docs/PLAN.md`` §3 (M2), because the pupil is a *product* (Eq. S7);
* **the common ramp** (:func:`add_common_ramp`) — the same frequency law added to every tone
  of a channel, which translates the whole ladder rigidly (Eq. S19's ``+ v X(t) / (2 lambda
  F)`` term) without changing its internal spacing, and later carries the ``f_Z`` co-chirp.

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
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..poly import PiecewisePoly
from ..units import ms
from .tones import ChannelWaveform, ConstantEnvelope, ToneTrack

#: Default programmed span of :func:`array_tones` [s] when no ``t1`` is given.  A static
#: ladder has nothing to schedule, so the span exists only to satisfy the
#: :class:`~aodl.waveform.tones.WaveformSet` "all tones cover the same span" rule and the
#: :func:`aodl.engine.simulate` coverage check; 1 ms is ~87 aperture transits, long enough
#: for any single-frame or short-move M2 scene.  Pass ``t1`` (or extend afterwards with
#: :meth:`~aodl.waveform.tones.ChannelWaveform.with_hold_until`) for a longer run.
DEFAULT_SPAN = 1.0 * ms

#: Phase conventions accepted by :func:`array_tones`.
PHASE_MODES: tuple[str, ...] = ("schroeder", "zero", "random")

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

    Returns
    -------
    ``(M,)`` array of phases in ``[0, 2 pi)``; ``M = 0`` gives an empty array.
    """
    count = int(n_tones)
    if count < 0:
        raise ValueError(f"n_tones must be non-negative, got {n_tones!r}")
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    n = np.arange(count, dtype=np.float64)
    return np.mod(_TWO_PI * n * (n - 1.0) / (2.0 * count), _TWO_PI)


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


__all__ = [
    "DEFAULT_SPAN",
    "PHASE_MODES",
    "add_common_ramp",
    "array_tones",
    "schroeder_phases",
]
