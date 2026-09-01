"""Waveform intermediate representation: envelopes, tone tracks, channels, waveform sets.

This is *the* exchange format of the package (``docs/ARCHITECTURE.md`` §1, layer L1).  A
channel drive is the real RF signal

.. math::

    V_\\mu(t) = \\sum_n A^{(n)}(t)\\,
                \\cos\\!\\Big(2\\pi f_\\text{center} t
                             + 2\\pi\\!\\int_0^t f^{(n)}(t')\\,dt' + \\phi^{(n)}\\Big),

but the carrier ``f_center`` is factored out everywhere inside the package: **frequencies
in the waveform IR are detunings from** :attr:`~aodl.params.AODParams.f_center` (the
rotating frame of Eq. S2).  The carrier is re-added exactly once, in
:func:`aodl.waveform.export.render_samples`, when literal AWG samples are produced.

Nothing here is ever sampled: a :class:`ToneTrack` stores a piecewise-polynomial frequency
law plus a parametric envelope, and hands out exact ``f``, ``fdot``, ``phase``, ``A``,
``dA``, ``d2A`` at arbitrary (vectorized) times.  ``phase`` is the *exact* antiderivative
of the frequency law, so tones are phase-continuous across segment boundaries by
construction, and ``fdot``/``d2A`` — which the device layer needs for chirp lensing
(Eq. S6) and acoustic irising (Eq. S5) — never come from numerical differentiation.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..params import CHANNELS, AODLParams
from ..poly import PiecewisePoly

#: Absolute tolerance [s] for calling two instants "the same time".  A picosecond is far
#: below anything the hardware resolves (1.6 ns at the default 625 MS/s) and far above
#: float64 round-off on microsecond-scale breakpoints.
TIME_TOL = 1e-12

#: Quantities returned by :meth:`ChannelWaveform.eval_table`, in a fixed order.
TABLE_KEYS: tuple[str, ...] = ("f", "fdot", "A", "dA", "d2A", "phase")

_TWO_PI = 2.0 * math.pi


def _shaped(values: ArrayLike, t: NDArray[np.float64]) -> NDArray[np.float64] | float:
    """Match :meth:`PiecewisePoly.__call__`: float for scalar input, array otherwise."""
    if t.ndim == 0:
        return float(np.asarray(values))
    return np.asarray(values, dtype=np.float64)


# --------------------------------------------------------------------------- envelopes


@runtime_checkable
class Envelope(Protocol):
    """Parametric amplitude envelope ``A(t)`` with two exact derivatives.

    All three methods are vectorized (scalar in -> float out, array in -> array out) and
    ``A`` takes values in ``[0, 1]``: it is a *relative* drive amplitude, with the
    absolute RF scale set once at export time.  ``dA`` and ``d2A`` are not decoration —
    the beam-centre Taylor expansion of the retarded amplitude (Eq. S5) turns them into
    the pupil's amplitude tilt and curvature, i.e. acoustic irising.
    """

    def A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Envelope value at ``t`` [dimensionless, in ``[0, 1]``]."""
        ...

    def dA(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """First time derivative ``dA/dt`` [1/s]."""
        ...

    def d2A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Second time derivative ``d2A/dt2`` [1/s^2]."""
        ...


@dataclass(frozen=True)
class ConstantEnvelope:
    """Flat envelope ``A(t) = amp`` (the default: a tone that is simply always on)."""

    amp: float = 1.0

    def __post_init__(self) -> None:
        amp = float(self.amp)
        if not math.isfinite(amp) or not 0.0 <= amp <= 1.0:
            raise ValueError(f"ConstantEnvelope.amp must lie in [0, 1], got {self.amp!r}")
        object.__setattr__(self, "amp", amp)

    def A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Envelope value ``amp``."""
        t_arr = np.asarray(t, dtype=np.float64)
        return _shaped(np.full(t_arr.shape, self.amp), t_arr)

    def dA(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Zero."""
        t_arr = np.asarray(t, dtype=np.float64)
        return _shaped(np.zeros(t_arr.shape), t_arr)

    def d2A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Zero."""
        t_arr = np.asarray(t, dtype=np.float64)
        return _shaped(np.zeros(t_arr.shape), t_arr)


@dataclass(frozen=True)
class SmoothOnOff:
    """Raised-sine gate: off, ``sin^2`` rise, flat, symmetric ``sin^2`` fall, off.

    With ``c = pi / (2 ramp)``:

    ===================================== ===========================
    interval                              ``A(t)``
    ===================================== ===========================
    ``t < t_on``                          ``0``
    ``t_on <= t < t_on + ramp``           ``sin^2(c (t - t_on))``
    ``t_on + ramp <= t < t_off - ramp``   ``1``
    ``t_off - ramp <= t < t_off``         ``sin^2(c (t_off - t))``
    ``t >= t_off``                        ``0``
    ===================================== ===========================

    so ``A`` and ``dA = +/- c sin(2 c u)`` are continuous everywhere, while
    ``d2A = 2 c^2 cos(2 c u)`` jumps at the four gate corners.  The half-open intervals
    above fix ``d2A`` there (right-continuous); the value is defined almost everywhere and
    the choice is immaterial to any integral.

    Requires ``ramp > 0`` and ``t_off - t_on >= 2 ramp`` (rise and fall may just touch,
    never overlap).
    """

    t_on: float
    t_off: float
    ramp: float

    def __post_init__(self) -> None:
        t_on, t_off, ramp = float(self.t_on), float(self.t_off), float(self.ramp)
        if not all(math.isfinite(v) for v in (t_on, t_off, ramp)):
            raise ValueError("SmoothOnOff parameters must be finite")
        if ramp <= 0.0:
            raise ValueError(f"SmoothOnOff.ramp must be positive, got {ramp!r}")
        if t_off - t_on < 2.0 * ramp:
            raise ValueError(
                f"SmoothOnOff needs t_off - t_on >= 2 * ramp (rise and fall would overlap): "
                f"t_on={t_on!r}, t_off={t_off!r}, ramp={ramp!r}"
            )
        object.__setattr__(self, "t_on", t_on)
        object.__setattr__(self, "t_off", t_off)
        object.__setattr__(self, "ramp", ramp)

    @property
    def _c(self) -> float:
        return math.pi / (2.0 * self.ramp)

    def _conditions(self, t: NDArray[np.float64]) -> list[NDArray[np.bool_]]:
        return [
            t < self.t_on,
            t < self.t_on + self.ramp,
            t < self.t_off - self.ramp,
            t < self.t_off,
        ]

    def A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Envelope value (0 outside the gate, 1 on the plateau)."""
        t_arr = np.asarray(t, dtype=np.float64)
        c = self._c
        rise = np.sin(c * (t_arr - self.t_on)) ** 2
        fall = np.sin(c * (self.t_off - t_arr)) ** 2
        values = np.select(
            self._conditions(t_arr),
            [np.zeros(t_arr.shape), rise, np.ones(t_arr.shape), fall],
            default=0.0,
        )
        return _shaped(values, t_arr)

    def dA(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """First derivative ``+/- c sin(2 c u)`` on the ramps, zero elsewhere."""
        t_arr = np.asarray(t, dtype=np.float64)
        c = self._c
        rise = c * np.sin(2.0 * c * (t_arr - self.t_on))
        fall = -c * np.sin(2.0 * c * (self.t_off - t_arr))
        values = np.select(
            self._conditions(t_arr),
            [np.zeros(t_arr.shape), rise, np.zeros(t_arr.shape), fall],
            default=0.0,
        )
        return _shaped(values, t_arr)

    def d2A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Second derivative ``2 c^2 cos(2 c u)`` on the ramps, zero elsewhere.

        Piecewise; the four gate corners take the value of the interval starting there.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        c = self._c
        rise = 2.0 * c**2 * np.cos(2.0 * c * (t_arr - self.t_on))
        fall = 2.0 * c**2 * np.cos(2.0 * c * (self.t_off - t_arr))
        values = np.select(
            self._conditions(t_arr),
            [np.zeros(t_arr.shape), rise, np.zeros(t_arr.shape), fall],
            default=0.0,
        )
        return _shaped(values, t_arr)


# -------------------------------------------------------------------------- tone track


@dataclass(frozen=True)
class ToneTrack:
    """One RF tone: a frequency law, an amplitude envelope and a starting phase.

    Parameters
    ----------
    freq:
        Detuning from ``f_center`` [Hz] versus time [s] (Eq. S2 rotating frame).
    env:
        Amplitude envelope; defaults to a flat, fully-on envelope.
    phase0:
        Constant phase offset [rad] (Schroeder phases for tone ladders live here).

    The phase is the exact antiderivative ``phase(t) = 2 pi * int freq dt + phase0``,
    cached on first use.  Outside ``freq.domain`` :class:`~aodl.poly.PiecewisePoly`
    clamp-holds, so the *frequency* keeps its terminal value but the *phase* stops
    advancing — extend the track with :meth:`with_hold_until` whenever it must stay
    coherent past its last programmed segment.
    """

    freq: PiecewisePoly
    env: Envelope = field(default_factory=ConstantEnvelope)
    phase0: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.freq, PiecewisePoly):
            raise TypeError(f"ToneTrack.freq must be a PiecewisePoly, got {type(self.freq)!r}")
        for name in ("A", "dA", "d2A"):
            if not callable(getattr(self.env, name, None)):
                raise TypeError(
                    f"ToneTrack.env must implement the Envelope protocol "
                    f"(A/dA/d2A); {type(self.env)!r} has no {name!r}"
                )
        phase0 = float(self.phase0)
        if not math.isfinite(phase0):
            raise ValueError(f"ToneTrack.phase0 must be finite, got {self.phase0!r}")
        object.__setattr__(self, "phase0", phase0)

    # -- cached derived polynomials (built lazily; frozen dataclasses keep a __dict__)

    @functools.cached_property
    def _phase_poly(self) -> PiecewisePoly:
        """``2 pi * int freq dt``, zero at the start of the domain (``phase0`` added on top)."""
        return self.freq.antiderivative().scale(_TWO_PI)

    @functools.cached_property
    def _fdot_poly(self) -> PiecewisePoly:
        """``d freq / dt`` [Hz/s] — the chirp rate that drives the lens term (Eq. S6)."""
        return self.freq.derivative()

    # -- evaluation

    @property
    def t_span(self) -> tuple[float, float]:
        """Programmed time span ``(t_start, t_end)`` [s] of the frequency law."""
        return self.freq.domain

    def f(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Instantaneous detuning ``f(t)`` [Hz] (clamp-held outside the domain)."""
        return self.freq(t)

    def fdot(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Chirp rate ``df/dt`` [Hz/s]."""
        return self._fdot_poly(t)

    def phase(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Rotating-frame phase ``2 pi int_{t_start}^{t} f dt' + phase0`` [rad]."""
        return self._phase_poly(t) + self.phase0

    def A(self, t: ArrayLike) -> NDArray[np.float64] | float:
        """Envelope value ``A(t)`` (shorthand for ``self.env.A(t)``)."""
        return self.env.A(t)

    # -- transforms

    def with_hold_until(self, t_end: float) -> ToneTrack:
        """Return a copy whose frequency law is held constant out to ``t_end``.

        Appends one constant segment at the terminal frequency, so the phase keeps
        advancing coherently (the antiderivative is continuous across the join) instead of
        freezing at the domain edge.  Returns ``self`` unchanged when ``t_end`` is not
        beyond the current end.
        """
        t_end = float(t_end)
        _, t1 = self.freq.domain
        if t_end <= t1 + TIME_TOL:
            return self
        tail = PiecewisePoly.constant(float(self.freq(t1)), t1, t_end)
        return replace(self, freq=PiecewisePoly.concat([self.freq, tail]))


# ----------------------------------------------------------------------- channel/set


@dataclass(frozen=True)
class ChannelWaveform:
    """All tones driving one AOD channel."""

    tones: tuple[ToneTrack, ...] = ()

    def __post_init__(self) -> None:
        tones = tuple(self.tones)
        for i, tone in enumerate(tones):
            if not isinstance(tone, ToneTrack):
                raise TypeError(f"ChannelWaveform.tones[{i}] must be a ToneTrack, got {tone!r}")
        object.__setattr__(self, "tones", tones)

    def __len__(self) -> int:
        return len(self.tones)

    @property
    def n_tones(self) -> int:
        """Number of tones on this channel."""
        return len(self.tones)

    @property
    def t_span(self) -> tuple[float, float]:
        """Union of the tone frequency domains ``(t_start, t_end)`` [s]."""
        if not self.tones:
            raise ValueError("an empty ChannelWaveform has no time span")
        starts, ends = zip(*(tone.t_span for tone in self.tones), strict=True)
        return min(starts), max(ends)

    def with_hold_until(self, t_end: float) -> ChannelWaveform:
        """Hold every tone's terminal frequency out to ``t_end`` (see
        :meth:`ToneTrack.with_hold_until`)."""
        return ChannelWaveform(tuple(tone.with_hold_until(t_end) for tone in self.tones))

    def eval_table(self, t: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """Evaluate every tone at ``t`` in one pass — the device layer's input.

        Returns a dict with the keys :data:`TABLE_KEYS` (``f`` [Hz], ``fdot`` [Hz/s],
        ``A``, ``dA`` [1/s], ``d2A`` [1/s^2], ``phase`` [rad]); each value is an array of
        shape ``(n_tones, *np.shape(t))``, so a scalar ``t`` yields one row of length
        ``n_tones`` per quantity.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        shape = (self.n_tones, *t_arr.shape)
        table = {key: np.empty(shape, dtype=np.float64) for key in TABLE_KEYS}
        for i, tone in enumerate(self.tones):
            table["f"][i] = tone.f(t_arr)
            table["fdot"][i] = tone.fdot(t_arr)
            table["A"][i] = tone.env.A(t_arr)
            table["dA"][i] = tone.env.dA(t_arr)
            table["d2A"][i] = tone.env.d2A(t_arr)
            table["phase"][i] = tone.phase(t_arr)
        return table


@dataclass(frozen=True)
class WaveformSet:
    """A complete, self-describing drive for (a subset of) the four AODL channels.

    Parameters
    ----------
    channels:
        ``{channel name: ChannelWaveform}``; keys must be a subset of
        :data:`aodl.params.CHANNELS`.  Absent channels mean "undriven" (the device layer
        treats them as an identity factor in the pupil product).
    params:
        Hardware snapshot the waveform was designed for — carried with the waveform so a
        loaded file needs no external context.
    description:
        Free-text provenance, echoed into the NPZ metadata.

    Every tone must cover the same time span; use
    :meth:`ChannelWaveform.with_hold_until` to extend short ones.
    """

    channels: dict[str, ChannelWaveform]
    params: AODLParams
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.params, AODLParams):
            raise TypeError(f"WaveformSet.params must be an AODLParams, got {type(self.params)!r}")
        if not isinstance(self.channels, Mapping):
            raise TypeError("WaveformSet.channels must be a mapping {name: ChannelWaveform}")
        channels = dict(self.channels)
        if not channels:
            raise ValueError("WaveformSet needs at least one channel")
        unknown = [name for name in channels if name not in CHANNELS]
        if unknown:
            raise ValueError(f"unknown channel name(s) {unknown}; valid names are {list(CHANNELS)}")
        for name, cw in channels.items():
            if not isinstance(cw, ChannelWaveform):
                raise TypeError(f"channel {name!r} must hold a ChannelWaveform, got {cw!r}")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "description", str(self.description))
        self._validate_spans()

    def _validate_spans(self) -> None:
        spans = [
            (name, i, tone.t_span)
            for name, cw in self.channels.items()
            for i, tone in enumerate(cw.tones)
        ]
        if not spans:
            raise ValueError("WaveformSet needs at least one tone")
        ref_name, ref_i, (ref_t0, ref_t1) = spans[0]
        for name, i, (t0, t1) in spans[1:]:
            if abs(t0 - ref_t0) > TIME_TOL or abs(t1 - ref_t1) > TIME_TOL:
                raise ValueError(
                    f"all tone frequency domains must cover the same span: "
                    f"tone {name}[{i}] spans ({t0!r}, {t1!r}) but tone "
                    f"{ref_name}[{ref_i}] spans ({ref_t0!r}, {ref_t1!r}). "
                    f"Extend the short ones with ToneTrack.with_hold_until(t_end) or "
                    f"ChannelWaveform.with_hold_until(t_end)."
                )

    @property
    def t_span(self) -> tuple[float, float]:
        """Union of all tone frequency domains ``(t_start, t_end)`` [s]."""
        spans = [cw.t_span for cw in self.channels.values() if cw.n_tones]
        starts, ends = zip(*spans, strict=True)
        return min(starts), max(ends)

    @property
    def n_tones(self) -> int:
        """Total tone count across all channels."""
        return sum(cw.n_tones for cw in self.channels.values())

    def with_hold_until(self, t_end: float) -> WaveformSet:
        """Hold every tone in every channel out to ``t_end``."""
        return replace(
            self,
            channels={name: cw.with_hold_until(t_end) for name, cw in self.channels.items()},
        )

    def eval_table(self, t: ArrayLike) -> dict[str, dict[str, NDArray[np.float64]]]:
        """``{channel: ChannelWaveform.eval_table(t)}`` for every driven channel."""
        return {name: cw.eval_table(t) for name, cw in self.channels.items()}

    # -- persistence (thin wrappers over aodl.waveform.serialize)

    def save(self, path: str | Path) -> Path:
        """Write the *parametric* NPZ (never samples).  See ``docs/waveform_format.md``."""
        from . import serialize

        return serialize.save(self, path)

    @staticmethod
    def load(path: str | Path) -> WaveformSet:
        """Read a parametric NPZ written by :meth:`save`."""
        from . import serialize

        return serialize.load(path)


def channel_waveform(tones: Iterable[ToneTrack]) -> ChannelWaveform:
    """Convenience constructor: ``ChannelWaveform`` from any iterable of tones."""
    return ChannelWaveform(tuple(tones))


__all__ = [
    "TABLE_KEYS",
    "TIME_TOL",
    "ChannelWaveform",
    "ConstantEnvelope",
    "Envelope",
    "SmoothOnOff",
    "ToneTrack",
    "WaveformSet",
    "channel_waveform",
]
