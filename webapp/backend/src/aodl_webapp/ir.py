r"""Waveform IR as JSON (API.md §4.1) and its pure-numpy reference evaluator.

:func:`to_ir` serialises a :class:`aodl.WaveformSet` tone by tone: the frequency law as a
normalized-local-time piecewise polynomial in **MHz**, the phase law ``2 pi int f dt`` in
**rad** (:class:`aodl.ToneTrack`'s own antiderivative, so the frontend never integrates), the
constant ``phase0`` and the parametric envelope.  Nothing is sampled.

:func:`evaluate_ir` is the single source of truth the TypeScript evaluator
(``frontend/src/physics``) is tested against: it reads only the JSON-level IR — never the
``aodl`` objects — and reproduces :meth:`aodl.ToneTrack.f` / ``phase`` / ``env.A`` and the
carrier-included voltage of :func:`aodl.waveform.export.render_samples`,

.. math::

    V(t) = \sum_n A_n(t)\cos\big(2\pi f_\text{center} t + \varphi_n(t) + \phi_{0,n}\big),

without the global renormalisation ``render_samples`` applies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aodl.device.conventions import geometry
from aodl.poly import PiecewisePoly
from aodl.units import MHz, mm, us
from aodl.waveform.shepard import FadeZoneEnvelope, SwitchRamped
from aodl.waveform.tones import ConstantEnvelope, SmoothOnOff, ToneTrack, WaveformSet

from .errors import ApiError
from .schemas import (
    ChannelIR,
    ConstantEnv,
    FadeZoneEnv,
    Poly,
    SmoothOnOffEnv,
    ToneIR,
    WaveformIR,
)

Float = NDArray[np.float64]

_TWO_PI = 2.0 * math.pi


# ------------------------------------------------------------------------------ to_ir


def poly_to_ir(poly: PiecewisePoly, value_scale: float = 1.0) -> Poly:
    """A :class:`aodl.PiecewisePoly` with breaks in µs and coefficients divided by ``scale``.

    The normalized local time ``tau`` is dimensionless, so changing the unit of the breaks
    does not touch the coefficients, and changing the unit of the *value* scales every
    coefficient alike.
    """
    return Poly(
        breaks_us=(poly.breaks / us).tolist(),
        coeffs=(poly.coeffs / value_scale).tolist(),
    )


def envelope_to_ir(env: object) -> ConstantEnv | SmoothOnOffEnv | FadeZoneEnv:
    """One ``aodl`` envelope -> its API record; ``SwitchRamped`` is refused (DESIGN §1 item 4)."""
    if isinstance(env, ConstantEnvelope):
        return ConstantEnv(amp=env.amp)
    if isinstance(env, SmoothOnOff):
        return SmoothOnOffEnv(t_on_us=env.t_on / us, t_off_us=env.t_off / us, ramp_us=env.ramp / us)
    if isinstance(env, FadeZoneEnvelope):
        return FadeZoneEnv(
            g=poly_to_ir(env.g, MHz),
            delta_f_MHz=env.delta_f / MHz,
            eta=env.eta,
            p=env.p,
            m=int(round(env.m)),
            amp=env.amp,
        )
    if isinstance(env, SwitchRamped):
        raise ApiError(
            "unsupported",
            "the drive carries a SwitchRamped envelope (switch_ramp > 0), which the waveform IR "
            "does not represent; synthesize with switch_ramp = 0.",
        )
    raise ApiError("unsupported", f"unknown envelope type {type(env).__name__!r}")


def tone_to_ir(tone: ToneTrack) -> ToneIR:
    """One :class:`aodl.ToneTrack`: frequency [MHz], phase [rad], ``phase0`` [rad], envelope."""
    return ToneIR(
        freq=poly_to_ir(tone.freq, MHz),
        phase=poly_to_ir(tone.freq.antiderivative().scale(_TWO_PI)),
        phase0=tone.phase0,
        env=envelope_to_ir(tone.env),
    )


def to_ir(wfs: WaveformSet) -> WaveformIR:
    """API.md §4.1 — the whole drive as JSON-ready parameters, channels in ``aodl`` order."""
    channels: dict[str, ChannelIR] = {}
    for name, cw in wfs.channels.items():
        aod = wfs.params.channels[name]
        geom = geometry(name)
        channels[name] = ChannelIR(
            f_center_MHz=aod.f_center / MHz,
            tau_us=aod.transit_time / us,
            sound_sign=geom.sound_sign,  # type: ignore[arg-type]
            axis=geom.axis_name,  # type: ignore[arg-type]
            aperture_mm=aod.aperture / mm,
            sound_speed_m_s=aod.sound_speed,
            band_MHz=(aod.band[0] / MHz, aod.band[1] / MHz),
            tones=[tone_to_ir(tone) for tone in cw.tones],
        )
    t0, t1 = wfs.t_span
    return WaveformIR(channels=channels, t_span_us=(t0 / us, t1 / us))


# ------------------------------------------------------------------------ evaluate_ir


def eval_poly(poly: Poly, t_us: ArrayLike) -> Float:
    """Evaluate an IR polynomial at ``t_us`` [µs], clamp-held outside its domain.

    Mirrors :meth:`aodl.poly.PiecewisePoly.__call__` step for step: clip ``t`` to the domain,
    ``searchsorted(side="right") - 1`` picks the segment (the last break belongs to the last
    segment), ``tau = (t - breaks[k]) / (breaks[k+1] - breaks[k])``, Horner in ``tau``.
    """
    breaks = np.asarray(poly.breaks_us, dtype=np.float64)
    coeffs = np.asarray(poly.coeffs, dtype=np.float64)
    t = np.asarray(t_us, dtype=np.float64)
    n_segments = coeffs.shape[0]
    tc = np.clip(t, breaks[0], breaks[-1])
    idx = np.searchsorted(breaks, tc, side="right") - 1
    idx = np.clip(idx, 0, n_segments - 1)
    tau = (tc - breaks[idx]) / (breaks[idx + 1] - breaks[idx])
    c = coeffs[idx]
    out = np.array(c[..., -1], dtype=np.float64, copy=True)
    for j in range(coeffs.shape[1] - 2, -1, -1):
        out = out * tau + c[..., j]
    return np.asarray(out, dtype=np.float64)


def eval_envelope(env: ConstantEnv | SmoothOnOffEnv | FadeZoneEnv, t_us: ArrayLike) -> Float:
    """``A(t)`` of an IR envelope at ``t_us`` [µs] (API.md §4.1 formulas).

    * ``constant`` — ``amp``;
    * ``smooth_on_off`` — :class:`aodl.waveform.tones.SmoothOnOff`'s raised-sine gate with its
      half-open intervals;
    * ``fade_zone`` — Eqs. S26/S27 on ``g(t)``: ``amp`` for ``|g| <= (m - eta) df / 2``, ``0``
      for ``|g| >= (m + eta) df / 2``, ``amp cos^p theta`` between, with
      ``theta = (pi / 2 eta)(|g| / df - m / 2) + pi / 4``.
    """
    t = np.asarray(t_us, dtype=np.float64)
    if isinstance(env, ConstantEnv):
        return np.full(t.shape, env.amp, dtype=np.float64)
    if isinstance(env, SmoothOnOffEnv):
        c = math.pi / (2.0 * env.ramp_us)
        rise = np.sin(c * (t - env.t_on_us)) ** 2
        fall = np.sin(c * (env.t_off_us - t)) ** 2
        return np.asarray(
            np.select(
                [
                    t < env.t_on_us,
                    t < env.t_on_us + env.ramp_us,
                    t < env.t_off_us - env.ramp_us,
                    t < env.t_off_us,
                ],
                [np.zeros(t.shape), rise, np.ones(t.shape), fall],
                default=0.0,
            ),
            dtype=np.float64,
        )
    g = np.abs(eval_poly(env.g, t))
    df = env.delta_f_MHz
    inner = 0.5 * (env.m - env.eta) * df
    outer = 0.5 * (env.m + env.eta) * df
    theta = np.clip(
        (math.pi / (2.0 * env.eta)) * (g / df - 0.5 * env.m) + 0.25 * math.pi, 0.0, 0.5 * math.pi
    )
    shoulder = (g > inner) & (g < outer)
    return np.asarray(
        np.where(g <= inner, env.amp, np.where(shoulder, env.amp * np.cos(theta) ** env.p, 0.0)),
        dtype=np.float64,
    )


def evaluate_ir(ir: WaveformIR, channel: str, t_us: ArrayLike) -> dict[str, Float]:
    """Reference evaluation of one channel of the IR at ``t_us`` [µs].

    Returns ``{"f_MHz": (n_tones, n_t), "A": (n_tones, n_t), "phase_rad": (n_tones, n_t),
    "V": (n_t,)}`` — detuning, envelope, total phase ``phase(t) + phase0`` and the
    carrier-included voltage ``sum_n A_n cos(2 pi f_center t + phase_n)`` (no normalisation).
    Pure numpy over the JSON-level IR; agrees with :class:`aodl.ToneTrack` to round-off.
    """
    if channel not in ir.channels:
        raise KeyError(f"channel {channel!r} is not in the IR ({list(ir.channels)})")
    chan = ir.channels[channel]
    t = np.atleast_1d(np.asarray(t_us, dtype=np.float64))
    n = len(chan.tones)
    f = np.empty((n, *t.shape), dtype=np.float64)
    a = np.empty_like(f)
    phase = np.empty_like(f)
    for i, tone in enumerate(chan.tones):
        f[i] = eval_poly(tone.freq, t)
        a[i] = eval_envelope(tone.env, t)
        phase[i] = eval_poly(tone.phase, t) + tone.phase0
    carrier = _TWO_PI * (chan.f_center_MHz * MHz) * (t * us)
    v = np.sum(a * np.cos(carrier[None, ...] + phase), axis=0) if n else np.zeros(t.shape)
    return {"f_MHz": f, "A": a, "phase_rad": phase, "V": np.asarray(v, dtype=np.float64)}


__all__ = [
    "envelope_to_ir",
    "eval_envelope",
    "eval_poly",
    "evaluate_ir",
    "poly_to_ir",
    "to_ir",
    "tone_to_ir",
]
