"""Render a :class:`~aodl.waveform.tones.WaveformSet` to literal AWG samples.

This is the *only* place in the package where the carrier is put back: everywhere else
frequencies are detunings from ``f_center`` (Eq. S2 rotating frame), but an arbitrary
waveform generator wants the absolute RF signal

.. math::

    V_\\mu(t) = \\sum_n A^{(n)}(t)\\,
                \\cos\\!\\big(2\\pi f_\\text{center}^\\mu t + \\varphi^{(n)}(t)\\big),
    \\qquad \\varphi^{(n)}(t) = 2\\pi\\!\\int f^{(n)} dt' + \\phi_0^{(n)} .

Samples are a *render target*, never the stored object (``docs/ARCHITECTURE.md`` §0.1):
they are regenerated from the parametric NPZ at whatever rate the hardware runs.  All
channels are divided by a single common factor — the global peak over the whole render —
so relative channel amplitudes, which set the diffraction balance of the four AODs,
survive normalization intact.  The factor is reported so the physical scale is recoverable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..units import MHz
from .tones import WaveformSet

#: Default AWG sample rate (Spectrum M4i.6631-x8, ``docs/PLAN.md`` §1.5).
DEFAULT_SAMPLE_RATE = 625.0 * MHz

#: Schema version of the ``*_samples.npz`` metadata block.
SAMPLES_SCHEMA_VERSION = 1

#: Required filename suffix for rendered-sample files — samples must never be mistaken
#: for the parametric exchange format.
SAMPLES_SUFFIX = "_samples.npz"

_TWO_PI = 2.0 * math.pi


def sample_times(t_span: tuple[float, float], sample_rate: float) -> tuple[int, float]:
    """``(n_samples, t_start)`` for a closed span sampled at ``sample_rate``.

    Sample ``k`` sits at ``t_start + k / sample_rate``; the count is rounded so that the
    last sample lands on ``t_span[1]`` when the span is a whole number of periods.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        raise ValueError(f"t_span must be increasing and finite, got {t_span!r}")
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive and finite, got {sample_rate!r}")
    return int(round((t1 - t0) * sample_rate)) + 1, t0


def render_samples(
    wfs: WaveformSet,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    t_span: tuple[float, float] | Sequence[float] | None = None,
    dtype: DTypeLike = np.float32,
    chunk: int = 2**20,
    return_scale: bool = False,
) -> dict[str, NDArray[Any]] | tuple[dict[str, NDArray[Any]], float]:
    """Expand ``wfs`` into normalized AWG samples, one array per driven channel.

    Parameters
    ----------
    wfs:
        The waveform set to render.  Each channel's carrier is taken from
        ``wfs.params.channels[name].f_center``.
    sample_rate:
        Samples per second (default 625 MS/s).
    t_span:
        ``(t_start, t_end)``; defaults to ``wfs.t_span``.  Beyond a tone's programmed
        domain the frequency clamp-holds but the *phase* stops advancing, so the tone
        collapses onto the bare carrier — call ``wfs.with_hold_until(t_end)`` first
        whenever the render is meant to run past the last programmed segment.
    dtype:
        Output floating dtype (default ``float32``, what AWG SDKs want).
    chunk:
        Samples evaluated per pass — bounds peak memory for long renders.
    return_scale:
        When true, also return the normalization factor (see below).

    Returns
    -------
    ``{channel: samples}`` with every array divided by the *global* peak
    ``max_channels max_t |V(t)|``, i.e. the full-scale sample is ``1.0`` on exactly one
    channel and the others keep their relative amplitude.  With ``return_scale=True`` the
    return value is ``({channel: samples}, scale)``; multiply the samples by ``scale`` to
    recover the un-normalized sum of tone amplitudes.
    """
    samples, scale = _render(wfs, sample_rate, _resolve_span(wfs, t_span), dtype, chunk)
    return (samples, scale) if return_scale else samples


def _resolve_span(
    wfs: WaveformSet, t_span: tuple[float, float] | Sequence[float] | None
) -> tuple[float, float]:
    if t_span is None:
        return wfs.t_span
    return float(t_span[0]), float(t_span[1])


def _render(
    wfs: WaveformSet,
    sample_rate: float,
    span: tuple[float, float],
    dtype: DTypeLike,
    chunk: int,
) -> tuple[dict[str, NDArray[Any]], float]:
    """Chunked render + global normalization; returns ``(samples, scale)``."""
    out_dtype = np.dtype(dtype)
    if not np.issubdtype(out_dtype, np.floating):
        raise ValueError(f"dtype must be a floating type, got {out_dtype}")
    if int(chunk) < 1:
        raise ValueError(f"chunk must be at least 1 sample, got {chunk!r}")
    chunk = int(chunk)
    n_samples, t0 = sample_times(span, sample_rate)

    out = {name: np.empty(n_samples, dtype=out_dtype) for name in wfs.channels}
    carriers = {name: wfs.params.channels[name].f_center for name in wfs.channels}
    peak = 0.0
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        t = t0 + np.arange(start, stop, dtype=np.float64) / float(sample_rate)
        for name, cw in wfs.channels.items():
            block = np.zeros(t.shape, dtype=np.float64)
            if cw.n_tones:
                table = cw.eval_table(t)
                carrier = _TWO_PI * carriers[name] * t
                for i in range(cw.n_tones):
                    block += table["A"][i] * np.cos(carrier + table["phase"][i])
                peak = max(peak, float(np.max(np.abs(block))))
            out[name][start:stop] = block

    scale = peak if peak > 0.0 else 1.0
    for arr in out.values():
        arr /= scale
    return out, scale


def save_samples(
    wfs: WaveformSet,
    path: str | Path,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    t_span: tuple[float, float] | Sequence[float] | None = None,
    dtype: DTypeLike = np.float32,
    chunk: int = 2**20,
) -> Path:
    """Render and write ``<name>_samples.npz``: one array per channel plus ``meta`` JSON.

    The mandatory ``_samples.npz`` suffix keeps rendered files visibly distinct from the
    parametric exchange format written by :func:`aodl.waveform.serialize.save`.  The
    metadata echoes ``sample_rate``, ``t_span``, ``n_samples``, the per-channel carriers
    and the ``normalization`` factor, so the physical signal is fully recoverable.
    """
    path = Path(path)
    if not path.name.endswith(SAMPLES_SUFFIX):
        raise ValueError(
            f"rendered sample files must be named '*{SAMPLES_SUFFIX}' (got {path.name!r}) "
            "so they cannot be confused with a parametric waveform NPZ"
        )
    span = _resolve_span(wfs, t_span)
    n_samples, t0 = sample_times(span, sample_rate)
    samples, scale = _render(wfs, sample_rate, span, dtype, chunk)
    meta: dict[str, Any] = {
        "schema_version": SAMPLES_SCHEMA_VERSION,
        "kind": "samples",
        "description": wfs.description,
        "sample_rate": float(sample_rate),
        "t_span": [float(span[0]), float(span[1])],
        "n_samples": int(n_samples),
        "t_start": float(t0),
        "normalization": float(scale),
        "channels": list(samples),
        "f_center": {name: wfs.params.channels[name].f_center for name in samples},
        "dtype": str(np.dtype(dtype)),
    }
    arrays: dict[str, Any] = dict(samples)
    np.savez(path, meta=json.dumps(meta), **arrays)
    return path


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "SAMPLES_SCHEMA_VERSION",
    "SAMPLES_SUFFIX",
    "render_samples",
    "sample_times",
    "save_samples",
]
