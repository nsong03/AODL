r"""RF spectrum, crest factor and peak modulation depth of a rendered channel drive.

The literal AWG samples come from :func:`aodl.waveform.export.render_samples` (the one place
the carrier is put back).  Over a window ``[t_c - W/2, t_c + W/2]``:

* **PSD** — Hann-windowed periodogram ``|FFT(w V)|^2`` of the real waveform, one-sided,
  normalised to 0 dB at the strongest line;
* **crest factor** — ``max |V| / rms(V)``; **PAPR** — ``20 log10(crest)`` dB;
* **peak modulation depth** — ``C * max|V| / A_single_tone`` [rad] with ``C`` the channel's
  ``drive_strength`` (the phase a unit-envelope tone imprints, Eq. S1) and ``A_single_tone =
  1`` the unit relative envelope: the largest instantaneous phase excursion of the thin
  grating, i.e. how far into the ``J1`` saturation the multi-tone drive pushes.

``render_samples`` normalises every channel by the *global* peak of the window; the returned
``scale`` undoes that, so all three quantities refer to the un-normalised sum of tone
amplitudes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from aodl.params import CHANNELS
from aodl.units import us
from aodl.waveform.export import render_samples
from aodl.waveform.tones import WaveformSet

Float = NDArray[np.float64]

#: Unit relative tone envelope the modulation depth is referred to.
A_SINGLE_TONE = 1.0


@dataclass(frozen=True)
class SpectrumResult:
    """One channel's spectrum over one window (SI: Hz, dB, rad)."""

    f: Float
    psd_db: Float
    crest_factor: float
    papr_db: float
    peak_depth_rad: float
    t_span: tuple[float, float]
    notes: list[str] = field(default_factory=list)


def spectrum(
    wfs: WaveformSet,
    channel: str,
    t_center: float,
    window: float,
    sample_rate: float,
) -> SpectrumResult:
    """Hann periodogram and crest statistics of ``channel`` around ``t_center`` [s].

    The window is clamped to the programmed drive span (a note says so).
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
    if channel not in wfs.channels:
        raise ValueError(f"channel {channel!r} is not driven by this plan")
    if not (math.isfinite(window) and window > 0.0):
        raise ValueError(f"window must be positive, got {window!r}")
    if not (math.isfinite(sample_rate) and sample_rate > 0.0):
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    notes: list[str] = []
    t0, t1 = wfs.t_span
    lo = float(t_center) - 0.5 * float(window)
    hi = float(t_center) + 0.5 * float(window)
    if lo < t0 or hi > t1:
        lo, hi = max(lo, t0), min(hi, t1)
        notes.append(
            f"window clamped to the programmed drive span [{lo / us:.4g}, {hi / us:.4g}] us"
        )
    if not hi > lo:
        raise ValueError("the spectrum window lies entirely outside the programmed drive")
    arrays, scale = cast(
        tuple[dict[str, NDArray[np.float64]], float],
        render_samples(wfs, sample_rate, (lo, hi), dtype=np.float64, return_scale=True),
    )
    v = np.asarray(arrays[channel], dtype=np.float64) * float(scale)
    n = v.size
    taper = np.hanning(n)
    psd = np.abs(np.fft.rfft(taper * v)) ** 2
    peak_bin = float(np.max(psd))
    with np.errstate(divide="ignore"):
        psd_db = 10.0 * np.log10(psd / peak_bin) if peak_bin > 0.0 else np.zeros(psd.shape)
    psd_db = np.maximum(psd_db, -200.0)
    freqs = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
    rms = float(np.sqrt(np.mean(v * v)))
    peak = float(np.max(np.abs(v)))
    crest = peak / rms if rms > 0.0 else 0.0
    papr = 20.0 * math.log10(crest) if crest > 0.0 else 0.0
    depth = float(wfs.params.channels[channel].drive_strength) * peak / A_SINGLE_TONE
    return SpectrumResult(
        f=np.asarray(freqs, dtype=np.float64),
        psd_db=np.asarray(psd_db, dtype=np.float64),
        crest_factor=crest,
        papr_db=papr,
        peak_depth_rad=depth,
        t_span=(lo, hi),
        notes=notes,
    )


__all__ = ["A_SINGLE_TONE", "SpectrumResult", "spectrum"]
