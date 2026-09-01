r"""Sampled RF -> complex baseband: one FFT per channel (Eqs. S1-S2).

The drive of Eq. S1 is a real signal riding a common carrier,

.. math::

    V_\mu(t) = \sum_n A^{(n)}(t)\cos\big(2\pi f_\text{center}^\mu t + \varphi^{(n)}(t)\big)
             = \mathrm{Re}\big[z_\mu(t)\, e^{i 2\pi f_\text{center}^\mu t}\big],
    \qquad z_\mu = \sum_n A^{(n)} e^{i\varphi^{(n)}},

and every pupil the checker builds is a function of that complex envelope ``z`` evaluated at
the *retarded* time of each aperture point (Eq. S4).  Recovering ``z`` from samples is the
one and only signal-processing step of the checker::

    z = 2 P+[V] * exp(-i 2 pi f_center t)

with ``P+`` the positive-frequency projection (complex FFT, zero the negative half, inverse
FFT) — i.e. the analytic signal.  The record's :attr:`~aodl.check.record.SampleRecord.
normalization` is folded in here, once, so ``|z|`` comes out in the envelope units the
simulator uses (``A`` in ``[0, 1]``) and ``drive_strength`` multiplies it unchanged.

Nothing about the parametric waveform is used: the tone frequencies, phases and envelopes are
*measured* off the buffer.  That is the whole point of M6.

Error budget
------------

Two effects set the floor, both measured on rendered ``float64`` records at the product
defaults (``docs/PLAN.md`` §1.5: ``f_center = 100 MHz``, band ±10 MHz, ``f_s = 625 MS/s``):

**1. Interpolation.**  :func:`sample_baseband` gathers ``z`` at arbitrary retarded times with
a cubic Hermite (Catmull-Rom) kernel, which is Keys' cubic convolution with ``a = -1/2`` and
therefore **third**-order accurate.  On a baseband line at detuning ``f_bb`` the worst-case
modulus error over the interpolation phase is

.. math::  \varepsilon_\text{interp} \simeq 0.0160\,\theta^3, \qquad
           \theta = 2\pi f_\text{bb}/f_s ,

measured 1.60e-2 x theta^3 to three digits over ``theta`` in [0.025, 0.4].  That is 4.4e-7 at
a 3 MHz detuning and 5.5e-5 at the 15 MHz band edge.  ``oversample=r`` divides ``theta`` by
``r`` and hence the error by ``r^3`` (measured 64x for ``r = 4``).  *Note for readers of
``docs/workorders/WO-21-check-core.md``*: the work order quotes a quartic budget
``(2 pi f_bb/f_s)^4/384 ~ 1.5e-6``; Catmull-Rom does not deliver a quartic law, and the cubic
one above is what this module actually achieves (it is pinned by
``tests/test_check_demod.py::test_cubic_hermite_error_follows_the_measured_cubic_law``).

**2. Analytic-signal edge ringing.**  A record is a finite window on the drive, so ``P+``
sees a truncation at each end.  The resulting error decays as

.. math::  \varepsilon_\text{edge} \simeq \frac{1}{\pi f_s \Delta t}
                                   = \frac{1}{\pi\,(\text{samples from the record end})},

measured to 5 % at ``f_s`` = 625, 1250 and 2500 MS/s (5.4e-5 at 10 us from the end at
625 MS/s).  It scales with the *sample* rate, not the carrier — the work order's
``1/(pi f_center dt)`` form happens to agree numerically at the product ``f_s / f_center =
6.25 ~ 2 pi`` but overstates the error by that factor.  Half of it is the FFT's circular wrap
and half is the genuine truncation of an abruptly-started drive; zero-padding the transform
removes only the first half (measured 1.8x), which is why it is not done here.  A drive whose
envelope ramps to zero at both record ends has no truncation at all and demodulates to ~1e-11.

This is why WO-22 marks early frames "transient": at a frame time ``t`` the aperture reads
drive times ``t_c -+ (u_max/v)``, and the earliest of those sits ``t_c - u_max/v`` after the
drive start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..params import CHANNELS, AODLParams
from ..units import us
from .record import SampleRecord

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]

_TWO_PI = 2.0 * math.pi

#: Absolute slack [s] when deciding whether a requested time runs past the record.  A
#: picosecond is far below the 1.6 ns sample period at 625 MS/s and far above float64
#: round-off on microsecond-scale times (mirrors ``aodl.waveform.tones.TIME_TOL``).
TIME_TOL = 1e-12

#: Window applied before the diagnostic record FFT of :func:`out_of_band_fraction`.
WINDOWS: tuple[str, ...] = ("hann", "boxcar")

__all__ = [
    "TIME_TOL",
    "WINDOWS",
    "Baseband",
    "demodulate",
    "out_of_band_fraction",
    "sample_baseband",
]


@dataclass(frozen=True)
class Baseband:
    """Complex envelopes ``z_mu(t)`` of every channel of a :class:`SampleRecord`.

    Attributes
    ----------
    z:
        ``{channel: envelope}``, equal-length 1D complex arrays.  ``z`` is in *envelope*
        units: for a single tone of programmed amplitude ``A`` and phase ``phi``,
        ``z = A exp(i phi)`` (the record's ``normalization`` is already folded in).
    sample_rate:
        Samples per second of ``z`` — ``oversample`` times the record's rate.
    t_start:
        Time of sample 0 [s]; the same instant as the record's sample 0.
    params:
        Hardware, carried through from the record.
    """

    z: dict[str, Complex]
    sample_rate: float
    t_start: float
    params: AODLParams

    def __post_init__(self) -> None:
        if not isinstance(self.params, AODLParams):
            raise TypeError(f"Baseband.params must be an AODLParams, got {type(self.params)!r}")
        if not self.z:
            raise ValueError("Baseband needs at least one channel")
        arrays: dict[str, Complex] = {}
        length: int | None = None
        for name, values in self.z.items():
            if name not in CHANNELS:
                raise ValueError(f"unknown channel name {name!r}; valid names are {list(CHANNELS)}")
            arr = np.asarray(values, dtype=np.complex128)
            if arr.ndim != 1:
                raise ValueError(f"channel {name!r} envelope must be 1-D, got shape {arr.shape}")
            if length is None:
                length = arr.size
            elif arr.size != length:
                raise ValueError("all channel envelopes must have the same length")
            arrays[name] = arr
        rate = float(self.sample_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"sample_rate must be positive and finite, got {self.sample_rate!r}")
        object.__setattr__(self, "z", arrays)
        object.__setattr__(self, "sample_rate", rate)
        object.__setattr__(self, "t_start", float(self.t_start))

    @property
    def n_samples(self) -> int:
        """Samples per channel envelope."""
        return int(next(iter(self.z.values())).size)

    @property
    def t_span(self) -> tuple[float, float]:
        """``(t_start, t_last)`` [s] of the envelope grid."""
        return self.t_start, self.t_start + (self.n_samples - 1) / self.sample_rate


def _analytic(v: Float, oversample: int) -> Complex:
    """``2 P+[v]`` on a grid ``oversample`` times finer.  Eq. S2's rotating-frame lift.

    The positive-frequency projection is the FFT with the negative half zeroed; doubling the
    kept bins (DC and Nyquist excepted, as in ``scipy.signal.hilbert``) makes the result the
    analytic signal, whose real part is ``v`` and whose modulus is the envelope.  Zero-padding
    the *spectrum* out to ``oversample`` times the length is exact band-limited interpolation
    onto the finer time grid, so the only thing ``oversample`` changes downstream is the
    cubic-Hermite gather's step size.
    """
    n = v.size
    spectrum = np.fft.fft(v)
    keep = n // 2 + 1  # bins 0 .. Nyquist for even n; 0 .. (n-1)/2 for odd n
    analytic = np.zeros(n * oversample, dtype=np.complex128)
    analytic[:keep] = spectrum[:keep]
    analytic[1 : (n + 1) // 2] *= 2.0  # DC and (even n) Nyquist stay un-doubled
    return np.asarray(np.fft.ifft(analytic) * oversample, dtype=np.complex128)


def demodulate(rec: SampleRecord, *, oversample: int = 1) -> Baseband:
    """Recover every channel's complex envelope from the sample buffers.  Eqs. S1-S2.

    ``z_mu[k] = 2 P+[V_mu][k] exp(-i 2 pi f_center_mu t_k)`` with
    ``V_mu = normalization * samples`` (:attr:`aodl.check.record.SampleRecord.drive`).

    Parameters
    ----------
    rec:
        The sample record.
    oversample:
        Spectral zero-padding factor, ``>= 1``.  The envelope is returned on a grid
        ``oversample`` times finer, which cuts :func:`sample_baseband`'s interpolation error
        by ``oversample**3`` (see the module docstring) at ``oversample`` times the memory.
        ``1`` is right for production; the tests use it as a convergence knob.

    Returns
    -------
    :class:`Baseband` on a grid starting at the record's own ``t_start``.
    """
    factor = int(oversample)
    if factor < 1:
        raise ValueError(f"oversample must be a positive integer, got {oversample!r}")
    envelopes: dict[str, Complex] = {}
    rate = rec.sample_rate * factor
    times = rec.t_start + np.arange(rec.n_samples * factor, dtype=np.float64) / rate
    for name in rec.channels:
        analytic = _analytic(rec.drive(name), factor)
        carrier = rec.params.channels[name].f_center
        envelopes[name] = analytic * np.exp(-1j * _TWO_PI * carrier * times)
    return Baseband(z=envelopes, sample_rate=rate, t_start=rec.t_start, params=rec.params)


def _coverage_error(bb: Baseband, channel: str, t_max: float) -> str:
    """Message for a requested time that runs past the end of the record (engine style)."""
    t0, t1 = bb.t_span
    return (
        f"channel {channel!r}: the checker needs the drive at t = {t_max / us:.4g} us but the "
        f"record only covers [{t0 / us:.4g}, {t1 / us:.4g}] us.  Past the end there is no "
        "drive at all, and clamp-holding the last sample would silently render a dead, "
        "incoherent aperture.  Render the samples over a longer span — a frame at time t reads "
        "drive times up to t - tau/2 + u_max/v, so the record must reach that far."
    )


def sample_baseband(bb: Baseband, channel: str, t: ArrayLike) -> Complex:
    """Gather ``z_channel`` at arbitrary times by cubic Hermite (Catmull-Rom) on I and Q.

    Vectorized over ``t`` (any shape).  The kernel is applied to the *baseband*, which varies
    at the detuning rate (≤ ~15 MHz), never to the carrier — see the module docstring for the
    ``0.016 (2 pi f_bb/f_s)^3`` error law.

    Two boundaries, treated asymmetrically on purpose (mirroring
    :func:`aodl.engine.simulate`'s coverage rules):

    * ``t < t_start`` returns **0** — the drive has not started, so that part of the aperture
      holds no sound (``docs/conventions.md`` §7).  This is physics, not extrapolation.
    * ``t`` past the last sample **raises** ``ValueError``.  A dead drive must never be
      clamp-held into a pupil.

    Within one sample of either end the four-point stencil is clipped to the record; that
    region is inside the edge-ringing zone anyway.
    """
    try:
        z = bb.z[channel]
    except KeyError:
        raise KeyError(
            f"channel {channel!r} is not in this baseband; it carries {tuple(bb.z)}"
        ) from None
    t_arr = np.asarray(t, dtype=np.float64)
    _, t_last = bb.t_span
    if t_arr.size and float(np.max(t_arr)) > t_last + TIME_TOL:
        raise ValueError(_coverage_error(bb, channel, float(np.max(t_arr))))

    x = (t_arr - bb.t_start) * bb.sample_rate
    i = np.floor(x).astype(np.intp)
    s = x - i
    last = z.size - 1
    y0 = z[np.clip(i - 1, 0, last)]
    y1 = z[np.clip(i, 0, last)]
    y2 = z[np.clip(i + 1, 0, last)]
    y3 = z[np.clip(i + 2, 0, last)]
    s2 = s * s
    s3 = s2 * s
    # Hermite basis with centred-difference tangents m1 = (y2 - y0)/2, m2 = (y3 - y1)/2.
    out = (
        (2.0 * s3 - 3.0 * s2 + 1.0) * y1
        + (s3 - 2.0 * s2 + s) * (0.5 * (y2 - y0))
        + (-2.0 * s3 + 3.0 * s2) * y2
        + (s3 - s2) * (0.5 * (y3 - y1))
    )
    return np.asarray(np.where(x < 0.0, 0.0, out), dtype=np.complex128)


def _window(name: str, n: int) -> Float:
    """Analysis window for the diagnostic record FFT."""
    if name == "boxcar":
        return np.ones(n, dtype=np.float64)
    if name == "hann":
        # Periodic Hann; its sidelobes fall as 1/df^3 in amplitude, so the band edge sees
        # none of the analysis window's own leakage (see out_of_band_fraction).
        return np.asarray(0.5 - 0.5 * np.cos(_TWO_PI * np.arange(n) / n), dtype=np.float64)
    raise ValueError(f"unknown window {name!r}; expected one of {WINDOWS}")


def out_of_band_fraction(rec: SampleRecord, *, window: str = "hann") -> dict[str, float]:
    """Fraction of each channel's RF power outside its ``AODParams.band`` — the splatter probe.

    A drive whose envelope *steps* radiates broadband: Table II's ``p_B = 0`` Shepard
    rectangles switch on and off instantaneously, and at the product defaults that costs about
    **-41 dB** of out-of-band power on ``Bx``/``By`` (``docs/ORCHESTRATION.md``, WO-19 F-3);
    ``ShepardConfig(switch_ramp=3 us)`` removes it.  This is a *report-only* diagnostic: the
    band is a hardware limit, not a physics prediction.

    Parameters
    ----------
    window:
        ``"hann"`` (default) or ``"boxcar"``.  The window is not cosmetic.  A boxcar's own
        leakage at ±10 MHz from a 100 MHz carrier is about **-45 dB** for a 280 us record —
        the same size as the splatter being measured, and it appears even on the perfectly
        smooth ``A`` channels (measured -44.6 dB boxcar vs -96.7 dB Hann on an unswitched
        Shepard ``Ax``).  Hann's sidelobes are far below the signal at that offset, so what
        the number reports is the drive, not the record's own ends.

    Returns
    -------
    ``{channel: fraction}`` in ``[0, 1]``; a silent channel reports ``0.0``.
    """
    n = rec.n_samples
    taper = _window(window, n)
    nu = np.fft.rfftfreq(n, 1.0 / rec.sample_rate)
    weight = np.full(nu.size, 2.0)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[-1] = 1.0  # Nyquist has no partner
    out: dict[str, float] = {}
    for name, samples in rec.channels.items():
        lo, hi = rec.params.channels[name].band
        power = weight * np.abs(np.fft.rfft(samples * taper)) ** 2
        total = float(power.sum())
        if total <= 0.0:
            out[name] = 0.0
            continue
        in_band = (nu >= lo) & (nu <= hi)
        out[name] = float(power[~in_band].sum() / total)
    return out
