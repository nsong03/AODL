r"""Aperture rebuild and diffraction-order selection, straight from Eqs. S1-S4.

Given the demodulated drives (:mod:`aodl.check.demod`) this module reconstructs the complex
pupil the AODL imprints on the beam, on a uniform aperture grid, with **no** Taylor expansion
anywhere: the acoustic field at aperture coordinate ``u`` is read at its own retarded time
``t_ret(u) = t_c - s u / v`` (``docs/conventions.md`` §7, via
:func:`aodl.device.conventions.retarded_time` — the sign authority is imported, never
re-derived).

Two models, sharing everything except the last step:

``weak``
    The first-order expansion of Eq. S3, ``exp(i C V) ~ 1 + i C V``, keeping the ``+1`` order:

    .. math::  P_\mu(u) = \tfrac{i}{2} C\, \overline{z_\mu(t_\text{ret}(u))} .

    Demodulating *at the retarded time* already puts this in the simulator's rotating frame:
    a static detuning ``f`` comes out with pupil phase slope ``s 2 pi f / v``, which is
    exactly :func:`aodl.device.conventions.theta1_contribution`.  This is the model the
    analytic simulator implements, so weak mode is the cross-validation path.

``bragg_band``
    The full nonlinear crystal.  The literal real drive is rebuilt at every aperture point,
    ``V(u) = Re[z(t_ret) e^{i 2 pi f_c t_ret}]`` (the carrier phase is evaluated analytically
    at ``t_ret``, never interpolated), the transmission is ``T = exp(i C V)``, and the ``+1``
    order is cut out **in the aperture's spatial-frequency domain** — which is what an AODL
    does optically, since the diffraction orders are separated angles.  This model carries
    compression (``2 J_1(C)/C`` on the fundamental) and every intermodulation product for
    free, with no expansion order to truncate.

Where the aperture holds no sound (``retarded_time < 0``) the crystal is *clear*, so
``bragg_band`` uses ``T = 1`` there, not ``0``.  A constant transmission has no ``+1``-band
content at all, so the band cut removes it cleanly; zeroing instead would plant a hard edge
whose spectrum smears across every order.  In ``weak`` mode the same region is exactly ``0``,
matching :func:`aodl.device.conventions.is_filled`.

Why ``du = Lambda / 8``
-----------------------

``exp(i C V)`` is a phase-modulated carrier: its aperture spectrum is a comb of orders at
``p s f_c / v``, ``p = 0, ±1, ±2, ...``, with amplitudes ``|J_p(C V)|`` for a single tone.
Sampling at ``8 / Lambda`` (``Lambda = v / f_c`` the acoustic wavelength) makes the alias
period exactly 8 order spacings, so **every alias lands on an order centre** instead of
smearing between them, and the orders that fold onto ``+1`` are ``p = 9`` and ``p = -7``.
The dominant one is ``|J_7|``, not ``|J_9|`` (``docs/workorders/WO-21-check-core.md`` names
only ``p = 9``; ``p = -7`` is closer in Bessel order and therefore larger).  At a peak
modulation index ``C V = 1.2`` that is ``J_7(1.2)/J_1(1.2) = 1.1e-5`` for a single tone and
several times smaller for a multi-tone ladder, whose peak is shared out among its rungs; at
the product default ``C = 0.3`` it is ``2e-9``.

Any change to ``du`` reopens this analysis — halving the sampling to ``Lambda / 4`` folds
``p = -3`` onto ``+1`` and the contamination jumps by four orders of magnitude.  Both numbers
are pinned by ``tests/test_check_pupil.py``.

Grid extent
-----------

The input beam is an *uncropped* Gaussian (``docs/PLAN.md`` decision 2), so the grid, not the
crystal, is what truncates it: 24576 cells of ``Lambda/8`` reach ±4.99 ``w_in`` at the product
defaults, where the pupil amplitude is ``e^{-24.9} = 1.5e-11``.  ``design`` refuses a grid
narrower than 4.2 ``w_in``.  Note that the grid runs *wider than the crystal*: it is filled
end to end only once ``v t - D/2 >= half_span``, i.e. from ``t = 1.83 tau`` at the defaults,
which is why the checker's own tests take their frames at ``t >= 2 tau``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..device import conventions
from ..params import CHANNELS, AODLParams
from ..units import mm
from .demod import Baseband, sample_baseband

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]

#: Which pupil model :func:`channel_pupil` builds.
PupilMode = Literal["bragg_band", "weak"]

#: Aperture cells per acoustic wavelength in ``bragg_band`` mode: ``du = Lambda / 8``.
CELLS_PER_PERIOD = 8

#: Cell count of the ``bragg_band`` grid (2^13 * 3 — an FFT-friendly length).
BRAGG_CELLS = 24576

#: Cell count of the ``weak`` grid, which spans the same aperture with 6x coarser cells
#: (no order selection happens there, so the carrier need not be resolved).
WEAK_CELLS = 4096

#: Smallest half-span, in units of ``w_in``, a grid may have.  At 4.2 the truncated Gaussian
#: tail is ``e^{-17.6} = 2.2e-8`` of the peak.
MIN_HALF_SPAN_W_IN = 4.2

_TWO_PI = 2.0 * math.pi

__all__ = [
    "BRAGG_CELLS",
    "CELLS_PER_PERIOD",
    "MIN_HALF_SPAN_W_IN",
    "WEAK_CELLS",
    "ApertureGrid",
    "PupilMode",
    "axis_pupil",
    "band_window",
    "channel_pupil",
]


@dataclass(frozen=True)
class ApertureGrid:
    """Uniform aperture sampling, centred on ``u = 0`` (the beam centre).

    Attributes
    ----------
    u:
        Aperture coordinates [m], strictly increasing and uniformly spaced, containing 0.
    du:
        Cell size [m].
    """

    u: Float
    du: float

    def __post_init__(self) -> None:
        u = np.array(self.u, dtype=np.float64, copy=True)
        if u.ndim != 1 or u.size < 4:
            raise ValueError(
                f"ApertureGrid.u must be a 1-D grid of at least 4 points, got {u.shape}"
            )
        du = float(self.du)
        if not math.isfinite(du) or du <= 0.0:
            raise ValueError(f"ApertureGrid.du must be positive and finite, got {self.du!r}")
        spacing = np.diff(u)
        if np.max(np.abs(spacing - du)) > 1e-9 * du:
            raise ValueError("ApertureGrid.u must be uniformly spaced by exactly du")
        u.flags.writeable = False
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "du", du)

    @property
    def n(self) -> int:
        """Number of aperture cells."""
        return int(self.u.size)

    @property
    def half_span(self) -> float:
        """Half the grid's extent, ``n du / 2`` [m]."""
        return 0.5 * self.n * self.du

    @property
    def nyquist(self) -> float:
        """Largest spatial frequency the grid resolves, ``1 / (2 du)`` [1/m]."""
        return 0.5 / self.du

    @classmethod
    def design(cls, params: AODLParams, mode: PupilMode) -> ApertureGrid:
        """The pinned grid for ``mode`` (see the module docstring for the ``Lambda/8`` rule).

        ``bragg_band`` uses ``du = v / (8 f_center)`` **exactly** with 24576 cells; ``weak``
        spans the same aperture with 4096 cells.  Raises when the four channels do not share
        one carrier or one sound speed (the grid is common to all of them), or when the
        resulting half-span would not reach ``4.2 w_in``.
        """
        if mode not in ("bragg_band", "weak"):
            raise ValueError(f"mode must be 'bragg_band' or 'weak', got {mode!r}")
        sound_speed = params.sound_speed  # raises if the channels disagree
        carriers = {name: params.channels[name].f_center for name in CHANNELS}
        if len(set(carriers.values())) != 1:
            raise ValueError(
                "ApertureGrid.design needs one common f_center — the aperture grid is shared by "
                f"all four channels and du = v / (8 f_center) is pinned to it; got {carriers}"
            )
        f_center = float(next(iter(carriers.values())))
        span = BRAGG_CELLS * sound_speed / (CELLS_PER_PERIOD * f_center)
        n = BRAGG_CELLS if mode == "bragg_band" else WEAK_CELLS
        du = span / n
        half_span = 0.5 * span
        w_in = params.optics.w_in
        if half_span < MIN_HALF_SPAN_W_IN * w_in:
            raise ValueError(
                f"the pinned aperture grid reaches only {half_span / w_in:.2f} w_in "
                f"({half_span / mm:.3f} mm at w_in = {w_in / mm:.3f} mm) but at least "
                f"{MIN_HALF_SPAN_W_IN} w_in is needed to truncate the input Gaussian below "
                "1e-8.  The half-span is 1536 v / f_center — 1536 acoustic wavelengths — so it "
                f"shrinks with a *slower* sound speed or a *higher* carrier, and this hardware "
                f"(v = {sound_speed:g} m/s, f_center = {f_center:g} Hz) has too short a "
                f"wavelength (Lambda = {sound_speed / f_center / mm:.4g} mm) for a beam this "
                "wide, so the grid would clip it.  Widen the grid (a lower f_center or a faster "
                "crystal), narrow the beam, or build an ApertureGrid explicitly if you really "
                "mean to."
            )
        u = (np.arange(n, dtype=np.float64) - n // 2) * du
        return cls(u=u, du=du)


def band_window(nu: ArrayLike, center: float, half: float, roll: float) -> Float:
    """Flat-top raised-cosine band filter: 1 within ``±half`` of ``center``, 0 outside the roll.

    ``roll`` is the width of the cosine shoulder as a fraction of ``half``, so the window
    reaches zero at ``|nu - center| = half (1 + roll)``.  ``roll = 0`` is a hard edge.

    The shoulder matters: the aperture's order comb is cut with this window and a hard edge
    would ring the selected order across the whole aperture (a sinc tail in ``u``).
    """
    d = np.abs(np.asarray(nu, dtype=np.float64) - float(center))
    half = float(half)
    roll = float(roll)
    if not math.isfinite(half) or half <= 0.0:
        raise ValueError(f"band_window half-width must be positive and finite, got {half!r}")
    if not math.isfinite(roll) or roll < 0.0:
        raise ValueError(f"band_window roll must be finite and non-negative, got {roll!r}")
    if roll == 0.0:
        return np.asarray(d <= half, dtype=np.float64)
    x = np.clip((d - half) / (roll * half), 0.0, 1.0)
    return np.asarray(0.5 * (1.0 + np.cos(math.pi * x)), dtype=np.float64)


def _axis_index(axis: int | str) -> int:
    """Accept ``0``/``1`` or ``"x"``/``"y"`` and return the convention's axis index."""
    if isinstance(axis, str):
        try:
            return conventions.AXIS_NAMES.index(axis)
        except ValueError:
            raise ValueError(
                f"unknown axis {axis!r}; expected one of {conventions.AXIS_NAMES} or 0/1"
            ) from None
    index = int(axis)
    if index not in (conventions.AXIS_X, conventions.AXIS_Y):
        raise ValueError(f"axis must be 0 (x) or 1 (y), got {axis!r}")
    return index


def channel_pupil(
    bb: Baseband,
    channel: str,
    t: float,
    grid: ApertureGrid,
    *,
    mode: PupilMode,
    band_margin: float = 1.15,
    roll: float = 0.25,
) -> Complex:
    """One channel's rotating-frame pupil factor at frame time ``t``.  Eqs. S1-S4.

    The input Gaussian is **not** included — :func:`axis_pupil` applies it once per axis, as
    the physical beam is shared by the two stacked crystals on that axis.

    Parameters
    ----------
    bb:
        Demodulated drives.
    channel:
        ``"Ax"``, ``"Bx"``, ``"Ay"`` or ``"By"``; its geometry comes from
        :func:`aodl.device.conventions.geometry`.
    t:
        Frame (observation) time [s].
    grid:
        Aperture sampling.  ``bragg_band`` needs one that resolves the carrier
        (:meth:`ApertureGrid.design`).
    mode:
        ``"weak"`` (linear, Eq. S3 to first order) or ``"bragg_band"`` (full ``exp(i C V)``
        with spatial-frequency order selection).
    band_margin:
        Flat half-width of the order window, in units of the channel's own half band
        ``(f_hi - f_lo) / 2 / v``.  1.15 keeps the whole usable band with headroom for the
        chirp excursion while staying far from the neighbouring orders, which sit a full
        ``f_center / v`` away — ten half-bands at the defaults, against a window that reaches
        ``1.15 * 1.25 = 1.44`` of one.
    roll:
        Raised-cosine shoulder of the window, as a fraction of the flat half-width.

    Notes
    -----
    ``bragg_band`` apodizes with the input Gaussian before the FFT (an un-tapered
    ``|T| = 1`` aperture would leak its wrap discontinuity straight into the selected band at
    the percent level) and divides it back out afterwards, so the return value has the same
    meaning in both modes.  The division amplifies the transform's round-off in the far
    tails, where the Gaussian is ``1e-11``; that region is multiplied back down by
    :func:`axis_pupil` and never carries light, but read a ``bragg_band`` pupil beyond
    ``|u| ~ 4 w_in`` at your own risk.

    The order selection also leaves a per-channel constant phase ``exp(i 2 pi f_c t_c)``
    relative to the work order's spatial-only deramp: the carrier is removed here as
    ``exp(+i 2 pi f_c t_ret(u))``, which is the same ramp times that constant.  Removing it
    too is what lands ``bragg_band`` on ``weak`` in phase as well as amplitude (they then
    differ by the real factor ``2 J_1(C)/C``); it is a unit-modulus constant, so no intensity
    anywhere depends on the choice.
    """
    if mode not in ("bragg_band", "weak"):
        raise ValueError(f"mode must be 'bragg_band' or 'weak', got {mode!r}")
    geom = conventions.geometry(channel)
    if channel not in bb.z:
        raise KeyError(f"channel {channel!r} is not in this baseband; it carries {tuple(bb.z)}")
    aod = bb.params.channels[channel]
    u = grid.u
    t_ret = conventions.retarded_time(t, u, geom, aod)
    filled = conventions.is_filled(u, t, geom, aod)
    z = sample_baseband(bb, channel, t_ret)

    if mode == "weak":
        return np.asarray(
            np.where(filled, 0.5j * aod.drive_strength * np.conj(z), 0.0), dtype=np.complex128
        )

    sound_speed = aod.sound_speed
    center = geom.sound_sign * aod.f_center / sound_speed
    lo, hi = aod.band
    half = band_margin * 0.5 * (hi - lo) / sound_speed
    if abs(center) + half * (1.0 + roll) > grid.nyquist:
        raise ValueError(
            f"this grid does not resolve the {channel!r} order comb: the +1 band reaches "
            f"{abs(center) + half * (1.0 + roll):.4g} 1/m but the grid's Nyquist frequency is "
            f"{grid.nyquist:.4g} 1/m (du = {grid.du:.4g} m).  Use "
            "ApertureGrid.design(params, 'bragg_band'), whose du = v / (8 f_center) puts four "
            "orders inside Nyquist."
        )
    carrier = np.exp(1j * _TWO_PI * aod.f_center * t_ret)
    drive = np.real(z * carrier)
    transmission = np.where(filled, np.exp(1j * aod.drive_strength * drive), 1.0 + 0.0j)
    apodization = np.exp(-((u / bb.params.optics.w_in) ** 2))
    spectrum = np.fft.fft(apodization * transmission)
    spectrum *= band_window(np.fft.fftfreq(grid.n, grid.du), center, half, roll)
    selected = np.fft.ifft(spectrum) * carrier
    inverse = np.zeros_like(apodization)
    positive = apodization > 0.0
    inverse[positive] = 1.0 / apodization[positive]
    return np.asarray(selected * inverse, dtype=np.complex128)


def axis_pupil(
    bb: Baseband,
    axis: int | str,
    t: float,
    grid: ApertureGrid,
    *,
    mode: PupilMode,
    **kwargs: Any,
) -> Complex:
    """The whole pupil of one transverse axis: its channels' product times the input beam.

    The AODs are stacked, so their pupils multiply (Eq. S7), and the illumination is applied
    **once** — the light crosses one Gaussian beam, not one per crystal.  An axis with no
    driven channel returns the bare Gaussian, which is the identity factor the simulator uses
    for an undriven axis.

    ``kwargs`` are forwarded to :func:`channel_pupil` (``band_margin``, ``roll``).
    """
    index = _axis_index(axis)
    names = [name for name in CHANNELS if name in bb.z and conventions.geometry(name).axis == index]
    out = np.ones(grid.n, dtype=np.complex128)
    for name in names:
        out = out * channel_pupil(bb, name, t, grid, mode=mode, **kwargs)
    return np.asarray(out * np.exp(-((grid.u / bb.params.optics.w_in) ** 2)), dtype=np.complex128)
