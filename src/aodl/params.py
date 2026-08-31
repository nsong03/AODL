"""Device and optics parameters (frozen dataclasses) plus hardware presets.

All quantities are SI (Hz, m, s, rad).  Defaults follow ``docs/PLAN.md`` §1.5:
AA Opto DTSX(Y)-400 tellurium-dioxide deflectors (v = 650 m/s, D = 7.5 mm,
f0 = 100 MHz, usable band ±10 MHz) behind an F = 6.5 mm effective objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .units import MHz, mm, nm

#: Channel names of the four-AOD stack, in the paper's order (``docs/PLAN.md`` §1.1).
#: ``Ax``/``Bx`` counter-propagate along x, ``Ay``/``By`` along y.
CHANNELS: tuple[str, str, str, str] = ("Ax", "Bx", "Ay", "By")


@dataclass(frozen=True)
class AODParams:
    """One acousto-optic deflector channel.

    Attributes
    ----------
    sound_speed:
        Acoustic velocity ``v`` [m/s] in the crystal.
    aperture:
        Active aperture ``D`` [m] along the sound-propagation axis.
    f_center:
        Rotating-frame carrier ``f_center`` [Hz].  Waveform-IR frequencies are
        *detunings* from this value (Eq. S2).
    band:
        Usable **absolute** RF band ``(f_lo, f_hi)`` [Hz], e.g. ``(90 MHz, 110 MHz)``.
    drive_strength:
        ``C·A`` at unit envelope: the peak phase modulation [rad] imprinted on the
        optical field by a unit-amplitude tone (Eq. S1).  Sets the weak-drive
        expansion order needed for intermodulation products.
    """

    sound_speed: float
    aperture: float
    f_center: float
    band: tuple[float, float]
    drive_strength: float = 0.30

    def __post_init__(self) -> None:
        if self.sound_speed <= 0.0:
            raise ValueError("sound_speed must be positive")
        if self.aperture <= 0.0:
            raise ValueError("aperture must be positive")
        if self.f_center <= 0.0:
            raise ValueError("f_center must be positive")
        lo, hi = self.band
        if not lo < hi:
            raise ValueError(f"band must be (lo, hi) with lo < hi, got {self.band!r}")

    @property
    def transit_time(self) -> float:
        """Acoustic transit time across the aperture, ``tau = D / v`` [s]."""
        return self.aperture / self.sound_speed


@dataclass(frozen=True)
class OpticsParams:
    """Illumination and objective.

    Attributes
    ----------
    wavelength:
        Optical wavelength ``lambda`` [m].
    focal_length:
        Effective objective focal length ``F`` [m].
    w_in:
        Input-beam 1/e^2 *intensity* radius at the AOD plane [m] (uncropped Gaussian).
    """

    wavelength: float
    focal_length: float
    w_in: float

    def __post_init__(self) -> None:
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive")
        if self.focal_length <= 0.0:
            raise ValueError("focal_length must be positive")
        if self.w_in <= 0.0:
            raise ValueError("w_in must be positive")

    @property
    def k(self) -> float:
        """Optical wavenumber ``k = 2*pi/lambda`` [rad/m]."""
        return 2.0 * math.pi / self.wavelength

    @property
    def waist0(self) -> float:
        """Focal 1/e^2 intensity radius of the uncropped Gaussian, ``lambda F / (pi w_in)`` [m]."""
        return self.wavelength * self.focal_length / (math.pi * self.w_in)

    @property
    def rayleigh(self) -> float:
        """Rayleigh range ``pi waist0^2 / lambda`` [m]."""
        return math.pi * self.waist0**2 / self.wavelength


@dataclass(frozen=True)
class AODLParams:
    """The whole 3D-AODL: four AOD channels sharing one objective.

    ``channels`` must have exactly the keys in :data:`CHANNELS`.
    """

    optics: OpticsParams
    channels: dict[str, AODParams]

    def __post_init__(self) -> None:
        if set(self.channels) != set(CHANNELS):
            raise ValueError(
                f"channels must have exactly the keys {CHANNELS}, got {tuple(self.channels)}"
            )

    @property
    def sound_speed(self) -> float:
        """Common acoustic velocity ``v`` [m/s]; raises if the channels disagree."""
        speeds = [self.channels[name].sound_speed for name in CHANNELS]
        v = speeds[0]
        if any(s != v for s in speeds[1:]):
            raise ValueError(
                "deflection/lens scales assume one common sound speed; "
                f"got {dict(zip(CHANNELS, speeds, strict=True))}"
            )
        return v

    @property
    def deflection_scale(self) -> float:
        """``lambda F / v`` [m per Hz of frequency difference] (paper Table I).

        ``X = deflection_scale * (f_Bx - f_Ax)``.
        """
        return self.optics.wavelength * self.optics.focal_length / self.sound_speed

    @property
    def lens_scale(self) -> float:
        """``lambda F^2 / v^2`` [m.s] (paper Table I).

        ``Zbar = 0.5 * lens_scale * (fdot_Ax + fdot_Bx + fdot_Ay + fdot_By)`` and
        ``Delta F = lens_scale * (fdot_Ax + fdot_Bx - fdot_Ay - fdot_By)``.
        """
        v = self.sound_speed
        return self.optics.wavelength * self.optics.focal_length**2 / v**2


def _preset(wavelength: float) -> AODLParams:
    """Paper hardware (DTSX(Y)-400 + F = 6.5 mm objective) at the given wavelength."""
    optics = OpticsParams(wavelength=wavelength, focal_length=6.5 * mm, w_in=2.0 * mm)
    aod = AODParams(
        sound_speed=650.0,
        aperture=7.5 * mm,
        f_center=100.0 * MHz,
        band=(90.0 * MHz, 110.0 * MHz),
    )
    return AODLParams(optics=optics, channels={name: aod for name in CHANNELS})


def default_1030() -> AODLParams:
    """Product default: paper hardware at lambda = 1030 nm (``docs/PLAN.md`` §1.5)."""
    return _preset(1030.0 * nm)


def paper_808() -> AODLParams:
    """Paper preset: identical hardware at lambda = 808 nm (for figure reproduction)."""
    return _preset(808.0 * nm)


__all__ = [
    "CHANNELS",
    "AODLParams",
    "AODParams",
    "OpticsParams",
    "default_1030",
    "paper_808",
]
