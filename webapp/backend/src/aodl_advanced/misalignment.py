r"""Per-AOD misalignment parameters (DESIGN §1.7).

Each AOD of the stack may be displaced, rotated, delayed, magnified or tilted with respect to
the ideal, perfectly overlaid device that ``aodl`` models.  :class:`AODMisalignment` holds one
channel's seven knobs in SI units and :class:`Misalignment` the four channels' (any channel
left out is aligned).

**Geometry conventions of this module** (the sound-direction, diffraction-order and defocus
signs are *not* decided here — they are read from :mod:`aodl.device.conventions`; what is new
here is only how a misaligned crystal's own aperture coordinate relates to the lab pupil
coordinates):

* a channel's *axis* unit vector ``e_axis`` is ``+x`` for the x channels and ``+y`` for the y
  channels (``conventions.ChannelGeometry.axis``), and its *perpendicular* unit vector is
  ``e_perp = z x e_axis`` (right-handed about the optical axis): ``+y`` for an x channel and
  ``-x`` for a y channel;
* ``offset_par`` / ``offset_perp`` displace the crystal's aperture centre by
  ``d = offset_par e_axis + offset_perp e_perp`` [m];
* ``rotation`` turns the sound axis by ``theta`` counter-clockwise about the optical axis
  (``+x`` toward ``+y``), so the channel's aperture coordinate at lab point ``r`` is
  ``u = (r - d) . R(theta) e_axis`` — equivalently ``r' = R(-theta)(r - d)``,
  ``u = r' . e_axis``;
* ``velocity_scale`` ``m`` is the 4f magnification mismatch: the crystal appears scaled by
  ``m``, so both its sound speed and its aperture look ``m`` times larger in the lab pupil
  (``v -> m v``, ``D -> m D``, transit time unchanged);
* ``delay`` is the acoustic delay error ``delta t`` of Eq. S29 (the drive reaches the
  transducer late by ``delta t``);
* ``tilt_x``, ``tilt_y`` are pointing tilts [rad] adding ``exp(i k (alpha_x x + alpha_y y))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from aodl.params import CHANNELS


@dataclass(frozen=True)
class AODMisalignment:
    """One channel's misalignment, SI units (m, rad, s, dimensionless)."""

    offset_par: float = 0.0
    offset_perp: float = 0.0
    rotation: float = 0.0
    delay: float = 0.0
    velocity_scale: float = 1.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = float(getattr(self, f.name))
            if not math.isfinite(value):
                raise ValueError(f"AODMisalignment.{f.name} must be finite, got {value!r}")
            object.__setattr__(self, f.name, value)
        if self.velocity_scale <= 0.0:
            raise ValueError(f"velocity_scale must be positive, got {self.velocity_scale!r}")

    @property
    def is_aligned(self) -> bool:
        """True when every knob is at its ideal value."""
        return self == AODMisalignment()

    @property
    def is_rotated(self) -> bool:
        """True when the sound axis is rotated (the pupil is then not separable in x, y)."""
        return self.rotation != 0.0


class Misalignment(dict[str, AODMisalignment]):
    """``{channel: AODMisalignment}`` for the four channels; missing channels are aligned."""

    def __init__(self, entries: dict[str, AODMisalignment] | None = None) -> None:
        super().__init__()
        for name in CHANNELS:
            self[name] = AODMisalignment()
        for name, value in (entries or {}).items():
            if name not in CHANNELS:
                raise KeyError(f"unknown channel {name!r}; expected one of {CHANNELS}")
            if not isinstance(value, AODMisalignment):
                raise TypeError(f"Misalignment[{name!r}] must be an AODMisalignment, got {value!r}")
            self[name] = value

    @classmethod
    def aligned(cls) -> Misalignment:
        """The ideal device: every channel at its defaults."""
        return cls()

    @property
    def is_aligned(self) -> bool:
        """True when every channel is aligned."""
        return all(m.is_aligned for m in self.values())

    @property
    def any_rotated(self) -> bool:
        """True when any channel's sound axis is rotated."""
        return any(m.is_rotated for m in self.values())

    def channel(self, name: str) -> AODMisalignment:
        """The channel's record (aligned when absent)."""
        return self.get(name, AODMisalignment())


__all__ = ["AODMisalignment", "Misalignment"]
