"""``HardwareSpec`` <-> :class:`aodl.AODLParams`, and the derived constants the UI draws with.

The one place display units (nm, mm, MHz) become SI on the way in.  Every number in
:func:`constants` is a property of :class:`aodl.AODLParams` or a one-line consequence of
Table I; nothing is re-derived here.
"""

from __future__ import annotations

from aodl.params import CHANNELS, AODLParams, AODParams, OpticsParams, default_1030, paper_808
from aodl.units import MHz, mm, ms, nm, um, us
from aodl.waveform.synthesis import max_z_integral

from .schemas import HardwareSpec, PlanConstants


def to_params(hw: HardwareSpec) -> AODLParams:
    """API.md §3.1: identical :class:`aodl.AODParams` on the four channels (SI)."""
    optics = OpticsParams(
        wavelength=hw.wavelength_nm * nm,
        focal_length=hw.focal_length_mm * mm,
        w_in=hw.w_in_mm * mm,
    )
    aod = AODParams(
        sound_speed=hw.sound_speed_m_s,
        aperture=hw.aperture_mm * mm,
        f_center=hw.f_center_MHz * MHz,
        band=(hw.band_MHz[0] * MHz, hw.band_MHz[1] * MHz),
        drive_strength=hw.drive_strength,
        mixing_order=hw.mixing_order,
    )
    return AODLParams(optics=optics, channels={name: aod for name in CHANNELS})


def from_params(params: AODLParams) -> HardwareSpec:
    """Inverse of :func:`to_params` for a stack with identical channels (the presets)."""
    aod = params.channels[CHANNELS[0]]
    return HardwareSpec(
        wavelength_nm=params.optics.wavelength / nm,
        focal_length_mm=params.optics.focal_length / mm,
        w_in_mm=params.optics.w_in / mm,
        sound_speed_m_s=aod.sound_speed,
        aperture_mm=aod.aperture / mm,
        f_center_MHz=aod.f_center / MHz,
        band_MHz=(aod.band[0] / MHz, aod.band[1] / MHz),
        drive_strength=aod.drive_strength,
        mixing_order=aod.mixing_order,  # type: ignore[arg-type]
    )


def transit_time(params: AODLParams) -> float:
    """The stack's aperture transit ``tau`` [s] — the longest of the four channels."""
    return max(aod.transit_time for aod in params.channels.values())


def constants(params: AODLParams) -> PlanConstants:
    """API.md §4 ``constants`` — Table I scales and the Eq. 1 budget in display units.

    * ``um_per_MHz`` — ``deflection_scale = lambda F / v`` (Table I);
    * ``chirp_per_um_MHz_per_ms`` — the co-chirp a sustained 1 µm of ``Z`` costs every
      channel, ``1 um / (2 lens_scale)`` (Eq. S19's ``f_Z``);
    * ``beam_transit_us`` — ``2 w_in / v``, the startup-transient time of ``docs/PLAN.md``
      §1.5 (the full 1/e^2 beam diameter crossing at the sound speed);
    * ``eq1_hold_10um_us`` — :func:`aodl.waveform.synthesis.max_z_integral` at ``Z = 10 µm``.
    """
    tau = transit_time(params)
    optics = params.optics
    return PlanConstants(
        tau_us=tau / us,
        half_tau_us=0.5 * tau / us,
        waist0_um=optics.waist0 / um,
        rayleigh_um=optics.rayleigh / um,
        um_per_MHz=params.deflection_scale * MHz / um,
        lens_scale_m_s=params.lens_scale,
        chirp_per_um_MHz_per_ms=(um / (2.0 * params.lens_scale)) / (MHz / ms),
        beam_transit_us=2.0 * optics.w_in / params.sound_speed / us,
        eq1_hold_10um_us=max_z_integral(params) / (10.0 * um) / us,
    )


#: The two hardware presets of API.md §2, as request models.
HARDWARE_PRESETS: dict[str, HardwareSpec] = {
    "default_1030": from_params(default_1030()),
    "paper_808": from_params(paper_808()),
}


__all__ = ["HARDWARE_PRESETS", "constants", "from_params", "to_params", "transit_time"]
