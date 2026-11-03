r"""The acoustic content on each crystal at one instant (API.md §5.3).

What sits at aperture coordinate ``u`` at frame time ``t`` is the drive at the retarded time
``t_ret(u) = t - tau/2 - s u / v`` (Eq. S4; :func:`aodl.device.conventions.retarded_time`),
and only where the sound has already arrived, ``s u <= v t - D/2``
(:func:`aodl.device.conventions.is_filled`, ``docs/conventions.md`` §7).  The literal RF
``V(u)`` — :func:`aodl.device.aod.aperture_window` — carries the 100 MHz carrier, ~1150
cycles across the aperture, which no few-hundred-point trace can draw honestly, so the
default picture is the **envelope** ``sum_n A_n(t_ret)`` and the amplitude-weighted
**instantaneous frequency** ``f_center + sum A_n f_n / sum A_n``; the carrier is drawn only
on request, on a dense zoom window.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from aodl.device.aod import aperture_window
from aodl.device.conventions import ChannelGeometry, geometry, is_filled, retarded_time
from aodl.params import AODParams
from aodl.units import MHz, mm, us
from aodl.waveform.tones import ChannelWaveform

from .planner import get_plan
from .schemas import ApertureChannel, ApertureRequest, ApertureResponse, ApertureVoltage

Float = NDArray[np.float64]

#: Fewest samples of the carrier-included zoom (API.md §5.3).
MIN_VOLTAGE_POINTS = 4096

#: Most samples the zoom will ever use.
MAX_VOLTAGE_POINTS = 32768

#: Samples per acoustic wavelength ``v / f_center`` the zoom guarantees.
VOLTAGE_SAMPLES_PER_WAVELENGTH = 32

#: Default zoom window: the central 0.1 mm of the aperture.
DEFAULT_VOLTAGE_WINDOW_MM = (-0.05, 0.05)


def aperture_content(
    cw: ChannelWaveform, aod: AODParams, geom: ChannelGeometry, t: float, n: int
) -> tuple[Float, Float, Float, NDArray[np.bool_], Float]:
    """``(u, envelope, f_inst, filled, t_ret)`` on ``n`` points of ``[-D/2, D/2]`` (SI).

    ``f_inst`` is **absolute** (``f_center`` added) and NaN where the envelope vanishes or the
    aperture is unfilled; the envelope is zero where unfilled.
    """
    half = 0.5 * aod.aperture
    u = np.linspace(-half, half, int(n))
    t_ret = retarded_time(t, u, geom, aod)
    filled = is_filled(u, t, geom, aod)
    envelope = np.zeros_like(u)
    weighted = np.zeros_like(u)
    for tone in cw.tones:
        a = np.asarray(tone.env.A(t_ret), dtype=np.float64)
        envelope += a
        weighted += a * np.asarray(tone.freq(t_ret), dtype=np.float64)
    envelope = np.where(filled, envelope, 0.0)
    live = envelope > 0.0
    f_inst = np.full(u.shape, np.nan)
    f_inst[live] = aod.f_center + weighted[live] / envelope[live]
    return u, envelope, f_inst, filled, t_ret


def aperture_voltage(
    cw: ChannelWaveform,
    aod: AODParams,
    geom: ChannelGeometry,
    t: float,
    u: Float,
) -> Float:
    """The literal RF on the crystal at the aperture coordinates ``u`` [m].

    Exactly :func:`aodl.device.aod.aperture_window`'s expression — ``sum_n A_n(t_ret)
    cos(2 pi f_center t_ret + phase_n(t_ret))``, zero where unfilled — evaluated on the
    caller's points instead of the full aperture, so a zoom window can be sampled densely.
    """
    t_ret = retarded_time(t, u, geom, aod)
    signal = np.zeros_like(u)
    for tone in cw.tones:
        env = np.asarray(tone.env.A(t_ret), dtype=np.float64)
        phase = np.asarray(tone.phase(t_ret), dtype=np.float64)
        signal += env * np.cos(2.0 * math.pi * aod.f_center * t_ret + phase)
    return np.asarray(np.where(is_filled(u, t, geom, aod), signal, 0.0), dtype=np.float64)


def voltage_points(aod: AODParams, window: tuple[float, float]) -> int:
    """Samples for a zoom over ``window`` [m]: at least :data:`MIN_VOLTAGE_POINTS`, and at
    least :data:`VOLTAGE_SAMPLES_PER_WAVELENGTH` per acoustic wavelength."""
    wavelength = aod.sound_speed / aod.f_center
    needed = int(math.ceil((window[1] - window[0]) / wavelength * VOLTAGE_SAMPLES_PER_WAVELENGTH))
    return int(min(max(MIN_VOLTAGE_POINTS, needed), MAX_VOLTAGE_POINTS))


def _nan_to_none(values: Float) -> list[float | None]:
    return [None if math.isnan(v) else float(v) for v in values.tolist()]


def aperture_response(req: ApertureRequest) -> ApertureResponse:
    """API.md §5.3 for every driven channel of the plan."""
    plan = get_plan(req.plan)
    t = req.time_us * us
    channels: dict[str, ApertureChannel] = {}
    for name, cw in plan.wfs.channels.items():
        aod = plan.params.channels[name]
        geom = geometry(name)
        u, envelope, f_inst, filled, t_ret = aperture_content(cw, aod, geom, t, req.n)
        voltage: ApertureVoltage | None = None
        if req.include_voltage:
            window_mm = req.voltage_window_mm or DEFAULT_VOLTAGE_WINDOW_MM
            window = (window_mm[0] * mm, window_mm[1] * mm)
            m = voltage_points(aod, window)
            half = 0.5 * aod.aperture
            whole = math.isclose(window[0], -half, rel_tol=1e-9) and math.isclose(
                window[1], half, rel_tol=1e-9
            )
            if whole:  # the full aperture: aodl's own diagnostic, point for point
                u_v, v = aperture_window(cw, aod, geom, t, m)
            else:
                u_v = np.linspace(window[0], window[1], m)
                v = aperture_voltage(cw, aod, geom, t, u_v)
            voltage = ApertureVoltage(u_mm=(u_v / mm).tolist(), v=v.tolist())
        channels[name] = ApertureChannel(
            u_mm=(u / mm).tolist(),
            envelope=envelope.tolist(),
            f_inst_MHz=_nan_to_none(f_inst / MHz),
            filled=filled.tolist(),
            t_ret_us=(t_ret / us).tolist(),
            voltage=voltage,
        )
    return ApertureResponse(time_us=req.time_us, channels=channels)


__all__ = [
    "DEFAULT_VOLTAGE_WINDOW_MM",
    "MAX_VOLTAGE_POINTS",
    "MIN_VOLTAGE_POINTS",
    "VOLTAGE_SAMPLES_PER_WAVELENGTH",
    "aperture_content",
    "aperture_response",
    "aperture_voltage",
    "voltage_points",
]
