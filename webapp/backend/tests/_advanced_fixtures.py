"""Shared helpers of the ``test_advanced_*`` / ``test_cross_engine`` suites (W2).

Not a ``conftest.py`` (that file belongs to the core package's tests): import what you need.
Everything here builds drives with ``aodl.plan_motion`` directly and reads the reference
numbers the cross-checks compare against from the ``aodl`` closed-form engine, so the tests
run without the web package.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aodl import ArraySpec, Hold, Lift, ShepardConfig, TrajectorySpec, Translate, plan_motion
from aodl.api import MotionPlan
from aodl.device.aodl import build_terms
from aodl.engine import simulate
from aodl.field.measure import SpotMetrics, measure
from aodl.params import AODLParams, default_1030
from aodl.units import MHz, um, us
from aodl.waveform.tones import WaveformSet
from aodl_advanced.engine import FrameResult, GridSpec, ModelSpec, XzSpec, simulate_frames
from aodl_advanced.misalignment import Misalignment

PARAMS = default_1030()
TAU = PARAMS.channels["Ax"].transit_time
W0 = PARAMS.optics.waist0
Z_R = PARAMS.optics.rayleigh
DEFLECTION = PARAMS.deflection_scale
LENS = PARAMS.lens_scale


def with_order(params: AODLParams, order: int, drive_strength: float | None = None) -> AODLParams:
    """``params`` with every channel at mixing order ``order`` (and optionally another ``C``)."""
    channels = {}
    for name, aod in params.channels.items():
        aod = replace(aod, mixing_order=order)
        if drive_strength is not None:
            aod = replace(aod, drive_strength=drive_strength)
        channels[name] = aod
    return AODLParams(optics=params.optics, channels=channels)


def pick_and_place(params: AODLParams = PARAMS) -> MotionPlan:
    """The landing page's ``pick_and_place``: lift 5 µm / 40 µs, 20 µm / 60 µs, lower."""
    spec = TrajectorySpec(
        array=ArraySpec(1, 1),
        moves=(Lift(5 * um, 40 * us), Translate(20 * um, 0.0, 60 * us), Lift(-5 * um, 40 * us)),
    )
    return plan_motion(spec, params)


def lift_hold(params: AODLParams = PARAMS, hold: float = 80 * us) -> MotionPlan:
    """10 µm lift, hold (a sustained co-chirp at 48.5 MHz/ms), lower — Eq. S19 mode."""
    spec = TrajectorySpec(
        array=ArraySpec(1, 1), moves=(Lift(10 * um, 60 * us), Hold(hold), Lift(-10 * um, 60 * us))
    )
    return plan_motion(spec, params)


def static_offset_trap(params: AODLParams = PARAMS, x: float = 20 * um) -> MotionPlan:
    """Translate to ``x`` and rest there (``f_Bx = -f_Ax = x / (2 deflection_scale)``)."""
    spec = TrajectorySpec(array=ArraySpec(1, 1), moves=(Translate(x, 0.0, 60 * us), Hold(100 * us)))
    return plan_motion(spec, params)


def static_array(mx: int, my: int, params: AODLParams = PARAMS) -> MotionPlan:
    """A static ``mx x my`` array at 1.0 / 1.3 MHz."""
    spec = TrajectorySpec(
        array=ArraySpec(mx, my, 1.0 * MHz if mx > 1 else 0.0, 1.3 * MHz if my > 1 else 0.0),
        moves=(Hold(100 * us),),
    )
    return plan_motion(spec, params)


def fading_hold(params: AODLParams = PARAMS, delta_f: float = 3 * MHz) -> MotionPlan:
    """``single_lift_hold``-like fading-Shepard drive with a small ladder spacing, so the
    Eq. S31 shadow pair at ``+- deflection_scale * delta_f`` fits a 256-point pupil grid."""
    spec = TrajectorySpec(
        array=ArraySpec(1, 1),
        moves=(Lift(10 * um, 60 * us), Hold(600 * us), Lift(-10 * um, 60 * us)),
    )
    return plan_motion(spec, params, shepard=ShepardConfig(delta_f, delta_f))


def reparam(wfs: WaveformSet, params: AODLParams) -> WaveformSet:
    """The same drive carried by other hardware parameters (mixing order, drive strength)."""
    return replace(wfs, params=params)


def aodl_spots(wfs: WaveformSet, t: float) -> list[SpotMetrics]:
    """``aodl``'s closed-form spot metrics of frame ``t`` (one per frequency group)."""
    return measure(build_terms(wfs, t), wfs.params.optics)


def aodl_spot(wfs: WaveformSet, t: float) -> SpotMetrics:
    """The brightest ``aodl`` spot of frame ``t``."""
    return max(aodl_spots(wfs, t), key=lambda m: m.power)


def tracked_plane(wfs: WaveformSet, t: float) -> float:
    """``aodl``'s tracked plane of frame ``t`` [m]."""
    return simulate(wfs, [t]).plane(0)


def run_frame(
    wfs: WaveformSet,
    t: float,
    mis: Misalignment | None = None,
    *,
    order_model: str = "weak",
    grid: GridSpec | None = None,
    xz: XzSpec | None = XzSpec(24, 8 * um),
    pupil_n: int = 256,
    comb_dft_points: int = 16,
    centre: tuple[float, float] | None = None,
    plane: float | None = None,
    reference: bool = True,
) -> FrameResult:
    """One advanced-engine frame on a grid centred on ``aodl``'s brightest spot by default."""
    params = wfs.params
    spot = aodl_spot(wfs, t)
    centre = (spot.x, spot.y) if centre is None else centre
    plane = tracked_plane(wfs, t) if plane is None else plane
    grid = GridSpec(6 * um, 6 * um, 96) if grid is None else grid
    model = ModelSpec(
        pupil_n=pupil_n, order_model=order_model, comb_dft_points=comb_dft_points, max_groups=128
    )
    return simulate_frames(
        wfs,
        params,
        [t],
        [plane],
        [centre],
        Misalignment.aligned() if mis is None else mis,
        grid,
        xz,
        model,
        reference,
    )[0]


def window_power(frame: FrameResult, x0: float, half: float, y0: float | None = None) -> float:
    """``sum I dx dy`` over ``|x - x0| <= half`` (and ``|y - y0| <= half`` when ``y0`` is given)."""
    dx = float(frame.x[1] - frame.x[0])
    dy = float(frame.y[1] - frame.y[0])
    cols = np.abs(frame.x - x0) <= half
    rows = np.ones(frame.y.shape, dtype=bool) if y0 is None else np.abs(frame.y - y0) <= half
    return float(frame.xy[np.ix_(rows, cols)].sum() * dx * dy)


def local_maxima(frame: FrameResult, threshold: float = 0.1) -> list[tuple[float, float, float]]:
    """``(x, y, peak)`` of every strict local maximum above ``threshold * max``, with a 3x3
    intensity-weighted centroid refinement — the spot finder of the array cross-check."""
    img = frame.xy
    peak = float(img.max())
    out: list[tuple[float, float, float]] = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            v = img[i, j]
            if v < threshold * peak:
                continue
            patch = img[i - 1 : i + 2, j - 1 : j + 2]
            if v < patch.max() or np.count_nonzero(patch == v) > 1:
                continue
            w = 3
            ys = slice(max(i - w, 0), i + w + 1)
            xs = slice(max(j - w, 0), j + w + 1)
            sub = img[ys, xs]
            total = sub.sum()
            xc = float((sub.sum(axis=0) * frame.x[xs]).sum() / total)
            yc = float((sub.sum(axis=1) * frame.y[ys]).sum() / total)
            out.append((xc, yc, float(v)))
    return out


__all__ = [
    "DEFLECTION",
    "LENS",
    "PARAMS",
    "TAU",
    "W0",
    "Z_R",
    "aodl_spot",
    "aodl_spots",
    "fading_hold",
    "lift_hold",
    "local_maxima",
    "pick_and_place",
    "reparam",
    "run_frame",
    "static_array",
    "static_offset_trap",
    "tracked_plane",
    "window_power",
    "with_order",
]
