"""AODL — waveform synthesis and closed-form optical simulation for a 3D AODL.

Physics reference: arXiv:2510.11451 (equation numbers ``S#`` cited throughout the code
refer to its Supplement).

The package front door re-exports the objects a lab notebook actually touches, in pipeline
order (``docs/ARCHITECTURE.md`` §1)::

    import numpy as np
    from aodl import ChannelWaveform, ToneTrack, WaveformSet, default_1030, ramps
    from aodl import array_tones, render_movie, simulate
    from aodl.units import MHz, us

    p = default_1030()
    tone = ToneTrack(freq=ramps.min_jerk(0.0, 100 * us, 0.0, 5 * MHz))
    wfs = WaveformSet({"Ay": ChannelWaveform((tone,))}, p).with_hold_until(100 * us)
    result = simulate(wfs, np.linspace(0.0, 100 * us, 120))
    render_movie(result, "sweep.mp4")

    # ... or a 5x5 array: a Schroeder-phased tone ladder per crossed channel (M2)
    ladder = array_tones(5, 1 * MHz, t1=50 * us)
    array = WaveformSet({"Ax": ladder, "Ay": ladder}, p)

Everything else stays one import away (``from aodl.device.aodl import build_terms``,
``from aodl.field import focal``, ...); SI units are used throughout, with the ``aodl.units``
constants for boundary code.
"""

from . import params, units
from .device.aodl import TermArray, build_terms
from .engine import SimResult, simulate
from .field.focal import FrameGrid, intensity_frame, intensity_slice_xz
from .field.measure import SpotMetrics, measure, track_z
from .params import (
    CHANNELS,
    AODLParams,
    AODParams,
    OpticsParams,
    default_1030,
    paper_808,
)
from .poly import PiecewisePoly
from .trajectory import ramps
from .viz.movie import auto_grid, render_movie
from .waveform.synthesis import add_common_ramp, array_tones, schroeder_phases
from .waveform.tones import (
    ChannelWaveform,
    ConstantEnvelope,
    Envelope,
    SmoothOnOff,
    ToneTrack,
    WaveformSet,
)

__version__ = "0.1.0"

__all__ = [
    "CHANNELS",
    "AODLParams",
    "AODParams",
    "ChannelWaveform",
    "ConstantEnvelope",
    "Envelope",
    "FrameGrid",
    "OpticsParams",
    "PiecewisePoly",
    "SimResult",
    "SmoothOnOff",
    "SpotMetrics",
    "TermArray",
    "ToneTrack",
    "WaveformSet",
    "__version__",
    "add_common_ramp",
    "array_tones",
    "auto_grid",
    "build_terms",
    "default_1030",
    "intensity_frame",
    "intensity_slice_xz",
    "measure",
    "paper_808",
    "params",
    "ramps",
    "render_movie",
    "schroeder_phases",
    "simulate",
    "track_z",
    "units",
]
