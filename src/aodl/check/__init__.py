r"""The independent FFT checker (milestone M6).

Everything else in :mod:`aodl` runs one way: a trajectory becomes a parametric waveform,
the waveform becomes pupil terms through the Eqs. S5-S6 Taylor expansion, and the terms
become closed-form Gaussian fields.  Fast, exact for quadratic phase — and entirely
self-consistent, which is precisely the problem: a sign error shared by the synthesizer and
the simulator is invisible to every test that goes through both.

``aodl.check`` closes that loop from outside.  It takes the **rendered RF samples** — the
literal AWG buffers, carrier included and globally normalized — measures the drive back off
them, rebuilds the aperture field with no Taylor expansion at all, and propagates it to the
image plane with a zoom (chirp-z) transform.  It shares with the simulator only two things:
:mod:`aodl.params` and :mod:`aodl.device.conventions`, the sign authority.  It imports
nothing from ``field/``, ``device/aod``, ``device/aodl``, ``device/mixing``, ``engine`` or
the waveform IR, and ``tests/test_check_independence.py`` enforces that by scanning the
source.  (This is also the one part of the package that is *allowed* to use FFTs; the
simulation path still has none — ``CLAUDE.md``.)

The pipeline::

    samples ── record.SampleRecord ─── demod.demodulate ──▶ per-channel z(t)
                                                              │
                          pupil.ApertureGrid + channel/axis_pupil (Eqs. S1-S4, retarded time)
                                                              │
                                    transform.zoom_field (Eq. S11 + defocus)
                                                              │
                            metrics: profile fits, best focus, blob audit

Two pupil models are available (:data:`~aodl.check.pupil.PupilMode`): ``weak`` is the linear
Eq. S3 model the simulator implements, so it cross-validates the analytic path directly, and
``bragg_band`` is the full ``exp(i C V)`` crystal with the ``+1`` order cut out in the
aperture's spatial-frequency domain, which carries compression and every intermodulation
product with no expansion order to truncate.

WO-22 builds the expectation, verdict and product wiring on top of these pieces.
"""

from __future__ import annotations

from .demod import Baseband, demodulate, out_of_band_fraction, sample_baseband
from .metrics import (
    Blob,
    TrapFit,
    accumulate_intensity,
    accumulate_marginals,
    best_focus,
    find_blobs,
    fit_gaussian_1d,
    profile_moments,
)
from .pupil import ApertureGrid, PupilMode, axis_pupil, band_window, channel_pupil
from .record import SampleRecord, from_arrays, load_samples
from .transform import subtimes, zoom_field

__all__ = [
    "ApertureGrid",
    "Baseband",
    "Blob",
    "PupilMode",
    "SampleRecord",
    "TrapFit",
    "accumulate_intensity",
    "accumulate_marginals",
    "axis_pupil",
    "band_window",
    "best_focus",
    "channel_pupil",
    "demodulate",
    "find_blobs",
    "fit_gaussian_1d",
    "from_arrays",
    "load_samples",
    "out_of_band_fraction",
    "profile_moments",
    "sample_baseband",
    "subtimes",
    "zoom_field",
]
