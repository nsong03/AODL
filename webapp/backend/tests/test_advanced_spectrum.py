"""RF spectrum, crest factor and modulation depth of rendered drives (W2 §5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _advanced_fixtures import PARAMS

from aodl import ArraySpec, Hold, TrajectorySpec, synthesize
from aodl.units import MHz, us
from aodl_advanced.spectrum import spectrum

RATE = 625 * MHz


def _static(mx: int, phases: str) -> object:
    spec = TrajectorySpec(
        array=ArraySpec(mx, 1, 1.0 * MHz if mx > 1 else 0.0, 0.0), moves=(Hold(200 * us),)
    )
    return synthesize(spec, PARAMS, phases=phases)


def test_single_tone_is_one_line_with_crest_sqrt2() -> None:
    wfs = _static(1, "zero")
    result = spectrum(wfs, "Ax", 100 * us, 40 * us, RATE)
    assert result.crest_factor == pytest.approx(math.sqrt(2.0), rel=0.01)
    assert result.papr_db == pytest.approx(20 * math.log10(math.sqrt(2.0)), abs=0.1)
    assert result.peak_depth_rad == pytest.approx(PARAMS.channels["Ax"].drive_strength, rel=0.01)
    peak_f = result.f[int(np.argmax(result.psd_db))]
    assert peak_f == pytest.approx(PARAMS.channels["Ax"].f_center, abs=50e3)
    # everything 1 MHz away from the carrier is far below the line (Hann side lobes)
    far = np.abs(result.f - peak_f) > 1 * MHz
    assert result.psd_db[far].max() < -60.0
    assert result.psd_db.max() == 0.0


def test_eight_tone_ladder_zero_phase_versus_schroeder_crest() -> None:
    zero = spectrum(_static(8, "zero"), "Bx", 100 * us, 40 * us, RATE)
    schroeder = spectrum(_static(8, "schroeder"), "Bx", 100 * us, 40 * us, RATE)
    assert zero.crest_factor == pytest.approx(4.0, rel=0.05)  # sqrt(2 M) for M in-phase tones
    assert schroeder.crest_factor < 2.3
    assert zero.peak_depth_rad == pytest.approx(8 * PARAMS.channels["Bx"].drive_strength, rel=0.02)
    # eight lines spaced 1 MHz, within 1 dB of each other
    lines = [
        schroeder.psd_db[int(np.argmin(np.abs(schroeder.f - f)))]
        for f in PARAMS.channels["Bx"].f_center + (np.arange(8) - 3.5) * MHz
    ]
    assert max(lines) - min(lines) < 1.0


def test_window_is_clamped_and_errors_are_clear() -> None:
    wfs = _static(1, "zero")
    result = spectrum(wfs, "Ax", 5 * us, 40 * us, RATE)
    assert result.t_span[0] == 0.0 and result.notes
    with pytest.raises(ValueError, match="not driven|unknown"):
        spectrum(wfs, "Cz", 100 * us, 40 * us, RATE)
    with pytest.raises(ValueError, match="outside the programmed drive"):
        spectrum(wfs, "Ax", 5000 * us, 10 * us, RATE)
