"""``aodl_advanced`` — the FFT pupil engine and diffraction-efficiency model of AODL Studio.

A *separate* physics path from ``aodl``'s closed-form simulation (DESIGN §1.5): it imports
from ``aodl`` only the parameters, units, the sign authority (``aodl.device.conventions``), the
waveform IR and envelopes, and the sample renderer — never ``aodl.field``, ``aodl.device.aod*``
or ``aodl.engine`` — so that agreement between the two engines at zero misalignment is a
genuine cross-check (``tests/test_cross_engine.py``, enforced by
``tests/test_advanced_independence.py``).

Modules: :mod:`.efficiency` (Bragg/TPM efficiency of the A and A–B pairs), :mod:`.misalignment`
(per-AOD knobs), :mod:`.pupil` (misaligned retarded-time pupils, the J1 law, frequency lines,
groups), :mod:`.propagate` (chirp-z zoom transforms), :mod:`.metrics`, :mod:`.spectrum`,
:mod:`.engine` (frame driver) and :mod:`.schemas` (the pydantic models of API.md §6, which
also import ``aodl_webapp.schemas`` and are therefore not imported here).
"""

from .efficiency import EfficiencyModel, default_model
from .engine import FrameResult, GridSpec, ModelSpec, XzSpec, simulate_frames
from .misalignment import AODMisalignment, Misalignment
from .spectrum import SpectrumResult, spectrum

__version__ = "0.1.0"

__all__ = [
    "AODMisalignment",
    "EfficiencyModel",
    "FrameResult",
    "GridSpec",
    "Misalignment",
    "ModelSpec",
    "SpectrumResult",
    "XzSpec",
    "__version__",
    "default_model",
    "simulate_frames",
    "spectrum",
]
