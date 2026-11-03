"""``GET /api/presets`` (API.md §2): hardware presets, ramp profiles, starter examples."""

from __future__ import annotations

from fastapi import APIRouter

from ..examples import EXAMPLES
from ..hardware import HARDWARE_PRESETS
from ..schemas import PROFILES, PresetsResponse

router = APIRouter()


@router.get("/presets", response_model=PresetsResponse)
def presets() -> PresetsResponse:
    return PresetsResponse(
        hardware_presets=dict(HARDWARE_PRESETS), profiles=list(PROFILES), examples=list(EXAMPLES)
    )


__all__ = ["router"]
