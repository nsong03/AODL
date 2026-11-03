"""``POST /api/aperture`` (API.md §5.3)."""

from __future__ import annotations

from fastapi import APIRouter

from ..aperture import aperture_response
from ..schemas import ApertureRequest, ApertureResponse

router = APIRouter()


@router.post("/aperture", response_model=ApertureResponse)
def aperture(req: ApertureRequest) -> ApertureResponse:
    return aperture_response(req)


__all__ = ["router"]
