"""``GET /api/health`` (API.md §1)."""

from __future__ import annotations

from fastapi import APIRouter

import aodl

from .. import __version__
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", aodl_version=aodl.__version__, version=__version__)


__all__ = ["router"]
