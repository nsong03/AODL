"""``POST /api/frame`` (API.md §5.2) — a plain ``def`` so FastAPI runs it in its threadpool."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import FrameRequest, FrameResponse
from ..simulation import frame_response

router = APIRouter()


@router.post("/frame", response_model=FrameResponse)
def frame(req: FrameRequest) -> FrameResponse:
    return frame_response(req)


__all__ = ["router"]
