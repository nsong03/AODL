"""``POST /api/simulate`` (API.md §5.1) — a plain ``def`` so FastAPI runs it in its threadpool."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import SimulateRequest, SimulateResponse
from ..simulation import simulate_response

router = APIRouter()


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    return simulate_response(req)


__all__ = ["router"]
