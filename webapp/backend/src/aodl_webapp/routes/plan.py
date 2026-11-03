"""``POST /api/plan``, ``GET /api/plan/{id}/waveform.npz``, ``POST /api/code`` (API.md §4-5)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Response

from ..codegen import generate_code
from ..errors import ApiError
from ..planner import cached_plan, plan_response
from ..schemas import CodeResponse, PlanRequest, PlanResponse

router = APIRouter()


@router.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    return plan_response(req)


@router.post("/code", response_model=CodeResponse)
def code(req: PlanRequest) -> CodeResponse:
    return CodeResponse(code_python=generate_code(req))


@router.get("/plan/{plan_id}/waveform.npz", response_class=Response)
def waveform_npz(plan_id: str) -> Response:
    """The parametric NPZ of a cached plan (``WaveformSet.save`` via a temporary file).

    404 ``plan_not_cached`` when the plan was evicted (the client re-posts the plan); 400
    ``unsupported`` when the drive cannot be serialised (``amp != 1`` under fading-Shepard,
    or a ``SwitchRamped`` envelope).
    """
    cached = cached_plan(plan_id)
    if cached is None:
        raise ApiError(
            "plan_not_cached",
            f"plan {plan_id!r} is not in the server cache; POST the plan again and retry.",
            status=404,
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"aodl_{plan_id}.npz"
        try:
            written = cached.wfs.save(path)
        except (ValueError, TypeError) as exc:
            raise ApiError("unsupported", str(exc)) from exc
        payload = written.read_bytes()
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="aodl_{plan_id}.npz"'},
    )


__all__ = ["router"]
