"""Domain errors as HTTP 400 bodies (API.md §0.1).

``aodl`` refuses impossible requests with ``ValueError`` messages written for humans; the API
passes them through verbatim under a machine-readable ``code``.  Validation failures never
reach this module — pydantic answers those with FastAPI's default 422.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

#: ``code`` values of API.md §0.1.
ERROR_CODES: tuple[str, ...] = (
    "band_violation",
    "invalid_spec",
    "too_many_terms",
    "plan_not_cached",
    "coverage",
    "unsupported",
)


class ApiError(Exception):
    """A domain refusal: ``{"error": {"code", "message", "detail"}}`` with HTTP ``status``."""

    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status = int(status)
        self.detail = detail

    def body(self) -> dict[str, Any]:
        """The JSON body of API.md §0.1."""
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


def classify(exc: Exception, default: str = "invalid_spec") -> ApiError:
    """Map an ``aodl`` refusal onto an API error code, keeping its message verbatim.

    * ``"leaves its usable band"`` — the Eq. 1 / Shepard band refusals of
      :mod:`aodl.waveform.synthesis` -> ``band_violation``;
    * ``"pupil term product"`` — :data:`aodl.device.aodl.MAX_TERMS` -> ``too_many_terms``;
    * ``"needs drive time"`` / ``"is before channel"`` — :func:`aodl.engine.simulate`'s
      coverage rule -> ``coverage``;
    * ``"cannot serialize"`` — the NPZ schema's refusals -> ``unsupported``;
    * everything else from the spec/synthesis layer -> ``default`` (``invalid_spec``).
    """
    if isinstance(exc, ApiError):
        return exc
    message = str(exc)
    if "leaves its usable band" in message:
        code = "band_violation"
    elif "pupil term product" in message:
        code = "too_many_terms"
    elif "needs drive time" in message or "is before channel" in message:
        code = "coverage"
    elif "cannot serialize" in message:
        code = "unsupported"
    else:
        code = default
    return ApiError(code, message)


def install_handlers(app: FastAPI) -> None:
    """Register the :class:`ApiError` -> JSON handler on ``app``."""

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=exc.body())


__all__ = ["ERROR_CODES", "ApiError", "classify", "install_handlers"]
