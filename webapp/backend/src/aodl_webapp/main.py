"""The FastAPI application factory (DESIGN §3).

``create_app()`` mounts every router under ``/api``, opens CORS for the Vite dev server,
installs the API.md §0.1 error handler, and serves the built frontend from
``webapp/frontend/dist`` when it exists (SPA fallback to ``index.html``); without a build the
root answers with a JSON pointer to the dev server.

The advanced router is W2's module.  Its import is guarded so the basic app still serves when
the module is absent, but a *broken* import can never ship silently: with
``AODL_STUDIO_STRICT_ADVANCED=1`` (the Makefile's ``test`` target and CI) the ``ImportError``
propagates.

Run: ``uvicorn aodl_webapp.main:create_app --factory --port 8000``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import DEV_ORIGIN, frontend_dist, strict_advanced
from .errors import install_handlers
from .routes import aperture, frame, health, plan, presets, simulate

try:
    from .routes.advanced import router as advanced_router
except ImportError as exc:  # W2 ships this module
    if strict_advanced():
        raise
    logging.getLogger(__name__).warning("advanced routes unavailable: %s", exc)
    advanced_router = None  # type: ignore[assignment]


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve ``dist`` at ``/``: real files as files, every other path as ``index.html``."""
    index = dist / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        candidate = (dist / full_path).resolve() if full_path else index
        if full_path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        if full_path == "api" or full_path.startswith(("api/", "assets/")):
            # an unknown API route or a missing bundle asset is a 404, never the SPA shell (W4)
            raise HTTPException(status_code=404, detail=f"no such route or file: /{full_path}")
        return FileResponse(index)


def create_app() -> FastAPI:
    """Build the application (the ``--factory`` entry point)."""
    app = FastAPI(title="AODL Studio", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[DEV_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_handlers(app)
    for module in (health, presets, plan, simulate, frame, aperture):
        app.include_router(module.router, prefix="/api")
    if advanced_router is not None:
        app.include_router(advanced_router, prefix="/api")

    dist = frontend_dist()
    if (dist / "index.html").is_file():
        _mount_frontend(app, dist)
    else:

        @app.get("/", include_in_schema=False)
        def root() -> JSONResponse:
            return JSONResponse(
                {
                    "service": "AODL Studio API",
                    "version": __version__,
                    "docs": "/docs",
                    "frontend": (
                        f"{DEV_ORIGIN} (run `make dev-frontend`; `make build` for a static bundle)"
                    ),
                }
            )

    return app


__all__ = ["create_app"]
