"""Service settings: cache sizes, the performance budget's knobs, and the few environment
variables the app reads (``DESIGN.md`` §1.2, §3).

Everything here is a plain constant or a function of ``os.environ``; nothing is physics.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Entries kept in the ``MotionPlan`` LRU (DESIGN §1.2 asks for at least 32).
PLAN_CACHE_SIZE = 64

#: Entries kept in the simulation LRU (one entry per ``(plan, times, mixing order)``).
SIM_CACHE_SIZE = 32

#: Frequency-group budget of the movie metrics: when the first frame (at ``t = 2 tau``) at the
#: requested mixing order holds more groups than this, the metrics fall back to order 1
#: (DESIGN §1.2) unless the request pins the order with ``movie_mixing_order``.
MOVIE_GROUP_BUDGET = 160

#: Frame-count ceiling of ``/api/simulate`` (API.md §5.1).
MAX_FRAMES = 240

#: Largest image side ``/api/frame`` renders (API.md §5.2).
MAX_IMAGE_SIDE = 512

#: Samples of the requested trajectory shipped with a plan (DESIGN §3).
TRAJECTORY_SAMPLES = 400

#: Significant digits kept in simulate frames (API.md §5.1).
SPOT_DIGITS = 6

#: Spots below this fraction of a frame's maximum power are dropped (API.md §5.1).
SPOT_POWER_CUT = 1e-3

#: Vite dev-server origin allowed by CORS.
DEV_ORIGIN = "http://localhost:5173"

#: Environment variable that turns a missing/broken advanced router into a hard failure.
STRICT_ADVANCED_ENV = "AODL_STUDIO_STRICT_ADVANCED"

#: Environment variable overriding where the built frontend lives.
FRONTEND_DIST_ENV = "AODL_STUDIO_FRONTEND_DIST"


def strict_advanced() -> bool:
    """Is the advanced-router import required to succeed?  (CI and ``make test`` set it.)"""
    return os.environ.get(STRICT_ADVANCED_ENV, "") == "1"


def frontend_dist() -> Path:
    """Directory of the built frontend (``webapp/frontend/dist`` unless overridden)."""
    override = os.environ.get(FRONTEND_DIST_ENV)
    if override:
        return Path(override)
    # src/aodl_webapp/config.py -> src/aodl_webapp -> src -> backend -> webapp
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


__all__ = [
    "DEV_ORIGIN",
    "FRONTEND_DIST_ENV",
    "MAX_FRAMES",
    "MAX_IMAGE_SIDE",
    "MOVIE_GROUP_BUDGET",
    "PLAN_CACHE_SIZE",
    "SIM_CACHE_SIZE",
    "SPOT_DIGITS",
    "SPOT_POWER_CUT",
    "STRICT_ADVANCED_ENV",
    "TRAJECTORY_SAMPLES",
    "frontend_dist",
    "strict_advanced",
]
