"""``PlanRequest`` moves -> :class:`aodl.TrajectorySpec` (SI), and the sampled request.

The display-unit boundary for geometry and time: µm and µs become metres and seconds here,
exactly (``value * um`` is one float multiplication), and :class:`aodl.Path` points keep
their meaning — displacements relative to where the path starts (DESIGN §1.3).
"""

from __future__ import annotations

import numpy as np

from aodl.trajectory.spec import ArraySpec, Hold, Lift, Move, Path, TrajectorySpec, Translate
from aodl.units import MHz, um, us

from .config import TRAJECTORY_SAMPLES
from .errors import classify
from .schemas import (
    HoldMove,
    LiftMove,
    PathMove,
    PlanRequest,
    TrajectorySamples,
    TranslateMove,
)


def to_move(move: LiftMove | TranslateMove | HoldMove | PathMove) -> Move:
    """One API move -> one ``aodl`` move, SI."""
    if isinstance(move, LiftMove):
        return Lift(dz=move.dz_um * um, duration=move.duration_us * us, profile=move.profile)
    if isinstance(move, TranslateMove):
        return Translate(
            dx=move.dx_um * um,
            dy=move.dy_um * um,
            duration=move.duration_us * us,
            profile=move.profile,
        )
    if isinstance(move, HoldMove):
        return Hold(duration=move.duration_us * us)
    return Path(
        points=tuple((p[0] * um, p[1] * um, p[2] * um) for p in move.points_um),
        duration=move.duration_us * us,
        closed=move.closed,
    )


def to_spec(req: PlanRequest) -> TrajectorySpec:
    """API.md §3.2–3.5 -> :class:`aodl.TrajectorySpec`; ``aodl``'s refusals become
    ``invalid_spec`` (zero spacing on a ladder axis, a degenerate ``Path``, ...)."""
    try:
        array = ArraySpec(
            mx=req.array.mx,
            my=req.array.my,
            delta_f_x=req.array.delta_f_x_MHz * MHz,
            delta_f_y=req.array.delta_f_y_MHz * MHz,
        )
        spec = TrajectorySpec(array=array, moves=tuple(to_move(m) for m in req.moves))
        spec.compile()  # a Path's zero-length chord only shows up here
    except ValueError as exc:
        raise classify(exc, "invalid_spec") from exc
    return spec


def seams(spec: TrajectorySpec) -> list[float]:
    """Move boundaries [s] including ``0`` and ``T``."""
    out = [0.0]
    for move in spec.moves:
        out.append(out[-1] + move.duration)
    return out


def sample_trajectory(spec: TrajectorySpec, n: int = TRAJECTORY_SAMPLES) -> TrajectorySamples:
    """The requested ``X(t), Y(t), Z(t)`` on ``n`` points of ``[0, T]`` plus the seams, in
    µm / µs — what the frontend overlays on the simulated spot (DESIGN §1 item 3)."""
    x, y, z = spec.compile()
    t = np.linspace(0.0, spec.duration, max(int(n), 2))
    return TrajectorySamples(
        t_us=(t / us).tolist(),
        x_um=(np.asarray(x(t), dtype=np.float64) / um).tolist(),
        y_um=(np.asarray(y(t), dtype=np.float64) / um).tolist(),
        z_um=(np.asarray(z(t), dtype=np.float64) / um).tolist(),
        seams_us=[s / us for s in seams(spec)],
    )


__all__ = ["sample_trajectory", "seams", "to_move", "to_spec"]
