# W1 — backend API package (`aodl_webapp`) and the `Path` move

**Role:** implementation agent. **Authority:** this work order; then `../DESIGN.md` and
`../API.md` (normative JSON contract). The repo's `CLAUDE.md` conventions apply to code under
`src/aodl` (SI units, equation citations, sign authority) and, by extension, to
`webapp/backend`. **No git commands** — leave the working tree for the closer.

Two other agents work concurrently in the same tree: W2 (`webapp/backend/src/aodl_advanced/**`,
`webapp/backend/src/aodl_webapp/routes/advanced.py`, `webapp/backend/tests/test_advanced_*.py`,
`test_cross_engine.py`) and W3 (`webapp/frontend/**`). Do not touch their files.

## Review corrections folded in (read these first)
The pre-implementation review (`../reports/REVIEW.md`) changed the following; they are already
reflected in `API.md`/`DESIGN.md` and below: (1) one `Spot` per *spatial cluster* within a
frequency group, not per group — shadow-tweezer pairs must come out as two spots; (2) simulate
frames are struct-of-arrays, rounded, with a `1e-3` power cut, and `bounds_um` comes from the
requested trajectory, not the spot table; (3) the movie mixing-order fallback is a *group-count*
budget (> 160 groups), not a trap count; (4) `/api/aperture` returns envelope + instantaneous
frequency (the literal carrier only on request at ≥ 4096 points); (5) `phases="random"` is
seeded (`phase_seed`); (6) `amp ≠ 1` + Shepard → NPZ 400 `unsupported`; (7) `constants.beam_transit_us
= 2 w_in / v`; (8) `frame_spots(...)` joins the frozen signature list; (9) the advanced-router
import is strict under `AODL_STUDIO_STRICT_ADVANCED=1`; (10) `codegen` emits the resolved
`ShepardConfig`/phases/rng and a fade-schedule comment; (11) example fixes: `sphere_loop` is
`closed=True`, `array_arbitrary_path`'s circle is tilted 30° about x. **Delivery order matters
for the other agents:** `Path` → `examples.py` → `ir.py`/`evaluate_ir` →
`gen_frontend_fixtures.py` (write the fixtures as early as possible) → everything else.

## Owned files
- `src/aodl/trajectory/spec.py` (add `Path`), `src/aodl/__init__.py` (export `Path`),
  `tests/test_path.py`, `docs/guide.md` (one paragraph in §3 + one row in the moves table if
  there is one; keep `tests/test_docs.py` green).
- `webapp/backend/pyproject.toml`, `webapp/backend/src/aodl_webapp/**` **except**
  `routes/advanced.py`, `webapp/backend/tests/test_api_*.py`, `tests/test_ir.py`,
  `tests/test_codegen.py`, `tests/test_trajectory.py`, `tests/test_path_api.py`, `tests/conftest.py`.
- `webapp/scripts/gen_frontend_fixtures.py` and its output `webapp/frontend/tests/fixtures/*.json`
  (W3 reads them; W3 owns everything else under `frontend/`).
- `webapp/Makefile` (all targets — see §6; W3/W4 will not edit it).

## 1. `Path` move in `aodl` (DESIGN §1.3)

```python
@dataclass(frozen=True)
class Path:
    """Smooth 3D waypoint path: clamped cubic spline through relative waypoints. ..."""
    points: tuple[tuple[float, float, float], ...]   # (dx, dy, dz) [m] relative to the path start
    duration: float                                   # [s]
    closed: bool = False
```
- Validation in `__post_init__`: ≥ 1 point, finite, positive duration; normalise `points` to a
  tuple of float triples. `closed=True` appends `(0, 0, 0)` unless the last point already is
  (within 1e-12 m).
- `compile` support in `TrajectorySpec.compile()`: waypoints `[0, p_1, …, p_n]` (absolute =
  running position + relative), knot times `t_0 + duration · cumlen_i / cumlen_n` from chord
  lengths; a zero-length chord raises `ValueError("Path waypoint i duplicates waypoint i-1 …")`;
  `scipy.interpolate.CubicSpline(knots, pts, axis=0, bc_type="clamped")`; convert each
  piece to normalized local time: `CubicSpline.c[j, k, axis]` is the coefficient of
  `(t − knot_k)^(3−j)`, so the normalized coefficient of `tau^m` is `c[3−m, k, axis] · T_k^m`.
  Build one `PiecewisePoly(breaks, coeffs)` per axis (degree 3, one row per spline piece).
  The existing `_check_continuity` must pass. Update the `Move` union, the `__post_init__`
  isinstance check, docstrings, `__all__`.
- Tests (`tests/test_path.py`, ≥ 8): value continuity at knots; zero velocity at both ends
  (derivative poly evaluated at `t0`, `t1` is `0` to 1e-9 relative to the peak speed);
  C² at interior knots (second derivative continuous to 1e-6 relative); passes through every
  waypoint to 1e-12 m; `closed` returns to start; one-point path equals the `constant_jerk`
  ramp exactly (clamped cubic between two rest points *is* `3τ² − 2τ³`); duplicate waypoint
  raises; chaining `Lift → Path → Hold` keeps the whole trajectory continuous and
  `synthesize(spec, default_1030())` band-checks a small circle (radius 5 µm, 8 points,
  200 µs) without error; `plan_motion` on it gives `mode == "s19"`.
- `ruff check src tests`, `ruff format --check src tests`, `python -m mypy src/aodl`, and the
  **whole** root suite (`pytest`) must stay green.

## 2. Package `aodl_webapp`

`webapp/backend/pyproject.toml`: project `aodl-studio` 0.1.0, Python ≥ 3.11, src layout with
`packages = ["aodl_webapp", "aodl_advanced"]` (W2 fills the second; create an empty
`aodl_advanced/__init__.py` **only if it does not exist** — check first, never overwrite),
deps: `aodl` (install editable from the repo root: document `pip install -e ../.. -e .[dev]`),
`fastapi>=0.110`, `uvicorn[standard]`, `numpy`, `scipy>=1.8`, `pydantic>=2`; dev:
`pytest`, `httpx`, `ruff`, `mypy`. Ruff line length 100, same rule set as the root.

Modules (DESIGN §2/§3). Signatures below are **frozen** (W2's routes call them):

```python
# aodl_webapp/errors.py
class ApiError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, detail: Any = None): ...
# + FastAPI exception handler producing API.md §0.1 bodies; map aodl ValueError messages
#   containing "leaves its usable band" -> band_violation, "pupil term product" -> too_many_terms,
#   simulate coverage messages ("needs drive time" / "is before channel") -> coverage,
#   everything else from spec/synthesis -> invalid_spec.

# aodl_webapp/hardware.py
def to_params(hw: HardwareSpec) -> AODLParams
def constants(params: AODLParams) -> PlanConstants      # API.md §4 "constants"
HARDWARE_PRESETS: dict[str, HardwareSpec]               # default_1030, paper_808

# aodl_webapp/trajectory.py
def to_spec(req: PlanRequest) -> TrajectorySpec        # moves -> Lift/Translate/Hold/Path, SI
def sample_trajectory(spec, n=400) -> TrajectorySamples # t_us, x_um, y_um, z_um + seams_us

# aodl_webapp/planner.py
def plan_id(req: PlanRequest) -> str                     # API.md §0.2
def get_plan(req: PlanRequest) -> MotionPlan             # LRU-cached by plan_id (>= 32 entries), raises ApiError
def plan_response(req: PlanRequest) -> PlanResponse
def movie_params(req: PlanRequest, movie_mixing_order: int | None) -> AODLParams  # DESIGN §1.2 rule

# aodl_webapp/ir.py
def to_ir(wfs: WaveformSet) -> WaveformIR
def evaluate_ir(ir: WaveformIR, channel: str, t_us: ArrayLike) -> dict[str, np.ndarray]
    # keys "f_MHz" (n_tones, n_t), "A", "phase_rad", "V" (n_t, carrier included) — pure numpy
    # reference implementation of the TS evaluator; must agree with ToneTrack/Envelope to 1e-9

# aodl_webapp/simulation.py
def get_sim(req: PlanRequest, times_s: np.ndarray, movie_mixing_order: int | None) -> SimResult  # cached
def simulate_response(req: SimulateRequest) -> SimulateResponse
def frame_response(req: FrameRequest) -> FrameResponse
def pack_image(arr: np.ndarray) -> PackedImage           # API.md §0.3, uint16 LE base64
def tracked_plane(plan: MotionPlan, t_s: float, params: AODLParams) -> float   # track_z of a one-frame simulate
def frame_spots(plan: MotionPlan, t_s: float, params: AODLParams) -> list[Spot]  # the clustered spots of one frame (W2 centres its grid on them)
def cluster_spots(terms: TermArray, optics: OpticsParams, tol: float) -> list[tuple[SpotMetrics, int, int]]
    # for each group_terms() group, split its terms by spot centre theta1*F/k into clusters within
    # 1 w0 (single-linkage), measure(engine._subset(terms, idx)) each cluster -> (metrics, group, n_terms)

# aodl_webapp/aperture.py
def aperture_response(req: ApertureRequest) -> ApertureResponse   # device.aod.aperture_window + fill mask + t_ret

# aodl_webapp/codegen.py
def generate_code(req: PlanRequest) -> str

# aodl_webapp/examples.py
EXAMPLES: list[ExampleSpec]      # ids in API.md §2, exactly those seven
```

Routes (`routes/*.py`, each exposing `router`): `health`, `presets`, `plan` (incl.
`GET /plan/{plan_id}/waveform.npz` and `POST /code`), `simulate`, `frame`, `aperture`.
`main.py: create_app()` includes them under `/api`, adds CORS for `http://localhost:5173`,
mounts `frontend/dist` at `/` when the directory exists (SPA fallback to `index.html`), and
includes W2's advanced router like this — keep it exactly so W2's module is the only owner:

```python
try:
    from .routes.advanced import router as advanced_router
except ImportError as exc:                # W2 ships this module
    logging.getLogger(__name__).warning("advanced routes unavailable: %s", exc)
    advanced_router = None
```

Pydantic class names (`schemas.py`): `HardwareSpec`, `ArraySpecModel`, `LiftMove`,
`TranslateMove`, `HoldMove`, `PathMove`, `MoveSpec` (Annotated discriminated union on `kind`),
`PlanOptions`, `PlanRequest`, `BandUsage`, `ZBudget`, `FadeEventModel`, `PlanConstants`,
`TrajectorySamples`, `Poly`, `ConstantEnv`, `SmoothOnOffEnv`, `FadeZoneEnv`, `EnvelopeModel`,
`ToneIR`, `ChannelIR`, `WaveformIR`, `PlanResponse`, `Spot`, `SpotFrame`, `SimulateRequest`,
`SimulateResponse`, `FrameGridModel`, `XzModel`, `PackedImage`, `ImageXY`, `ImageXZ`, `Cuts`,
`FrameRequest`, `FrameResponse`, `ApertureRequest`, `ApertureChannel`, `ApertureVoltage`,
`ApertureResponse`, `ExampleSpec`, `PresetsResponse`, `CodeResponse`, `ErrorBody`.
`aodl_webapp/__init__.py` stays **bare** (version string only, no `main` import): W2's route
module imports `aodl_webapp.planner/simulation/schemas/errors`, and a package-level import of
`main` would make a cycle. The advanced-router import in `main.py` must raise instead of
warning when the environment variable `AODL_STUDIO_STRICT_ADVANCED=1` is set (the Makefile's
`test` target and CI set it), so a broken W2 import can never ship silently.

Behavioural notes:
- `simulate`: default times `linspace(0, min(T + τ, t_span[1] − τ/2), n_frames)`; before `τ/2`
  `build_terms` returns no terms → an empty `SpotFrame`. Build the frames with `cluster_spots`
  on `result.terms(i)` (so a fading drive's shadow pair is two spots at `±10.3·Δf µm`, not a
  phantom at the trap), pack them struct-of-arrays, round to 6 significant digits, drop spots
  below `1e-3` of the frame's max power. `bounds_um` from the **requested trajectory** samples
  ± the array half-extent (`(m−1)/2 · pitch`) ± 6 w0 laterally and ± 1 z_R axially; `power_ref`
  = median `power` over all kept spots. Mixing order for the metrics: the requested order unless
  the first frame at `t = 2τ` yields > 160 groups, then 1 (`movie_mixing_order` overrides either
  way); `notes` says what happened. Make the route a plain `def` (threadpool).
- `frame`: `plane="tracked"` uses `tracked_plane`; build `FrameGrid` from the request;
  `intensity_frame` and `intensity_slice_xz` on `result.terms(0)` of a one-frame simulate;
  `cuts` through the brightest spot (`max power`), `i_x` along the XY grid row nearest its `y`,
  `i_z` from `intensity_slice_xz` at its `x` column.
- `codegen`: the script must run with plain `aodl` (no `aodl_webapp` import). Emit hardware
  overrides only for fields that differ from `default_1030`; emit `Path(points=..., duration=...)`
  for path moves; emit the **resolved** synthesis options explicitly so the fade design is
  visible and the script is deterministic: when the plan is `"shepard"`, pass
  `shepard=ShepardConfig(delta_f_x=…, delta_f_y=…, eta=…)` with the spacings `auto_config`
  actually chose (read them off the plan's `FadeZoneEnvelope`s), `phases=…`,
  `rng=np.random.default_rng(seed)` for random phases, and the M5 options when non-default;
  precede the call with a comment block: mode, per-channel live rung counts, the fade schedule
  (`fade_events` times and axes) and the shadow offsets (`±deflection_scale·Δf`), and the Eq. 1
  budget line. End with the five product lines of `docs/guide.md` §2 (`summary`, `save`,
  `render_samples`, `simulate`, commented `movie`).
- `examples.py`: exactly the seven ids with these specs (µm, µs; defaults otherwise):
  `pick_and_place` 1×1: Lift +5/40, Translate (20, 0)/60, Lift −5/40 (expected `s19`) ·
  `square_xy` 1×1: Lift +3/30, four Translates of 15 µm/50 µs around a square, Lift −3/30
  (`s19`) · `sphere_loop` 1×1: one `path` of 24 points on a spherical spiral of radius 6 µm
  centred at (0, 0, 6) (start at the origin = south pole, θ 0→π, φ = 3θ), **`closed=True`** so
  it returns to the origin (as an open spiral it ended 12 µm up and planned as a Shepard drive
  with 0.97 MHz margin), 400 µs (expected `s19`) · `single_lift_hold` 1×1: Lift +10/60, Hold 600,
  Lift −10/60 (`shepard`, auto Δf 12 MHz) · `array_lift_hold` 4×4 (1.0/1.3 MHz): Lift +8/80,
  Hold 500, Lift −8/80 (`shepard`, 92 rungs) · `array_arbitrary_path` 3×3 (1.0/1.3 MHz): Lift
  +4/50, path of 12 points on a circle of radius 8 µm **tilted 30° about the x axis** (300 µs,
  closed), Lift −4/50 (`s19`, By margin ≈ 1.9 MHz) · `flagship_10x10` 10×10 (1.0/1.3): Lift
  +10/150, Translate (40, 25)/250, Lift −10/150 (`shepard`, 93 rungs; tag `slow`).
  Every example must plan without error at default options; add a test that asserts this and
  the expected `mode` of each.
- `gen_frontend_fixtures.py`: for `pick_and_place` and `array_lift_hold` write
  `frontend/tests/fixtures/<id>.json` = `{"ir": WaveformIR, "constants": {...}, "t_us": [48 times
  over t_span incl. before τ/2], "eval": {channel: {"f_MHz": [[...]], "A": [[...]], "phase_rad":
  [[...]], "V": [...]}}, "voltage_window": {"t_us": [2001 pts over ±0.2 µs around 0.4·T], channel:
  [...]}, "aperture": [{"time_us": 0.6·τ | 2τ, channel: {"u_mm": [...], "envelope": [...],
  "f_inst_MHz": [...], "filled": [...]}}]}` using `evaluate_ir` and `aperture_response`. Also
  write `fixtures/examples.json` (the presets). Do this **before** the routes.

## 3. Tests (`webapp/backend/tests`, pytest, ≥ 40 total)
- `test_api_plan.py`: every example plans (status 200, `mode` as expected), `plan_id` stable
  and equal to the formula, `band_usage` equals `aodl.band_usage` in MHz, out-of-band request
  → 400 `band_violation` with the `aodl` message, `shepard="always"` without spacings → 422,
  `PathMove` with a duplicate point → 400 `invalid_spec`, NPZ download round-trips through
  `WaveformSet.load`.
- `test_ir.py`: `evaluate_ir` vs `ToneTrack.f/phase/env.A` on 200 random times for an S19 drive
  and a fading drive: `f` 1e-9 relative, `A` 1e-12 absolute, `phase` 1e-9 relative of the
  largest; `V` vs `render_samples(..., dtype=np.float64, return_scale=True)` with `V/scale`
  compared at 1e-9; a `SwitchRamped` envelope → `ApiError("unsupported")`.
- `test_api_simulate.py`: frame count, empty `SpotFrame` for `t < τ/2`, spot x of the
  pick-and-place at `t = T + τ/2` (request `t_start_us = t_end_us = T + τ/2`, `n_frames = 2`, or
  use `/api/frame`'s `spots`) equals 20 µm to 1e-3 µm; the shadow pair of `single_lift_hold` at
  an x hand-over (`fade_events[0].time_us + τ/2`) appears as **two** spots at `±10.3·Δf` µm each
  with ≈ half the main spot's power and the main spot keeps ≈ unit power (the collapsed-group
  phantom must not appear); struct-of-arrays shape and the `1e-3` cut; `movie_mixing_order`
  note for the 10×10 and for a 3×3 at order 3 (49 groups → stays at 3, no note); `frame` returns
  images of the requested shape, `peak > 0`, packed round-trip exact to `peak/65535`; the exact
  frame's peak equals `2·power/(π·wx·wy)` of the brightest spot within 2 % for a static single
  tweezer at `t = 3τ` (the shared-scale claim); cuts peak at the brightest spot; `coverage`
  error for a late frame.
- `test_aperture.py`: `envelope`, `f_inst_MHz`, `filled`, `t_ret_us` against direct
  `ToneTrack`/`retarded_time` evaluation; `voltage` (when requested) equals `aperture_window`
  on the same points exactly; zero/NaN where unfilled.
- `test_codegen.py`: generated code for each example `compile()`s and, with `.save`/`.movie`
  lines removed, `exec`s in a fresh namespace producing `plan` whose `report.band_usage` matches
  the API response to 1e-6 Hz.
- `test_trajectory.py`: `to_spec` on every move kind; µm/µs → SI exact; `sample_trajectory` ends
  at the requested displacement.

## 4. Lint / type
`ruff check webapp/backend`, `ruff format --check webapp/backend`, `mypy webapp/backend/src/aodl_webapp`
clean (pydantic plugin not required; annotate everything).

## 5. Report
When done, write `webapp/docs/reports/W1.md`: what was built, test counts, timings of each
route on the examples (plan / simulate 80 frames / frame 192px), deviations from this order
and why, and anything the frontend must know that API.md does not say (put API changes into
API.md itself and flag them at the top of the report).

## 6. Makefile targets (frozen names; W3/W4 rely on them)
`install` (pip -e root + backend, `npm ci` in frontend), `dev` (uvicorn :8000 --reload and
vite :5173 concurrently via a tiny shell loop; print both URLs), `dev-backend`, `dev-frontend`,
`test` (= `test-backend test-frontend`), `test-backend` (pytest), `test-frontend`
(`npm test -- --run`), `lint` (ruff + mypy + `npm run lint` + `npm run typecheck`), `build`
(`vite build` → `frontend/dist`), `fixtures` (the generator), `serve` (uvicorn serving the
built frontend on :8000).
