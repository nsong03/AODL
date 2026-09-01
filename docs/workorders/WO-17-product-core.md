# WO-17 — Product core: api front door + synthesis options (M5)

**Role:** implementation agent, Wave N (solo). **You are the wave closer**: commit and
push. **Read first:** `CLAUDE.md`, `docs/PLAN.md` §3 (M5) + §1.5, `docs/ORCHESTRATION.md`
(backlog + Wave M log), `docs/workorders/WO-16-verify-m4.md` findings F-2/F-3,
`src/aodl/waveform/{synthesis,shepard}.py`, `src/aodl/{engine,api}.py` (api.py may not
exist yet), then this. M4 is merged and verified (291 tests at HEAD).

## Owned files

```
src/aodl/api.py                    (new)
src/aodl/waveform/synthesis.py     (edit: options §2)
src/aodl/waveform/shepard.py       (edit: lattice alignment §2.1, switch_ramp §2.4)
src/aodl/engine.py                 (edit: spot_table power_coherent column only)
src/aodl/__init__.py               (edit: front-door exports)
tests/test_api.py  tests/test_synthesis_options.py
```

## 1. `api.py` — the one-call front door

Thin composition of existing pieces (no new physics):

```python
@dataclass(frozen=True) class PlanReport:
    mode: str                      # "s19" | "shepard"
    band_usage: dict[str, tuple[float, float, float]]   # per channel: (min, max, margin) Hz, live tones
    tone_counts: dict[str, int]
    z_budget: tuple[float, float]  # (|∫Z dt| requested, ceiling) for s19 mode; (…, inf) shepard
    fade_events: list[...]         # shepard: (t, axis) handover schedule + shadow offsets
    notes: tuple[str, ...]         # e.g. extended-grid parity note, splatter caveat if switch_ramp=0
    def summary(self) -> str       # short human-readable block
    def figure(self) -> Figure     # band-usage + tone-track overview panel

@dataclass class MotionPlan:
    spec: TrajectorySpec; params: AODLParams; wfs: WaveformSet; report: PlanReport
    def save(self, path)                              # parametric NPZ (wfs.save)
    def render_samples(self, rate=625*MHz, **kw)
    def simulate(self, times=None, **kw) -> SimResult # times=None → sensible default grid
                                                      #   (~40 frames over [τ, duration+τ/2])
    def movie(self, path, times=None, **kw) -> path   # simulate + render_movie

def plan_motion(spec: TrajectorySpec, params: AODLParams = None, *,
                shepard="auto", **synth_opts) -> MotionPlan
```

`plan_motion` is THE product entry point; docstring carries the 8-line quickstart
(array + moves → plan → save/movie). Export from `aodl.__init__`.

## 2. Synthesis options (each independent, each defaulting to current behavior)

### 2.1 Lattice alignment for even M (WO-15/WO-16 finding F-2 — architect ruling)
Shepard trap columns sit at integer multiples of Δf (index differences), while Eq. S19
puts an even-M array at half-integer multiples — `shepard="auto"` currently moves the
user's traps by half a pitch. Fix by **decoupling rung frequency from fade coordinate**:
add a per-channel constant `comb_offset` to the B rung *frequencies* only
(offset = Δf/2 when M is even, 0 when odd), leaving g, the fade windows, the schedule,
and the A channels untouched. Rationale (put in the docstring): nothing ties the fade
coordinate to the frequency offset — the schedule only needs to hand over as f_Z
advances; co-location and optical-frequency degeneracy structure are preserved because
every B rung shifts equally. Consequences to assert: Shepard trap lattice now coincides
with the S19 lattice for even AND odd M (same spec, both modes, positions equal to
1% waist); Shepard band bound gains |comb_offset| (update `shepard_band_bound`); power
flatness and shadow offsets unchanged (regression vs current values).

### 2.2 Retardation pre-compensation
`synthesize(..., retard_compensate=False)`: when True, evaluate the compiled profiles at
`min(t + τ/2, duration)` so the *atom-plane* motion matches the request at t (not
t − τ/2). Tests: measured trajectory matches request at t to M3 tolerances; default
False reproduces current behavior bit-for-bit.

### 2.3 f_Z edge pre-bias (plain-S19 mode only)
`synthesize(..., f_z_bias=0.0 | "auto")`: constant added to all four channels' start
frequency. "auto" = −½ · (v²/2λF²)·∫Z dt clamped to the per-channel headroom, so the
f_Z excursion is centered in the band — doubling the usable Eq. 1 budget (PLAN §1.5's
412 µs figure). Band check accounts for the bias; lattice positions unaffected (common
to all channels — assert). Error message when even the biased budget fails should quote
the doubled ceiling.

### 2.4 `switch_ramp` for the p_B = 0 rectangles (WO-16 F-3 mitigation)
`ShepardConfig(switch_ramp: float = 0.0)` [seconds]: when > 0, multiply each p = 0
rung's rectangle by smooth raised-cosine on/off ramps of that duration in *time*,
anchored at the rung's precomputed entry/exit instants (compose with the existing
envelope machinery; entry/exit times come from the g-poly crossings already computed).
Default 0 = Table II faithful (keep the −40 dB splatter caveat in `PlanReport.notes`
when 0 and arrays are present). Tests: interior-column flatness unchanged; the extended
column's brightness now ramps continuously over `switch_ramp`; total in-band claim left
to WO-19's FFT (your test just checks envelope continuity and that τ-scale ramps don't
break the power identity by more than the documented ρ-law).

### 2.5 `spot_table` gains a `power_coherent` column (engine.py)

## 3. Tests

`test_synthesis_options.py`: each §2 item as specced (defaults bit-compatible with
current behavior — snapshot a WaveformSet before/after where feasible); lattice
coincidence for M ∈ {2,3,4,5}; pre-bias bisect: max feasible hold duration doubles
(±2%) vs unbiased.
`test_api.py`: plan_motion on (a) a static 3×3, (b) the fast M3 story, (c) the unhurried
M4 story — mode chosen correctly, report numbers match the underlying wfs (band usage vs
a direct sweep of the tone laws), save/simulate/movie smoke (movie ≤ 12 frames, tiny
grid), summary() contains the mode and the worst band margin.

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `python -m mypy src/aodl` exit 0
(always `python -m mypy`); commit (`M5: api front door and synthesis options`, footer
per dispatch) and push. Report: pytest summary verbatim, the lattice-coincidence and
pre-bias-doubling numbers, switch_ramp behavior summary, deviations (or "none").
