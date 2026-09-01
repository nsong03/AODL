# WO-22 — M6 product: expectation, verdict, api hook, notebook 07, docs

**Role:** implementation agent, Wave S. **You are the wave closer**: commit and push.
**Read first:** `CLAUDE.md`, `docs/workorders/WO-21-check-core.md` (merged before you
start — its §interfaces are frozen) + its report in `docs/ORCHESTRATION.md`,
`src/aodl/check/` as built, `src/aodl/api.py`, `docs/guide.md`, notebooks 01–06 house
style, then this.

## Owned files

```
src/aodl/check/expect.py   src/aodl/check/report.py   src/aodl/check/__init__.py
src/aodl/api.py            src/aodl/__init__.py
tests/test_check_expect.py tests/test_check_weak_vs_sim.py tests/test_check_bragg.py
tests/test_check_verdict.py tests/test_check_flagship.py
examples/07_fft_checker.ipynb
CLAUDE.md  README.md  docs/PLAN.md  docs/ARCHITECTURE.md  docs/guide.md
src/aodl/engine.py  src/aodl/field/gaussian.py     (docstring scoping edits ONLY, §5)
```

## 1. `expect.py`

```python
@dataclass(frozen=True)
class Expectation:
    spec: TrajectorySpec
    params: AODLParams
    retard_compensated: bool = False       # NOT discoverable from samples — explicit
    amp: float = 1.0
    shadows: tuple[tuple[float, str, float], ...] = ()   # (time, axis, offset) fade whitelist
    fade_pad: float = 0.0                                # half-width [s] of fade exclusion
    @classmethod def from_table(cls, times, x, y, z, array, params) -> Expectation  # lab path
    def eval_time(self, t, aod) -> float   # clamp(t − τ/2, 0, T); clamp(t,0,T) if compensated
    def traps(self, t) -> ExpectedTraps    # x/y centers from spec.compile()(eval_time) +
                                           # deflection_scale·ArraySpec.detunings(); z; and
                                           # lateral velocities from the compiled polys'
                                           # .derivative() (for the beat-window cap)
    def lattice(self, t, extend=1)         # array grid ± `extend` pitches (blob whitelist:
                                           # commensurate IM3 + Shepard extended columns)
    def in_fade(self, t) -> bool
class SimResultLike(Protocol): ...         # structural: times; metrics[i][g].x/y/z_lab/wx/wy/power/df_opt
def sim_delta(rows, sim: SimResultLike, tol_match) -> dict   # nearest-trap matched diffs, report-only
```

Independence: `expect.py` imports `trajectory.spec`, `params`, `conventions` only; the
`SimResult` comparison is structural (Protocol + `if TYPE_CHECKING` import), mirroring
the repo's `TermLike` precedent — never compute anything via `engine`.

## 2. `report.py` — tolerances, verdict, driver

```python
@dataclass(frozen=True) class Tolerances:
    lateral: float = 0.05       # × waist0
    axial: float = 0.05         # × rayleigh (also gates |delta_f| growth)
    waist: float = 0.02         # relative, non-transient non-fade frames only
    uniformity: float = 0.03    # per-trap relative intensity vs frame pattern, non-fade frames
    blob_off_lattice: float = 0.01   # × median trap peak
    blob_on_lattice: float = 0.10
    missing_trap: float = 0.25
@dataclass(frozen=True) class CheckReport:
    passed: bool; mode: PupilMode; times; table: dict[str, array]   # long per (frame, trap):
        # time, ix, iy, x, y, z_lab, delta_f, sigma_astig, wx, wy, peak, power, beat_std,
        # dx, dy, dz, verdict_frame; plus raw-moment columns
    blobs: tuple[Blob, ...]; failures: tuple[str, ...]; notes: tuple[str, ...]
    out_of_band: dict[str, float]; sim_delta: dict | None; tolerances: Tolerances
    def summary(self) -> str    # PlanReport.summary() style: verdict line, worst residual
                                # per metric vs tolerance, worst offender frame/trap, blobs,
                                # exclusions applied
def check_samples(samples: SampleRecord | Mapping | str | Path, expect: Expectation, *,
                  times=None, mode: PupilMode = "bragg_band", tolerances=None,
                  sim: SimResultLike | None = None, k_subtimes=64, n_z=7,
                  z_half_range=None,             # default 1.0 × rayleigh
                  sample_rate=None, params=None, normalization=None) -> CheckReport
```

Pipeline per frame (using WO-21 pieces): beat window
`W = min(2/δf_min_live, 0.2·waist0/max|v_lat(t)|)` (motion cap — without it fast
translates fail good drives via smear; δf_min from the array spacings/Shepard ladder);
K golden-ratio sub-times; per sub-time 2 axis pupils → zoom fields on per-trap fine
patches (center ± 3–4·waist0, ~waist0/16 pitch) + coarse full-FOV canvas
(waist0/3 pitch) + Z-stack (n_z planes over expected Z ± z_half_range, w²-parabola per
axis); accumulate outer products. Fits → TrapFit rows; blob audit vs
`lattice(t, extend=1)` + shadow whitelist during `|t − shadow.time| ≤ fade_pad`.

**Verdict-bearing** (model-gap-immune): lateral, axial, |ΔF|, waists + uniformity on
non-transient/non-fade frames, missing traps, blob thresholds. **Report-only**:
absolute intensity scale (a uniform per-channel gain is optically invisible after
global normalization — document as the known blind spot), sim_delta, compression
effects, beat_std, splatter, fade/transient metrics. Frames with t < 2τ or with any
axis mid-fill are marked transient (excluded from waist/uniformity gates; positions
still gated except while filling).

## 3. `api.py` hook (+ the options field)

- `MotionPlan` gains `options: dict[str, Any] = field(default_factory=dict)` (keyword-
  constructed dataclass — backward compatible); `plan_motion` fills it with the synth
  kwargs it received (`retard_compensate`, `amp`, `f_z_bias`, `switch_ramp`, resolved
  shepard mode).
- `MotionPlan.check(*, times=None, rate=DEFAULT_SAMPLE_RATE, mode="bragg_band",
  tolerances=None, sim=None, **kw) -> CheckReport`: renders float64 samples with
  `return_scale=True`, builds `SampleRecord` with the true normalization, builds
  `Expectation` from spec/params/options + `report.fade_events` shadows, default times
  = ~9 deterministic frames (2τ, move seams and midpoints shifted +τ/2 or +0 per
  retard mode, T+τ/2), validated against the record end using the TRUE aperture reach
  (**WO-21 correction: `t ≤ t_end + τ/2 − grid.half_span/v`** — the grid spans
  ≈ 5·w_in so a frame gathers drive up to t + 9.6 µs; the earlier `4·w_in/v` bound is
  3.1 µs too loose and trips the coverage error), engine-style message. Render with
  `t_pad` sufficient for the last default frame, or trim the last frame accordingly.
- `aodl/__init__.py` exports `check_samples`, `CheckReport`, `Tolerances`,
  `Expectation` (+ `load_samples`).

## 4. Tests

- `test_check_expect.py`: eval-time clamping both retard modes; trap tables vs
  hand-built Table I positions; lattice/shadow whitelists; `from_table` round trip.
- `test_check_weak_vs_sim.py` (the cross-validation gate): weak-mode checker vs
  `simulate(..., mixing_order=1 params)` on canonical drives — M1 static + chirped Ay,
  the M3 2×2 lift–traverse–lower, one M4 Shepard-hold frame away from fades: fields
  ≤ 1e-4 relative, positions ≤ 1e-3·waist0, powers ≤ 3e-4. **WO-21 correction: the
  interpolation law is cubic (ε ≈ 0.016·θ³ → 5.5e-5 at the 15 MHz band edge), so run
  these tight gates with `demodulate(..., oversample=2)`** (8× better → ≥ 16× headroom
  even at band edge; typical few-MHz detunings have ~200× regardless). One transient
  frame at 0.75τ compared loosely (≤ 1e-2) and marked. The verdict-path default stays
  `oversample=1` (its tolerances are 0.05·waist0-scale — huge margin).
- `test_check_bragg.py`: bragg at drive_strength=0.01 → weak (≤ 3e-5, error ∝ C²); at
  C=0.3 the `(2J₁(C)/C)²` compression measured; K 64→128 and W×2 move no verdict
  metric by more than 10% of its tolerance (time-averaging convergence).
- `test_check_verdict.py`: 3×3 mini-story PASSes via `MotionPlan.check()`; three
  corruptions FAIL with failure strings naming the metric:
  (1) Ax chirp-sign flip (rebuild tone with `freq.scale(-1)`, re-render) → lateral +
  astig FAIL; (2) one Bx ladder tone dropped → missing-trap FAIL; (3) **5% amplitude on
  one Bx tone** (not a whole channel — a uniform channel gain is invisible after global
  normalization) → uniformity FAIL. Tolerances overrides respected; `sim=` diff
  populated.
- `test_check_flagship.py` (the CI gate): the guide-quickstart flagship (Shepard mode)
  `plan.check()` at ~7 frames, `k_subtimes=48` → PASS + residual ceilings spot-checked;
  the hurried 25/30/25 µs S19 variant at tighter tolerances. Combined new-test runtime
  budget ≤ 60 s (cache CZT plans per frame; batch baseband gathers over sub-times;
  band-FFT once per sub-time, defocus after selection).

## 5. Docs (complete edit list — the "No FFTs" convention gets scoped)

- `CLAUDE.md` conventions bullet → "No FFTs in the *simulation* path (`field/`,
  `device/`, `engine`); `field/reference.py` is tests-only quadrature;
  `src/aodl/check/` is the deliberately FFT-based independent checker and must not
  import the simulation's field/device internals."
- `README.md`: ":5" tagline → "no FFT anywhere in the simulation"; the "No FFTs"
  bullet reworded; a new key-modeling bullet for the checker; Status → M6, new test
  count, notebook 07.
- `docs/PLAN.md`: §1.3 heading scoped; new **M6 milestone block** in §3 (goal, the §4
  acceptance tolerances, the three corruption FAILs as acceptance).
- `docs/ARCHITECTURE.md`: principle 3 scoped; tree + §4 deps (`scipy>=1.8`, czt) + §5
  decision entry (default pupil model = full exp(iCV) + band selection; weak mode for
  cross-validation).
- `docs/guide.md`: new §5.5 "Checking a rendered drive" (call pattern, report fields,
  tolerances table, what PASS does/doesn't certify incl. the channel-gain blind spot
  and the equal-Δf anti-diagonal coherence note), FAQ entry, §8 checked-numbers rows.
- `src/aodl/engine.py` and `src/aodl/field/gaussian.py` module docstrings: scope the
  no-FFT claim to "this simulation path".
- Notebook 07 `examples/07_fft_checker.ipynb` (≤ 3 min, thin cells, outputs cleared):
  1 why an independent checker + pipeline sketch; 2 flagship `plan.check()` →
  `summary()` PASS; 3 inside view: record spectrogram + aperture spectrum with ±1/±2
  orders and the band window drawn (the repo's first legitimate FFT figures);
  4 weak-vs-bragg single tone: `(2J₁(C)/C)²` measured from samples; 5 measured-vs-
  requested trajectory overlay with residuals against the tolerance band (t − τ/2
  alignment shown); 6 blob audit during a hand-over (shadows whitelisted on-lattice);
  7 breaking it on purpose: chirp-flip → FAIL report.

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `python -m mypy src/aodl` exit 0;
`pytest --nbmake examples/` green (all seven); pre-M6 behavior untouched (MotionPlan
constructible without `options`; no existing test edited except where a doc-test count
changes); nothing binary staged; outputs cleared; commit (`M6: checker expectation,
verdict, api hook, notebook, docs`, footer per dispatch) and push. Report: pytest/
nbmake summaries verbatim, flagship check runtime + worst residuals, the three
corruption failure strings, notebook 07 runtime, doc-edit checklist confirmation,
deviations (or "none").
