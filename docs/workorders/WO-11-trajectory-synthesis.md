# WO-11 — Trajectory spec + full Eq. S19 waveform synthesis

**Role:** implementation agent, Wave H (parallel with WO-10 — it owns `field/gaussian.py`,
`device/aodl.py`, `field/focal.py`; you must not touch those, and your tests must not
depend on partial-fill behavior). **No git.**
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.4, `docs/conventions.md`,
`src/aodl/trajectory/ramps.py`, `src/aodl/waveform/synthesis.py` (WO-08's array helpers),
`src/aodl/waveform/tones.py`, then this. M2 merged and verified (188 tests at HEAD).

## Owned files

```
src/aodl/trajectory/spec.py
src/aodl/waveform/synthesis.py     (extend; do not break existing helpers/tests)
src/aodl/__init__.py               (export the new public names)
tests/test_spec.py  tests/test_synthesis_s19.py
```

## 1. `trajectory/spec.py`

```python
@dataclass(frozen=True) class ArraySpec:
    mx: int = 1; my: int = 1
    delta_f_x: float = 0.0; delta_f_y: float = 0.0    # tone spacings [Hz] (0 ⇒ single)
    # convenience constructor: ArraySpec.from_pitch(mx, my, pitch_x, pitch_y, params)
    #   → delta_f = pitch / params.deflection_scale

# Moves (each with ramp profile name from trajectory/ramps.py, default "min_jerk"):
@dataclass(frozen=True) class Lift:      dz: float; duration: float; profile: str = "min_jerk"
@dataclass(frozen=True) class Translate: dx: float; dy: float; duration: float; profile: str = "min_jerk"
@dataclass(frozen=True) class Hold:      duration: float
# Lower = Lift with dz < 0 (document; no separate class)

@dataclass(frozen=True) class TrajectorySpec:
    array: ArraySpec
    moves: tuple[Lift | Translate | Hold, ...]
    def compile(self) -> tuple[PiecewisePoly, PiecewisePoly, PiecewisePoly]
        # (X(t), Y(t), Z(t)) of the array center, meters vs seconds, t from 0,
        # segments concatenated; every axis covers the full span (holds where unused)
    @property def duration(self) -> float
```

Ramp application: a Lift changes only Z; Translate only X/Y (simultaneous, same
profile/duration); Hold freezes all three. Use `trajectory/ramps.py` functions and
`PiecewisePoly.concat`; endpoint continuity asserted at compile.

## 2. Eq. S19 synthesis in `waveform/synthesis.py`

```python
def synthesize(spec: TrajectorySpec, params: AODLParams, *, amp: float = 1.0,
               phases: {"schroeder","zero","random"}|arrays = "schroeder",
               t_pad: float = None, rng=None) -> WaveformSet
```

With X, Y, Z from `spec.compile()` and F, λ, v from params (assert all four channel v
equal — use `params.sound_speed`):

- `f_Z(t) = (v²/(2λF²)) · Z.antiderivative()`   (a PiecewisePoly; Eq. S19)
- `f_Ax = −(v/(2λF))·X + f_Z`, single tone; `f_Ay = −(v/(2λF))·Y + f_Z`, single tone
- `f_Bx^(n) = f_x0^(n) + (v/(2λF))·X + f_Z`, n = 0..Mx−1 with
  `f_x0^(n) = (n − (Mx−1)/2)·Δf_x` (detunings; rotating frame) and Schroeder phases
  (reuse `schroeder_phases` / `array_tones` + `add_common_ramp` where they fit); same
  for By. All frequencies are detunings from each channel's `f_center`.
- Sanity identities to assert in tests (they follow from Table I; derive in a comment):
  `deflection_scale·(f_Bx − f_Ax) = X(t)` exactly (polynomial identity — compare
  coefficients, not samples), and `2·lens_scale·ḟ_Z = Z(t)` exactly.
- `t_pad` (default `2·transit_time`): extend all tones with `with_hold_until
  (duration + t_pad)` so simulations can probe past the end; document that the waveform
  time axis starts at 0 and the atom-plane response lags by τ/2 (**no retardation
  pre-compensation in v1** — architect decision; tests compare at t − τ/2).

**Band checking** (Eq. 1): for every channel/tone, min/max of `f_center + f(t)` over the
domain must lie inside `aod.band`; violation → `ValueError` reporting: the offending
channel, the excursion in MHz, the limit, and the max feasible |∫Z dt| from Eq. 1
(`lens_scale·(f_max − f_min)` … derive the exact factor-of-2 bookkeeping for four
channels in a comment and report the number). Include a `check_band=False` escape hatch
(documented as "for plotting infeasible drives only").

## 3. Tests

`test_spec.py`: compile continuity/durations/holds; from_pitch round-trip
(pitch ↔ Δf·deflection_scale); profile names dispatch; frozen dataclass hygiene.

`test_synthesis_s19.py`:
- Polynomial identities above (coefficient-level, 1e-12).
- Static 3×3: synthesize + `simulate` at t = 2τ → 9 groups on the Table-I grid (1% waist)
  — mind the equal-Δf anti-diagonal degeneracy (WO-08 finding): use Δf_x ≠ Δf_y for the
  position assertions.
- **Mini user story** (the M3 core check; keep it fast — 2×2 array, Δf 1.0/1.3 MHz,
  Lift(+5 µm, 60 µs) → Translate(+15 µm, +10 µm, 80 µs) → Lift(−5 µm, 60 µs), t_pad
  default): sample ~12 probe times in [τ, duration]; measured mean (X, Y, Z̄)(t) tracks
  the requested profiles evaluated at t − τ/2 within (1% waist lateral, 2% z_R axial);
  **|ΔF| < 0.02·z_R at every probe** (the astigmatism-free claim); per-trap spread of Z̄
  < 2% z_R. Run with `mixing_order=1` params for speed and determinism; one order-3
  spot-check at a single t.
- Band check: Hold at Z = +10 µm for 2 ms → ValueError naming the limit; the same spec
  with `check_band=False` synthesizes; a feasible fast version passes `check_band=True`.
- Pure-Z spec (Lift only): f_Bx − f_Ax is constant (coefficient check), all four ḟ equal.

Note: at t < τ/2 a counter-propagating pair has zero aperture overlap; WO-10 (parallel)
is making that return dark frames instead of raising. Your tests probe t ≥ τ only, so
they must pass regardless of WO-10's landing state — verify yours pass on current HEAD.

## Definition of done

Full `pytest` green **excluding** any WO-10-owned test file present in the tree
(`pytest --ignore=tests/test_window.py` if it exists); `ruff check` clean on owned files;
no git. Report: files, test summary verbatim, the mini-user-story tracking errors
(worst lateral/axial/ΔF numbers), band-limit message example, deviations (or "none").
