# WO-18 — Product tour notebook, user guide, README (M5)

**Role:** implementation agent, Wave O. **You are the wave closer**: commit and push.
**Read first:** `CLAUDE.md`, `docs/PLAN.md`, `docs/ORCHESTRATION.md` (all wave logs —
they carry the measured numbers you will quote), `docs/workorders/WO-17-product-core.md`
(merged: `plan_motion` exists), notebooks 01–05, `docs/waveform_format.md`, WO-16's
findings F-3/F-7/F-8, then this.

## Owned files

```
examples/06_product_tour.ipynb
docs/guide.md                       (new: the user guide)
README.md                           (rewrite)
src/aodl/waveform/shepard.py        (edit: docstring fixes ONLY, §3)
tests/test_docs.py                  (new, small: §4)
```

## 1. `examples/06_product_tour.ipynb` — the deliverable demo (≤ 90 s runtime)

The notebook a lab reads first. Narrative: *"I want a 10×10 array to lift 10 µm,
traverse, and drop — give me the RF waveforms."*

1. The ask, in ≤ 10 lines: `ArraySpec` + moves → `plan_motion` → `report.summary()`.
2. What you got: `plan.save("move.npz")` (show size + that it's parameters, not
   samples); `plan.render_samples(...)` for the AWG (show shape/rate); the report
   figure (band usage + tone tracks).
3. Quick look: `plan.simulate()` metrics plots (tracking + ΔF); a *small* movie
   (~30 frames, coarse grid, ≤ 30 s render) — refer to notebooks 04/05 for the full
   renders.
4. Where the physics lives: one markdown map of notebooks 01–05 (what each verifies).
5. Fidelity & limits, honestly: the model's validity conditions with the measured
   numbers (weak-drive IM3 order; ρ fade-speed law, 1% at ρ ≈ 0.057; α₁ tilt-term;
   p_B = 0 splatter −40 dB and the `switch_ramp` option; even/odd-M extended grid;
   pick-up scheduling). Keep it one screen.

## 2. `docs/guide.md` + `README.md`

`guide.md` (the lab-facing manual, ~250–400 lines): install; five-minute quickstart
(same code as notebook 06 §1); concepts (channels, Table I mapping, parametric
waveforms, rotating frame, retardation and `retard_compensate`, Eq. 1 budget +
`f_z_bias`, fading-Shepard + when "auto" engages); parameter reference (AODLParams
fields, presets, how to model *your* hardware); outputs (NPZ schema pointer,
`render_samples`, movies, `SpotMetrics` fields incl. `power` vs `power_coherent`);
fidelity & limitations (same content as notebook 06 §5, with the measured numbers and
citations to the paper's equation numbers); FAQ (why is my spot dark before τ/2; why
did synthesis refuse; why 11 columns during fades; equal vs unequal Δf interlacing).
Every number quoted must be reproducible from the current code — WO-19 will check.

`README.md` rewrite: what it is (2 sentences), the 8-line quickstart, a bullet map of
docs/notebooks, physics reference + status line (M1–M5 verified; tests count), install,
license placeholder line "License: TBD" (the architect/owner decides — do not choose
one). Keep the existing modeling-choices bullets, updated. Do not change `version`.

## 3. Docstring fixes (WO-16 F-7, F-8)

- `FadeZoneEnvelope`: state the residual-weight estimate's general form
  `S²(1+S²)^{-p}·(p(π/2)ρ)²/4`, note it is ~30% optimistic at the design point
  (measured 8.6e-4 vs 6.45e-4 at ρ = 0.0373) and say "estimate", not "bound".
- `auto_config`: document that it can pick Δf_x ≠ Δf_y (whenever lateral spans differ),
  that interlacing is then approximate (schedules beat; some hand-overs light 16 rays),
  and how to force equal spacings via an explicit `ShepardConfig`.

## 4. `tests/test_docs.py` (keep tiny)

- README + guide quickstart code blocks execute (extract fenced python, run in a
  namespace, assert a MotionPlan results — keeps docs from rotting).
- guide.md and README contain no stale numbers for the four canonical constants
  (10.3 µm/MHz, 206 µs, 1.0655 µm, 11.54 µs) — grep-and-compare against `params`.

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `python -m mypy src/aodl` exit 0;
`pytest --nbmake examples/` green (all six); outputs cleared; nothing binary staged;
commit (`M5: product tour notebook, user guide, README`, footer per dispatch) and push.
Report: nbmake summary verbatim, notebook 06 runtime, guide.md section list, deviations
(or "none").
