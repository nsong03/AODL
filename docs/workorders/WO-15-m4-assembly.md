# WO-15 — M4 assembly: Shepard notebook, coherent power, backlog cleanups

**Role:** implementation agent, Wave L. **You are the wave closer**: commit and push.
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §3 (M4), `docs/workorders/WO-14-fading-shepard.md`
(merged before you start) + its report highlights in `docs/ORCHESTRATION.md`, notebooks
01–04 (house style), then this.

## Owned files

```
examples/05_fading_shepard.ipynb
src/aodl/field/measure.py           (edit: coherent power option, §2)
src/aodl/viz/movie.py               (edit only if a small hook is needed)
tests/test_integration_m4.py
tests/test_measure.py               (edit: coherent-power tests only)
examples/02_crossed_pair_diagonal.ipynb  (edit ONLY §3 cleanup)
```

## 1. `examples/05_fading_shepard.ipynb`

House style. Sections:

1. **The problem**: plain S19 refusal for a long Z-hold (reuse the band-check message);
   Eq. 1 recap with the 206 µs number.
2. **The scheme**: spectrogram-style figure of the four channels' tone ladders during a
   1 ms Z = 10 µm hold (frequency vs time, alpha ∝ amplitude — the cascading-tones
   picture of paper Fig. 3b); markdown walk-through of S26/S27 fade windows and the
   interlaced ξ offsets.
3. **It works**: tracking + total-power flatness through many fade cycles; max-|f|
   bound plot (band occupancy vs time against the band edges).
4. **Shadow tweezers**: mid-fade frames (log scale) showing the ±deflection_scale·Δf
   companions; for a 3×3 array, the (Mx+2)×My extended grid during x-fades; markdown on
   the pick-up-scheduling caveat (Supplement: start/end moves in non-fading zones).
5. **Interlaced vs simultaneous** (ξ_y = 0.5 vs 0): with `power_coherent` (§2) and a
   deliberate optical-path phase offset on one channel, show the simultaneous scheme's
   center-intensity sensitivity (static Mach–Zehnder, Fig. S6) vs interlaced immunity.
6. **The user story, unhurried**: the original 10×10 `Lift(+10 µm, 150 µs) →
   Translate(+40, +25, 250 µs) → Lift(−10, 150 µs)` — refused in notebook 04 — now
   synthesized with `shepard="auto"` and simulated; tracking plots; **movie 05**
   (tracked mode, hue ↔ Z, XZ panel; budget ≤ 120 s render). This is the product's
   closing argument — say so in one line, not a sales pitch.
7. **Design-caveat quantifications** (backlog): (a) fast-fade α₁ tilt-term power
   correction vs fade-zone crossing time (sweep Δf·η/ḟ_Z; mark where it crosses 1%);
   (b) the compression-correction envelope approximation (WO-09 finding 3) evaluated at
   a fade edge — plot pupil error vs `l1·w_in/v`, mark the mid-band ~1.2e-3 regime.
   Keep both cells cheap (< 10 s each).

Runtime budget: ≤ 3.5 min total.

## 2. Coherent per-group power (backlog: M4 item)

`measure.py`: add `SpotMetrics.power_coherent` — exact Gram form
`P = Σ_jk c_j c_k* · O_x(j,k) · O_y(j,k)` with per-axis pupil overlaps
`O(j,k) = ∫ p_j(u) p_k*(u) du` in closed form via `gauss_moments` at
`a = a_j + conj(a_k)`, `b = i(θ1_j − θ1_k)` … derive carefully (windowed terms use the
window moments; α polys multiply — truncate the product at u², consistent with the
field path) and validate against the frame integral for a *degenerate destructive* pair
(power_coherent → 0 where the incoherent `power` stays finite — the WO-04-era
known-limitation this closes). Incoherent `power` stays the default everywhere; document
when each is the right question. Tests in `test_measure.py`.

## 3. Notebook 02 cleanup (WO-09 finding 4 / WO-12 deviation 5)

Cell 13's `spread["schroeder"] < 0.5*spread["random"]` → `< spread["random"]`, keeping
the `< 0.25·spread["zero"]` claim; one markdown sentence on why (random-phase spread is
itself random; Schroeder always wins but not always by 2×).

## 4. `tests/test_integration_m4.py` (M4 acceptance, PLAN §3)

- Sustained Z: 10 µm hold, duration ≥ 3× the Eq. 1 ceiling: tracking bounds as M3;
  total power flat to < 1%; plain S19 on the same spec raises.
- Shadow tweezers at ±deflection_scale·Δf during fades, absent on plateaus; array case
  shows the (Mx+2)-column extended grid during an x-fade.
- Interlaced vs simultaneous: simultaneous config's degenerate shadow pair responds to
  a channel phase offset (power_coherent changes by > 10%); interlaced does not (< 0.1%).
- The unhurried user story synthesizes, tracks (M3 bounds), and stays inside the band.
- Runtime guard: the 1 ms single-tweezer hold `simulate` (24 probes, no frames) < 10 s.

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `python -m mypy src/aodl` exit 0
(always `python -m mypy`); `pytest --nbmake examples/` green (all five); nothing binary
staged; outputs cleared; commit (`M4: fading-Shepard notebook, coherent power,
integration`, footer per dispatch) and push. Report: pytest/nbmake summaries verbatim,
notebook runtime, movie size + description, the §1.3 flatness and §1.5 sensitivity
numbers, deviations (or "none").
