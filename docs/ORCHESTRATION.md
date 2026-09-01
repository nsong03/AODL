# Orchestration model

How this codebase gets built (per Nathan's direction, 2026-08-31):

- **Architect/orchestrator (Fable 5, max effort)** — owns the physics decisions, the frozen
  interfaces, and the work orders in `docs/workorders/`. Sequences waves, integrates
  reports, reports to Nathan. Does not implement, and does not verify code itself.
- **Implementation agents (Opus)** — one per work order. Each receives a self-contained
  spec: owned files, frozen interfaces, exact formulas, acceptance tests, self-verification
  steps. They implement, test their own scope, and report.
- **Verification agent (Opus, fresh eyes)** — separate from the implementers; re-runs the
  full suite, adversarially reviews signs/formulas against the paper excerpts embedded in
  its work order, executes notebooks, fixes small defects, reports larger ones.

## Rules of engagement (all agents)

1. The work order is the task authority; `CLAUDE.md` carries repo conventions.
2. Touch only the files your work order owns. Do not edit `src/aodl/__init__.py`
   (public export wiring belongs to assembly work orders).
3. Only the wave closer runs git. Implementation agents leave their changes in the tree.
4. Self-verify before reporting: your own tests pass, `ruff check` clean on owned files.
5. Report: what was built, test summary, any deviation from the work order (with reason),
   any interface friction encountered (the architect uses this to fix specs, not you).

## Wave plan — M0 + M1

```
Wave A   WO-01  core scaffold + poly + Gaussian integral kernel + reference integrator
                (solo; closer: commits + pushes)
             │
Wave B   ┌───┴──────────────┬───────────────────┐        (parallel, disjoint files,
         WO-02 waveform     WO-03 device        WO-04 field       no git)
         tones/ramps/       conventions/aod/    focal/measure
         serialize/export   single-channel aodl
             └──────────────┴───────────────────┘
Wave C   WO-05  M1 assembly: engine + viz/movie + notebook 01 + integration
                (closer: commits + pushes)
Wave D   WO-06  verification pass (fresh eyes; may commit small fixes + push)
```

M2–M4 repeat the same pattern; their work orders are authored by the architect after the
M1 verification report lands.

## Status log

| Date | Event |
|------|-------|
| 2026-08-31 | Work orders WO-01…WO-06 authored; Wave A dispatched. |
| 2026-08-31 | Wave A (WO-01) accepted: 20 tests, S11 mapping vs quadrature ~1e-15; commit `1398fc8`. Wave B (WO-02/03/04) dispatched in parallel. |
| 2026-09-01 | Wave B accepted: WO-02 (46 tests, `ad1d09f`), WO-03 + WO-04 (40 tests, `6d34333`). Architect rulings: TermArray.alpha is envelope-normalized (α0=1, envelope in `c` via line amps); FrameGrid stays in `field/focal.py` (engine re-exports); per-group `power` stays incoherent until M4. Wave C (WO-05) dispatched. |
| 2026-09-01 | Wave C accepted (WO-05, `0392e1b`): 122 tests, zero cross-module reconciliations; M1 notebook + movie. Wave D verdict: **M1 ACCEPTED** (WO-06, fix `b2a277d`: fill-transient patch cropping). |
| 2026-09-01 | M2 work orders WO-07/08/09 authored. Architect rulings: IM3 line table fixed in WO-07 §2 (Bessel-consistent); GROUP_TOL 10 kHz → 1 kHz + cluster-diameter cap; TermLike members become read-only properties + mypy gate (WO-08). Wave E (WO-07) dispatched. |

| 2026-09-01 | Wave E (WO-07, `5905bd6`): IM3 table confirmed by independent derivation + frozen-time projection; 147 tests. Ruling folded into WO-08 §0b (`ca93279`): default mixing_order → 3; serialize carries it. |
| 2026-09-01 | Wave F accepted (WO-08, `ef09cb0`): 188 tests, mypy gate, notebook 02, Schroeder ghost suppression 57–437×. Wave G verdict: **M2 ACCEPTED** (WO-09, fix `fd1778c`). |
| 2026-09-01 | M3 work orders WO-10…WO-13 authored; ARCHITECTURE.md drift fixed. Wave H (WO-10 ∥ WO-11) dispatched. |
| 2026-09-01 | Wave H accepted: WO-11 (`32d0e4c`, exact-to-float64 S19 round trip; PLAN Eq. 1 headroom corrected to one-sided ≈206 µs), WO-10 (`b1c0b63`, τ/2-darkness + erf² fill + 646k-term guard case). Ruling: measure.py two-sided power fix assigned to WO-12 §0. |
| 2026-09-01 | Wave I accepted (WO-12, `f0ac4cf`): 255 tests, 4 notebooks, user story at 25/30/25 µs (original durations band-infeasible — kept as the notebook's Eq. 1 teaser; M4 reprises them with fading-Shepard), 3D-AODL vs 2-AOD contrast 0.00 vs 1.90 z_R. Wave J (WO-13) dispatched. |
| 2026-09-01 | Wave J verdict: **M3 ACCEPTED** (WO-13; fixes `5d30409` XZ-panel row snapping, `f8d4d71` band-message headroom wording). S19 inverted from waveforms at 1e-15; 2-AOD control fails the astig bound by 317×; Eq. 1 boundary bisected to 1 Hz. |
| 2026-09-01 | M4 work orders WO-14…WO-16 authored (fading-Shepard); ARCHITECTURE test-tree drift fixed. Wave K (WO-14) dispatched. |
| 2026-09-01 | Wave K accepted (WO-14, `5aad94f`): 280 tests; 10 µm × 1 ms hold (5.15× Eq. 1) at 0.43 % flatness, Shepard band bound to 34 Hz. Rulings ratified: shadow ratio 1/2 (implementer's derivation over the WO), slope clamp replaces the ineffective A-clamp; WO-15 amended (`3841c28`). |
| 2026-09-01 | Wave L accepted (WO-15, `21b5969`): 291 tests, 5 notebooks, movie 05 = the unhurried user story; coherent Gram power validated (destructive pair → 0, frame black); interlaced 0 % vs simultaneous 100 % phase sensitivity. Findings logged: even-M extended grid is (M+1)-wide always; even-M Shepard lattice offset pitch/2 vs S19 (→ M5 backlog); interlacing exact only for Δf_x = Δf_y. Wave M (WO-16) dispatched. |

## Backlog (tracked findings, not yet scheduled)

- **M4**: coherent (Gram-matrix) per-group `power` option for degenerate shadow-tweezer
  pairs; fast-fade α₁ tilt-term power correction (~4% at 1 ms ramps → grows for Shepard
  fades); compression-correction envelope shape approximation (WO-09 finding 3 — exact
  below `l1·w_in/v ≲ 0.1`, ~1.2e-3 pupil error in the mid band) — quantify all three in
  the M4 notebook and tighten if fades demand it.
- **M5**: even-M Shepard lattice sits half a pitch off the S19 layout (WO-15 finding —
  compensate in synthesis so shepard="auto" preserves trap positions); optional
  retardation pre-compensation (τ/2) and f_Z edge pre-bias (doubles Eq. 1 budget);
  `power_coherent` in `spot_table`; envelope steps (p_B = 0 rung switch-on) are
  instantaneous in the model rather than traveling across the aperture — document or
  model in a fidelity pass.
- Done in M3: two-sided aperture window + term-product guard (WO-10); seed-lucky
  Schroeder-vs-random assertion correction (WO-12). Done in M4: coherent Gram power,
  fast-fade + compression-envelope quantifications, notebook-02 assertion (WO-15).
