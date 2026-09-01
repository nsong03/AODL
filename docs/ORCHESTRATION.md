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
| 2026-09-02 | Wave M verdict: **M4 ACCEPTED** (WO-16, fix `16a9fd6`). Fade algebra to 1e-16; Shepard bound tight to 0.4 kHz at 7.46× ceiling; F-3 (p_B=0 splatter −40 dB, switch period < transit) prioritized for M5. |
| 2026-09-02 | M5 work orders WO-17…WO-19 authored; PLAN M4 shadow text corrected. Wave N (WO-17) dispatched. |
| 2026-09-02 | Wave N accepted (WO-17, `177d8b2`): 322 tests; api front door; even-M lattice coincidence exact (was 0.5 pitch); f_z_bias doubles the ceiling exactly (hold ratio 2.0098 with 2 µs ramps); switch_ramp with (πρ_r)² interior-dip law. Wave O (WO-18) dispatched. |
| 2026-09-02 | Wave O accepted (WO-18, `4aaa76e`): 329 tests, 6 notebooks, guide.md + README with executable doc-tests; flagship ρ = 0.30 interior ripple 7.6 % measured. Wave P (WO-19) dispatched. |
| 2026-09-02 | Wave P verdict: **M5 ACCEPTED — RELEASE CANDIDATE** (WO-19, fix `737eb07`). Founding story verified both modes at 1e-13/1e-15; pre-M5 bit-compatibility proven against `16a9fd6`; switch_ramp FFT −41 → −103 dB. Open: F-2 switch_ramp scope (ruled: p = 0 only → WO-20), F-4 license (owner), doc drift (WO-20/architect). |
| 2026-09-02 | Wave Q accepted (WO-20, `b4f7e58`): switch_ramp scoped to p = 0 rungs — splatter fix identical (−114 dB either way, A ramps contributed 0.0 dB), interior columns bit-flat; audit's flatness claim refined by parity (even M: every column exact; odd M: the two edge columns pay the (πρ_r)² law, closed form to 5 digits). Guide F-9 clause; CI workflow. **334 tests.** Remaining before tag: F-4 license (owner decision). |

## Backlog (tracked findings, not yet scheduled)

- **M4**: coherent (Gram-matrix) per-group `power` option for degenerate shadow-tweezer
  pairs; fast-fade α₁ tilt-term power correction (~4% at 1 ms ramps → grows for Shepard
  fades); compression-correction envelope shape approximation (WO-09 finding 3 — exact
  below `l1·w_in/v ≲ 0.1`, ~1.2e-3 pupil error in the mid band) — quantify all three in
  the M4 notebook and tighten if fades demand it.
- **Awaiting owner**: LICENSE (WO-19 F-4 — pyproject says MIT, README says TBD, no file;
  note the paper's patent application when choosing).
- **Post-release**: schema v3 slot for `SwitchRamped` envelopes (`save()` currently
  refuses switch_ramp > 0 drives by name — WO-17 deviation 2); traveling-edge modeling
  of envelope steps (the remaining F-3 fidelity item once switch_ramp > 0 is used);
  PlanReport ghost-prediction/astig-metric extensions (PLAN §3 M5 wording deferred —
  WO-19 F-7).
- Done in M3: two-sided aperture window + term-product guard (WO-10); seed-lucky
  Schroeder-vs-random assertion correction (WO-12). Done in M4: coherent Gram power,
  fast-fade + compression-envelope quantifications, notebook-02 assertion (WO-15).
  Done in M5 (WO-17): even-M lattice alignment, retard_compensate, f_z_bias,
  switch_ramp, spot_table power_coherent.
