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
