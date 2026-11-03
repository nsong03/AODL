# R2-frontend-b — the fading-Shepard tutorial page

**Read first:** `R2-DESIGN-ADDENDUM.md` §F3, `webapp/docs/API.md` (§2 examples — R2-backend adds
`tutorial_hold_plain` and `tutorial_hold_shepard`; §4 plan; §5.1 simulate; §6.3
`trajectory_efficiency`), `webapp/docs/DESIGN.md` §1 item 3 (retardation), `docs/guide.md` §3.5–3.6
(Eq. 1 budget, when Shepard engages) and §6.7; the existing `pages/Basic.tsx`, `pages/Advanced.tsx`
(how the efficiency panel calls `/api/advanced/trajectory_efficiency`), `components/rf/RfPanel.tsx`,
`components/tweezer/TweezerView.tsx`, `components/tweezer/Playback.tsx`, `state/store.ts`,
`api/client.ts`. Owned files: `pages/Tutorial.tsx`, `components/tutorial/**`, `router.tsx`,
`App.tsx` (nav link "Tutorial"), `components/landing/ModeCards.tsx` (a third card). Do not edit
`api/types.ts` (use the existing types; `trajectory_efficiency` types exist). No git. Build
against the API contract; the backend serves the two example ids once R2-backend lands (poll
`/api/presets`).

## 1. Route `/tutorial/shepard`
Loads both examples from `/api/presets`, plans and simulates both (80 frames) with **local
page state** (two independent request/plan/sim triples; do not reuse the global workbench
store for the two plans, but share one time cursor and one `Playback`), and fetches
`/api/advanced/trajectory_efficiency` for both with the default `EfficiencyModel`.

## 2. Layout (two columns, "without fading-Shepard" | "with fading-Shepard"; stack on narrow screens)
1. **Intro** (one short paragraph + the Eq. 1 budget as numbers from `plan.constants`:
   206 µs at 10 µm, the hold asked for 600 µs, `z_budget.fraction` ≈ 3.2): a sustained Z offset
   costs a co-chirp on all four channels, so a single tone must leave the band; fading-Shepard
   ladders hand the chirp from rung to rung so the *live* tones never leave it.
2. **RF drives** per column (reuse `RfPanel` in a compact mode or a thin wrapper): the plain
   drive's tracks cross the band edge — compute from the IR the first time any live tone leaves
   `band_MHz` at the beam centre and mark it ("leaves the band at t = … µs"); the Shepard
   column shows the rungs fading in and out inside the band, with the hand-over ticks.
3. **Trap power vs time** (uPlot, shared axes): three curves per column — ideal Gaussian trap
   depth from the simulation (the closed-form engine assumes flat efficiency, so the plain
   drive stays bright: say so in the caption), the in-band fraction of live tone amplitude²
   at `t − τ/2` from the IR (hard band mask), and the efficiency-weighted power from
   `trajectory_efficiency` (Bragg/TPM model; label as a model). The plain column collapses
   after the band edge; the Shepard column stays near 1 with the shadow events.
4. **Movies**: a `TweezerView` (XY + XZ) per column sharing the cursor, with the shadow badge.
5. **Numbers table**: tones per channel, band margin (negative for the plain drive), number
   of hand-overs, shadow offset `±10.3·Δf µm`, the `|∫Z dt|` requested vs ceiling.
6. **What to take away** (three bullets): band budget, hand-over transfer keeps the summed
   power constant (`p_A + p_B = 1`), shadows exist only while a pair hands over — do not pick
   up atoms then.
Each number comes from the API responses, never hard-coded.

## 3. Quality
`npm run lint`, `npm run typecheck`, `npm test -- --run`, `npm run build` clean; no console
errors on `/tutorial/shepard`; a `vitest` test for the "leaves the band at" helper against a
synthetic IR. Report → `webapp/docs/reports/R2-frontend-b.md`; final message ≤ 25 lines.
