# R2-frontend-b — the fading-Shepard tutorial page (report)

Route **`/tutorial/shepard`** (`/tutorial` redirects there), nav link "Tutorial", third landing
mode card. Built against the live backend: `/api/presets` already served
`tutorial_hold_plain` and `tutorial_hold_shepard` when I started, so no local stand-in was needed.

## Files

| file | what |
|---|---|
| `src/pages/Tutorial.tsx` (new) | page: intro + Eq. 1 budget, shared sticky time bar, paired panels (RF, trap power, movies), numbers table, take-aways |
| `src/components/tutorial/useTutorialData.ts` (new) | loads presets (polls every 4 s until both ids exist), then per side: `/api/plan` → `/api/simulate` (80 frames) ‖ `/api/advanced/trajectory_efficiency` (default `EfficiencyModel`, n = 300); local state, aborts on unmount |
| `src/components/tutorial/cursor.ts` (new) | page-local zustand cursor (t, playing, speed) shared by both columns |
| `src/components/tutorial/tutorialMath.ts` (new) | `firstBandExit` (first live tone outside `band_MHz`, sampled + bisected, drive time and `+τ/2` atom time), `inBandFraction` (hard-mask Σ_in A² / Σ A² per channel, product over the four), `trapDepthSeries` (Σ 2P/(π w_x w_y) of spots within w0 of the requested site at t − τ/2, normalised to the median lit frame), hold/lift/ladder-spacing/hand-over helpers, grid helpers |
| `src/components/tutorial/trapSeries.ts`, `TrapPowerChart.tsx` (new) | three curves on a union time grid (no interpolation; `null` gaps): ideal trap depth, in-band fraction at t − τ/2, η-product normalised to its own max (raw max/min printed); amber hand-over lines and red band-exit line at atom time |
| `src/components/tutorial/TutorialRf.tsx` (new) | compact per-channel RF tracks on a shared frequency axis (both drives' `band_usage` ∪ band), out-of-band shading, band-exit marker + label, hand-over ticks, beam-centre tick with live tones, cursor |
| `src/components/tutorial/TutorialTweezer.tsx` (new) | XY (tracked focus) + XZ per column with the shadow badge; one brightness scale for both columns |
| `src/components/tutorial/TutorialPlayback.tsx` (new) | the single shared time bar (seams, Shepard hand-overs + τ/2, plain band exit + τ/2, scrub, play, speed) |
| `src/components/tutorial/derive.ts`, `tutorial.css` (new) | per-column derived numbers; page CSS (two columns → one below 1000 px) |
| `src/router.tsx`, `src/App.tsx`, `src/components/landing/ModeCards.tsx` | route, nav link, card |
| `tests/tutorialMath.test.ts` (new) | vitest for the "leaves the band at" helper on synthetic IRs (crossing to bisection accuracy, earliest channel / lower edge, gated tone ignored, in-band → null, start-outside → t0), the in-band fraction and the grid helpers |

## Wrappers instead of edits (shared components are store-bound)

`RfPanel`, `TweezerView` and `Playback` read the single plan / sim / cursor of the workbench
store, and the work order forbids reusing that store for two plans. I did not edit them
(R2-frontend-a owns `components/tweezer/**`); instead:
* `TutorialRf` reuses `useRfTracks` and RfPanel's drawing scheme (alpha-binned tracks, beam-centre tick);
* `TutorialTweezer` reuses `XyView`, `XzView`, `viewScale`, `spotsAt`, `offCanvas`, `dedupeFades`;
* `TutorialPlayback` mirrors `Playback` on the page-local cursor;
* `TrapPowerChart` reuses `UPlotChart` + `uplotTheme`.
A `cursor`/`plan`/`sim` prop on those three shared components would remove the duplication later.

**Ownership note:** the work order requires a vitest test, but vitest only collects
`tests/**/*.test.ts` and `frontend/tests/**` belongs to R2-frontend-a; I added one *new* file,
`tests/tutorialMath.test.ts`, and touched nothing else there.

## What the page shows (live values, default hardware)

Intro: Eq. 1 = 206 µs at 10 µm, hold 600 µs, `z_budget.fraction` 3.20, |∫Z dt| 6600 µm·µs.
Plain: s19, 1 tone/channel, margin −22.04 MHz (100–132.04 MHz), leaves the band at drive
236.0 µs → t = 241.8 µs at the beam centre; ideal trap depth stays ≥ 0.954 (flat-η engine, said
so in the caption and in a "why the engines disagree" note), in-band fraction → 0, η-weighted
→ ≤ 0.010 of its max. Shepard: 4 rungs/channel, Δf 12 MHz, margin +1.00 MHz, 6 hand-overs
(x 3, y 3; one y at t = 0), shadows ±10.3 × 12 = ±123.6 µm (matches `fade_events`), brightest
shadow P/P_ref 0.49, ideal depth ≥ 0.94, η-product min 0.316 (raw). The `assumptions` string of
`trajectory_efficiency` is shown verbatim in a disclosure. No number is hard-coded in prose
(the only literal is `n_frames = 80` / `n = 300`, which are request parameters).

Note for reviewers: the Shepard η-weighted curve dips to ~0.63–0.75 of its max at each
hand-over (live rungs far from f_tpm), so "stays near 1" in the work order is approximate;
the caption says "dips … but never collapses" rather than claiming ≈ 1.

## Checks

* `npm run lint` clean (0 warnings) · `npm run typecheck` clean · `npm test -- --run` 10 files / 83 tests pass · `npm run build` ok.
* Playwright (Chromium) on `/tutorial/shepard` at 1500 px and 420 px: no console errors or warnings,
  no horizontal overflow at 420 px, landing card navigates, play advances the shared cursor,
  `/tutorial` redirects.

## Not verified

* Visual check after R2-frontend-a's concurrent changes to `XyView`/`XzView`/`physics/**` land
  (typecheck passed against the tree as of this run).
* Light colour scheme (canvases are dark like the rest of the app).
