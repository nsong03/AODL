# Round 2 — design addendum (owner feedback of 2026-11-03)

Four findings from the owner's review of the first release, with the diagnosis the architect
made before dispatch. Each becomes one work order in this folder. DESIGN.md §1 rules still
apply; this file adds to them.

## F1 · The landing diagram has the AODs in the wrong order (R2-frontend-a)

The physical layout of arXiv:2510.11451 Fig. 1(a) is **laser → Ax → Ay → 4f relay → Bx →
By → objective**: the two first deflectors (the A pair, crossed x and y) sit together before
the relay, the two second deflectors (the B pair) together after it, and the 4f system images
the A acoustic plane onto the B plane so that all four apertures are superposed (magnification
−1, which is why the pairs counter-propagate in the common pupil frame). The first release
drew Ax → relay → Bx → Ay → By. Nothing in the physics code depends on the drawing order
(`aodl` overlays the four apertures; `aodl_advanced` treats "first" = A pair, "second" = B
pair, which is right), so this is a drawing and prose fix.

## F2 · Even-M fading-Shepard arrays walk: requested traps go dark once per hand-over (R2-core)

**Measured** (per-site power of the requested lattice sites, summed over coincident spots,
relative to the plateau; `Lift 8 µm / 80 µs, Hold 500 µs`, Δf = 1.0/1.3 MHz, mixing order 1):

| array | as shipped: worst requested-site power | with the fix |
|---|---|---|
| 4×4 | **0.001** | 0.857 |
| 2×2 | **0.001** | 0.928 |
| 1×4 | **0.001** | 0.910 |
| 4×1 | **0.003** | 0.902 |
| 3×3 | 0.917 | 0.917 (unchanged) |
| 4×4 at Δf 1.0/1.0 | **0.002** | 0.935 |

The residual 7–14 % dip of the fixed cases is the fade-speed ripple `ρ` already documented in
`docs/guide.md` §6.4 (the 3×3 shows the same), not a defect.

**Mechanism.** Table II gives the A and B ladders of an axis the same offset `ξ`. Rung `n` of
either ladder sits at fade coordinate `g_n = f_Z + (n + ξ)Δf`. The A window (Eq. S26, `m = 1`)
hands over while `frac((f_Z/Δf) + ξ) ∈ (¼, ¾)` — two A rungs live. The B rectangle (Eq. S27,
`p_B = 0`, half-width `(M + η)Δf/2`) holds `M + 1` rungs when the rung nearest the window
centre lies within `¼Δf` of it and `M` rungs otherwise. For **odd** `M`, `(M + η)/2` and
`(1 + η)/2` share their fractional part, so B's `M + 1`-rung phase coincides with A's
hand-over: both live A rungs see the same `M + 1` B rungs, the `M` interior sites get
`cos²θ + sin²θ = 1`, and the two extra columns are the hard-switched edge columns of
`docs/guide.md` §6.7. For **even** `M` the two fractional parts differ by ½, so B holds `M`
rungs *during* A's hand-over: the sites fed by the old A rung are `S = {b − a}` and those fed
by the new rung are `S + 1` — the lit block shifts by one pitch over the fade, the leftmost
requested column fades to zero (`cos²θ → 0`) and only comes back when B's next rung switches
on a quarter-cycle later. Hence "(M+1) columns at every instant" in the old docs: it was the
block oscillating between `S` and `S + 1`, not a flat array.

**Fix.** On an even-count axis offset the B ladder by half a rung relative to A:
`ξ_B = ξ_A + ½` (`table_ii`). Then B's `M + 1`-rung phase coincides with A's fade, the `M`
requested sites are fed by both A rungs throughout, and the trap lattice
`(b + ξ_B) − (a + ξ_A) = b − a + ½` lands on Eq. S19's half-integer even-M lattice by itself —
so `lattice_comb_offset` (the WO-17 `Δf/2` frequency-only correction) must become `0`: the
half pitch now comes from `ξ_B`, which moves the *window timing* as well as the frequency,
which is exactly the difference between the two. Interlacing between axes is untouched
(`ξ_Ax = 0`, `ξ_Ay = ½`; the A windows set the hand-over times). Odd `M` and the single-tweezer
row are unchanged. Consequences to carry into docs and tests: even `M` now behaves like odd `M`
— `M + 2` columns during a hand-over (two hard-switched edge columns), `M` otherwise; the
`switch_ramp` parity claim ("even M pays nothing") inverts; the api `_notes` parity sentence,
`docs/guide.md` §6.6/§6.7/FAQ, `docs/PLAN.md` M4 line and the `SwitchRamped` docstring all change.
The paper's own 4×4 demonstration (±9 % intensity fluctuation, all 16 traps tracked) is only
consistent with the fixed timing.

## F3 · A tutorial comparing a held tweezer with and without fading-Shepard tones (R2-backend, R2-frontend-b)

Same trajectory twice — `Lift +10 µm / 60 µs, Hold 600 µs, Lift −10 µm / 60 µs`, single
tweezer: once as a plain Eq. S19 drive with `shepard = "never"` and `check_band = false`
(the co-chirp walks every channel out of the 90–110 MHz band after Eq. 1's 206 µs at 10 µm),
once with the fading ladders. The closed-form engine assumes flat diffraction efficiency, so
it keeps the out-of-band trap bright; the tutorial must therefore show, next to the ideal
trap depth, the efficiency-weighted trap power from `/api/advanced/trajectory_efficiency`
(Bragg/TPM model, labelled as a model) and a hard in-band fraction computed from the IR,
and say in one sentence why the two engines disagree there.

## F4 · Plots of the actual tweezer intensity (R2-backend, R2-frontend-a)

The first release plotted integrated *power* per spot and showed exact frames only as
images. Add: (a) trap depth vs time per trap — `Σ 2P/(π w_x w_y)` over the spots coincident
at a site (the astigmatic-Gaussian peak; exact for an unwindowed single-term trap), normalised
to the plateau; (b) closed-form line cuts `I(x)`, `I(y)`, `I(z)` through the brightest trap at
the cursor time from `/api/frame` (new `i_y`), overlaid with the Gaussian approximation and
labelled "closed form (Eq. S11)" vs "Gaussian approx"; (c) the same cuts in the exact-frame
view. The exact peak and its ratio to the Gaussian peak are shown as numbers.

## Ownership (disjoint; no git; architect commits)

| order | model | owns |
|---|---|---|
| R2-core | Opus | `src/aodl/waveform/shepard.py`, `src/aodl/api.py` (`_notes` text only), `docs/guide.md`, `docs/PLAN.md`, `tests/test_shepard.py`, `tests/test_synthesis_options.py`, `tests/test_integration_m4.py`, `tests/test_docs.py`, `webapp/frontend/tests/fixtures/*.json` (regenerated), `webapp/backend/tests/test_api_plan.py` note wording if pinned |
| R2-backend | Opus | `webapp/backend/src/aodl_webapp/{examples,simulation,schemas}.py`, `routes/frame.py`, `webapp/backend/tests/test_api_simulate.py`, `test_api_plan.py` (examples list), `webapp/docs/API.md` |
| R2-frontend-a | Opus | `webapp/frontend/src/physics/**`, `components/tweezer/**`, `components/landing/SetupDiagram.tsx`, `pages/Landing.tsx` (prose only), `pages/Basic.tsx`, `components/intensity/**` (new), `api/types.ts` (add `i_y`), `frontend/tests/**` (not fixtures) |
| R2-frontend-b | Opus | `webapp/frontend/src/pages/Tutorial.tsx` (new), `components/tutorial/**` (new), `router.tsx`, `App.tsx` (nav link), `components/landing/ModeCards.tsx` (tutorial card) |
| R2-verify | Opus | `webapp/README.md`, `webapp/docs/VERIFICATION.md`, `webapp/docs/screenshots/`, `webapp/scripts/e2e.mjs`, `.github/workflows/webapp.yml`; may fix any file, listing every change |
