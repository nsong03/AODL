# R2-frontend-a — movie fidelity, the corrected setup diagram, the intensity panel

**Read first:** `R2-DESIGN-ADDENDUM.md` §F1, §F2 (what the movie must show once the core fix
lands: a flat even-M array with two hard-switched extra columns per hand-over), §F4;
`webapp/docs/DESIGN.md` §1 items 2–4 and §4; `webapp/docs/API.md` §5.1–5.2 (R2-backend adds
`i_y`, `gauss_peak`, `exact_peak` to `/api/frame` — build against that contract);
`webapp/frontend/src/physics/{spots,gaussianBeam,zColor}.ts`, `components/tweezer/*`,
`components/landing/SetupDiagram.tsx`, `pages/{Landing,Basic}.tsx`, `state/store.ts`,
`api/types.ts`, and `src/aodl/viz/style.py` (`composite`) for the reference compositing rule.
Owned files: addendum table. No git. Run the backend locally to test (`make -C webapp
dev-backend`); the `array_lift_hold` example will only look right after R2-core lands —
until then test the compositing rules on `array_arbitrary_path` (3×3) and `single_lift_hold`.

## 1. Movie: spots must not slide between frames (`physics/spots.ts`)
`spotsAt` interpolates by array index when the counts match. On a fading drive the backend
orders spots by frequency group and the ladder slides, so index `k` is a different trap in
consecutive frames and traps appear to glide across the lattice. Replace with position
matching: for bracketing frames `a`, `b`, pair spot `i` of `a` with the nearest unpaired spot
of `b` within `0.5·w0` (greedy, brightest first); paired spots interpolate position, `z`,
`ΔF` and power; unpaired spots of `a` keep their position with power `(1 − w)·P`, unpaired
spots of `b` with `w·P` (so switched-on columns fade in over one frame interval instead of
jumping, and nothing ever moves that the physics did not move). `w0` is in
`sim.constants.waist0_um`. Unit tests: two frames with the same spots in different order
produce no motion; a spot present only in frame `b` ramps from 0 to `P`; a spot that really
moved by `3·w0` is not paired and cross-fades.

## 2. Movie: coincident spots must add in intensity before the gamma (`physics/gaussianBeam.ts`)
`renderXY`/`renderXZ` apply `pow(I/peak, γ)` per spot and add — two coincident half-power
groups (every Shepard hand-over) render 23 % brighter than one full trap. Accumulate per
pixel the linear intensity `ΣI` and the intensity-weighted colour `ΣI·rgb` over all spots,
then write `(ΣI·rgb / ΣI) · (ΣI/peak)^γ` — the rule of `aodl.viz.style.composite` applied to
the merged intensity, with the hue the power-weighted mean Z. `peakNormalization` must use
the same merged intensity. Unit test: two coincident spots of power `P/2` render identically
(to 1e-6) to one spot of power `P`; distinct spots unchanged.

## 3. Landing diagram (`components/landing/SetupDiagram.tsx`, `pages/Landing.tsx` prose)
Order left → right: laser → **Ax** (sound −x) → **Ay** (sound −y) → 4f relay (two lenses,
"images the A plane onto the B plane, magnification −1") → **Bx** (sound +x) → **By**
(sound +y) → objective → focal volume. Group the A pair and the B pair visually ("first 2D
deflector", "second 2D deflector") and keep the transducer faces and sound arrows as they are
(`conventions.CHANNEL_GEOMETRY`). Fix the `aria-label`, the legend, the "How to read the
diagram" paragraph and any caption that says the relay sits between Ax and Bx. Keep the
channel colours.

## 4. Intensity panel (new `components/intensity/IntensityPanel.tsx`, mounted in `pages/Basic.tsx` under the tweezer view)
Two parts, both with the cursor:
- **Trap depth vs time** (uPlot): per site-merged trap (merge spots within `0.5·w0` per
  frame, then track sites across frames by position within `0.5·w0`), the astigmatic-Gaussian
  peak `Σ 2P/(π w_x w_y)` normalised to the plateau (median over frames of the brightest site),
  one line per trap (cap 25 lines; show min/median/max envelopes beyond), labelled "Gaussian
  approximation of the closed-form metrics: exact for an unwindowed single-term trap". Shadows
  and extended columns appear as their own lines when above 5 %.
- **Line cuts at the cursor**: `I(x)`, `I(y)`, `I(z)` through the brightest trap from
  `/api/frame` (debounced 250 ms after the cursor stops; immediately on pause; cached by
  frame time), drawn with the Gaussian approximation dashed on top, in units of the plateau
  peak; badges for `exact_peak`, `gauss_peak` and their ratio. Labels: "closed form (Eq. S11)"
  vs "Gaussian approx". Reuse this in `ExactFrame` (show the cuts beside the images).
Add `i_y`, `gauss_peak`, `exact_peak` to `api/types.ts` per API.md.

## 5. Quality
`npm run lint`, `npm run typecheck`, `npm test -- --run` (new tests for §1–§2), `npm run build`
clean; bundle still ≤ 400 kB gzip; no console errors on `/`, `/basic?example=single_lift_hold`,
`/basic?example=array_arbitrary_path`. Report → `webapp/docs/reports/R2-frontend-a.md`; final
message ≤ 30 lines.
