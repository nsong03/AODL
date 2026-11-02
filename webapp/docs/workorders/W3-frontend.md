# W3 — frontend (Vite + React 18 + TypeScript)

**Role:** implementation agent. **Authority:** this work order; then `../DESIGN.md` §1–§2, §4
and `../API.md` (the JSON you consume — mirror it exactly in `src/api/types.ts`). No git.
W1 and W2 build the backend concurrently; it may not be running while you work, so develop
against the shapes in API.md and the fixtures W1 drops into `frontend/tests/fixtures/`
(`pick_and_place.json`, `array_lift_hold.json`, `examples.json`). If they are not there yet
when you need them, write your tests against the schema and finish them when the files appear
(W4 will run everything end to end).

## Owned files
`webapp/frontend/**` except `tests/fixtures/*.json` (W1). Do not edit the `Makefile` (W1 owns
it; it will call `npm ci`, `npm run dev -- --port 5173`, `npm test -- --run`, `npm run lint`,
`npm run typecheck`, `npm run build` — provide exactly these scripts).

## 1. Stack and constraints
`npm create vite@latest frontend -- --template react-ts` (Vite 6, React 18, TS strict), plus
`react-router-dom@6`, `zustand`, `uplot`; dev: `vitest`, `eslint` (template), `typescript`.
No Tailwind, Three.js, Plotly, MUI or D3. Bundle budget: `vite build` total JS ≤ 400 kB gzip
(print the numbers in your report). Dev proxy `/api → http://localhost:8000`. Node 22, npm
registry is reachable. `index.html` title "AODL Studio".

## 2. Physics module (`src/physics/`, pure functions, unit-tested against the fixtures)
- `units.ts`: `MHz, us, um, mm, nm`; helpers to convert API display units → SI.
- `piecewisePoly.ts`: `evalPoly(poly: Poly, t_us: number): number` and a vectorised form;
  clamp-hold outside the domain, `searchsorted` semantics identical to `aodl.poly` (segment `k`
  is `[breaks[k], breaks[k+1])`, the last break belongs to the last segment); `derivative(poly)`.
- `envelope.ts`: `envA(env, t_us)`: `constant`, `smooth_on_off` (sin² ramps, half-open
  intervals as in `aodl.waveform.tones.SmoothOnOff`), `fade_zone` (API.md §4.1 formula with
  `g` from its own `Poly`).
- `waveform.ts`: `toneFreqMHz(tone, t)`, `tonePhaseRad(tone, t)` (= `evalPoly(phase) + phase0`),
  `voltage(channel, t_us[])` (= `Σ A cos(2π f_center[Hz]·t[s] + phase)`, **no** renormalisation —
  the fixtures are generated with the same convention), `liveTracks(channel, n=1500)` → arrays
  for plotting with `alpha = A` and `NaN` where `A == 0` so lines break.
- `aperture.ts`: `apertureVoltage(channel, t_us, n)` → `{u_mm, v, filled}` with
  `t_ret = t − τ/2 − s·u/v` and `filled = s·u ≤ v·t − D/2` (signs from the IR: `sound_sign`).
- `gaussianBeam.ts`: `renderXY(spots, grid, plane_z, w0, zR, peakNorm, gamma=0.7)` →
  `Float32Array` RGB (additive, hue by z through `zColor`), patch-based (±4 w per spot),
  separable 1D profiles; `renderXZ(spots, xgrid, zgrid, rowY, …)`; `spotRadii(spot, z, w0, zR)`
  (DESIGN §1 formulas); `peakNormalization(frames, planes, w0, zR)` = max over frames of
  `Σ_spots P·2/(π wx wy)` at the frame's plane (global, honest dimming).
- `zColor.ts`: port of `aodl.viz.style.z_color`: `RdBu_r` from the 17 anchors below
  (linear interpolation), `z_norm = clip(0.5 + z/(2 z_max), 0, 1)`, then divide RGB by its max
  channel so brightness carries only intensity; `z_max = max(|z|, 1 µm)`.
  ```
  x, r, g, b: 0 5 48 97 | .0625 23 82 144 | .125 42 113 178 | .1875 63 142 192 | .25 107 172 209 |
  .3125 155 201 224 | .375 194 221 236 | .4375 224 236 243 | .5 247 246 246 | .5625 251 229 216 |
  .625 251 204 180 | .6875 245 170 137 | .75 228 128 102 | .8125 208 85 72 | .875 186 40 50 |
  .9375 147 14 38 | 1 103 0 31
  ```
Tests (`vitest`, environment node): `f`, `A`, `phase` vs fixtures (1e-9 relative on the
largest value, 1e-12 absolute on `A`), `V(t)` vs `voltage_window` (1e-6), aperture vs fixture
(1e-6), Gaussian render peak location and integral (≈ P within 2 % on a fine grid), `zColor(0)`
is white-ish `(1,1,1)`-normalised, poly clamp-hold.

## 3. State (`src/state/store.ts`, zustand)
`hardware`, `array`, `moves`, `options` (the editable `PlanRequest`), `plan: PlanResponse | null`,
`planStatus`, `planError` (the API `message` verbatim), `sim: SimulateResponse | null`, `simStatus`,
`t_us` (global cursor), `playing`, `speed_us_per_s` (default `duration/8`), `viewMode:
"tracked"|"fixed"`, `showExact`, `exactFrame`, advanced-page state. Actions: `loadExample(id)`,
`setMove(i, patch)`, `addMove(kind)`, `removeMove`, `reorderMove`, `requestPlan()` (debounced
400 ms after any edit; aborts in-flight), `requestSim()` (after a plan, `n_frames = 80`, or
`48` for > 36 traps), `fetchExact(t)`, `setTime`, `play/pause`, `setSpeed`.

## 4. Pages and components (DESIGN §4)
- **Landing `/`**: hero (one sentence: trajectory in → four RF drives + a preview of the
  tweezers), `SetupDiagram` — one inline SVG, ~900×360, left→right: laser → **Ax** (sound −x,
  transducer on the +x face, arrow) → 4f relay (two lenses) → **Bx** (sound +x) → **Ay** (sound
  −y) → **By** (sound +y) → objective → focal volume with a small 3D trap trail; each crystal a
  slab in its channel colour (Ax `#4cc9f0`, Bx `#4361ee`, Ay `#8ac926`, By `#ffb703`), labelled
  with its role ("single tone / lateral", "ladder / columns", …); a legend; under it the Table I
  mapping as HTML (`X = λF/v·(f_Bx − f_Ax)`, `Y = …`, `Z̄ = ½·λF²/v²·Σ ḟ`, `ΔF = … = 0`) and
  three short "how it works" cards (deflection = frequency difference; lens = chirp sum;
  fading-Shepard for sustained Z). Then `ModeCards` (Basic / Advanced) and `ExampleCards`
  (from `/api/presets`, the seven ids; each card → `/basic?example=<id>`; the three the brief
  names — `single_lift_hold`, `array_lift_hold`, `array_arbitrary_path` — go first and are
  tagged "quick link").
- **Basic `/basic`**: three-column desktop layout — left: editor (`HardwarePanel` preset
  select + fields, `ArrayPanel` mx/my/Δf with the pitch shown live as `um_per_MHz·Δf`,
  `MoveList` with kind-specific fields and the `PathGenerators` modal: square (side, plane,
  points), circle (radius, plane, turns, points), sphere spiral (radius, points), helix (radius,
  pitch, turns) → emits a `path` move's `points_um`; `OptionsPanel`); centre: `TweezerView`
  (XY canvas ≤ 320 px long side, XZ canvas, `IsoView`, colour bar, mode toggle, "Exact frame"
  button → `ExactFrame` side by side with the same colour scale), under it `Playback` (time bar
  over `t_span_us` with seam ticks, fade-event ticks, the shaded `[t − τ, t]` window, play/pause
  (space), speed, step); right: `ReportCard` (mode, `summary_text` in a `<pre>`, `BandUsage`
  bars per channel, notes, fade events), `RfPanel`, `AperturePanel`, `MetricsPlots`, `CodePanel`.
  On narrow screens stack them. Errors: a banner showing `planError`; the previous plan stays
  on screen.
- `RfPanel`: four stacked canvases (one per channel, channel colour), tone tracks with opacity
  = envelope, dotted band edges, shaded `[t − τ, t]` ("inside the crystal"), a tick at `t − τ/2`
  ("beam centre"), the cursor; a toggle "voltage zoom" draws `V(t')` over `[t − 1 µs, t + 1 µs]`
  (401 points) under the tracks. Tracks computed once per plan via `liveTracks`.
- `AperturePanel`: four small canvases, `V(u)` for the cursor time, unfilled region greyed and
  labelled "not yet filled" while `t < τ`, transducer side marked (`−s·D/2`), `±w_in` shaded.
- `TweezerView` rendering: `renderXY` at the frame's tracked plane (or 0), gamma 0.7, global
  `peakNormalization`; XZ through the brightest spot's y (`spot_row` logic: snap to the spot
  nearest the power-weighted mean y among spots ≥ half the max power); `IsoView`: isometric
  projection of the bounds box, each spot's 1/e² caustic as 9 ellipses over `z ± 2 z_R` tinted
  by z, plus the requested trajectory as a trail and the current requested point as a hollow
  marker (so the τ/2 lag is visible). Interpolate spots between frames linearly by group
  index when counts match; otherwise nearest frame.
- `MetricsPlots` (uPlot): requested vs simulated `x, y, z` (simulated lags by τ/2 — draw the
  requested curve shifted by +τ/2 as a dashed "expected at the atoms" line too), power per
  spot (normalised by `power_ref`), `sigma_astig`, `wx/wy`; all share the cursor.
- `CodePanel`: `code_python` in a `<pre>` with copy, download `.py`, download `.npz`
  (`/api/plan/{plan_id}/waveform.npz`).
- **Advanced `/advanced`**: `EfficiencyPanel` (model fields; η vs f with a second x-axis in
  mrad and the band shaded; pair heat map canvas with the ridge marked; "along this
  trajectory" plot of the four channels + product), `MisalignmentPanel` (per-AOD sliders with
  the seven knobs, reset, "Run" → `/api/advanced/simulate` at 6 times spread over the
  trajectory; XY/XZ images for misaligned vs aligned with a shared colour scale; metrics
  table), `SpectrumPanel` (channel select, window around the cursor, PSD plot, crest/PAPR/peak
  depth badges). Every panel has a one-line "model assumptions" caption from the response.

## 5. Quality
`npm run lint`, `npm run typecheck` (`tsc --noEmit`), `npm test -- --run`, `npm run build`
clean. Keyboard: space play/pause, ←/→ step one frame. No console errors in dev. Loading
skeletons for plan/sim. Numbers formatted with units (`toFixed` by magnitude).

## 6. Report → `webapp/docs/reports/W3.md`: built, test counts, bundle sizes, what you could not
verify without the backend, and any API ambiguity you resolved (and how).
