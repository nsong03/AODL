# R2-verify — integration and verification of round 2 (report)

Work order: `workorders/round2/R2-verify.md`, plus the five extra items of the dispatch (stale
even-M text, the addendum's order-1/order-3 correction, R2-frontend-a's rigid shift, R2-frontend-b's
wrappers, the dev servers). The four R2 reports were audited as claims. **No git commands were
run**; every changed file is listed below for the architect. Measured numbers are in
`VERIFICATION.md` § "Round 2" (R2.0–R2.5).

## 1. Final status per check

| check | status | evidence |
|---|---|---|
| root `python -m pytest -q` | **green** | 464 passed, 1 xfailed (149 s) after all edits |
| root `ruff check/format --check src tests` | **green** | clean, 75 files |
| root `mypy src/aodl` | **green (tolerated)** | the 10 pre-existing numpy-2.5 overload errors only |
| notebooks 05/06/07 under `--nbmake` | **green after fix** | 05 failed on arrival (D1); all three pass now; outputs cleared |
| `make -C webapp lint test build` | **green** | backend 123 passed, frontend 84 tests, bundle 157 kB gzip |
| `make -C webapp fixtures` no-op | **green** | R2-core's fixtures byte-identical; `examples.json` changed only by D8's blurbs, then no-op |
| §2 even-M per-site power ≥ 0.80 (4×4, 2×2 + 1×4, 4×1, 4×4 @1.0/1.0) | **green** | worst 0.851 (4×4, order 3); 0.949 at order 1 |
| §2 extended columns switch hard | **green** | edge column jumps 0.966 in one 0.167 µs frame; requested columns move ≤ 0.035 |
| §2 `single_lift_hold` shadows 0.5 | **green** | 0.5004 at ±123.6 µm at all five hand-overs |
| §2 `pick_and_place` unchanged | **green** | s19, band identical, tracking 5e-4 µm, fixture byte-identical |
| §2 plain tutorial: margin < 0 on all channels, η-power < 0.05 after band edge | **green** | −22.04 MHz ×4; ≤ 0.0158 after 241.8 µs |
| §2 Shepard tutorial η-power ≥ 0.8 away from hand-overs | **red as phrased (model behaviour, not a defect)** | min 0.629, dips *between* hand-overs on the B channels; ≥ 0.94 within ±20 µs of each hand-over (O3) |
| §3 e2e: tutorial and 4×4 pages, 0 errors | **green** | 11 pages, 0 console/page errors, 0 failed requests |
| §3 no trap moves > 0.5·w0 per frame in the hold | **green** | max 0.0000 w0 over 57 intervals; ≥ 16 traps every frame |
| item 3: `pick_and_place` / `array_arbitrary_path` glide | **green** | 0 cross-fades; steps 1.124 / 0.961 w0 interpolated |
| item 4: tutorial = workbench physics | **green** | identical `/api/simulate` bodies; identical brightest-spot badge at the same cursor |
| playback ≥ 30 fps | **green after fix** | 4×4 47.1, flagship 30.0 fps (were 32.7 / **20.6**, D5) |
| §4 README, DESIGN §1.4 | **done** | see §3 |
| screenshots | **done** | all nine regenerated; `intensity_panel.png`, `tutorial.png` new |

## 2. Defects found and fixed

| # | file(s) | symptom | fix |
|---|---|---|---|
| D1 | `examples/05_fading_shepard.ipynb` | `pytest --nbmake` **failed**: cell 15 asserted `set(nodes_lit) == {(11, 11)}` (the retired "M + 1 at every instant" rule); §4 and §6 prose explained the walk as parity and the Δf/2 comb as the lattice fix | §4 last paragraph rewritten (phasing, `ξ_B = ξ_A + ½`, M ↔ M+2 for both parities); §6 bullets rewritten (10×10 between hand-overs, +2 during; the half rung carries the lattice and the timing; `lattice_comb_offset` = 0); cell 15 now indexes the requested columns −5..4 and edges −6/5, asserts widths ⊆ {10, 12}, requested columns flat to 1e-9 and the two edges summing to ≤ 1 (measured: requested 1.000000, edges 0.00 on a plateau / 1.00 in a hand-over, residual 7e-14 w0); cell 16 marks edges −6/5 and retitles the plot |
| D2 | `examples/06_product_tour.ipynb` | 93 tones, 1.3 MHz margin, "even-M comb offset" caveat, "11 x 11 extended grid", "M+1 at every instant", 97 kB | 94 tones, 2.0 MHz (measured 1.9614), "half-rung offset of the B ladder", "10 x 10, plus edge columns/rows in a fade", M+2 rule, 98 kB (97.7) |
| D3 | `examples/07_fft_checker.ipynb` | M+1 rule; 93 tones; "spread by about a fifth", "82 %", "Astigmatism 0.02 z_R", "uniformity_median 0.137" — all re-measured by its own cells | M+2 rule; 94; "about a sixth (0.171 worst)", 78 % (R2-core's measurement), 0.01 z_R (measured 0.0130), 0.102 |
| D4 | `src/aodl/check/report.py` | the checker's own runtime notes (shown in nb 07's output) still said "clean and 0.80-rung flagships both median 0.137" and "an M + 1 (even M) or M + 2 (odd M) wide array"; the `uniformity_median` docstring table had the pre-fix 0.213/0.137, 0.280/0.137 | notes: "clean flagship medians 0.102, a 0.80-rung fault 0.098", "an array M + 2 wide during a hand-over (odd and even M alike)"; docstring table 0.171/0.102 and 0.24/0.098 (`tests/test_check_flagship.py` pins), "−0.24 excursion" |
| D5 | `webapp/frontend/src/physics/gaussianBeam.ts`, `components/tweezer/IsoView.tsx`, `tests/gaussianBeam.test.ts` | flagship playback **20.6 fps** (gate 30; round 1 measured 37.6), 4×4 32.7 (was 52–56): the fix lights M+2 columns and each hand-over site carries two coincident rung-pair spots, so frames hold 484 / 100 spots (were 400 / 64); `renderXY` allocated ~3 MB of accumulators per frame; `IsoView` allocated an object per ellipse point (9 ellipses × 25 points × spot) | new `mergeCoincident(spots)` (exact duplicates in x, y, z, ΔF to 1e-6 µm → one spot with summed power; exact for renderers that sum linear intensity before the gamma) used by `renderXY`, `renderXZ` and `IsoView` (halves the drawn spots: 391 → 199 on the flagship); `renderXY` reuses module-level accumulators; `IsoView` inlines the projection over a precomputed unit circle. Result 47.1 / 30.0 fps. Unit test added (84 tests). Badges and spot counts are unchanged (they read `spotsAt`, not the merged list) |
| D6 | `webapp/backend/src/aodl_webapp/simulation.py` (`trajectory_bounds`), `backend/tests/test_api_simulate.py`, `docs/API.md` §5.1 | on a fading array the M+2 hand-over columns sit one pitch outside the requested array, but `bounds_um` = array half-extent + 6 w0 (6.4 µm < 10.3 µm pitch), so the movie **never showed** them: mid-hand-over the 4×4 canvas showed 16 traps and the badge called the edge columns "36 shadow tweezers off canvas" | a fading-Shepard array axis (M > 1) gets one extra pitch per side (single-tone axes unchanged, so the ±123.6 µm single-tweezer shadows stay off canvas). New test `test_fading_array_bounds_include_the_extended_columns` (bounds value + every spot of every frame inside); API.md comment. The fixtures carry no bounds (no-op) |
| D7 | `webapp/scripts/e2e.mjs` | an app-cancelled `/api/frame` line-cut fetch (`net::ERR_ABORTED`, superseded by a newer cursor time) counted as a failed request | aborted requests are listed separately (`abortedRequests`) and reported, not failed; plus the new round-2 checks (§3 below) |
| D8 | `webapp/backend/src/aodl_webapp/examples.py`, `src/aodl/api.py`, `src/aodl/waveform/synthesis.py` | blurbs "the lit grid grows by a column", "93 fading-Shepard rungs, 483 frequency groups"; comment "A 93-tone fading drive"; `_shepard_band_message` called the comb "the comb offset that puts an even-M ladder on Eq. S19's lattice" | "two edge columns light up during each hand-over while all 16 requested traps stay lit"; "94 … about 460 frequency groups" (measured 461 at the first lit frame); "94-tone"; "the explicit … comb offset (ShepardConfig.comb_offset; the automatic one is 0, since table_ii's half-rung xi_B already puts an even-M ladder on Eq. S19's lattice)" — the clause is only emitted for a non-zero explicit offset, as R2-core said |
| D9 | `webapp/docs/workorders/round2/R2-DESIGN-ADDENDUM.md` §F2 | "with the fix" column was order 3, labelled order 1 | table now has as-shipped / fixed order 1 / fixed order 3 columns (R2-core §2 values, re-measured to ±0.001), and the residual-dip sentence says 2–5 % at order 1 (ρ ripple), 7–15 % at order 3 (IM3) |
| D10 | `webapp/docs/DESIGN.md` §1.2 | "a 10×10 has 483" groups | "~460, 483 before the round-2 even-M fix" |

## 3. Other edits (work-order deliverables)

* `webapp/scripts/e2e.mjs`: new `checkMotion` (per-interval rendered displacement through
  `spotsAt`, rigid shift and cross-fades through `pairFrames`, trap counts through `frameSites`);
  run on `pick_and_place`, `array_arbitrary_path` (glide: 0 cross-fades required) and the 4×4 hold
  (≤ 0.5·w0 per interval, ≥ 16 traps required); the 4×4 screenshot is now taken at a mid-hold x
  hand-over + τ/2 and the Trap-intensity panel is captured as `intensity_panel.png`; a tutorial
  block loads `/tutorial/shepard`, captures its `/api/simulate` bodies, compares them and the
  brightest-spot badge with the workbench's `/basic?example=tutorial_hold_*` at the tutorial's
  start cursor, checks the shadow badge and the `/tutorial` redirect, and takes `tutorial.png`.
* `webapp/README.md`: setup order in the intro (laser → Ax → Ay → 4f relay → Bx → By →
  objective), the intensity panel and the tutorial, the two new screenshots and corrected
  captions, the nine examples and the tutorial route, the physics caveats rewritten as a table
  with the even-M fix row, the mixing-order row (4×4 now order 3), flat-efficiency and
  interpolation rows.
* `webapp/docs/DESIGN.md`: new §1.4 "Even-M fading arrays (round 2)" pointing at the addendum.
* `webapp/docs/VERIFICATION.md`: new "Round 2" section R2.0–R2.5.
* `webapp/docs/screenshots/`: all nine regenerated by `e2e.mjs`.
* `.github/workflows/webapp.yml`: reviewed, no change needed (fixture-freshness step already
  there).

## 4. Audit of the R2 reports' claims

* **R2-core**: table reproduced to ±0.001 (both orders) through HTTP; 94 tones / 1.96 MHz / 34
  hand-overs confirmed; root suite count confirmed. Its "left for other owners" list was
  complete except for `src/aodl/check/report.py`'s two runtime notes and docstring table (D4)
  and nb 07's "a fifth / 82 % / 0.02 / 0.137" (D3), and it did not run the notebooks — nb 05's
  assertion failed (D1).
* **R2-backend**: margins, z-budget fraction 3.20, band exit, `i_y`/`gauss_peak`/`exact_peak`
  confirmed (backend tests + e2e line cuts).
* **R2-frontend-a**: index-free pairing and the rigid shift confirmed in the browser (0 w0
  motion in the hold; 0 cross-fades on the carries; shift ≤ 1.124 w0 < 2·w0 cap). Its
  "≥ 0.85 interior" figure is the order-3 number, now the 4×4's default movie order. Missed:
  the extended columns were off canvas (D6) and the fps regression (D5).
* **R2-frontend-b**: the wrappers render the same physics (identical simulate bodies, same
  `spotsAt`, same badge at the same cursor) and the page still works after R2-frontend-a's
  changes (no console errors, shadows at the start cursor). Its η-curve note (dips ~0.63–0.75)
  is right; where the dips are is in O3.

## 5. Open issues and recommendations

* **O1 — 4×4 movie now at mixing order 3 and slow.** The fix dropped the 4×4's first-lit-frame
  group count from 165 to 152 (< budget 160), so the movie runs at order 3: `/api/simulate`
  5.9 s, movie 7.6 s after navigation (was 2.8 s at order 1). Correct per DESIGN §1.2, but the
  rule samples one frame and hand-over frames hold more groups. Recommend sampling the budget on
  a hand-over frame (or the max of the first few frames), or lowering `MOVIE_GROUP_BUDGET` to 150,
  if the architect prefers the order-1 speed.
* **O2 — flagship playback at the 30-fps edge** (30.0 headless software rendering). `IsoView`
  (9 ellipses per spot) is still the largest JS cost. Recommend drawing only the focal ellipse
  during playback, or skipping ellipses for spots below a few % of `power_ref`.
* **O3 — Shepard tutorial efficiency criterion.** Work order R2-verify §2 expected the
  Shepard η-weighted power ≥ 0.8 "away from hand-overs"; measured, the model's dips (min 0.629)
  sit *between* hand-overs on the B channels, through the `γ` coupling term, and ≥ 0.94 near
  them. The tutorial's caption is accurate; recommend the caption also say where the dips are
  ("between hand-overs, on the second AOD of each pair, through the model's γ coupling") so a
  reader does not attribute them to the fading.
* **O4 — requested-position marker differs between the two movies.** `TweezerView` marks the
  request at drive time `t` (the lag is shown on purpose, DESIGN "retardation shown"), the
  tutorial's `TutorialTweezer` at `t − τ/2` (on the trap). Both are defensible; pick one and
  label it (the spots are identical either way).
* **O5 — duplicated wrappers.** `TutorialRf` / `TutorialTweezer` / `TutorialPlayback` mirror
  `RfPanel` / `TweezerView` / `Playback` because those read the global store; recommend
  `plan`/`sim`/`cursor` props on the shared components (R2-frontend-b's suggestion).
* **O6 — R2-core's own unsure items stand**: the guide's 82 % → 78 % C²-share was measured by a
  different method than the original; the one-channel skew thresholds were not re-measured; the
  F-2 rung-fault hole is still open (strict xfail). `docs/ORCHESTRATION.md` keeps the historical
  0.1373 (a log, left as is).

## 6. Commands run (all from `/home/user/AODL`)

```
python -m pytest -q                                   # 464 passed, 1 xfailed (149 s)
ruff check src tests && ruff format --check src tests # clean
python -m mypy src/aodl                               # 10 errors in 5 files (pre-existing)
python -m pytest --nbmake examples/05_fading_shepard.ipynb -q   # failed on arrival → 1 passed
python -m pytest --nbmake examples/06_product_tour.ipynb -q     # 1 passed
python -m pytest --nbmake examples/07_fft_checker.ipynb -q      # 1 passed
make -C webapp lint test build fixtures               # green; backend 123, frontend 84
node webapp/scripts/e2e.mjs --out results.json        # all e2e checks passed
python <scratch>/r2v_audit.py http://127.0.0.1:8001   # the §R2.2/R2.3 HTTP audit
```

The dev servers left by earlier agents (uvicorn :8000, two Vite :5173) were stopped; the ones
started for this pass were stopped as well.
