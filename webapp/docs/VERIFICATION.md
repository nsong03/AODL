# AODL Studio — verification (W4)

Measured on the dev container: Python 3.13.18, numpy 2.5.3, scipy 1.18.1, fastapi 0.162,
pydantic 2.13.5, mypy 2.4.1, ruff 0.17.3, Node 22, Vite 6.4.5, Playwright 1.63 with Chromium
153 (headless, software rendering), `aodl` 0.1.0 at the `default_1030` hardware
(τ = 11.5385 µs, w0 = 1.0655 µm, z_R = 3.463 µm, λF/v = 10.30 µm/MHz). Everything below was
produced by `webapp/scripts/e2e.mjs` (browser + API) and a Python audit script that calls the
backend's own modules; the reports in `reports/W1–W3.md` were treated as claims and re-measured.

## 0. Gates

| gate | command | result |
|---|---|---|
| root suite | `python -m pytest` | **457 passed, 1 xfailed** (2 min 01 s) |
| root lint | `ruff check src tests`, `ruff format --check src tests` | clean (74 files formatted) |
| root types | `python -m mypy src/aodl` | 10 errors in 5 files, **all pre-existing** (§6) |
| backend | `AODL_STUDIO_STRICT_ADVANCED=1 python -m pytest webapp/backend/tests -q` | **114 passed** from the repo root and from `webapp/backend`; `-k "advanced or cross_engine"` selects 44 from either directory (W2's collection error is gone: W1's tests import `conftest` as a top-level module) |
| backend lint/types | `ruff check/format webapp/backend webapp/scripts`; `python -m mypy src/aodl_webapp src/aodl_advanced` (from `webapp/backend`) and `python -m mypy webapp/backend/src/aodl_webapp webapp/backend/src/aodl_advanced` (from the root) | clean, 31 source files |
| frontend | `npm run lint`, `npm run typecheck`, `npm test -- --run`, `npm run build` | eslint 0 problems; tsc clean; **8 files / 67 tests passed**; bundle 138 kB gzip (index 46.6 + vendor 68.3 + uplot 23.3) |
| Makefile | `make -C webapp install lint test build fixtures` | all green; `make fixtures` regenerates the three golden JSON files **byte-identically** (md5 unchanged), so CI can assert fixture freshness |
| Python 3.11 compatibility | `/usr/bin/python3.11 -m compileall` over `webapp/backend`, `webapp/scripts`, `src/aodl/trajectory/spec.py`, `tests/test_path.py` | 50 files compile; no 3.12-only syntax (the root CI pins 3.11) |
| e2e | `node webapp/scripts/e2e.mjs` (uvicorn :8000 + `vite preview` :4173, Chromium) | **all checks passed**, 0 console errors / page errors / failed requests on every page |

## 1. Sign of the aperture highlight and the beam-centre marker (W4 §3.1)

`docs/conventions.md` §7: the sample at aperture coordinate `u` at frame time `t` was emitted
at `t_ret(u) = t − τ/2 − s·u/v`; the aperture holds drive content where `s·u ≤ v·t − D/2`; the
beam centre sees `t_c = t − τ/2`; the whole aperture holds the drive segment `[t − τ, t]`.

| place | what it draws | verdict |
|---|---|---|
| `frontend/src/physics/aperture.ts` `retardedTimeUs` | `t − τ/2 − s·u/v` (SI inside) | matches `aodl.device.conventions.retarded_time` sign for sign |
| `aperture.ts` `isFilled` / `apertureContent` | `s·u ≤ v·t − D/2` | matches `conventions.is_filled` |
| `components/aperture/AperturePanel.tsx` | transducer at `u = −s·D/2`, leading wavefront at `u_edge = s(v·t − D/2)`, "not yet filled" on the far side | matches §7's table (`+1`: filled `u ≤ u_edge`; `−1`: filled `u ≥ u_edge`) |
| `components/rf/RfPanel.tsx` `drawDynamic` | shades `[max(t_span[0], t − τ), t]`, dotted tick at `t − τ/2` with live-tone markers | the shaded window is in the **past** of the cursor and clipped at the drive start, as DESIGN §1.3 / REVIEW A4 require |
| `components/tweezer/Playback.tsx` | shaded `[t − τ, t]`, dotted `t − τ/2`, fade ticks at `t_fade + τ/2` (dedupe 1 ns) | correct; the fade events are drive times, so the atoms see them `τ/2` later |
| `backend/src/aodl_webapp/aperture.py` | delegates to `conventions.retarded_time` / `is_filled`; no sign written in the web layer | correct; `voltage` on the full aperture is `aperture_window` point for point |

Numerically (e2e check (b), pick_and_place, 512 points per channel, all four channels):

| instant | envelope (rel. to max) | `f_inst` [MHz] | fill mask | carrier `V(u)` on the central 0.1 mm (4096 pts, rel. to max) |
|---|---|---|---|---|
| `t = 75.769 µs` (= 70 + τ/2, aperture full) | 0 | 1.4e-14 | exact (and `null` ↔ `NaN` agree) | 2.9e-11 |
| `t = 6.923 µs` (= 0.6 τ, aperture filling) | 0 | 1.4e-14 | exact | 3.4e-13 |

The frontend's `apertureContent` / `apertureVoltage` and `/api/aperture` agree far inside the
1e-6 tolerance of DESIGN §5 (b). The panel itself draws 320 samples and restricts the tone loop
to tones that can be live in `[t − τ, t]`; both are display optimisations of the same function.

## 2. Gaussian approximation vs closed form, static tweezer at `t = 3τ` (W4 §3.2)

One `Hold(100 µs)`, `t = 34.615 µs`, tracked plane `Z = 0`, 193 × 193 grid over ±6 w0:

| quantity | value |
|---|---|
| spot | `x = y = z = 0`, `ΔF = 0`, `wx = wy = 1.06554 µm` |
| peak of `intensity_frame` | 3.696935e-17 |
| peak of `2P/(π wx wy)` | 3.696935e-17 (ratio **1.000000**) |
| max relative pixel difference | **4.6e-15** |
| `peak · wx · wy · π/2 / power` | 1.000000 |

The last row is the unit check behind API.md §0.3: a packed basic-mode image is **power per m²**
on the `SpotMetrics.power` scale (`wx`, `wy` in metres). The frontend evaluates `2P/(π wx wy)`
with the `_um` radii, i.e. per µm², and therefore multiplies decoded exact-frame values by
`PER_M2_TO_PER_UM2 = 1e-12` (`physics/units.ts`). With that factor the two are on one scale: in
the browser (check (a) below) exact peak / approximation peak = 0.987 at mid-translate (the
closed form includes the acoustic irising of the chirped drive; the approximation does not),
and the brightest pixels coincide. API.md §0.3 now says the unit explicitly.

## 3. `Path` spline: 24-point circle, R = 10 µm, 300 µs (W4 §3.3)

Waypoints `(R cos φ_k − R, R sin φ_k, 0)`, `φ_k = 2πk/24`, the 24th being the origin
(`closed=True` is then a no-op, as DESIGN §1.3 specifies). `plan_motion` → **`s19`**, no notes
beyond the standard retardation note.

| check | value |
|---|---|
| simulated spot at `t + τ/2` vs requested `X(t), Y(t)` over the middle 80 % of the move (41 instants) | max **1.0e-14 µm** (mean 2.3e-15 µm) |
| distance of the simulated spot from the ideal circle | max 0.0188 µm (the clamped cubic through 24 chords, not a tracking error; a polyline would miss by 0.086 µm) |
| band usage Ax / Bx | [100.0000, 100.9709] / [99.0291, 100.0000] MHz, width 0.9709 MHz |
| band usage Ay / By | [99.5146, 100.4854] MHz each, width 0.9709 MHz |
| expected lateral span / (2 λF/v) | X ∈ [−20, 0] µm → 0.9709 MHz; Y ∈ [−10, 10] µm → 0.9709 MHz |

The band check reports exactly the lateral span of the compiled spline on every channel
(`poly_range` on the degree-3 segments), with the x pair split into the `+`/`−` halves of
Eq. S19 and the y pair symmetric about `f_center`, as the circle is.

## 4. Advanced engine: `trajectory_efficiency` (W4 §3.4)

| case | result |
|---|---|
| static tweezer (`Hold 100 µs`, default `EfficiencyModel`: L = 4 mm, n = 2.26, η_peak = 0.85, γ = 0.2, `f_tpm = f_center`) | product = **0.522006 = η_peak⁴** at every one of 200 samples inside the live window (min = max); η = η_peak on all four channels because `f_A = f_tpm` puts B on its ridge |
| `flagship_10x10` (`/api/advanced/trajectory_efficiency`, n = 300, 187 ms) | product 0.5170 at the start (the 10 × 10 ladders already spread B's incidence); **min 0.3875 at t = 383 µs** (observation time; inside the 40 × 25 µm translate, 156–406 µs), per channel at the minimum Ax 0.850, Bx 0.705, Ay 0.850, By 0.761 |
| `pick_and_place` from the browser (`EfficiencyPanel`) | 3 dB bandwidth **27.90 MHz**, first-AOD usable width **12.05 MHz**, trajectory min product 0.4357 |

The first-AOD usable width is 12.05 MHz at the defaults, not the 10.4 MHz of API.md §6.2's
example JSON (W2 flagged it); the example number was corrected.

## 5. Generated code under plain `aodl` (W4 §3.5)

`POST /api/code` for every starter example, each script written to a scratch directory and run
with `python -I <script>` in a subprocess (isolated mode: no site customisation, no cwd on
`sys.path`); the scripts run `plan_motion`, `print(plan.report.summary())`, `plan.save(...)`,
`plan.render_samples(625e6)` and `plan.simulate()`:

| example | rc | wall time | first output line |
|---|---|---|---|
| pick_and_place | 0 | 1.7 s | `AODL motion plan - mode: s19` |
| square_xy | 0 | 1.5 s | `… mode: s19` |
| sphere_loop | 0 | 1.8 s | `… mode: s19` |
| single_lift_hold | 0 | 6.9 s | `… mode: shepard` |
| array_lift_hold | 0 | 29.9 s | `… mode: shepard` |
| array_arbitrary_path | 0 | 1.7 s | `… mode: s19` |
| flagship_10x10 | 0 | 26.2 s | `… mode: shepard` |

Seven NPZ files were written (`aodl_<plan_id>.npz`), one per script.

## 6. Root `mypy` (verification item 1)

`python -m mypy src/aodl` reports 10 `call-overload` errors ("All overload variants of
`max`/`min` of `ndarray` require at least one argument") in `api.py:114`, `check/metrics.py`,
`check/report.py:1289`, `viz/movie.py`, `waveform/shepard.py`. None of these files changed in
this work (`git diff d8d9e2a -- src/aodl` touches only `__init__.py`, `py.typed`,
`trajectory/spec.py`; `tests/test_path.py` is new), and `python -m mypy src/aodl/trajectory/spec.py
tests/test_path.py` is clean. The error reproduces on a bare
`a: NDArray[np.float64] = np.zeros(3); a.max()` with numpy **2.5.3**'s stubs and does **not**
reproduce with numpy **2.4.6**'s stubs (checked by pointing `MYPYPATH` at the unpacked 2.4.6
wheel). numpy 2.5 ships no Python 3.11 wheel, so the root CI (Python 3.11) resolves numpy ≤ 2.4.6
and is unaffected; the errors are a local Python 3.13 + numpy 2.5 + mypy 2.4 artefact.
Recommended fix (not applied, files untouched by this work): spell the ten calls as
`np.min(x)` / `np.max(x)` (which type-check on every numpy), or pin `numpy<2.5` in the `dev`
extra until the stubs settle.

## 7. End-to-end checks (W4 §2, DESIGN §5)

Every page below: **0 console errors, 0 page errors, 0 failed requests** (status ≥ 400 or
network failure).

| page | route | time to plan | time to movie | notes |
|---|---|---|---|---|
| landing | `/` | — | — | example cards from `/api/presets` |
| basic | `/basic?example=pick_and_place` | 0.85 s | 0.85 s | `/api/plan` 23 ms / 26 kB, `/api/simulate` 156 ms / 21 kB (80 frames) |
| basic | `/basic?example=single_lift_hold` | 1.22 s | 1.22 s | `/api/plan` 48 ms / 39 kB, `/api/simulate` 532 ms / 38 kB |
| basic | `/basic?example=array_lift_hold` | 2.80 s | 2.80 s | `/api/plan` 167 ms / 135 kB, `/api/simulate` 1.98 s / 264 kB (order-1 fallback noted, 165 groups at order 3) |
| basic | `/basic?example=array_arbitrary_path` | 0.95 s | 0.95 s | `/api/plan` 28 ms / 62 kB, `/api/simulate` 287 ms / 78 kB |
| advanced | `/advanced?example=pick_and_place` | 1.33 s | 1.33 s | `/api/advanced/efficiency` 98 ms / 308 kB, `/api/advanced/trajectory_efficiency` 7 ms / 34 kB (plan and simulate served from the cache) |
| basic | `/basic?example=flagship_10x10` | 2.66 s | 2.66 s | `/api/plan` 185 ms / 152 kB, `/api/simulate` **1.83 s / 1.0 MB** (48 frames, order-1 fallback) |

Times are from navigation start as seen by Playwright (they include the page load and the
`vite preview` proxy hop); the per-route numbers are the browser's Resource Timing durations.

**(a) Brightest pixel, pick_and_place at `t = 70 + τ/2 = 75.769 µs`** — the approximation
rendered through `window.__aodl.physics.renderXY` on the exact frame's own 192 × 112 grid vs
the decoded `/api/frame` image: approximation brightest pixel (9.914, 0.000) µm, exact
brightest pixel (9.914, 0.000) µm → distance **0.000 w0** (grid step 0.146 µm = 0.14 w0, so
the two argmaxes fall in the same pixel); planes 5.000 / 5.000 µm; exact peak / approximation
peak 0.987. Both the lateral centroid and the tracked plane are exercised (`X ≈ 10 µm`, `Z = 5 µm`).

**(b) Aperture** — §1 above.

**(c) Shadow pair, single_lift_hold at the first x hand-over + τ/2.** `fade_events[0]` is the
`y` event at `t = 0` (`ξ_y = ½`, as W1 documented) and the x events appear twice at the same
instant (153.6, 153.6 µs — the 1e-16 duplicate the UI dedupes), so the check uses the first
`axis == "x"` event: drive `t = 153.600 µs`, frame `t = 159.369 µs`. `/api/frame` returns **four
spots**: two coincident main-trap spots at (0, 0, 10 µm) in groups 0 and 2 (the dying `(a, a)`
and rising `(a−1, a−1)` rung pairs), and two shadows at **x = −123.600 and +123.600 µm**
(`shadow_um = 123.60 = 10.3 × 12 MHz`), each with **P/P_main = 0.5004** (Eq. S31: at the
hand-over centre each of the four rung pairs carries ¼ of the plateau power, so main = ½,
shadows = ¼ each; shadow/main = ½). The UI badge at that instant reads
"2 shadow tweezers off canvas at ±123.6 µm (x fade)". Relative to the API's `power_ref` (the
median spot power over the 80 frames) the main trap reads **2.43** and each shadow 1.22, not
"≈ 1 and ≈ 0.5": this example spends most of its hold inside fades (each 124 µs fade every
247 µs, on both axes), so the median spot is a quarter-power rung pair — see the open issue in
`reports/W4.md`. Against the plateau power (max frame-summed power) the main trap is 0.4996 and
each shadow 0.2500.

**Playback** (rAF timestamps over 2 s, all panels redrawing, headless software rendering):
4 × 4 fading example **52–56 fps** over three runs (105–113 frames, worst gap 33–50 ms, up to 64 spots per frame);
flagship 10 × 10 **37.6 fps** with 400 spots at the cursor (worst gap 50 ms). Both ≥ 30 fps.

**flagship_10x10 in the browser** — plans (`shepard`, 93 rungs), simulates with the order-1
fallback note ("movie metrics fell back to mixing order 1: the first frame at order 3 holds 462
frequency groups (budget 160)"), 48 frames (the > 36-trap rule), ≤ 400 spots per frame, plays
at 35–38 fps (two runs), no console errors. Route timings as the browser saw them: `/api/plan`
185 ms (152 kB), `/api/simulate` 1.83 s (1.0 MB, 48 frames); the movie is ready 2.7 s after
navigation (cold backend caches). This matches W1's in-process 2.7 s for 80 frames scaled to 48
and the 1.7 MB → 1.0 MB payload the 48-frame rule was introduced for.

**Production path** — `uvicorn aodl_webapp.main:create_app --factory` serves the built bundle
at `/`: `http://127.0.0.1:8000/basic?example=pick_and_place` loads, survives a hard reload at
the same URL (title "AODL Studio", plan and movie ready), assets are served as files
(`text/javascript`), unknown non-API routes return `index.html`; after the W4 fix an unknown
`/api/...` path is a 404 instead of the SPA shell.

## 8. Screenshots (`webapp/docs/screenshots/`)

`landing.png` · `basic_pick_and_place.png` (t = τ) · `basic_single_lift_hold_fade.png` (first
x hand-over + τ/2, shadow badge) · `basic_array_lift_hold.png` · `basic_array_arbitrary_path.png`
(mid-circle) · `advanced_efficiency.png` · `basic_flagship_10x10.png` (mid-traverse).

---

# Round 2 — verification (R2-verify)

Same container and versions as above (Python 3.13.18, numpy 2.5.3, Node 22, Playwright 1.63 /
Chromium 153 headless, `default_1030`: τ = 11.5385 µs, w0 = 1.0655 µm, λF/v = 10.30 µm/MHz).
Round 2 = the four work orders of `workorders/round2/` (addendum §F1–F4). The four reports in
`reports/R2-*.md` were treated as claims; everything below was re-measured. Defects found and
fixed in this pass are listed in `reports/R2-verify.md`.

## R2.0 Gates

| gate | command | result |
|---|---|---|
| root suite | `python -m pytest -q` | **464 passed, 1 xfailed** (149 s), after every R2-verify edit |
| root lint / types | `ruff check src tests`, `ruff format --check src tests`, `python -m mypy src/aodl` | clean (75 files); mypy: the 10 pre-existing numpy-2.5 `call-overload` errors only (§6) |
| notebooks | `python -m pytest --nbmake examples/05_fading_shepard.ipynb` (and `06_…`, `07_…`, one at a time) | 05 **failed** on arrival (`assert set(nodes_lit) == {(11, 11)}` — the retired even-M rule); after the fix all three pass (final run: 05 93 s, 06 35 s, 07 30 s); outputs cleared |
| webapp | `make -C webapp lint test build` | ruff/mypy clean (31 files), eslint/tsc clean; backend **123 passed**, frontend **10 files / 84 tests**; bundle 157 kB gzip (index 61.9 + vendor 68.4 + uPlot 23.3 + CSS 3.7) |
| fixtures | `make -C webapp fixtures` | byte-identical to R2-core's regeneration for `array_lift_hold.json` and `pick_and_place.json`; `examples.json` differs only in the two blurbs R2-verify corrected (regenerated, then a second run is a no-op) |
| Python 3.11 | `python3.11 -m py_compile` on every edited `.py` | OK |
| e2e | `node webapp/scripts/e2e.mjs` (uvicorn :8000 + `vite preview` :4173) | **all checks passed**, 0 console errors / page errors / failed requests on all 11 pages |

## R2.1 F1 — landing diagram

`landing.png`: laser → **Ax → Ay** (boxed "first 2D deflector", sound −x, −y) → 4f relay
("images the A plane onto the B plane, magnification −1") → **Bx → By** ("second 2D
deflector", sound +x, +y) → objective → focal volume, matching arXiv:2510.11451 Fig. 1(a);
the "How to read the diagram" prose says the same.

## R2.2 F2 — even-M arrays: every requested site lit

Through the live `POST /api/simulate` (HTTP, not the test client), `array_lift_hold`'s
trajectory (Lift 8 µm / 80 µs, Hold 500 µs, Lift −8 µm / 80 µs) with the array replaced,
120 frames over `[2τ, 580 µs + τ/2]`, `movie_mixing_order` pinned; `power` summed within
0.25 pitch of each Eq. S19 site; normalised to the median over all (frame, requested site)
values (R2-core normalised to the first frame's median — the bracketed value, which agrees
with R2-core's table to ±0.001):

| array (Δf MHz) | worst requested site, order 1 | worst, order 3 | brightest requested, order 3 | extended (edge) site max, order 1 | off-lattice max | hand-overs |
|---|---|---|---|---|---|---|
| 4×4 (1.0/1.3) | **0.949** | **0.851** (0.855) | 1.038 | 0.985 | 0 | 41 |
| 2×2 (1.0/1.3) | 0.949 | 0.927 (0.926) | 1.024 | 0.985 | 0 | 41 |
| 1×4 (1.0/1.3) | 0.979 | 0.914 (0.916) | 1.039 | 0.979 | 0.50 (x shadows) | 21 |
| 4×1 (1.0/1.3) | 0.965 | 0.892 (0.907) | 1.035 | 0.989 | 0.50 (y shadows) | 25 |
| 3×3 (1.0/1.3), odd control | 0.949 | 0.916 (0.915) | 1.038 | 0.985 | 0 | 41 |
| 4×4 (1.0/1.0) | 0.964 | 0.933 (0.935) | 1.032 | 0.988 | 0 | 46 |

Every requested site is **≥ 0.80 of the plateau at every frame** (worst 0.851, the 4×4 at
order 3), and the even arrays now behave exactly like the odd 3×3 control. The "off-lattice"
0.50 on the 1×4 / 4×1 is the single-tone axis's Eq. S31 shadow pair (free-axis Δf = 12 MHz,
±123.6 µm), as expected. The addendum's "with the fix" column was order 3, not order 1 (R2-core
§2); `R2-DESIGN-ADDENDUM.md` §F2 now shows both columns.

**Extended columns switch hard.** A dense probe of the 4×4 around the x hand-over at
`t = 161.6 µs` (240 frames over ±20 µs, Δt = 0.167 µs, order 1): the four requested columns
stay in [0.965, 1.001] with a largest frame-to-frame change of 0.035, while each extended
column jumps by **0.966 in a single 0.167 µs frame** — column `x = +2.5 pitch` switches off at
`−19.3 µs`, column `x = −2.5 pitch` on at `+19.2 µs` (the Table II `p_B = 0` rectangles), and
fades smoothly through the A window in between.

**`single_lift_hold` shadows** (`/api/frame` at each of the five hand-overs + τ/2,
t = 159.4, 283.0, 406.6, 530.2, 653.8 µs, alternating x/y): 4 spots each time, two shadows at
**±123.6 µm** with **P/P_main = 0.5004** at every one (Eq. S31: ½).

**`pick_and_place` unchanged**: mode `s19`, one tone per channel, band
Ax [100, 101.4563], Bx [100, 103.3981], Ay/By [100, 102.4272] MHz (identical to round 1),
one spot per frame, brightest spot vs requested `X, Y, Z(t − τ/2)` within 5.0e-4 µm (the
trajectory-sample interpolation), frame power constant (spread 0). Its golden fixture is
byte-identical after `make fixtures`.

**`flagship_10x10`**: 94 rungs (Ax 21, Bx 30, Ay 17, By 26; was 93), worst band margin
+1.961 MHz on By (was +1.31), 34 hand-overs, the first fully lit frame holds 461 groups at
order 3 (budget 160 → order-1 movie, as before).

## R2.3 F3 — the tutorial drives

`/api/plan` + `/api/advanced/trajectory_efficiency` (default `EfficiencyModel`, n = 300):

| | `tutorial_hold_plain` | `tutorial_hold_shepard` |
|---|---|---|
| mode, tones | `s19`, 1 per channel | `shepard`, 4 per channel (Δf 12 / 12 MHz) |
| band margin | **−22.04 MHz on all four channels** (100–132.04 MHz) | **+1.00 MHz** on all four (91–109 MHz) |
| first tone out of band (from the IR) | drive 236.0 µs → atoms **241.8 µs** | never |
| η-weighted power / its own max | 0.0158 max after 241.8 µs; 0.0096 max after 241.8 µs + τ; first below 0.05 at 212 µs | min **0.629**; ≥ 0.94 within ±20 µs of each hand-over + τ/2, ≥ 0.82 within ±40 µs; ≥ 0.80 for 79 % of the live time |
| raw η product max | 0.522 (= η_peak⁴) | 0.502 |

The plain drive meets the work-order criterion (< 0.05 after the band edge). The Shepard
drive does **not** meet "≥ 0.8 away from hand-overs" as phrased: the Bragg/TPM model's dips
are *between* hand-overs (minima at ≈ 105, 229, 354, 478, 602 µs, midway between the x and y
events at 159, 283, 407, 530, 654 µs), not at them, and they are on the **B** channels
(Bx 0.687 at 229 µs while Ax is 0.849): the model's B mismatch carries the first AOD's
offset through `−2γ f_tpm (f_A − f_tpm)`, which is ~40× more sensitive than the B channel's
own `(f_B − f_tpm)²`, so a live A rung a quarter ladder off `f_tpm` costs the B crystal most.
This is the model (γ is a user-set stand-in, README "advanced engine" table), not the drive;
the tutorial's caption says "dips where the live rungs sit away from the TPM frequency but
never collapses", which is accurate. See open issue O3 in `reports/R2-verify.md`.

## R2.4 F4 and the movie — browser checks (`e2e.mjs`, new in round 2)

**Rendered motion** through `window.__aodl` (`physics.spotsAt` just after frame *i* and just
before frame *i* + 1 — one spot list per interval, so spot *k* is the same rendered spot;
`physics.pairFrames` for the pairing and its rigid shift; sites by `physics.frameSites`):

| movie | frame interval | intervals | max rendered step per interval | rigid shift | bright cross-fades (≥ 20 % of frame max) | traps per frame (sites ≥ ¼ of the brightest) |
|---|---|---|---|---|---|---|
| `array_lift_hold`, hold `[85.8, 585.8]` µs | 8.50 µs | 57 | **0.0000 w0** (≤ 0.5 required) | 0 | 1011 in place, of which 243 switching on / 227 off with no spot within 0.5·w0 in the other frame (the hard-switched edge columns) | 16 (16 frames), 20 (16), 24 (12), 25–34 (13) — **never fewer than 16** |
| `pick_and_place` | 1.92 µs | 75 | 1.124 w0 (a real carry, interpolated) | ≤ 1.124 w0 | **0** | 1 |
| `array_arbitrary_path` | 5.21 µs | 77 | 0.961 w0 | ≤ 0.961 w0 | **0** | 9 |

The 4×4's `bounds_um` now include one extra pitch per side on each fading array axis
(`reports/R2-verify.md` D6): before, the M + 2 hand-over columns sat 10.3 / 13.4 µm outside the
requested array but only 6 w0 = 6.4 µm of margin was drawn, so the movie never showed them and
the badge counted them as "36 shadow tweezers off canvas". Every spot of every 4×4 frame is now
inside the canvas (backend test `test_fading_array_bounds_include_the_extended_columns`).

So during the 4×4 hold no rendered trap moves at all between frames (the requested sites do
not move; light switching at the edge columns cross-fades in place), and both carrying movies
glide with zero cross-fades — R2-frontend-a's rigid-shift addition does what it claims, and
the 2·w0 cap is not reached (largest real step 1.12 w0).

**Tutorial vs workbench** (R2-frontend-b's wrappers): the tutorial page's two `/api/simulate`
bodies (captured from the network) are **byte-identical** (`JSON.stringify`) to the
workbench's for `/basic?example=tutorial_hold_plain` and `…_shepard`; at the tutorial's start
cursor (first x hand-over + τ/2 = 159.369 µs) both render "brightest P/P_ref 1.00 · z 10.00 µm
· 1 spot" (plain) and "brightest P/P_ref 0.53 · z 10.00 µm · 4 spots" (Shepard), and the
tutorial shows "2 shadow tweezers off canvas at ±123.6 µm (x hand-over)". `/tutorial`
redirects to `/tutorial/shepard`. Tracks: both RF panels use `useRfTracks` on the same plan.

**Intensity panel** (`intensity_panel.png`, 4×4 at the hand-over t = 341.9 µs): 36 trap
tracks, line cuts with exact peak 0.868, Gaussian 0.977 (exact/Gaussian 0.888, order-3
metrics) — the Gaussian approximation ignores the interference of the IM3 terms that land on
the trap; static single tweezer exact/Gaussian = 1.0000 (R2-backend, re-checked by the
existing backend test).

**Playback** (rAF over 2 s, headless software rendering): 4×4 **47.1 fps** (100 spots/frame),
flagship **30.0 fps** (484 spots at the cursor). On arrival they were 32.7 and **20.6 fps**
(flagship below the 30-fps gate): the fix halves the drawn spots — exact coincident duplicates
(the two rung pairs of a hand-over) are merged before drawing, and `renderXY` reuses its
accumulators and `IsoView` stops allocating per ellipse point (`reports/R2-verify.md` D5).

**Pages**: `/`, `/basic?example=` pick_and_place / single_lift_hold / array_lift_hold /
array_arbitrary_path / tutorial_hold_plain / tutorial_hold_shepard / flagship_10x10,
`/tutorial/shepard`, `/advanced?example=pick_and_place`, served SPA — 0 console errors,
0 page errors, 0 failed requests. Two `/api/frame` line-cut fetches were cancelled by the app
itself (a newer cursor time superseded them); `e2e.mjs` now lists `net::ERR_ABORTED` as
`abortedRequests` instead of failing on it. Time to movie: pick_and_place 1.2 s,
single_lift_hold 1.7 s, **array_lift_hold 7.6 s** (`/api/simulate` 5.9 s / 300 kB — the 4×4's
first lit frame now holds 152 groups at order 3, under the 160 budget, so its movie runs at
order 3; it was 165 → order 1 before the fix), flagship 4.4 s, tutorial 1.0 s.

## R2.5 Screenshots (`webapp/docs/screenshots/`, all regenerated by `e2e.mjs`)

`landing.png` (corrected diagram) · `basic_array_lift_hold.png` (4×4 at a mid-hold x
hand-over + τ/2: the 16 requested traps plus the switched edge columns, now on the canvas) ·
`intensity_panel.png` (new) · `tutorial.png` (new) · `basic_pick_and_place.png` ·
`basic_single_lift_hold_fade.png` · `basic_array_arbitrary_path.png` ·
`advanced_efficiency.png` · `basic_flagship_10x10.png`.
