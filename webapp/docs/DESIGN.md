# AODL Studio — design document

A web application that turns a 3D tweezer trajectory into the four RF drive waveforms of a
3D acousto-optic deflector lens (the four-AOD stack of arXiv:2510.11451), shows the waveforms
and the acoustic window on each crystal as time advances, and plays a lightweight movie of the
tweezers built from the astigmatic-Gaussian metrics of the `aodl` closed-form engine. An
**advanced** mode runs a *separate*, FFT-based pupil engine (`aodl_advanced`) for per-AOD
misalignment, Bragg diffraction-efficiency roll-off, RF spectra and related diagnostics.

Everything here is a **contract**: three implementation work orders (backend, advanced engine,
frontend) run in parallel against it, so a change to any frozen interface below must be made
here first and announced in the work order that depends on it.

Companion documents: [`API.md`](API.md) (the JSON contract, normative),
[`workorders/`](workorders/) (one per agent).

---

## 0. Scope and non-goals

In scope (v1):

* **Basic mode** — trajectory editor (array + moves, incl. arbitrary 3D paths), Eq. S19 /
  fading-Shepard synthesis through `aodl.plan_motion`, report, generated Python code, NPZ
  download, RF panel with time cursor + acoustic-aperture highlight, on-crystal waveform panel,
  tweezer movie (XY / XZ / isometric 3D) from `SpotMetrics` under the astigmatic Gaussian beam
  approximation, "exact" closed-form intensity frames on demand, metric plots.
* **Advanced mode** — `aodl_advanced`: Bragg/TPM diffraction efficiency of the first (A) AODs
  and of the A–B pair, efficiency along a trajectory, FFT pupil propagation with per-AOD
  misalignment (aperture offset, sound-axis rotation, acoustic delay mismatch — Eq. S29 —,
  4f magnification mismatch, pointing tilt), beat-averaged intensity, Strehl / waist / astigmatism
  fits against the aligned reference, RF spectrum + crest factor of the rendered drive.
* **Landing page** — a high-level SVG diagram of the four-AOD setup, the Table I control
  mapping, and quick links to the starter examples.

Out of scope (v1): atom dynamics, user accounts, persistence beyond an in-memory cache,
multi-user job queues, GPU, mp4 export (the `aodl` package already does that offline — the
generated code shows how).

## 1. Physics choices (binding)

1. **Basic mode uses `aodl` unchanged** as the physics authority: `plan_motion` → `MotionPlan`
   (synthesis, band checks, Shepard auto-fallback, report), `simulate` → `SimResult` /
   `SpotMetrics`, `intensity_frame` / `intensity_slice_xz` for exact frames,
   `device.aod.aperture_window` for the on-crystal waveform. No physics is re-derived in the
   web layer. The one core addition is a `Path` move (§1.3).
2. **The movie is the astigmatic Gaussian beam approximation**, rendered client-side from
   per-frame `SpotMetrics`. For a group with centre `(x, y)`, best focus `z`, astigmatic
   interval `ΔF = Z_x − Z_y`, power `P`, and the hardware constants `w0`, `z_R`:

   ```
   Z_x = z + ΔF/2,        Z_y = z − ΔF/2                    (Table I: Zbar, ΔF)
   w_x(Z) = w0 sqrt(1 + ((Z − Z_x)/z_R)^2),   w_y(Z) likewise with Z_y
   I(X, Y; Z) = P · 2/(π w_x(Z) w_y(Z)) · exp(−2 (X−x)^2/w_x^2 − 2 (Y−y)^2/w_y^2)
   ```

   (`aodl.field.focal.spot_params` is the same formula; `SpotMetrics.wx/wy` are these radii
   at `Z = z`.) Groups add in intensity (they are distinct optical frequencies — `docs/PLAN.md`
   §1.3). Hue encodes `z` exactly as `aodl.viz.style` does (blue below, white at 0, red above;
   luminance = intensity). The approximation drops the fill-edge truncation, irising and
   intra-group interference; the **"exact" frame** button fetches the closed-form frame from the
   backend so the two can be compared side by side. The frontend must label the view
   "astigmatic Gaussian approximation" and the exact view "closed form (Eq. S11)".
3. **Retardation is shown, not hidden.** A frame at observation time `t` is driven by the
   waveform at `t_c = t − τ/2`; the acoustic content inside the aperture at time `t` is the
   drive segment `[t − τ, t]` (`docs/conventions.md` §7). The RF panel therefore shades
   `[t − τ, t]` ("inside the crystal"), marks `t − τ/2` ("beam centre"), and the simulated
   trajectory plot overlays the *requested* `X(t), Y(t), Z(t)` against the simulated spot so the
   `τ/2` lag is visible. Before `τ/2` a pair-driven tweezer is dark; the movie must show that.
4. **Fade envelopes are the package's.** The frontend evaluates the exact parametric envelopes
   (`ConstantEnvelope`, `SmoothOnOff`, `FadeZoneEnvelope`) from the waveform IR shipped by the
   backend; nothing is re-sampled. `SwitchRamped` is not exposed by the web UI (the
   `switch_ramp` option is not offered) and the IR serializer must refuse it loudly.
5. **Advanced engine is separate code.** `aodl_advanced` may use FFT / chirp-z transforms
   freely (the "no FFT" rule of `CLAUDE.md` scopes `src/aodl`'s simulation path, not this
   package). It imports from `aodl` only `params`, `units`, `device.conventions` (the sign
   authority), `waveform.tones`/`shepard` envelopes for evaluation, and `WaveformSet` as its
   input. It must **not** import `aodl.field`, `aodl.device.aod*`, `aodl.engine` — so that an
   agreement between the two engines at zero misalignment is a genuine cross-check
   (`tests/test_cross_engine.py`).
6. **Diffraction efficiency model** (`aodl_advanced.efficiency`): coupled-wave Bragg efficiency
   `η = Γ² sin²(√(Γ² + δ²)) / (Γ² + δ²)` with `δ = Δk L / 2` (Kogelnik; Xu & Stroud ch. 3).
   For a TeO₂ slow-shear deflector at tangential phase matching the first-order mismatch
   vanishes and the leading term is quadratic in the frequency offset. Derivation to put in the
   docstring: the anisotropic Bragg incidence angle is
   `θ_i(f) = λ f /(2 n v) + v (n_i² − n_d²) / (2 n λ f)`, stationary at
   `f_0 = (v/λ) sqrt(n_i² − n_d²)` (the TPM frequency, a design parameter of the crystal cut);
   at fixed incidence `θ_i(f_0)` the mismatch is `Δk ≈ K [θ_i(f) − θ_i(f_0)]` with
   `K = 2π f / v`, and `θ_i''(f_0) = λ /(n v f_0)`, hence

   ```
   δ(f) = Δk L / 2 ≈ (π λ L / (2 n v²)) · (f − f_0)²        [rad]
   ```

   With `n ≈ 2.26`, `v = 650 m/s`, `λ = 1030 nm`, `L = 4 mm` this gives a 3 dB full bandwidth of
   ≈ 29 MHz — the right order for a DTSXY-400. **Do not** use the `C₂ ∝ n_o³/(v⁴ (n_e² − n_o²))`
   expression from `integratedphotonicsinvestigation/AODLsimulator/aodl_sim/diffraction.py`;
   it is dimensionally off by ~10⁴ and gives GHz bandwidths. Parameters exposed: `f_tpm`
   (default = `f_center`), interaction length `L`, refractive index `n`, peak efficiency
   `η_peak = sin²Γ` (sets `Γ`). The second AOD of a pair (B) sees an incidence angle shifted by
   the first AOD's deflection: `θ_i,B = θ_design,B + γ · λ (f_A − f_0) / v` with the angular
   coupling `γ ∈ [−1, 1]` (default `+1`: the 4f relay of magnification −1 inverts the ray, and
   the counter-propagating B grating is matched so that A's deflection *compensates* B's Bragg
   drift — this is what produces the paper's wider B-band ridge in Fig. S1/S8). The pair map
   is `η_pair(f_A, f_B) = η_A(f_A) η_B(f_B; f_A)`. Report deflection angle
   `Δθ = λ (f − f_center) / v` [mrad] alongside frequency on every efficiency plot.
   *Trajectory efficiency*: at each frame, per channel, the amplitude-weighted mean of
   `η(f_center + f_n(t_c))` over the live tones; the product over the four channels is the
   predicted relative trap power. This is a *model* with user-set parameters, not a measurement
   — the UI must say so.
7. **Misalignment model** (`aodl_advanced.pupil`): the pupil of AOD `μ` is evaluated on a 2D
   grid `(x, y)` at the retarded time of its own, possibly misaligned, sound axis:

   ```
   u_μ = cos θ_μ (x − δ_∥,μ ê_μ·…)        — in words: rotate the grid by the sound-axis
                                             rotation θ_μ about the optical axis, then shift by
                                             the aperture offset (δ_∥ along the sound axis,
                                             δ_⊥ across it)
   t_ret,μ(x, y) = t − τ/2 − s_μ u_μ / v_μ − δt_μ          (Eq. S4 + Eq. S29 delay error δt_μ)
   v_μ = v · m_μ                                            (4f magnification mismatch m_μ)
   T_μ = i J₁(C |E_μ(t_ret)|) · E_μ*(t_ret) / |E_μ(t_ret)|   (exact first-order envelope law;
                                                            reduces to (iC/2) A e^{−i phase}
                                                            as J₁(x) → x/2)
   ```

   where `E_μ(t) = Σ_n A_n(t) e^{i phase_n(t)}` is the complex analytic drive of the channel
   (rotating frame, carrier dropped). Fill masks (`s_μ u_μ ≤ v_μ (t − δt_μ) − D/2`) apply as in
   `aodl.device.conventions.is_filled`. Optional pointing tilt adds `exp(i k (α_x x + α_y y))`.
   The four factors multiply, times the input Gaussian `exp(−(x²+y²)/w_in²)` (uncropped, as in
   `aodl`), and the field at lab plane `Z` is the Eq. S11 integral with `Z_S11 = Z_LAB_SIGN·Z`,
   evaluated with a 2D chirp-z (zoom) transform onto a focal patch. Beat averaging: the
   intensity is averaged over `K` sub-times spread uniformly over one period of the smallest
   tone spacing present (default `K = 8`; `K = 1` = instantaneous). Signs come from
   `aodl.device.conventions` — never hardcode one.
8. **Units**: SI inside both backend packages; the JSON API uses the *display* units named in
   every field (`_um`, `_us`, `_MHz`, `_mm`, `_nm`, `_mrad`, `_ns`) and nothing else. The
   frontend converts to SI for the physics module (`physics/units.ts`).

### 1.1 What is approximate in the movie, stated for users

The basic movie is exact for spot *positions*, *best-focus Z*, *astigmatic interval* and
*power* (all closed form, Table I), and approximate for the intensity *shape*: it ignores the
aperture fill edge (so the first `τ` looks like a dim Gaussian rather than a truncated one),
acoustic irising, and interference inside a frequency group (shadow-tweezer pairs). The exact
frame exists to show the difference. The advanced engine adds the fill edge, irising,
compression/IM3 (through the `J₁` law), misalignment, and efficiency roll-off.

### 1.2 Performance budget (measured on the dev container, `aodl` 0.1.0, numpy 2.5)

| call | 1 tweezer | 4×4 fading | 10×10 fading, mixing 3 |
|---|---|---|---|
| `plan_motion` | 6 ms | 0.2 s | 0.2 s |
| `simulate`, 40 frames | 70 ms | ~1.5 s (30 fr) | 2.3 s (24 fr) |
| exact XY frame 192 px | 2 ms | 0.13 s | 0.7 s |

Rules: the backend caches `MotionPlan` and `SimResult` by request hash (LRU, ≥ 32 entries);
`simulate` caps `n_frames` at 240; arrays above 64 traps default to `mixing_order=1` for the
*movie metrics* unless the user opts in (the plan/report always use the requested order); the
frontend renders the movie at 60 fps from cached metrics with no network traffic during
playback; exact frames are fetched only on demand.

### 1.3 The `Path` move (core addition to `aodl`)

`aodl.trajectory.spec.Path(points, duration, closed=False)` — a smooth 3D waypoint path:
`points` are `(dx, dy, dz)` displacements **relative to the position where the path starts**
(the implicit first waypoint is `(0, 0, 0)`), `duration` the total time, `closed=True` appends
a return to the start. Knot times are allocated proportionally to chord length; the position
law is a **clamped cubic spline** (`scipy.interpolate.CubicSpline(..., bc_type="clamped")`):
zero velocity at both ends (rest-to-rest, so it chains with any other move with a continuous
frequency law), C² at interior knots, C¹ at its two ends (acceleration steps there, exactly as
the `constant_jerk` profile does). Each spline piece becomes one degree-3 segment per axis in
normalized local time (`c_j^norm = c_j T_seg^j`). A single point is a straight clamped-cubic
move. Degenerate (zero-length) chords are rejected with a clear error. Exported from `aodl`,
documented in `docs/guide.md` §3 with a one-paragraph note, tested in `tests/test_path.py`.
Everything downstream — `synthesize`, band checks (`_poly_extrema` works on any polynomial),
`shepard="auto"`, `plan_motion`, `check_times` — works unchanged because they only use
`spec.compile()`, `spec.duration`, `spec.array` and `move.duration`.

## 2. Repository layout

```
AODL/
├── src/aodl/trajectory/spec.py      # + Path (W1)            tests/test_path.py (W1)
└── webapp/
    ├── README.md                    # how to run, screenshots, architecture summary
    ├── Makefile                     # install / dev / test / lint / build / fixtures
    ├── docs/                        # this file, API.md, workorders/
    ├── scripts/gen_frontend_fixtures.py   # golden JSON for the TS evaluator tests (W1)
    ├── backend/
    │   ├── pyproject.toml           # project "aodl-studio"; packages aodl_webapp, aodl_advanced
    │   ├── src/aodl_webapp/
    │   │   ├── __init__.py  main.py (create_app, static mount)  config.py  cache.py
    │   │   ├── schemas.py           # pydantic v2 models == API.md (W1 owns; W2 adds advanced models in schemas_advanced.py)
    │   │   ├── hardware.py          # HardwareSpec <-> AODLParams, derived constants
    │   │   ├── trajectory.py        # MoveSpec list -> TrajectorySpec (Lift/Translate/Hold/Path), sampled trajectory
    │   │   ├── planner.py           # PlanRequest -> MotionPlan (+ cache), report fields, fade events
    │   │   ├── ir.py                # WaveformSet -> WaveformIR (JSON) + evaluate_ir() reference evaluator
    │   │   ├── simulation.py        # frames (SpotMetrics), exact frame/slice encoding, line cuts
    │   │   ├── aperture.py          # on-crystal V(u) per channel at time t
    │   │   ├── codegen.py           # PlanRequest -> runnable Python using aodl
    │   │   ├── examples.py          # starter trajectories (ids used by the landing page)
    │   │   └── routes/ health.py presets.py plan.py simulate.py frame.py aperture.py advanced.py (W2)
    │   ├── src/aodl_advanced/
    │   │   ├── __init__.py  efficiency.py  misalignment.py  pupil.py  propagate.py
    │   │   ├── metrics.py  spectrum.py  engine.py  schemas.py (pydantic models for /api/advanced)
    │   └── tests/                   # pytest; test_api_*.py (W1), test_advanced_*.py + test_cross_engine.py (W2)
    └── frontend/                    # Vite + React 18 + TypeScript (strict) (W3)
        ├── package.json vite.config.ts tsconfig.json index.html .eslintrc
        ├── public/
        └── src/
            ├── main.tsx App.tsx router.tsx
            ├── api/ client.ts types.ts        # types mirror API.md exactly
            ├── physics/ units.ts piecewisePoly.ts envelope.ts waveform.ts gaussianBeam.ts aperture.ts zColor.ts
            ├── state/ store.ts (zustand)       # plan, sim, time cursor, playback, settings
            ├── components/
            │   ├── landing/ SetupDiagram.tsx (inline SVG) ExampleCards.tsx ModeCards.tsx
            │   ├── editor/ HardwarePanel.tsx ArrayPanel.tsx MoveList.tsx PathGenerators.tsx OptionsPanel.tsx
            │   ├── report/ ReportCard.tsx BandUsage.tsx
            │   ├── rf/ RfPanel.tsx ToneTracks.tsx EnvelopePlot.tsx VoltageZoom.tsx
            │   ├── aperture/ AperturePanel.tsx
            │   ├── tweezer/ XyView.tsx XzView.tsx IsoView.tsx ExactFrame.tsx Playback.tsx
            │   ├── metrics/ TrajectoryPlot.tsx PowerPlot.tsx AstigPlot.tsx
            │   ├── code/ CodePanel.tsx
            │   └── advanced/ EfficiencyPanel.tsx MisalignmentPanel.tsx AdvancedFrames.tsx SpectrumPanel.tsx
            ├── pages/ Landing.tsx Basic.tsx Advanced.tsx
            └── styles/ tokens.css app.css
        └── tests/                   # vitest: physics evaluator vs fixtures, gaussian render, poly eval
```

## 3. Backend

* **FastAPI** app factory `create_app()`; routers under `/api`; CORS open for the Vite dev
  origin; `GET /` serves `frontend/dist` when it exists (production), otherwise a JSON pointer
  to the dev server. Request validation = pydantic v2 (`schemas.py`). Domain errors from `aodl`
  (`ValueError` band refusals etc.) are returned as HTTP 400 with
  `{"error": {"code": "band_violation" | "invalid_spec" | "too_many_terms" | ..., "message": "..."}}`
  (see API.md §0) — never a 500 for a user-caused refusal.
* **Plan identity**: `plan_id = sha1(canonical JSON of PlanRequest)[:16]`. Every later call
  carries the full `PlanRequest` (stateless API) and the backend looks it up in the LRU before
  re-synthesizing. `/api/plan/{plan_id}/waveform.npz` streams `WaveformSet.save` of a cached plan
  (404 if evicted: the client simply re-posts the plan).
* **Waveform IR** (`ir.py`): the parametric drive as JSON — per channel `f_center_Hz`, `tau_s`,
  `sound_sign`, `axis`, and per tone the frequency polynomial, the phase polynomial
  (`freq.antiderivative().scale(2π)`), `phase0`, and the envelope record (§API.md). Also
  `evaluate_ir(ir, t)` — a pure-numpy reference evaluator used by tests and by the fixture
  generator, so the TS port has a single source of truth to match.
* **Trajectory sampling** for plots: `X, Y, Z` from `spec.compile()` on 400 points, plus the
  per-move seam times, so the frontend can draw requested-vs-simulated.
* **Code generation** (`codegen.py`): emits a self-contained script (imports, hardware via
  `dataclasses.replace` only for non-default fields, `ArraySpec`, moves incl. `Path`,
  `plan_motion` with the chosen options, `print(plan.report.summary())`, `plan.save`,
  `plan.render_samples`, `plan.simulate`, commented `plan.movie`). Test: the generated code for
  every starter example compiles and executes in-process (`exec` in a fresh namespace with the
  `.save`/`.movie` lines stripped) and reproduces the same `plan_id`-equivalent band usage.
* **Lint/type**: `ruff` (same config as root: line length 100) and `mypy` on both packages.

## 4. Frontend

* **Stack**: Vite 6, React 18, TypeScript strict, `react-router-dom` 6, `zustand`, `uplot`
  (line plots; ~45 kB). Everything else is hand-written canvas 2D. No Tailwind, no Three.js,
  no Plotly — the brief asks for lightweight and fast. Bundle target < 400 kB gzipped total.
* **Theme**: dark lab theme with CSS custom properties in `styles/tokens.css` (background
  `#0b0d12`, panel `#141821`, text `#d7dde5`, accent per channel using `aodl.viz.style.CHANNEL_COLORS`
  — Ax `#4cc9f0`, Bx `#4361ee`, Ay `#8ac926`, By `#ffb703`). Light mode is **not** required;
  provide `prefers-color-scheme` only for the landing page text contrast. Mobile: single-column
  stack at < 900 px; the workbench is desktop-first.
* **Global time cursor** `t` (µs) in the store; `Playback` drives it with `requestAnimationFrame`
  at a user-set speed (µs of drive time per real second), loops; a slider scrubs it. Every panel
  reads `t` and redraws from cached data — no network during playback.
* **RF panel**: per channel a canvas with tone frequency tracks `f_center + f_n(t)` [MHz] vs
  drive time, line opacity = envelope `A_n(t)` (hidden where `A = 0`), band edges dotted, the
  shaded aperture window `[t − τ, t]`, a marker at `t − τ/2`; below it the summed envelope and,
  optionally, the voltage zoom `V(t')` for `t' ∈ [t − 1 µs, t + 1 µs]` evaluated from the IR
  (carrier included). Keep ≤ 2000 samples per track; recompute tracks only when the plan changes.
* **Aperture panel**: four small canvases, `V(u)` for `u ∈ [−D/2, D/2]` at the cursor time via
  `t_ret(u) = t − τ/2 − s u / v`, zero where unfilled, transducer side labelled, beam
  (`±w_in`) shaded. This is the "acoustic delay region visible on the AOD surface".
* **Tweezer view**: XY canvas (tracked or fixed plane; hue = z), XZ canvas through the brightest
  row, isometric 3D canvas drawing each trap's 1/e² caustic as stacked ellipses over `±2 z_R`
  plus the requested path as a trail; shared colour bar. "Exact frame" fetches `/api/frame` and
  shows it beside the approximation with the same colour scale.
* **Metrics**: requested vs simulated `X, Y, Z` (the simulated curve lags by `τ/2`), per-trap
  power (normalised to the plateau), `σ_astig`, `w_x, w_y` — all vs time with the cursor.
* **Code panel**: generated Python with copy + "download .npz" + "download .py".
* **Advanced page**: efficiency panel (single AOD η vs f and vs deflection angle with the band
  marked; pair heat map `η(f_A, f_B)`; efficiency along the current trajectory and predicted
  trap-power ripple), misalignment panel (per-AOD sliders; run button; XY/XZ exact frames from
  the FFT engine beside the aligned reference; metrics table: centroid shift in `w0`, waists,
  `Z_x`, `Z_y`, Strehl), spectrum panel (PSD per channel of the rendered samples around the
  cursor, crest factor). Every advanced plot states its model assumptions in one line.

## 5. Testing and CI

* Backend: `pytest webapp/backend/tests` — API contract tests through `httpx` `TestClient`
  for every route and every starter example; physics tests listed in the work orders;
  cross-engine test (advanced vs `aodl` at zero misalignment). `ruff check`, `ruff format
  --check`, `mypy`.
* Frontend: `vitest` — `physics/*` against the golden fixtures produced by
  `scripts/gen_frontend_fixtures.py` (tolerance 1e-9 relative on `f`, `A`, `phase`; 1e-6 on
  `V(t)`), Gaussian renderer unit tests (peak at the centre, power normalisation), store logic;
  `tsc --noEmit`, `eslint`. Build must succeed (`vite build`).
* CI: `.github/workflows/webapp.yml` (W4) running both suites; the root CI is untouched except
  that the `Path` tests live under `tests/` and run with it.
* Verification (W4): start both servers, drive the UI with Playwright (pre-installed Chromium),
  capture screenshots into `webapp/docs/screenshots/`, and check that (a) the XY approximation
  and the exact frame agree on spot centroids to 0.05 w0 for the pick-and-place example at
  `t = 2τ`, (b) the aperture panel's `V(u)` matches `/api/aperture` to 1e-6, (c) no console
  errors.

## 6. Orchestration

Fable architect (this document, API.md, work orders, integration, git) → Fable *plan review*
agent (physics + architecture critique before any code) → W1 ∥ W2 ∥ W3 implementation agents
(disjoint file ownership, no git) → W4 verification/integration agent (fresh eyes; runs
everything end to end, fixes cross-cutting defects, reports) → architect commits and pushes
`claude/tender-volta-kti6ex`. Loop: any W4 finding that needs a design change comes back here.
