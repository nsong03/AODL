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
   at `Z = z`.) `P = SpotMetrics.power` puts the approximation on the **same absolute intensity
   scale** as `intensity_frame` (`measure.py`: an unwindowed term's power is `peak·wx·wy·π/2`),
   so the approximation and the exact frame are directly comparable. Groups add in intensity
   (they are distinct optical frequencies — `docs/PLAN.md` §1.3). **One spot per spatial cluster,
   not per frequency group**: a frequency group can hold spatially separate degenerate terms —
   the Eq. S31 shadow-tweezer pair at `±deflection_scale·Δf` during a fade, or the two array
   traps two pitches apart of `docs/guide.md` §6.7 — and `measure()` would report their
   power-weighted centroid (a phantom spot on the main trap carrying the shadows' power). The
   backend therefore splits every group by the term centres `theta1·F/k` (clusters within
   `1 w0`) and measures each cluster separately; spatially separated degenerate terms have
   negligible pupil overlap, so adding the clusters in intensity is exact to that overlap. Hue
   encodes `z` exactly as `aodl.viz.style` does (blue below, white at 0, red above; luminance =
   intensity). The approximation drops the fill-edge truncation, irising and intra-cluster
   interference; the **"exact" frame** button fetches the closed-form frame from the backend so
   the two can be compared side by side. The frontend must label the view "astigmatic Gaussian
   approximation" and the exact view "closed form (Eq. S11)".
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

   With `n ≈ 2.26`, `v = 650 m/s`, `λ = 1030 nm`, `L = 4 mm`, `η_peak = 0.85` this gives a 3 dB
   full bandwidth of ≈ 28 MHz (27.9 MHz; `Γ = asin√η_peak = 1.17` narrows the sinc slightly
   relative to the small-Γ 28.7) — the right order for a DTSXY-400. **Do not** use the
   `C₂ ∝ n_o³/(v⁴ (n_e² − n_o²))` expression from
   `integratedphotonicsinvestigation/AODLsimulator/aodl_sim/diffraction.py`; it is off by
   ~10⁴ and gives GHz bandwidths. Parameters exposed: `f_tpm` (default = `f_center`; it is a
   *crystal-cut* parameter — pure [110] propagation would put it at 511 nm-equivalent
   frequencies far above the band, commercial cuts place it near band centre), interaction
   length `L`, refractive index `n`, peak efficiency `η_peak = sin²Γ` (sets `Γ`). Caveats for
   the docstring and the UI: the plane-wave single-`L` model omits acoustic walk-off and
   divergence (which widen the angular acceptance), optical activity near [001] and the TeO₂
   mid-band degeneracy dip; the measured maps of the paper's Fig. S1c/S8 are the calibration
   target.

   **The second AOD of a pair (B) sees an incidence angle shifted by the first AOD's
   deflection, and that shift is an additive mismatch, not a frequency shift.** At tangential
   phase matching `θ_i'(f_tpm) = 0`, so nothing about `f_B` can compensate an incidence change
   to first order. With `K_B = 2π f_B / v`, the internal deflection of A
   `Δθ_A = λ (f_A − f_tpm) / (n v)`, and an effective angular coupling `γ`:

   ```
   δ_B(f_B; f_A) = (L/2) K_B [θ_in − θ_i(f_B)]
                 = C₂ [ (f_B − f_tpm)² − 2 γ f_tpm (f_A − f_tpm) ],      C₂ = π λ L / (2 n v²)
   ```

   Consequences, which are exactly the paper's "Design caveats → limited first-AOD bandwidth":
   at fixed `f_B` the pair efficiency falls off in `f_A` on a scale `∝ 1/(L γ)` (1 MHz of A
   detuning is one full B half-width at `L = 4 mm`, `γ = 1`), while in `f_B` it keeps the
   TPM-broadened width — **the first AOD's usable band is narrow because its deflection
   detunes the second**; nothing is compensated. The map is asymmetric in `f_A`: on the side
   where `γ (f_A − f_tpm) > 0` the incidence angle rises above the Bragg-curve minimum and B
   re-matches on the parabolic ridge `(f_B − f_tpm)² = 2 γ f_tpm (f_A − f_tpm)` (the two matched
   frequencies of anisotropic diffraction), on the other side it is cut off. `γ ∈ [−1, 1]` is
   the signed fraction of A's deflection that reaches B's Bragg plane: `|γ| = 1` for an ideal
   4f relay of magnification −1 and plane acoustic waves; smaller values stand in for the
   angular-acceptance widening by acoustic divergence that the single-`L` model lacks. The
   single-`L` model cannot give both B ≈ 30 MHz and A "a few times narrower" at `γ = 1`, so the
   **default is `γ = 0.2`** (A usable ≈ ±5 MHz at `L = 4 mm`); the sign is a geometry
   convention the user may flip. The pair map is `η_pair(f_A, f_B) = η_A(f_A) η_B(f_B; f_A)`.
   Report `first_aod_usable_MHz` — the full width in `f_A` over which `max_{f_B ∈ band}
   η_pair ≥ ½ η_pair,max`, computed numerically from the map (the analytic one-sided value is
   `δ₃ / (C₂ f_tpm |γ|)` with `η(Γ, δ₃) = ½ η(Γ, 0)`) — this is the number the brief's
   "efficiency vs deflection angle of the first pair" is about. Report deflection angle
   `Δθ = λ (f − f_center) / v` [mrad] alongside frequency on every efficiency plot.
   *Trajectory efficiency*: at each frame, per channel, the amplitude²-weighted mean of
   `η(f_center + f_n(t_c))` over the live tones (for a B channel at the A channel's live
   frequency — exact for Eq. S19, amplitude²-weighted over the A ladder for Shepard); the
   product over the four channels is the predicted relative trap power. This is a *model* with
   user-set parameters, not a measurement — the UI must say so.
7. **Misalignment model** (`aodl_advanced.pupil`): the pupil of AOD `μ` is evaluated on a 2D
   grid `r = (x, y)` at the retarded time of its own, possibly misaligned, sound axis. With the
   channel's axis unit vector `ê_μ` (x or y, from `conventions`), its perpendicular `ê_⊥`, the
   aperture offset `d_μ = δ_∥ ê_μ + δ_⊥ ê_⊥`, the sound-axis rotation `θ_μ` about the optical
   axis, the 4f magnification mismatch `m_μ` and the acoustic delay error `δt_μ` (Eq. S29):

   ```
   r'    = R(−θ_μ) (r − d_μ),        u_μ = r' · ê_μ
   v_μ   = m_μ v
   t_ret = t − τ/2 − s_μ u_μ / v_μ − δt_μ                       (Eq. S4 + Eq. S29)
   fill  : s_μ u_μ ≤ m_μ [ v (t − δt_μ) − D/2 ]                 (conventions.is_filled, scaled)
   E_μ(t) = Σ_n a_n A_n(t) e^{i (phase_n(t) + phase0_n)}        (complex analytic drive, rotating
                                                                frame, carrier dropped)
   T_μ   = i J₁(C |E_μ(t_ret)|) · E_μ*(t_ret) / |E_μ(t_ret)|     ("j1": the +1 order of the thin
                                                                phase grating exp(iCV) in the
                                                                quasi-static limit, all orders in
                                                                C; → (iC/2) E* as J₁(x) → x/2,
                                                                i.e. aodl's (iC/2) A e^{−i phase})
         = (iC/2) E_μ*(t_ret)                                   ("weak": Eq. S3, aodl's model)
   ```

   `a_n = √(η(f)/η_peak)` is the *relative* Bragg-efficiency amplitude of tone `n` when an
   `EfficiencyModel` is attached (relative, so the aligned centred `weak` case reproduces `aodl`
   exactly and the Raman–Nath-type `J₁` saturation and Kogelnik's `sin²Γ` are not
   double-counted); for a B channel `η_B` depends on the A channel's live frequency at `t_c`
   (closure; amplitude²-weighted over a Shepard A ladder). Expanding the `J₁` law to `C³`
   reproduces `aodl.device.mixing`'s IM3 and compression table exactly. Optional pointing tilt
   adds `exp(i k (α_x x + α_y y))`. Under rotation or `m_μ ≠ 1` a channel's carrier tilt
   `s 2π f_c u_μ / v_μ` no longer cancels its partner's; it is common to every term — a static
   pointing offset (~10 µm at 10 mrad) — and is dropped as "alignment", but reported as
   `static_offset_um` per channel. The four factors multiply, times the input Gaussian
   `exp(−(x²+y²)/w_in²)` (uncropped, as in `aodl`), and the field at lab plane `Z` is the
   Eq. S11 integral with `Z_S11 = Z_LAB_SIGN·Z`, evaluated with a 2D chirp-z (zoom) transform
   onto a focal patch. Signs come from `aodl.device.conventions` — never hardcode one.

   **Beats are handled by frequency decomposition, not by time averaging.** Uniform sub-time
   averaging over one period of the smallest spacing cancels only harmonics of that period
   (measured: 19–79 % of the cross terms survive for an (8, 6.5) MHz pair, and the
   0.1 MHz-commensurate 1.0/1.3 MHz arrays need a 10 µs window). Instead each channel's
   transmission is decomposed into *lines*: under `"weak"` the lines are the tones themselves
   (`T_μ = Σ_n (iC/2) a_n A_n e^{−i phase_n}`, one 2D factor per live tone); under `"j1"` the
   channel's tones sit on a comb `f_ref(t_c) + m Δf_μ` (true for every drive `plan_motion`
   produces: Eq. S19 ladders, Shepard ladders, single tones), so `T_μ(u, t_ret(u) + k/(K Δf_μ))`
   is evaluated at `K` sub-times per pixel and a `K`-point DFT along `k` yields the lines at
   `f_ref + m Δf_μ`, `|m| ≤ (K−1)/2` (fundamentals, compression, IM3 …), keeping lines above
   `1e-3` of the strongest (`K = 1` is the frozen, instantaneous pupil). The Cartesian product
   of the four channels' lines is grouped by optical frequency sum (1 kHz tolerance — exactly
   `aodl.field.focal.group_terms`' rule), each group's pupil is the sum of its products, one
   zoom transform per group, and the groups add in intensity. This *is* the beat-averaged
   intensity, with the same approximation `aodl` makes. Groups are capped (default 128; beyond
   that the brightest are kept and a note says so); the XZ scan uses `nz ≤ 32` and only the
   brightest ≤ 6 groups; the aligned reference frame is cached per
   `(plan_id, time, grid, model)` since it does not depend on the misalignment.

8. **Units**: SI inside both backend packages; the JSON API uses the *display* units named in
   every field (`_um`, `_us`, `_MHz`, `_mm`, `_nm`, `_mrad`, `_ns`) and nothing else. The
   frontend converts to SI for the physics module (`physics/units.ts`).

### 1.1 What is approximate in the movie, stated for users

The basic movie is exact for spot *positions*, *best-focus Z*, *astigmatic interval* and
*power* (all closed form, Table I; shadow tweezers appear as their own spots at the right
places with the right power), and approximate for the intensity *shape*: it ignores the
aperture fill edge (so the first `τ` looks like a dim Gaussian rather than a truncated one),
acoustic irising, and interference between co-located degenerate terms (the simultaneous-fading
Mach–Zehnder case). The exact frame exists to show the difference. The advanced engine adds the fill edge, irising,
compression/IM3 (through the `J₁` law), misalignment, and efficiency roll-off.

### 1.2 Performance budget (measured on the dev container, `aodl` 0.1.0, numpy 2.5)

| call | 1 tweezer | 4×4 fading | 10×10 fading, mixing 3 |
|---|---|---|---|
| `plan_motion` | 6 ms | 0.2 s | 0.2 s |
| `simulate`, 40 frames | 70 ms | ~1.5 s (30 fr) | 2.3 s (24 fr) |
| exact XY frame 192 px | 2 ms | 0.13 s | 0.7 s |

Rules: the backend caches `MotionPlan` and `SimResult` by request hash (LRU, ≥ 32 entries);
`simulate` caps `n_frames` at 240; the *movie metrics* fall back to `mixing_order=1` when the
busiest of three probed frames (first lit frame, trajectory midpoint, first hand-over) at the
requested order holds more than 80 frequency groups — a cost budget, ~0.5 ms per group and
frame, so the default 80-frame movie stays under ~3 s (a 3×3 at order 3 has 49 — 9 traps and
40 IM3 ghosts — and stays; a fading 4×4 has ~220 in a hand-over and a 10×10 ~460, and both
fall back) unless the user opts in with
`movie_mixing_order=3` (the plan/report always use the requested order); simulate responses
are struct-of-arrays, rounded to 6 significant digits, with spots below `1e-3` of the frame's
maximum power dropped (invisible at gamma 0.7); long routes (`/api/simulate`, `/api/frame`,
`/api/advanced/*`) are plain `def` endpoints so FastAPI runs them in its threadpool; the
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

### 1.4 Even-M fading arrays (round 2)

A fading-Shepard array lights **`M + 2` columns during a hand-over and `M` otherwise, for odd
and even M alike**, and every requested site stays lit (`cos²θ + sin²θ = 1`). For even M this
needs `aodl.waveform.shepard.table_ii` to offset the B ladder by half a rung, `ξ_B = ξ_A + ½`;
the same half rung puts the traps on Eq. S19's half-integer lattice, so the WO-17 frequency-only
`Δf/2` comb (`lattice_comb_offset`) is now `0`. Before the fix an even-M array lit `M + 1`
columns and its requested block walked a pitch through each hand-over. The diagnosis, the
measured before/after table and the consequences for docs and tests are in
[`workorders/round2/R2-DESIGN-ADDENDUM.md`](workorders/round2/R2-DESIGN-ADDENDUM.md) §F2; the
re-measurement through the API is in [`VERIFICATION.md`](VERIFICATION.md) §R2.2. The web layer
re-derives nothing: the movie, the trap-intensity panel and the shadow badge read the spots the
backend returns.

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
  shaded aperture window `[t − τ, t]` (clipped at 0 while `t < τ`), a marker at `t − τ/2`;
  below it the summed envelope and, optionally, the voltage zoom `V(t')` for
  `t' ∈ [t − 0.2 µs, t + 0.2 µs]` on 2001 points (50 samples per 10 ns carrier period) evaluated
  from the IR (carrier included). Keep ≤ 2000 samples per track; recompute tracks only when the
  plan changes.
* **Aperture panel**: four small canvases, the acoustic content on the crystal for
  `u ∈ [−D/2, D/2]` at the cursor time via `t_ret(u) = t − τ/2 − s u / v`: the **envelope**
  `Σ_n A_n(t_ret(u))` as a filled curve coloured by the local instantaneous frequency
  `f_c + Σ A_n f_n / Σ A_n` (the carrier itself — 1150 cycles across 7.5 mm — cannot be drawn
  honestly on a 200 px canvas and is aliased at 512 samples; a zoom toggle draws the literal
  `V(u)` over a 100 µm window at ≥ 4096 samples), zero and greyed where unfilled, transducer
  side labelled, beam (`±w_in`) shaded. This is the "acoustic delay region visible on the AOD
  surface".
* **Tweezer view**: XY canvas (tracked or fixed plane; hue = z), XZ canvas through the brightest
  row, isometric 3D canvas drawing each trap's 1/e² caustic as stacked ellipses over `±2 z_R`
  plus the requested path as a trail; shared colour bar. "Exact frame" fetches `/api/frame` and
  shows it beside the approximation with the same colour scale.
* **Metrics**: requested vs simulated `X, Y, Z` (the simulated curve lags by `τ/2`; a dashed
  "expected at the atoms" copy of the request shifted by `+τ/2` is drawn unless
  `retard_compensate` is on), per-trap power (normalised to `power_ref`), `σ_astig`,
  `w_x, w_y` — all vs time with the cursor. Shadow tweezers that fall outside the canvas are
  announced by a badge driven by `fade_events`.
* **Code panel**: generated Python (with the *resolved* synthesis options — `ShepardConfig`,
  phases, seeded `rng` — and a comment block listing the fade schedule and shadow offsets) with
  copy + "download .npz" + "download .py", and a "waveform recipe" tab tabulating every tone
  from the IR (channel, rung, frequency range, segment count, envelope kind and parameters,
  `phase0`) — the fade-in/fade-out design made explicit.
* **Advanced page**: efficiency panel (single AOD η vs f and vs deflection angle with the band
  marked and `first_aod_usable_MHz` called out; pair heat map `η(f_A, f_B)` with the ridge and
  the usable rectangle; efficiency along the current trajectory and predicted trap-power
  ripple), misalignment panel (per-AOD sliders; run button; XY/XZ exact frames from
  the FFT engine beside the aligned reference; metrics table: centroid shift in `w0`, waists,
  `Z_x`, `Z_y`, Strehl), spectrum panel (PSD per channel of the rendered samples around the
  cursor, crest factor — *if time permits*). Every advanced plot states its model assumptions
  in one line. Priority order if the schedule bites: efficiency → misalignment → spectrum;
  helix generator and advanced XZ focus fits are also optional.

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
  capture one screenshot per page into `webapp/docs/screenshots/`, and check that (a) the XY
  approximation and the exact frame agree on spot centroids to 0.05 w0 for the pick-and-place
  example at `t = 70 µs + τ/2` (mid-translate: `X ≈ 10 µm`, tracked plane `≈ 5 µm`, so both
  centroid and plane are exercised — at `t = 2τ` the check would be trivial), (b) the aperture panel's `V(u)` matches `/api/aperture` to 1e-6, (c) no console
  errors.

## 6. Orchestration

Fable architect (this document, API.md, work orders, integration, git) → Fable *plan review*
agent (physics + architecture critique before any code) → W1 ∥ W2 ∥ W3 implementation agents
(disjoint file ownership, no git) → W4 verification/integration agent (fresh eyes; runs
everything end to end, fixes cross-cutting defects, reports) → architect commits and pushes
`claude/tender-volta-kti6ex`. Loop: any W4 finding that needs a design change comes back here.
