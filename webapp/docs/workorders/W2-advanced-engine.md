# W2 — `aodl_advanced` engine and `/api/advanced` routes

**Role:** implementation agent. **Authority:** this work order; then `../DESIGN.md` §1.5–1.7
and `../API.md` §6 (normative). No git. W1 (backend core) and W3 (frontend) work concurrently
in the same tree; W1's package `aodl_webapp` may not be importable when you start — the engine
package does not depend on it, so build and test `aodl_advanced` first and write the routes
against the frozen W1 signatures listed below, then run the route tests once `aodl_webapp`
imports.

## Review corrections folded in (read these first)
The pre-implementation review (`../reports/REVIEW.md`) changed: (1) the **second-AOD model** —
an incidence shift is an *additive* mismatch, `δ_B = C₂[(f_B − f_tpm)² − 2γ f_tpm (f_A − f_tpm)]`,
default `γ = 0.2`, and `first_aod_usable_MHz` is a required output (DESIGN §1.6 has the full
derivation and narrative — read it, the old "compensating ridge" story was backwards); (2)
**beats are handled by frequency-line decomposition**, not sub-time averaging (DESIGN §1.7);
(3) the misalignment coordinate formula, the magnification-scaled fill mask, the relative
efficiency amplitude `√(η/η_peak)` with the B-channel closure over `f_A`, and `static_offset_um`;
(4) the rotation and velocity-scale tests had wrong expected values (fixed below); (5) the
aligned reference is cached, XZ is capped, routes are sync `def`; (6) `frame_spots` is now in
W1's frozen list; (7) your pytest fixtures live in `tests/_advanced_fixtures.py`, not in W1's
`conftest.py`; your tests may import anything from `aodl` (the import scan covers the package
only). API.md §6 is updated accordingly.

## Owned files
`webapp/backend/src/aodl_advanced/**`, `webapp/backend/src/aodl_webapp/routes/advanced.py`,
`webapp/backend/tests/test_advanced_*.py`, `webapp/backend/tests/test_cross_engine.py`,
`webapp/docs/reports/W2.md`. Create `aodl_advanced/__init__.py` (if W1 left an empty one,
replace its contents — that file is yours).

## 1. Import discipline (enforced by `tests/test_advanced_independence.py`, which you write)
`aodl_advanced` imports from `aodl` only: `aodl.params`, `aodl.units`,
`aodl.device.conventions`, `aodl.waveform.tones`, `aodl.waveform.shepard`,
`aodl.waveform.export` (for the spectrum's `render_samples`), `aodl.poly`. **Never**
`aodl.field`, `aodl.device.aod`, `aodl.device.aodl`, `aodl.device.mixing`, `aodl.engine`,
`aodl.api`, `aodl.viz`, `aodl.check` — a source scan over the package asserts it. The *route*
module lives in `aodl_webapp` and may use anything.

## 2. `efficiency.py` (DESIGN §1.6 — the derivation goes in the module docstring)
```python
@dataclass(frozen=True)
class EfficiencyModel: interaction_length: float; f_tpm: float; refractive_index: float; gamma_peak: float; angular_coupling: float
def bragg_efficiency(gamma: ArrayLike, delta: ArrayLike) -> np.ndarray      # Γ² sin²√(Γ²+δ²)/(Γ²+δ²)
def tpm_mismatch(f: ArrayLike, f_tpm: float, L: float, n: float, v: float, wavelength: float) -> np.ndarray
    # δ = π λ L (f − f_tpm)² / (2 n v²)
def deflection_angle(f: ArrayLike, f_center: float, v: float, wavelength: float) -> np.ndarray  # λ (f − f_c)/v  [rad]
def first_aod_efficiency(f, model, hw) -> np.ndarray
def second_aod_efficiency(f_B, f_A, model, hw) -> np.ndarray
    # B sees incidence θ_in = θ_i(f_tpm) + γ·λ(f_A − f_tpm)/(n v); at TPM θ_i'(f_tpm) = 0 so this is an
    # ADDITIVE mismatch: δ_B = (L/2) K_B [θ_in − θ_i(f_B)] = C₂ [(f_B − f_tpm)² − 2 γ f_tpm (f_A − f_tpm)]
    # (document the derivation and the asymmetry in f_A — DESIGN §1.6)
def pair_map(f_A, f_B, model, hw) -> np.ndarray                       # η_A(f_A) η_B(f_B; f_A), shape (len A, len B)
def bandwidth_3db(model, hw) -> float
def first_aod_usable_width(model, hw, band) -> float                 # full width in f_A where max_{f_B in band} η_pair ≥ ½ its max (numeric)
def trajectory_efficiency(wfs: WaveformSet, model, t: np.ndarray) -> dict[str, np.ndarray]
    # per channel: Σ_n A_n² η(f_c + f_n(t − τ/2)) / Σ_n A_n² over live tones; "product" key too
```
Validation: `f_tpm` default `f_center`; `L > 0`; `n > 1`; `gamma_peak = asin(√η_peak)`.

## 3. The FFT pupil engine
- `misalignment.py`: `AODMisalignment(offset_par, offset_perp, rotation, delay, velocity_scale, tilt_x, tilt_y)` (SI), `Misalignment = dict[str, AODMisalignment]` with `.aligned()`.
- `pupil.py`:
  ```python
  def pupil_grid(w_in: float, n: int, span_w_in: float) -> tuple[x1d, y1d]      # ±span·w_in
  def channel_coordinate(X, Y, geom: ChannelGeometry, mis: AODMisalignment) -> u  # r' = R(−θ)(r − d), u = r'·ê_axis, d = δ_par ê_axis + δ_perp ê_perp
  def retarded_time(t, u, geom, aod, mis) -> t_ret                                # t − τ/2 − s u / (v·scale) − delay   (conventions.beam_center_time + Eq. S29)
  def filled(u, t, geom, aod, mis) -> mask                                        # s u ≤ scale·[v (t − delay) − D/2]  (D scales with the magnification too)
  def complex_drive(cw: ChannelWaveform, t_ret, rel_amp: Callable[[f], a] | None) -> E  # Σ_n a_n A_n(t_ret) e^{i(phase_n + phase0)}, a_n = √(η(f_c+f_n)/η_peak) or 1
  def transmission(E, C, order_model) -> T   # "j1": i J1(C|E|) E*/|E| (0 where |E|=0); "weak": (iC/2) E*
  def channel_lines(cw, t, params, geom, aod, mis, model, rel_amp) -> list[Line]   # DESIGN §1.7: "weak" → one 2D factor per live tone tagged f_n(t_c);
      # "j1" → evaluate T_μ at K sub-times t_ret(u) + k/(K Δf_μ), K-point DFT along k, lines tagged f_ref(t_c) + m Δf_μ,
      # keep |line| ≥ 1e-3·max; K = 1 → the frozen pupil as one line tagged with the amplitude²-weighted mean frequency
  def pupil_groups(lines_per_channel, params, max_groups) -> list[Group]           # Cartesian product, group by Σ f within 1 kHz (same rule as
      # aodl.field.focal.group_terms: neighbour chaining + diameter cap), each group's pupil = Gaussian · Σ products · tilt phases;
      # keep the brightest max_groups groups (report how many were dropped)
  def static_offset(params, mis) -> dict[str, tuple[float, float]]               # the dropped common carrier-tilt pointing offset per channel [m]
  ```
  The comb of a channel for `"j1"`: round `(f_n(t_c) − f_0(t_c)) / Δf_μ` to integers `m_n` with
  `Δf_μ` the channel's smallest tone spacing (`plan_motion` drives always sit on one comb:
  S19 ladders, Shepard ladders, single tones); if the residual exceeds 1 kHz fall back to
  `"weak"` for that channel and note it.
  Skip tones whose envelope is identically zero over the grid's `t_ret` range (check `A` at
  `min/max t_ret`; for `FadeZoneEnvelope` use `is_active` plus a coarse 64-point scan).
- `propagate.py`: `zoom_dft(P, dx, dy, X_out, Y_out, k_over_F)` — the Eq. S11 kernel
  `exp(−i k (xX + yY)/F)` via `scipy.signal.czt` along each axis (derive `w`, `a` from the
  output grid; test against a direct `einsum` DFT on a 32×32 pupil to 1e-10 relative);
  `defocus_phase(x, y, z_lab, k, F)` = `exp(−i k Z_S11 (x²+y²)/(2F²))` with
  `Z_S11 = conventions.z_s11_from_lab(z_lab)`; `focal_intensity(P, …, z_lab)`;
  `xz_slice(P, …, x_out, z_out, y_row)` (czt along x for all rows, then a 1-point czt along y);
  `group_intensity(groups, …, z_lab)` = Σ_groups |zoom_dft(group pupil · defocus)|² — this *is*
  the beat-averaged intensity (DESIGN §1.7); no sub-time averaging anywhere. XZ (and the YZ scan
  for `z_best_y`) only for the brightest ≤ 6 groups, `nz ≤ 32`.
- `metrics.py`: centroid, second-moment radii (`2σ`), peak, power (`Σ I dx dy`); `best_focus`
  from the `w(Z)` minimum (parabolic refinement) on XZ and YZ scans; `AdvMetrics` dataclass.
- `spectrum.py`: Hann periodogram of `render_samples` over the window; crest factor; PAPR;
  `peak_depth_rad = drive_strength · max|V·scale| / A_single_tone` (`return_scale=True`).
- `engine.py`: `simulate_frames(wfs, params, times, planes, centres, mis, grid, xz, model,
  reference=True)` → per-frame dict with the images, `n_groups`, `groups_dropped`, metrics and
  the aligned reference metrics (`Misalignment.aligned()`), plus `static_offset_um`; the
  tracked plane and grid centre are **passed in** by the route (W1's `simulation.tracked_plane`
  and `simulation.frame_spots`), so the engine never imports `aodl.engine`. Cache the aligned
  reference per `(plan_id, time, grid, xz, model)` (an LRU in the route module keyed by the
  request's canonical JSON minus `misalignment`).
- `schemas.py`: pydantic models of API.md §6 (`EfficiencyModelModel`, `EfficiencyRequest/Response`,
  `TrajectoryEfficiencyRequest/Response`, `AODMisalignmentModel`, `MisalignmentModel`,
  `AdvancedGrid`, `AdvancedXz`, `AdvancedModel`, `AdvancedSimulateRequest/Response`,
  `AdvFrame`, `AdvMetricsModel`, `SpectrumRequest/Response`). Import `PlanRequest`,
  `HardwareSpec`, `PackedImage` from `aodl_webapp.schemas`.

## 4. Routes (`aodl_webapp/routes/advanced.py`, `router = APIRouter(prefix="/advanced")`)
Use W1's frozen helpers: `aodl_webapp.planner.get_plan(req)`, `aodl_webapp.hardware.to_params(hw)`,
`aodl_webapp.simulation.pack_image(arr)`, `aodl_webapp.simulation.tracked_plane(plan, t_s, params)`,
`aodl_webapp.errors.ApiError`. Routes: `efficiency`, `trajectory_efficiency`, `simulate`,
`spectrum` exactly per API.md §6, all plain `def` (FastAPI threadpool). Performance guards
from API.md (times ≤ 24, n ≤ 192, pupil_n ∈ {128,256,512}, xz.nz ≤ 32, comb_dft_points ≤ 64,
max_groups ≤ 256) → 422. Until W1's `aodl_webapp.planner`/`simulation` import, develop the
engine against `aodl.plan_motion` directly in your own fixtures (`tests/_advanced_fixtures.py`).

## 5. Tests (≥ 30)
- `test_advanced_efficiency.py`: `η(f_tpm) = η_peak`; even in `(f − f_tpm)`; 3 dB bandwidth
  scales as `L^{-1/2}` (L, 4L → ratio 2 ± 2 %); with the default model and `L = 4 mm` the
  bandwidth is 27.9 ± 0.5 MHz; `pair_map` with `γ = 0` is separable (`η_A ⊗ η_B`); with `γ = 1`
  the row `f_A = f_tpm + 1 MHz` is matched on the parabolic ridge `(f_B − f_tpm)² = 2 γ f_tpm·1 MHz`
  (its max equals `η_A(f_A)` within 1 %) while the row `f_A = f_tpm − 1 MHz` has `max_{f_B}
  η_pair / η_A ≤ 0.6` (the cut-off side); `first_aod_usable_width` at `γ = 0.2` is within 5 %
  of the analytic one-sided value `δ₃ / (C₂ f_tpm |γ|)` *doubled on the ridge side only if the
  ridge stays inside the band* — simply compute both sides numerically and assert the numeric
  routine matches a brute-force evaluation of the map; `trajectory_efficiency` of a static
  single tweezer at `f_center` is `η_peak⁴`.
- `test_advanced_propagate.py`: `zoom_dft` vs direct DFT; Gaussian pupil → waist `λF/(πw_in)`
  within 0.5 % and Rayleigh behaviour `w(z_R) = √2 w0` within 1 %; defocus sign: a pupil with
  curvature `+π·fdot/v²`… simpler — compare `best_focus` of a pure co-chirp drive (all four
  channels `fdot`) to `aodl`'s Table I `Z = 2·lens_scale·fdot` **through conventions only**
  (`params.lens_scale`), within 0.1 z_R.
- `test_cross_engine.py` (the load-bearing one; this file *may* import `aodl.engine`/`aodl.api`
  because it is a test): `pick_and_place` plan via `aodl.plan_motion`; at `t = 2τ` and
  `t = 70 µs + τ/2` (mid-translate), aligned, `order_model="weak"`, with `xz.nz = 24`: centroid
  within 0.02 w0 of `SpotMetrics.x/y`, waists within 3 % (the coma aodl drops is ~0.03 rad at
  `w_in` here — negligible), `z_best` within 0.1 z_R of `z_lab`, power ratio between the two
  times within 2 % of `aodl`'s; a 2×2 S19 array: four spots found within 0.05 w0 each; a
  `single_lift_hold`-like fading drive at an x hand-over: the group decomposition gives two
  shadow spots at `±deflection_scale·Δf` with ≈ ½ the main spot's power (compare against
  `aodl`'s `build_terms` → per-term power); `order_model="j1"` vs `"weak"` at `C = 0.3`: peak
  ratio `(2J₁(C)/C)²` within 1e-3 (single tone); `"j1"` on a 4-tone Schroeder ladder: the IM3
  ghost one pitch outside the array has the power `aodl` (mixing order 3) gives it within 5 %.
- `test_advanced_misalignment.py`: aligned `Misalignment` reproduces the reference bit-exactly;
  a delay `δt` on `Ax` during a constant chirp `ḟ` moves the spot by
  `−s_Ax · deflection_scale · ḟ · δt` (nothing for a static tone) with the sign fixed by
  `conventions` (derive the expectation from `theta1_contribution`, not by hand); a lateral
  aperture offset of `Ax` by `d` along x with a static tone leaves the position unchanged but at
  `t = 0.6τ` changes the fill edge (power differs from aligned); `velocity_scale = 1.02` on
  **`Bx` only** (scaling both B channels keeps the x and y chirp sums equal and `ΔF = 0`
  exactly) during a 10 µm co-chirp gives `ΔF = lens_scale · ḟ · (1/1.02² − 1) ≈ −0.039 ·
  lens_scale · ḟ` (≈ −0.19 µm) within 15 % and an X scale error of `(1/1.02 − 1)/2 ≈ −0.98 %`
  (only one pair member is scaled) within 10 % of itself; rotation of `Bx` by `θ = 10 mrad` for
  an S19 trap at rest at `X = 20 µm` (`f_Bx = −f_Ax = Δ/2`) moves the spot in Y by
  `tan θ · deflection_scale · f_Bx(t_c) = tan θ · X / 2` (≈ 0.1 µm) within 10 %, with
  `static_offset_um` reporting the dropped common carrier tilt; the reference cache returns
  the same object for two requests differing only in `misalignment`.
- `test_advanced_spectrum.py`: single tone → one line at `f_center + f`, crest `√2` ± 1 %;
  8-tone zero-phase ladder crest `≈ √16 = 4` vs Schroeder `< 2.3`.
- `test_advanced_api.py`: each route 200 on `pick_and_place`; guards → 422; shapes per API.md.
- `test_advanced_independence.py`: the import scan.

## 6. Lint/type: `ruff check`, `ruff format --check`, `mypy webapp/backend/src/aodl_advanced` clean.

## 7. Report → `webapp/docs/reports/W2.md`: built, tests, timings (pupil 256, one frame, K = 8),
measured cross-engine residuals, every physics decision you had to make that DESIGN did not
fix (sign choices especially), deviations.
