# W2 — `aodl_advanced` engine and `/api/advanced` routes

**Role:** implementation agent. **Authority:** this work order; then `../DESIGN.md` §1.5–1.7
and `../API.md` §6 (normative). No git. W1 (backend core) and W3 (frontend) work concurrently
in the same tree; W1's package `aodl_webapp` may not be importable when you start — the engine
package does not depend on it, so build and test `aodl_advanced` first and write the routes
against the frozen W1 signatures listed below, then run the route tests once `aodl_webapp`
imports.

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
    # incidence of B shifted by γ·λ(f_A − f_tpm)/v: equivalent frequency offset for B's mismatch is
    # (f_B − f_tpm) − γ (f_A − f_tpm)  ⇒ δ_B = C₂ [(f_B − f_tpm) − γ (f_A − f_tpm)]²   (document why: the
    # anisotropic θ_i(f) is quadratic about f_tpm, so an incidence shift equals a frequency shift of the
    # matched point to first order)
def pair_map(f_A, f_B, model, hw) -> np.ndarray                       # η_A(f_A) η_B(f_B; f_A), shape (len A, len B)
def bandwidth_3db(model, hw) -> float
def trajectory_efficiency(wfs: WaveformSet, model, t: np.ndarray) -> dict[str, np.ndarray]
    # per channel: Σ_n A_n² η(f_c + f_n(t − τ/2)) / Σ_n A_n² over live tones; "product" key too
```
Validation: `f_tpm` default `f_center`; `L > 0`; `n > 1`; `gamma_peak = asin(√η_peak)`.

## 3. The FFT pupil engine
- `misalignment.py`: `AODMisalignment(offset_par, offset_perp, rotation, delay, velocity_scale, tilt_x, tilt_y)` (SI), `Misalignment = dict[str, AODMisalignment]` with `.aligned()`.
- `pupil.py`:
  ```python
  def pupil_grid(w_in: float, n: int, span_w_in: float) -> tuple[x1d, y1d]      # ±span·w_in
  def channel_coordinate(X, Y, geom: ChannelGeometry, mis: AODMisalignment) -> u  # rotate by mis.rotation about the optical axis (the channel's own axis = ê_axis), shift by offsets: u = (ê·R⁻¹(r − d))
  def retarded_time(t, u, geom, aod, mis) -> t_ret                                # t − τ/2 − s u / (v·scale) − delay   (conventions.beam_center_time + Eq. S29)
  def filled(u, t, geom, aod, mis) -> mask                                        # s u ≤ v·scale·(t − delay) − D/2
  def complex_drive(cw: ChannelWaveform, t_ret, efficiency: Callable[[f], η] | None) -> E  # Σ_n A_n(t_ret) √η(f_c+f_n) e^{i(phase_n + phase0)}
  def transmission(E, C, order_model) -> T   # "j1": i J1(C|E|) E*/|E| (0 where |E|=0); "weak": (iC/2) E*
  def pupil_field(wfs, t, params, mis, model) -> P   # Gaussian · Π_channels T_μ · fill masks · tilt phases
  ```
  Skip tones whose envelope is identically zero over the grid's `t_ret` range (check `A` at
  `min/max t_ret`; for `FadeZoneEnvelope` use `is_active` plus a coarse 64-point scan).
- `propagate.py`: `zoom_dft(P, dx, dy, X_out, Y_out, k_over_F)` — the Eq. S11 kernel
  `exp(−i k (xX + yY)/F)` via `scipy.signal.czt` along each axis (derive `w`, `a` from the
  output grid; test against a direct `einsum` DFT on a 32×32 pupil to 1e-10 relative);
  `defocus_phase(x, y, z_lab, k, F)` = `exp(−i k Z_S11 (x²+y²)/(2F²))` with
  `Z_S11 = conventions.z_s11_from_lab(z_lab)`; `focal_intensity(P, …, z_lab)`;
  `xz_slice(P, …, x_out, z_out, y_row)` (czt along x for all rows, then a 1-point czt along y);
  `beat_average(fn, t, K, wfs)` averaging over `K` sub-times across one period of the
  smallest tone spacing present (1/min |f_i − f_j| over all pairs, all channels; cap 2 µs).
- `metrics.py`: centroid, second-moment radii (`2σ`), peak, power (`Σ I dx dy`); `best_focus`
  from the `w(Z)` minimum (parabolic refinement) on XZ and YZ scans; `AdvMetrics` dataclass.
- `spectrum.py`: Hann periodogram of `render_samples` over the window; crest factor; PAPR;
  `peak_depth_rad = drive_strength · max|V·scale| / A_single_tone` (`return_scale=True`).
- `engine.py`: `simulate_frames(wfs, params, times, mis, grid, xz, model, reference=True)`
  → per-frame dict with the images and metrics (and the aligned reference metrics computed with
  `Misalignment.aligned()`); the tracked plane is **passed in** by the route (W1's
  `simulation.tracked_plane`), so the engine never imports `aodl.engine`.
- `schemas.py`: pydantic models of API.md §6 (`EfficiencyModelModel`, `EfficiencyRequest/Response`,
  `TrajectoryEfficiencyRequest/Response`, `AODMisalignmentModel`, `MisalignmentModel`,
  `AdvancedGrid`, `AdvancedXz`, `AdvancedModel`, `AdvancedSimulateRequest/Response`,
  `AdvFrame`, `AdvMetricsModel`, `SpectrumRequest/Response`). Import `PlanRequest`,
  `HardwareSpec`, `PackedImage` from `aodl_webapp.schemas`.

## 4. Routes (`aodl_webapp/routes/advanced.py`, `router = APIRouter(prefix="/advanced")`)
Use W1's frozen helpers: `aodl_webapp.planner.get_plan(req)`, `aodl_webapp.hardware.to_params(hw)`,
`aodl_webapp.simulation.pack_image(arr)`, `aodl_webapp.simulation.tracked_plane(plan, t_s, params)`,
`aodl_webapp.errors.ApiError`. Routes: `efficiency`, `trajectory_efficiency`, `simulate`,
`spectrum` exactly per API.md §6. Performance guards from API.md (times ≤ 24, n ≤ 192,
pupil_n ∈ {128,256,512}, beat_subtimes ≤ 16) → 422.

## 5. Tests (≥ 30)
- `test_advanced_efficiency.py`: `η(f_tpm) = η_peak`; even in `(f − f_tpm)`; 3 dB bandwidth
  scales as `L^{-1/2}` (L, 4L → ratio 2 ± 2 %); with the default model and `L = 4 mm` the
  bandwidth is between 20 and 40 MHz (sanity, documented); `pair_map` with `γ = 1` has its
  ridge along `f_B − f_tpm = f_A − f_tpm` and with `γ = 0` is separable; `trajectory_efficiency`
  of a static single tweezer at `f_center` is `η_peak⁴`.
- `test_advanced_propagate.py`: `zoom_dft` vs direct DFT; Gaussian pupil → waist `λF/(πw_in)`
  within 0.5 % and Rayleigh behaviour `w(z_R) = √2 w0` within 1 %; defocus sign: a pupil with
  curvature `+π·fdot/v²`… simpler — compare `best_focus` of a pure co-chirp drive (all four
  channels `fdot`) to `aodl`'s Table I `Z = 2·lens_scale·fdot` **through conventions only**
  (`params.lens_scale`), within 0.1 z_R.
- `test_cross_engine.py` (the load-bearing one; this file *may* import `aodl.engine`/`aodl.api`
  because it is a test): `pick_and_place` plan via `aodl.plan_motion`; at `t = 2τ` and
  `t = T + τ/2`, aligned, `order_model="weak"`, `beat_subtimes=1`: centroid within 0.02 w0 of
  `SpotMetrics.x/y`, waists within 3 %, `z_best` within 0.1 z_R of `z_lab`, power ratio between
  the two times within 2 % of `aodl`'s; a 2×2 S19 array: four spots found within 0.05 w0 each;
  `order_model="j1"` vs `"weak"` at `C = 0.3`: peak ratio `(2J₁(C)/C)²` within 1e-3 (single tone).
- `test_advanced_misalignment.py`: aligned `Misalignment` reproduces the reference bit-exactly;
  a delay `δt` on `Ax` during a constant chirp `ḟ` moves the spot by
  `deflection_scale · ḟ · δt` with the sign fixed by `conventions` (derive expected from
  `theta1_contribution`, not by hand); a lateral aperture offset of `Ax` by `d` along x with a
  static tone leaves the position unchanged but at `t = 0.6τ` changes the fill edge (power
  differs from aligned); `velocity_scale = 1.02` on the B pair produces a residual `astig_um`
  ≠ 0 during a co-chirp and a lateral scale error of 2 % on X; rotation by 10 mrad of `Bx`
  tilts the X deflection into Y by `tan(10 mrad)` of the deflection (within 5 %).
- `test_advanced_spectrum.py`: single tone → one line at `f_center + f`, crest `√2` ± 1 %;
  8-tone zero-phase ladder crest `≈ √16 = 4` vs Schroeder `< 2.3`.
- `test_advanced_api.py`: each route 200 on `pick_and_place`; guards → 422; shapes per API.md.
- `test_advanced_independence.py`: the import scan.

## 6. Lint/type: `ruff check`, `ruff format --check`, `mypy webapp/backend/src/aodl_advanced` clean.

## 7. Report → `webapp/docs/reports/W2.md`: built, tests, timings (pupil 256, one frame, K = 8),
measured cross-engine residuals, every physics decision you had to make that DESIGN did not
fix (sign choices especially), deviations.
