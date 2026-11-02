# AODL Studio — HTTP API contract (normative)

Base path `/api`. JSON in, JSON out. All numbers are finite floats unless stated. Units are in
the field names: `_um`, `_us`, `_ns`, `_MHz`, `_mm`, `_nm`, `_mrad`, `_m_s`. Angles in mrad.
Arrays are JSON lists of numbers unless a field says `b64u16` (§0.3). The pydantic models in
`aodl_webapp/schemas.py` (and `aodl_advanced/schemas.py` for §6) **are** this document; the
frontend `api/types.ts` mirrors it field for field. Changes go here first.

## 0. Conventions

### 0.1 Errors
Validation failures: HTTP 422 (FastAPI default body). Domain refusals: HTTP 400 with

```json
{"error": {"code": "band_violation", "message": "channel 'Bx' tone 0 leaves its usable band: ...", "detail": null}}
```

`code` ∈ `band_violation` (an `aodl` band refusal), `invalid_spec` (array/move semantics, e.g.
zero spacing on a multi-tone axis, degenerate Path), `too_many_terms` (`aodl` MAX_TERMS),
`plan_not_cached` (404, only for the NPZ download), `coverage` (frame time outside the
programmed drive), `unsupported` (e.g. switch_ramp). `message` is the original `aodl` message
verbatim where one exists — they are written for humans.

### 0.2 Plan identity
`plan_id = sha1(json.dumps(PlanRequest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))[:16]`.
Every route that needs a plan takes the full `PlanRequest` (the API is stateless); the server
caches by `plan_id`.

### 0.3 Packed arrays (`b64u16`)
A 2-D intensity image is `{"shape": [ny, nx], "peak": <float>, "data": "<base64>"}` where `data`
is little-endian `uint16`, row-major (`y` outer, `x` inner, `imshow` convention: row 0 = `y0`),
and `value = peak * code / 65535`. `peak` is in the engine's own (prefactor-dropped) intensity
units; only ratios between images returned from the same call are meaningful.

### 0.4 Channel order
Always `["Ax", "Bx", "Ay", "By"]` (`aodl.params.CHANNELS`). Objects keyed by channel use these
keys.

---

## 1. `GET /api/health` → `{"status": "ok", "aodl_version": "0.1.0", "version": "<studio version>"}`

## 2. `GET /api/presets` → `PresetsResponse`

```json
{
  "hardware_presets": {"default_1030": HardwareSpec, "paper_808": HardwareSpec},
  "profiles": ["min_jerk", "constant_jerk", "constant_accel", "switching_constant_jerk", "linear"],
  "examples": [ExampleSpec, ...]
}
```

`ExampleSpec = {"id": str, "title": str, "blurb": str, "tags": [str], "request": PlanRequest}`.
Required ids (used by the landing page): `pick_and_place`, `square_xy`, `sphere_loop`,
`single_lift_hold`, `array_lift_hold`, `array_arbitrary_path`, `flagship_10x10`.

## 3. Shared request models

### 3.1 `HardwareSpec`
```json
{"wavelength_nm": 1030.0, "focal_length_mm": 6.5, "w_in_mm": 2.0,
 "sound_speed_m_s": 650.0, "aperture_mm": 7.5, "f_center_MHz": 100.0,
 "band_MHz": [90.0, 110.0], "drive_strength": 0.3, "mixing_order": 3}
```
`mixing_order ∈ {1, 3}`; `band_MHz[0] < band_MHz[1]`; all positive. Maps 1:1 onto
`aodl.AODLParams` with identical `AODParams` on the four channels (per-channel bands are a
v2 item).

### 3.2 `ArraySpec`
`{"mx": 1, "my": 1, "delta_f_x_MHz": 1.0, "delta_f_y_MHz": 1.3}` — `mx, my ≥ 1` integers;
a multi-tone axis needs a non-zero spacing (else `invalid_spec`, message from `aodl`).

### 3.3 Moves (`MoveSpec`, discriminated on `kind`)
```json
{"kind": "lift",      "dz_um": 5.0, "duration_us": 40.0, "profile": "min_jerk"}
{"kind": "translate", "dx_um": 20.0, "dy_um": 0.0, "duration_us": 60.0, "profile": "min_jerk"}
{"kind": "hold",      "duration_us": 100.0}
{"kind": "path",      "points_um": [[dx, dy, dz], ...], "duration_us": 200.0, "closed": false}
```
`profile` defaults to `"min_jerk"`; `path.points_um` has ≥ 1 point, relative to the position
where the path starts (DESIGN §1.3). Durations positive.

### 3.4 `PlanOptions`
```json
{"shepard": "auto", "shepard_delta_f_MHz": null, "eta": 0.5,
 "phases": "schroeder", "amp": 1.0, "retard_compensate": false,
 "f_z_bias": "none", "t_pad_us": null, "check_band": true}
```
`shepard ∈ {"auto", "never", "always"}` → `plan_motion(shepard="auto" | None | ShepardConfig)`;
`shepard_delta_f_MHz` (`[dfx, dfy]`) is required when `"always"` (for an array axis it must
equal the array spacing — `aodl` enforces it) and ignored otherwise; `eta ∈ (0, 0.5]`;
`phases ∈ {"schroeder", "zero", "random"}`; `f_z_bias ∈ {"none", "auto"}` → `0.0 | "auto"`;
`t_pad_us: null` = `aodl` default (2τ). `check_band=false` is allowed for plotting infeasible
drives (the report carries the `aodl` note).

### 3.5 `PlanRequest`
`{"hardware": HardwareSpec, "array": ArraySpec, "moves": [MoveSpec, ...], "options": PlanOptions}`
— `moves` non-empty; `options` defaults as in 3.4.

## 4. `POST /api/plan` (`PlanRequest`) → `PlanResponse`

```json
{
  "plan_id": "a1b2c3d4e5f60718",
  "mode": "s19" | "shepard",
  "description": "<WaveformSet.description>",
  "summary_text": "<PlanReport.summary()>",
  "band_usage": {"Ax": {"f_min_MHz": 100.0, "f_max_MHz": 101.46, "margin_MHz": 8.54}, ...},
  "tone_counts": {"Ax": 1, "Bx": 1, "Ay": 1, "By": 1},
  "z_budget": {"requested_m_s": 5e-10, "ceiling_m_s": 2.06e-9, "fraction": 0.243},   // ceiling null when infinite
  "fade_events": [{"time_us": 123.4, "axis": "x", "shadow_um": 10.3}, ...],
  "notes": ["...", ...],
  "constants": {"tau_us": 11.54, "half_tau_us": 5.77, "waist0_um": 1.0655, "rayleigh_um": 3.463,
                "um_per_MHz": 10.3, "lens_scale_m_s": 1.03e-16, "chirp_per_um_MHz_per_ms": 4.854,
                "beam_transit_us": 3.077, "eq1_hold_10um_us": 206.0},
  "t_span_us": [0.0, 163.08],                // programmed drive span incl. hold tail
  "duration_us": 140.0,                      // trajectory duration T
  "seams_us": [0.0, 40.0, 100.0, 140.0],     // move boundaries
  "trajectory": {"t_us": [...400...], "x_um": [...], "y_um": [...], "z_um": [...]},
  "waveform_ir": WaveformIR,
  "code_python": "<runnable script>"
}
```

### 4.1 `WaveformIR`
```json
{
  "channels": {
    "Ax": {
      "f_center_MHz": 100.0, "tau_us": 11.54, "sound_sign": -1, "axis": "x",
      "aperture_mm": 7.5, "sound_speed_m_s": 650.0, "band_MHz": [90.0, 110.0],
      "tones": [
        {"freq": Poly, "phase": Poly, "phase0": 0.0, "env": Envelope}
      ]
    }, "Bx": {...}, "Ay": {...}, "By": {...}
  },
  "t_span_us": [0.0, 163.08]
}
```
`Poly = {"breaks_us": [K+1 floats], "coeffs": [K rows × (D+1) floats]}` evaluated as
`sum_j c[k][j] * tau^j`, `tau = (t − breaks[k]) / (breaks[k+1] − breaks[k])`, **clamp-held
outside the domain** exactly like `aodl.poly.PiecewisePoly.__call__`. Units: `freq` in **MHz**
(detuning from `f_center`), `phase` in **rad** (already `2π ∫ f dt` with `f` in Hz and `t` in
s — the serializer converts; the frontend must *not* re-scale), `phase0` in rad. Total tone
phase `= phase(t) + phase0`; voltage `V(t) = Σ A_n(t) cos(2π f_center t + phase_n(t) + phase0_n)`.

`Envelope` (discriminated on `kind`):
```json
{"kind": "constant", "amp": 1.0}
{"kind": "smooth_on_off", "t_on_us": 0.0, "t_off_us": 100.0, "ramp_us": 20.0}
{"kind": "fade_zone", "g": Poly, "delta_f_MHz": 8.0, "eta": 0.5, "p": 0.5, "m": 1, "amp": 1.0}
```
`fade_zone.g` is the fade coordinate in **MHz** (`aodl.waveform.shepard.FadeZoneEnvelope.g`);
`A = amp` for `|g| ≤ (m − η)Δf/2`, `0` for `|g| ≥ (m + η)Δf/2`, `amp·cos^p θ` between with
`θ = (π/2η)(|g|/Δf − m/2) + π/4`. A `SwitchRamped` envelope → HTTP 400 `unsupported`.

## 5. Basic-mode routes

### 5.1 `POST /api/simulate` (`SimulateRequest`) → `SimulateResponse`
```json
{"plan": PlanRequest, "n_frames": 60, "t_start_us": null, "t_end_us": null,
 "movie_mixing_order": null}
```
Defaults: `t_start_us = 0`, `t_end_us = duration + τ` (never beyond `t_span[1] − τ/2`, which the
server clamps to), `n_frames ∈ [2, 240]`. `movie_mixing_order` (`1 | 3 | null`) overrides the
hardware order for the metrics only (DESIGN §1.2); when `null` and the array has > 64 traps the
server uses 1 and says so in `notes`.
```json
{
  "plan_id": "...", "times_us": [...], "tracked_z_um": [...],
  "frames": [[Spot, ...], ...],          // frames[i] = groups of frame i (may be [] before τ/2)
  "constants": {"waist0_um": ..., "rayleigh_um": ..., "tau_us": ...},
  "bounds_um": {"x0": -3.0, "x1": 25.0, "y0": -8.0, "y1": 8.0, "z0": -2.0, "z1": 8.0},  // spots ± 6 w0 / 1 z_R
  "power_ref": 6.59e-29,                  // median spot power over frames where > 0 (for normalising)
  "notes": ["..."]
}
Spot = {"x_um", "y_um", "z_um", "delta_f_um", "sigma_astig", "wx_um", "wy_um",
        "power", "power_coherent", "df_opt_MHz"}
```

### 5.2 `POST /api/frame` (`FrameRequest`) → `FrameResponse`
```json
{"plan": PlanRequest, "time_us": 30.0, "plane": "tracked" | "fixed" | {"z_um": 2.0},
 "grid": {"x0_um": -5, "x1_um": 25, "nx": 192, "y0_um": -8, "y1_um": 8, "ny": 112},
 "xz": {"nx": 160, "nz": 96, "z_half_um": 8.0, "row_y_um": null} | null,
 "movie_mixing_order": null}
```
```json
{"plan_id": "...", "time_us": 30.0, "plane_z_um": 2.0,
 "xy": {"x_um": [...nx], "y_um": [...ny], "image": b64u16},
 "xz": {"x_um": [...], "z_um": [...], "row_y_um": 0.0, "image": b64u16} | null,
 "cuts": {"x_um": [...], "i_x": [...], "z_um": [...], "i_z": [...], "through": {"x_um":..,"y_um":..}},
 "spots": [Spot, ...]}
```
`nx, ny, nz ≤ 512`. `time_us` must satisfy `aodl.simulate`'s coverage rule (else 400
`coverage`). Cuts go through the brightest spot of the frame.

### 5.3 `POST /api/aperture` (`ApertureRequest`) → `ApertureResponse`
`{"plan": PlanRequest, "time_us": 30.0, "n": 512}` →
`{"time_us": 30.0, "channels": {"Ax": {"u_mm": [...n], "v": [...n], "filled": [bool...n], "t_ret_us": [...n]}, ...}}`
— `aodl.device.aod.aperture_window` verbatim (carrier included, zero where unfilled), plus
the fill mask and retarded time so the frontend can be tested against it.

### 5.4 `GET /api/plan/{plan_id}/waveform.npz` → the parametric NPZ (`WaveformSet.save`) as
`application/octet-stream`, filename `aodl_<plan_id>.npz`; 404 `plan_not_cached` if evicted.

### 5.5 `POST /api/code` (`PlanRequest`) → `{"code_python": "..."}` (same as in the plan
response; exists so the code panel can refresh without re-planning).

## 6. Advanced-mode routes (owned by W2; models in `aodl_advanced/schemas.py`)

### 6.1 `EfficiencyModel`
```json
{"interaction_length_mm": 4.0, "f_tpm_MHz": null, "refractive_index": 2.26,
 "peak_efficiency": 0.85, "angular_coupling": 1.0}
```
`f_tpm_MHz: null` → the hardware `f_center`. `peak_efficiency ∈ (0, 1]` → `Γ = asin(√η_peak)`.
`angular_coupling ∈ [−1, 1]` (DESIGN §1.6).

### 6.2 `POST /api/advanced/efficiency` → curves and the pair map
Request `{"hardware": HardwareSpec, "model": EfficiencyModel, "f_range_MHz": null, "n": 241}`
(`f_range` default `f_center ± 1.5 × half band`).
```json
{"f_MHz": [...n], "deflection_mrad": [...n], "eta_first": [...n],
 "eta_second_at_center": [...n],              // η_B(f_B; f_A = f_center)
 "bandwidth_3dB_MHz": 29.1, "band_MHz": [90, 110], "f_tpm_MHz": 100.0,
 "pair": {"f_A_MHz": [...m], "f_B_MHz": [...m], "eta": [[...m]...m]},   // m = 121, row = f_A
 "ridge": {"f_A_MHz": [...], "eta_max": [...], "f_B_at_max_MHz": [...]},
 "assumptions": "..." }
```

### 6.3 `POST /api/advanced/trajectory_efficiency` `{"plan": PlanRequest, "model": EfficiencyModel, "n": 300}` →
`{"t_us": [...], "per_channel": {"Ax": [...], ...}, "product": [...], "min_product": 0.73, "assumptions": "..."}`
(per-channel amplitude²-weighted mean of `η` over live tones at the beam-centre retarded time).

### 6.4 `Misalignment`
```json
{"Ax": {"offset_par_um": 0, "offset_perp_um": 0, "rotation_mrad": 0, "delay_ns": 0,
        "velocity_scale": 1.0, "tilt_x_mrad": 0, "tilt_y_mrad": 0}, "Bx": {...}, "Ay": {...}, "By": {...}}
```
All default to the aligned device; any channel may be omitted.

### 6.5 `POST /api/advanced/simulate` → FFT-engine frames
```json
{"plan": PlanRequest, "misalignment": Misalignment, "times_us": [20.0, 40.0],
 "grid": {"half_x_um": 6.0, "half_y_um": 6.0, "n": 96, "center": "tracked" | {"x_um":..,"y_um":..}},
 "xz": {"nz": 64, "z_half_um": 8.0} | null,
 "model": {"pupil_n": 256, "span_w_in": 2.5, "order_model": "j1" | "weak", "beat_subtimes": 8,
           "efficiency": EfficiencyModel | null},
 "reference": true}
```
`len(times_us) ≤ 24`, `n ≤ 192`, `pupil_n ∈ {128, 256, 512}`.
```json
{"plan_id": "...",
 "frames": [{"time_us": 20.0, "plane_z_um": 1.2,
             "xy": {"x_um": [...], "y_um": [...], "image": b64u16},
             "xz": {"x_um": [...], "z_um": [...], "row_y_um": 0.0, "image": b64u16} | null,
             "metrics": AdvMetrics, "reference_metrics": AdvMetrics | null}, ...],
 "notes": ["..."]}
AdvMetrics = {"x_um", "y_um", "wx_um", "wy_um", "z_best_x_um", "z_best_y_um", "astig_um",
              "peak_rel", "power_rel", "strehl"}      // *_rel relative to the aligned reference at the same time
```
`strehl` = peak / aligned peak (1.0 for the reference itself). Waists are second-moment radii
(`2σ`) of the XY image; `z_best_*` from a parabolic fit of the on-axis `w_x(Z)`, `w_y(Z)` minima
over the XZ/YZ scans (so `xz` may be requested with `nz ≥ 16` for them; otherwise `null`).

### 6.6 `POST /api/advanced/spectrum` `{"plan": PlanRequest, "channel": "Ax", "t_center_us": 50.0, "window_us": 20.0, "sample_rate_MSps": 625.0}` →
`{"f_MHz": [...], "psd_dB": [...], "crest_factor": 1.9, "papr_dB": 5.6, "peak_depth_rad": 0.42, "band_MHz": [90,110], "notes": [...]}`
(Hann-windowed periodogram of `render_samples` over the window, normalised to 0 dB at the
strongest line; crest factor = peak / rms of the real waveform in the window; `peak_depth_rad =
drive_strength × max|V|/A_single_tone`).
