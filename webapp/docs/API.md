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
and `value = peak * code / 65535`. For the basic-mode routes (§5) `peak` is on the
`SpotMetrics.power` scale of `aodl.field.focal.intensity_frame` (an unwindowed spot's peak is
`2·power/(π·wx·wy)`), so the Gaussian approximation and an exact frame are directly
comparable. **The unit is SI: power per m²** (`wx`, `wy` in metres), so a client that evaluates
`2·power/(π·wx·wy)` with the spot radii in µm (the `_um` fields) must multiply the decoded image
by `1e-12` (µm²/m²) before comparing — the frontend's `PER_M2_TO_PER_UM2`. For the advanced
engine (§6) `peak` is in that engine's own prefactor-dropped units (in practice the same
`intensity_frame` scale to 1e-3, see `reports/W2.md` §5.8) and only ratios between images from
the same call are guaranteed meaningful.

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
Required ids, in this order (used by the landing page and the tutorial): `pick_and_place`,
`square_xy`, `sphere_loop`, `single_lift_hold`, `array_lift_hold`, `array_arbitrary_path`,
`flagship_10x10`, `tutorial_hold_plain`, `tutorial_hold_shepard`. The two tutorial examples
share one single-tweezer trajectory (`Lift +10 µm / 60 µs`, `Hold 600 µs`, `Lift −10 µm /
60 µs`): `tutorial_hold_plain` has `options.shepard = "never"`, `options.check_band = false`
(tags `["tutorial", "no-shepard"]`; plans as `mode = "s19"` with the `check_band=False` note
and negative `band_usage` margins on all four channels — the drive leaves the band after
Eq. 1's 206 µs, which the flat-efficiency closed-form engine does not see);
`tutorial_hold_shepard` has default options (tags `["tutorial", "shepard"]`; `mode =
"shepard"`).

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
 "phases": "schroeder", "phase_seed": 0, "amp": 1.0, "retard_compensate": false,
 "f_z_bias": "none", "t_pad_us": null, "check_band": true}
```
`shepard ∈ {"auto", "never", "always"}` → `plan_motion(shepard="auto" | None | ShepardConfig)`;
`shepard_delta_f_MHz` (`[dfx, dfy]`) is required when `"always"` (for an array axis it must
equal the array spacing — `aodl` enforces it) and ignored otherwise; `eta ∈ (0, 0.5]`;
`phases ∈ {"schroeder", "zero", "random"}`; `f_z_bias ∈ {"none", "auto"}` → `0.0 | "auto"`;
`phases="random"` draws from `numpy.random.default_rng(phase_seed)` so the plan, its id and the
generated code are reproducible; `t_pad_us: null` = `aodl` default (2τ) — `t_pad_us = 0` means
the array's final rest position is never observable (the simulate grid is clamped to
`t_span[1] − τ/2`), which the plan `notes` must say; `check_band=false` is allowed for plotting
infeasible drives (the report carries the `aodl` note). `amp ≠ 1` on a drive that ends up in
`"shepard"` mode is a valid plan but cannot be serialised (`FadeZoneEnvelope.amp ≠ 1` has no
schema slot), so the NPZ download answers 400 `unsupported`; the frontend greys the `amp`
field out when the current plan's `mode == "shepard"`.

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
  "z_budget": {"requested_m_s": 5e-10, "ceiling_m_s": 2.06e-9, "fraction": 0.243},   // ceiling null when infinite (Shepard);
                // fraction is always finite: requested over the *plain* Eq. 1 ceiling (doubled under f_z_bias),
                // so under Shepard it reads how far past the budget the request went (2.9 = 290 %)
  "fade_events": [{"time_us": 123.4, "axis": "x", "shadow_um": 10.3}, ...],
  "notes": ["...", ...],
  "constants": {"tau_us": 11.54, "half_tau_us": 5.77, "waist0_um": 1.0655, "rayleigh_um": 3.463,
                "um_per_MHz": 10.3, "lens_scale_m_s": 1.03e-16, "chirp_per_um_MHz_per_ms": 4.854,
                "beam_transit_us": 6.154, "eq1_hold_10um_us": 206.0},
  // beam_transit_us = 2 w_in / v (PLAN §1.5's startup-transient time), not w_in / v
  "t_span_us": [0.0, 163.08],                // programmed drive span incl. hold tail
  "duration_us": 140.0,                      // trajectory duration T
  "seams_us": [0.0, 40.0, 100.0, 140.0],     // move boundaries
  "trajectory": {"t_us": [...400...], "x_um": [...], "y_um": [...], "z_um": [...], "seams_us": [...]},  // seams repeated here
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
hardware order for the metrics only (DESIGN §1.2); when `null` and the busiest of three probed
frames (first lit frame, trajectory midpoint, first hand-over) at the requested order holds more
than 80 frequency groups (a cost budget: ~0.5 ms per group and frame, so the default 80-frame
movie stays under ~3 s) the server uses 1 and says so in `notes`.
```json
{
  "plan_id": "...", "times_us": [...], "tracked_z_um": [...],
  "frames": [SpotFrame, ...],            // frames[i]: struct-of-arrays, all lists equal length
                                         // (length 0 before τ/2); one entry per *spatial cluster*
                                         // of a frequency group (DESIGN §1 item 2)
  "constants": PlanConstants,              // the full §4 block (a superset of waist0_um, rayleigh_um, tau_us)
  "bounds_um": {"x0": -3.0, "x1": 25.0, "y0": -8.0, "y1": 8.0, "z0": -2.0, "z1": 8.0},
                // requested trajectory ± array half-extent ± 6 w0 laterally, ± 1 z_R axially;
                // on a fading-Shepard drive each array axis (M > 1) is one pitch wider per side,
                // so the M + 2 hand-over columns are on the canvas (round 2, R2-verify)
                // (NOT the spot table: shadow tweezers at ±10.3·Δf µm would blow the canvas up)
  "power_ref": 6.59e-29,                  // median over lit frames of the brightest trap's power, spots
                                          // within one w0 merged first (so a trap reads 1.0 on its plateau
                                          // and through a hand-over; shadows read 0.5)
  "notes": ["..."]
}
SpotFrame = {"x_um": [...], "y_um": [...], "z_um": [...], "delta_f_um": [...], "sigma_astig": [...],
             "wx_um": [...], "wy_um": [...], "power": [...], "power_coherent": [...],
             "df_opt_MHz": [...], "group": [int...], "n_terms": [int...]}
Spot      = the same twelve fields as scalars (used by /api/frame and /api/advanced)
```
Floats are rounded to 6 significant digits; spots below `1e-3` of the frame's maximum power
are dropped (both rules exist to keep an 80-frame 4×4 fading response under ~1 MB).
`group` is the frequency-group index of `aodl.field.focal.group_terms` for that frame,
`n_terms` the number of pupil terms in the cluster. Spots of *different* groups may coincide
and must be added in intensity by the client: at a fading hand-over centre the main trap is
**two** coincident spots (the dying `(a, a)` and rising `(a−1, a−1)` rung pairs, `2Δf` apart
in optical frequency) whose powers sum to the trap's, beside the two shadows at `±10.3·Δf µm`
(one group, two clusters) — four spots in all. `notes` also says when frames before `τ/2`
are included (they are empty: the pair-driven tweezer is dark) and when `t_end_us` was
clamped. The first `fade_events` entry of a lifted single tweezer is a *y* event at
`t = 0` (`ξ_y = ½` puts a y fade centre exactly where `f_Z = 0`); pick the first `x` event to
look at an x hand-over.

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
 "cuts": {"x_um": [...], "i_x": [...], "i_y": [...ny], "z_um": [...], "i_z": [...], "through": {"x_um":..,"y_um":..}},
 "spots": [Spot, ...],
 "gauss_peak": <float>, "exact_peak": <float>}
```
`nx, ny, nz ≤ 512`. `time_us` must satisfy `aodl.simulate`'s coverage rule (else 400
`coverage`). The XZ slice reuses the XY grid's `x` range on `nx` points; its `z_um` axis is
`plane_z_um ± z_half_um` on `nz` points (centred on the evaluation plane), and `row_y_um:
null` snaps to the tweezer row nearest the power-weighted centre (`SimResult.spot_row`).
Cuts go through the brightest spot of the frame: `i_x` is the XY image row nearest its `y`
(over `cuts.x_um`, `nx` points), `i_y` the XY image column nearest its `x` (over `xy.y_um`,
`ny` points), `i_z` an exact axial cut at its `(x, y)` over `cuts.z_um` (the XZ `z` axis when
`xz` is requested, else `plane_z_um ± 8 µm` on 96 points). All three are closed-form
(Eq. S11) intensities, unpacked floats on the §0.3 scale (SI power per m²). `gauss_peak` is
the astigmatic-Gaussian peak `Σ 2·power/(π·wx·wy)` (radii in metres) over the spots whose
lateral centres lie within one waist `w0` of the brightest spot — the coincident groups of a
hand-over add — on the same SI per-m² scale; `exact_peak` is the maximum of the XY image
(`== xy.image.peak`). Their ratio `exact_peak / gauss_peak` is ≈ 1 for an unwindowed
single-term trap on its own plane (within 2 % for a static tweezer at `t = 3τ`); a dark frame
(no spots) has `gauss_peak = 0`. With a fixed or explicit plane the Gaussian peak is still the
spot's in-focus value, so the ratio then also measures defocus. `movie_mixing_order` is resolved exactly
as in §5.1, so the exact frame shows the same spot set as the movie (`null` → the hardware
order unless the > 80-group fallback applies). `spots` are the §5.1 clusters of this frame,
unrounded and uncut.

### 5.3 `POST /api/aperture` (`ApertureRequest`) → `ApertureResponse`
`{"plan": PlanRequest, "time_us": 30.0, "n": 512, "include_voltage": false, "voltage_window_mm": null}` →
```json
{"time_us": 30.0, "channels": {"Ax": {"u_mm": [...n], "envelope": [...n], "f_inst_MHz": [...n],
 "filled": [bool...n], "t_ret_us": [...n], "voltage": {"u_mm": [...m], "v": [...m]} | null}, ...}}
```
`envelope = Σ_n A_n(t_ret(u))` and `f_inst_MHz = f_center + Σ A_n f_n / Σ A_n` (NaN where the
envelope is 0), both zero/NaN where unfilled, with `t_ret(u) = t − τ/2 − s u / v`. The literal
carrier-included `V(u)` of `aodl.device.aod.aperture_window` cannot be drawn at `n = 512`
(Δu = 14.7 µm against an acoustic wavelength of 6.5 µm → aliased moiré), so it is returned only
when `include_voltage=true`, on `m ≥ 4096` points over `voltage_window_mm = [u0, u1]` (default
the central 0.1 mm; `m = max(4096, 32 samples per acoustic wavelength v/f_center)`, capped at
32768). Asking for the full aperture (`[−D/2, D/2]`) returns `aperture_window` point for
point. **JSON has no NaN**: `f_inst_MHz` entries where the envelope is zero are `null`
(pydantic's serialisation of NaN), and the frontend maps `null` back to `NaN`.

### 5.4 `GET /api/plan/{plan_id}/waveform.npz` → the parametric NPZ (`WaveformSet.save`, via a
temporary file) as `application/octet-stream`, filename `aodl_<plan_id>.npz`; 404
`plan_not_cached` if evicted; 400 `unsupported` for a Shepard drive with `amp ≠ 1` (§3.4).

### 5.5 `POST /api/code` (`PlanRequest`) → `{"code_python": "..."}` (same as in the plan
response; exists so the code panel can refresh without re-planning).

## 6. Advanced-mode routes (owned by W2; models in `aodl_advanced/schemas.py`)

### 6.1 `EfficiencyModel`
```json
{"interaction_length_mm": 4.0, "f_tpm_MHz": null, "refractive_index": 2.26,
 "peak_efficiency": 0.85, "angular_coupling": 0.2}
```
`f_tpm_MHz: null` → the hardware `f_center`. `peak_efficiency ∈ (0, 1]` → `Γ = asin(√η_peak)`.
`angular_coupling γ ∈ [−1, 1]`: the signed fraction of the first AOD's deflection that reaches
the second AOD's Bragg plane (DESIGN §1.6; `|γ| = 1` ideal relay, default 0.2 reproduces the
paper's "a few times narrower" first-AOD band at `L = 4 mm`). The second-AOD mismatch is
`δ_B = C₂[(f_B − f_tpm)² − 2 γ f_tpm (f_A − f_tpm)]`.

### 6.2 `POST /api/advanced/efficiency` → curves and the pair map
Request `{"hardware": HardwareSpec, "model": EfficiencyModel, "f_range_MHz": null, "n": 241}`
(`f_range` default `f_center ± 1.5 × half band`).
```json
{"f_MHz": [...n], "deflection_mrad": [...n], "eta_first": [...n],
 "eta_second_at_center": [...n],              // η_B(f_B; f_A = f_tpm) — equals eta_first when f_tpm = f_center
 "bandwidth_3dB_MHz": 27.9, "first_aod_usable_MHz": 12.05, "band_MHz": [90, 110], "f_tpm_MHz": 100.0,
 "pair": {"f_A_MHz": [...m], "f_B_MHz": [...m], "eta": [[...m]...m]},   // m = 121, row = f_A
 "ridge": {"f_A_MHz": [...], "eta_max": [...], "f_B_at_max_MHz": [...]},  // max over f_B within the band, per f_A
 "assumptions": "..." }
```

`first_aod_usable_MHz` is the full width in `f_A` over which `ridge.eta_max ≥ ½·max(ridge.eta_max)`
with `f_B` restricted to the hardware band (DESIGN §1.6).

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
 "xz": {"nz": 24, "z_half_um": 8.0} | null,
 "model": {"pupil_n": 256, "span_w_in": 2.5, "order_model": "j1" | "weak", "comb_dft_points": 16,
           "max_groups": 128, "efficiency": EfficiencyModel | null},
 "reference": true}
```
`len(times_us) ≤ 24`, `n ≤ 192`, `pupil_n ∈ {128, 256, 512}`, `xz.nz ≤ 32`,
`comb_dft_points ∈ [1, 64]` (`K` of DESIGN §1.7; `1` = frozen instantaneous pupil, no line
decomposition), `max_groups ≤ 256`. `center: "tracked"` centres the grid on the
power-weighted mean of the frame's spots (from `aodl_webapp.simulation.frame_spots`) and the
plane on its tracked Z.
```json
{"plan_id": "...",
 "frames": [{"time_us": 20.0, "plane_z_um": 1.2, "n_groups": 7, "groups_dropped": 0,
             "xy": {"x_um": [...], "y_um": [...], "image": b64u16},
             "xz": {"x_um": [...], "z_um": [...], "row_y_um": 0.0, "image": b64u16} | null,
             "metrics": AdvMetrics, "reference_metrics": AdvMetrics | null}, ...],
 "static_offset_um": {"Ax": {"x": 0.0, "y": 0.0}, ...},   // dropped common carrier-tilt pointing offset per channel
 "notes": ["..."]}
AdvMetrics = {"x_um", "y_um", "wx_um", "wy_um", "z_best_x_um", "z_best_y_um", "astig_um",
              "peak_rel", "power_rel", "strehl"}      // *_rel relative to the aligned reference at the same time
```
`strehl` = peak / aligned peak (1.0 for the reference itself). Waists are second-moment radii
(`2σ`) of the XY image; `z_best_*` from a parabolic refinement of the `w_x(Z)`, `w_y(Z)` minima
over the XZ/YZ scans, computed only when `xz` is requested with `nz ≥ 16` (otherwise `null`,
and `astig_um` with it). The XZ image and the focus scans cover only the brightest ≤ 6 groups
(`notes` says so when groups were left out).

### 6.6 `POST /api/advanced/spectrum` `{"plan": PlanRequest, "channel": "Ax", "t_center_us": 50.0, "window_us": 20.0, "sample_rate_MSps": 625.0}` →
`{"f_MHz": [...], "psd_dB": [...], "crest_factor": 1.9, "papr_dB": 5.6, "peak_depth_rad": 0.42, "band_MHz": [90,110], "notes": [...]}`
(Hann-windowed periodogram of `render_samples` over the window, normalised to 0 dB at the
strongest line; crest factor = peak / rms of the real waveform in the window; `peak_depth_rad =
drive_strength × max|V|/A_single_tone`).
