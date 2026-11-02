# Pre-implementation review (Fable reviewer) — accepted findings

The full reviewer report was delivered to the architect in-session. This file records what
was accepted and folded into `DESIGN.md`, `API.md` and the work orders, so implementers can
see why the documents say what they say.

| # | finding | severity | disposition |
|---|---|---|---|
| A1 | Second-AOD efficiency model inverted: at TPM an incidence shift is an *additive* mismatch, `δ_B = C₂[(f_B − f_tpm)² − 2γ f_tpm (f_A − f_tpm)]`; the first AOD's band is narrow *because* its deflection detunes the second (paper's "Design caveats") | blocker | accepted; DESIGN §1.6, API §6.1–6.2, W2 §2/§5; default `γ = 0.2`; `first_aod_usable_MHz` added |
| A2 | One `Spot` per frequency group collapses the shadow-tweezer pair onto the main trap (phantom spot with the shadows' power) | blocker | accepted; one spot per spatial cluster (`cluster_spots`), `bounds_um` from the trajectory, shadow badge |
| A3 | `zColor` anchors: 11 ColorBrewer anchors beat 17 resampled ones | should-fix | accepted (W3) |
| A4 | Retardation display correct; clip `[t−τ, t]` at 0; no +τ/2 overlay under `retard_compensate` | ok / nits | accepted (W3) |
| A5 | `aperture_window` at 512 points aliases the 100 MHz carrier; draw envelope + instantaneous frequency; voltage zoom ±0.2 µs / 2001 pts | should-fix | accepted (API §5.3, W1, W3) |
| A6 | TPM derivation checks out; 3 dB bandwidth is 27.9 MHz at L = 4 mm, η_peak = 0.85 (not 29.1); caveats listed | ok | numbers and caveats updated |
| A7 | Misalignment coordinate formula garbled; fill mask must scale with magnification; J₁ law is the quasi-static +1 order (reduces to mixing.py's table); efficiency as relative amplitude `√(η/η_peak)` with B's closure over `f_A`; report dropped common carrier tilt | should-fix | accepted (DESIGN §1.7, W2 §3) |
| A8 | Uniform sub-time beat averaging leaves 19–79 % of cross terms; replace by frequency-line decomposition (weak: tones; j1: comb DFT) grouped by Σf as aodl does | should-fix | accepted (DESIGN §1.7, API §6.5, W2 §3) |
| A9 | `Path` spline sound (no frequency-law discontinuity, degree ≤ 5, serialisation untouched); `sphere_loop` must be closed, `array_arbitrary_path` tilt must be specified (30° about x) | ok / nits | accepted (W1 §2) |
| A10 | W2 tests: rotation moves Y by `tanθ·X/2` for a symmetric S19 trap; velocity scale on *one* B channel only (ΔF ≈ −0.039·lens_scale·ḟ, X error ≈ −0.98 %) | should-fix | accepted (W2 §5) |
| B1 | W2 needs the frame's spot centre → `frame_spots` in W1's frozen list | blocker | accepted |
| B2 | Simulate payload up to 5 MB → struct-of-arrays, 6 significant digits, 1e-3 power cut, group-count budget (> 160) instead of the 64-trap rule | should-fix | accepted (API §5.1, DESIGN §1.2) |
| B3 | `beam_transit_us` must be `2 w_in / v` = 6.154 µs | should-fix | accepted |
| B4 | `amp ≠ 1` under Shepard cannot be serialised → NPZ 400 `unsupported`; seed random phases (`phase_seed`); note `t_pad = 0` | should-fix | accepted (API §3.4/§5.4) |
| B5 | `peak` of basic-mode frames is on the `SpotMetrics.power` scale | should-fix | accepted (API §0.3) |
| B7 | Silent `try/except ImportError` on the advanced router would ship 404s → strict under `AODL_STUDIO_STRICT_ADVANCED=1`; bare `aodl_webapp/__init__.py` | should-fix | accepted (W1, W4) |
| B8 | W2 fixtures outside W1's `conftest.py`; NPZ via tempfile | nit | accepted |
| C | Pupil engine XZ cost (5 s/frame as specified) → cache aligned reference, `nz ≤ 32`, brightest ≤ 6 groups, sync routes | should-fix | accepted |
| D | "Clean fade in/out tone code": codegen must emit resolved `ShepardConfig`/phases/rng + fade schedule; add a waveform-recipe table | should-fix | accepted (W1, W3) |
| E | W1 delivery order (fixtures first); `dtype=np.float64` in `test_ir`; explicit frame time in `test_api_simulate`; W4 check at mid-translate | should-fix | accepted |

Measured reference numbers from the review (default_1030, `aodl` 0.1.0): τ = 11.54 µs,
w0 = 1.0655 µm, z_R = 3.46 µm, 10.3 µm/MHz, 4.854 MHz/ms per µm of Z, Eq. 1 hold 206 µs;
example modes: pick_and_place s19, square_xy s19, sphere_loop (closed) s19, single_lift_hold
shepard (Δf 12 MHz, 16 rungs), array_lift_hold shepard (92 rungs, 155–169 groups/frame),
array_arbitrary_path (30° tilt) s19 with 49 groups at mixing order 3, flagship_10x10 shepard
(93 rungs, 483 groups at order 3 / 121 at order 1).
