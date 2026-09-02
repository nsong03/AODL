# AODL — Code Architecture

Companion to [`PLAN.md`](PLAN.md) (physics + milestones). This document fixes the package
layout, core data types, and dependencies before implementation.

---

## 0. Design principles

1. **Parametric waveforms are the canonical object.** A waveform is stored and exchanged as
   *functions with parameters* (piecewise-polynomial frequency laws, parametric amplitude
   envelopes), never as raw samples. Samples are one *render target* (for a real AWG);
   analytic evaluation of $f(t), \dot f(t), \varphi(t), A(t)$ at retarded times is another
   (for the simulator). Both derive from the same stored parameters, so what you simulate is
   exactly what you export.
2. **Polynomial closure.** Trajectory profiles, tone frequencies, and tone phases are all
   piecewise polynomials: a min-jerk position segment maps through synthesis (Eq. S19) to a
   polynomial frequency law, whose integral (phase) and derivative (chirp) are again
   polynomials — exact everywhere, no numerical differentiation anywhere in the pipeline.
   One well-tested `PiecewisePoly` class is the workhorse for all three roles.
3. **Terms, not grids, until the last moment.** Device physics reduces each frame to a list
   of *terms* (complex amplitude + per-axis Taylor coefficients + optical frequency). Fields
   are closed-form Gaussians per term, evaluated only on small patches around each spot.
   No FFTs *in the simulation path*; grids appear only at rendering. The M6 checker
   (`check/`) is the deliberate exception — it is FFT-based on purpose and shares none of
   this code (principle 6).
4. **One sign authority.** All orientation/sign conventions (sound directions, +1-order
   signs, axis handedness) live in `device/conventions.py` and are pinned by tests against
   the paper's Table I. Nothing else hardcodes a sign.
5. **Physics functions cite equations.** Every function implementing paper physics carries
   the equation number (`Eq. S8`) in its docstring; tests assert the formulas numerically.
6. **One independent verification path.** `check/` re-derives the tweezers from the *rendered
   samples* and is forbidden, by a source scan, from importing anything of the simulator's
   beyond `params`, `units`, `poly`, `trajectory.spec`, `device.conventions` and the
   samples-file schema constants. Truth enters it only as samples plus the compiled trajectory.

## 1. Layer diagram

```
                 user intent                          lab hardware
                     │                                     ▲
        ┌────────────▼─────────────┐            ┌──────────┴─────────────┐
  L2    │ trajectory/  (spec, ramps)│           │ waveform/export.py     │
        │  ArraySpec + Move list    │           │  .npz (parametric) +   │
        └────────────┬─────────────┘            │  render_samples(rate)  │
                     │ synthesize (Eq. S19)     └──────────▲─────────────┘
        ┌────────────▼─────────────────────────────────────┴───┐
  L1    │ waveform/  WaveformSet = {Ax,Bx,Ay,By} → [ToneTrack] │  ◄── the exchange format
        │  ToneTrack = PiecewisePoly freq + Envelope + phase   │
        └────────────┬─────────────────────────────────────────┘
                     │ f, ḟ, φ, A, Ȧ, Ä at frame time t (retarded)
        ┌────────────▼─────────────┐
  L3    │ device/  aod, mixing,    │  tones → emission lines (IM3) → 4-channel
        │          aodl            │  product → Term[] (amp, tilt, lens, ω)
        └────────────┬─────────────┘
                     │ Term arrays
        ┌────────────▼─────────────┐
  L4    │ field/  gaussian, focal, │  closed-form U(X,Y,Z) per term; frequency-
        │         measure          │  group intensities; spot metrics
        └────────────┬─────────────┘
                     │ frames + metrics
        ┌────────────▼─────────────┐
  L5/6  │ engine.py → viz/movie.py │  SimResult → focus-tracked movie (hue ↔ Z),
        │ api.py (front door)      │  XZ slice panel, spectrogram panel
        └──────────────────────────┘
```

## 2. Folder structure

```
AODL/
├── pyproject.toml            # package: aodl; deps pinned loosely; ruff + pytest config
├── README.md
├── docs/
│   ├── PLAN.md               # physics + milestones
│   ├── ARCHITECTURE.md       # this file
│   ├── ORCHESTRATION.md      # build process: waves, rulings, backlog
│   ├── guide.md              # the lab-facing user guide                 [M5]
│   ├── waveform_format.md    # NPZ schema for parametric WaveformSet
│   └── conventions.md        # axes, signs, retarded time, units
├── src/aodl/
│   ├── __init__.py           # public API re-exports
│   ├── params.py             # AODParams, OpticsParams, AODLParams; presets
│   ├── units.py              # MHz/µm/µs constants (calibrations live on AODLParams)
│   ├── poly.py               # PiecewisePoly: eval/derivative/antiderivative/shift
│   ├── trajectory/
│   │   ├── spec.py           # ArraySpec, Move (lift/translate/lower/hold/waypoints)
│   │   └── ramps.py          # min-jerk, const-jerk, const-accel, SCJ, linear (S14–S17)
│   ├── waveform/
│   │   ├── tones.py          # Envelope kinds; ToneTrack; ChannelWaveform; WaveformSet
│   │   ├── synthesis.py      # Eq. S19 solver + Schroeder phases + band checks
│   │   ├── shepard.py        # fading-Shepard ladders (S24–S28)          [M4]
│   │   ├── serialize.py      # WaveformSet ↔ .npz (parameters, not samples)
│   │   └── export.py         # render_samples(rate) for AWG use
│   ├── device/
│   │   ├── conventions.py    # sound directions, order signs, axis map (sign authority)
│   │   ├── aod.py            # per-channel: retarded-time eval, aperture fill window,
│   │   │                     #   diagnostic window extraction V(t ∓ u/v)
│   │   ├── mixing.py         # weak-drive expansion → emission lines; IM3   [M2]
│   │   └── aodl.py           # 4-channel term product → Term arrays (Eq. S7/S8)
│   ├── field/
│   │   ├── gaussian.py       # closed-form ∫(poly)·e^{−au²+bu} du (+ erf-edge variants)
│   │   ├── focal.py          # per-term U(X,Y,Z); frequency grouping; patch accumulation
│   │   ├── measure.py        # centroid, waists, Z̄, ΔF, σ_astig per spot (analytic)
│   │   └── reference.py      # direct quadrature of Eq. S11 (tests only)
│   ├── check/                # the independent FFT checker (M6) — reads *samples*, not the IR
│   │   ├── record.py         # SampleRecord: the AWG buffers + rate + normalization
│   │   ├── demod.py          # one FFT per channel → complex baseband z(t) (Eqs. S1–S2)
│   │   ├── pupil.py          # aperture rebuild at du = Λ/8; +1-order band selection (S1–S4)
│   │   ├── transform.py      # zoom (chirp-z) Eq. S11 + defocus; golden-ratio sub-times
│   │   ├── metrics.py        # profile fits, w²(Z) best focus, blob audit, accumulation
│   │   ├── expect.py         # what was *asked* for: Table I positions from the spec
│   │   └── report.py         # tolerances, verdict, check_samples() driver
│   ├── engine.py             # simulate(wfs, times) → SimResult (params from wfs.params)
│   ├── viz/
│   │   ├── style.py          # colormaps (Z-hue), panel layout defaults
│   │   └── movie.py          # focus-tracked XY view + XZ slice + spectrograms → mp4/gif
│   └── api.py                # one-call front door for the product workflow
├── examples/                         # documented Jupyter notebooks (see §3a)
│   ├── 01_single_aod_sweep.ipynb        # M1
│   ├── 02_crossed_pair_diagonal.ipynb   # M2
│   ├── 03_aodl_3d_motion.ipynb          # M3
│   ├── 04_array_lift_traverse.ipynb     # M3 (the user story)
│   ├── 05_fading_shepard.ipynb          # M4
│   ├── 06_product_tour.ipynb            # M5 (the lab-facing demo)
│   └── 07_fft_checker.ipynb             # M6 (the independent checker)
└── tests/                            # (as built; one file per module plus per-milestone
    ├── test_poly.py … test_tones.py  #  integration suites)
    ├── test_conventions.py  test_device_single_aod.py  test_mixing.py  test_window.py
    ├── test_focal.py  test_measure.py  test_focal_geometry.py  test_grouping.py
    ├── test_ramps.py  test_serialize.py  test_export.py  test_spec.py
    ├── test_synthesis.py  test_synthesis_s19.py  test_synthesis_options.py
    ├── test_shepard.py  test_engine.py  test_api.py  test_docs.py
    ├── test_check_demod.py  test_check_pupil.py  test_check_transform.py
    ├── test_check_metrics.py  test_check_independence.py  test_check_expect.py
    ├── test_check_weak_vs_sim.py  test_check_bragg.py  test_check_verdict.py
    ├── test_check_flagship.py                          # the M6 CI gate
    └── test_integration_m1.py  _m2.py  _m3.py  _m4.py   # per-milestone acceptance
                                                 # (spec → waveforms → sim → measured)
```

## 3. Core data types (contracts)

### L1 — waveform IR

```python
PiecewisePoly      # breakpoints t[k], per-segment coeffs (normalized time);
                   # .__call__(t), .derivative(), .antiderivative(), exact & vectorized

Envelope           # kind ∈ {const, cos_fade(p, η, zone), raised_edge}; A(t), Ȧ(t), Ä(t)
                   # (2nd derivative feeds acoustic irising)

ToneTrack          # freq: PiecewisePoly [Hz], env: Envelope, phase0: float [rad]
                   # phase(t) = 2π∫freq  (antiderivative, phase-continuous by construction)

ChannelWaveform    # tones: list[ToneTrack]; compiled to column arrays for vector eval

WaveformSet        # {"Ax","Bx","Ay","By"} → ChannelWaveform
                   # + params snapshot + spec echo + schema_version
                   # .save(path.npz) / .load(path.npz)     ← parameters only
                   # sample rendering lives in waveform/export.py:
                   #   render_samples(wfs, rate) → {channel: float32 array}  ← AWG target
                   #   (also exposed as MotionPlan.render_samples)
```

NPZ schema (detailed in `waveform_format.md`): one structured array per channel — rows =
(tone_id, seg_index, t0, T, freq_coeffs[·], env_kind, env_params[·], phase0) — plus a JSON
metadata string. Loading reconstructs `WaveformSet` exactly; no information lives in samples.

### L3 — device

```python
EmissionLine       # one virtual first-order tone after mixing: complex amp,
                   # f_eff, ḟ_eff (evaluated at frame time), source tone indices
Term               # one 4-channel combination: complex amplitude c,
                   # tilt (bx, by), quad (qx, qy)  [from Σ±f/v, Σḟ/2v² per axis],
                   # amp-poly coeffs (irising), optical freq offset ω, fill factor
```

`device/aod.py` evaluates each channel's tones at the beam-center retarded time (aperture
center; matched delays per design brief) and reports the aperture *fill state* so the
leading-edge transient (waveform entering the crystal) appears in the first ≈ τ/2. The
diagnostic `window(channel, t, n_points)` returns the literal waveform segment on the
crystal for plotting.

### L4/L5 — field & results

```python
SimResult          # times, per-frame Term arrays, per-spot metrics table
                   # (position X/Y/Z̄, waists, ΔF, σ_astig, intensity, group id),
                   # lazy intensity_frame(t, plane=...) evaluator
```

`field/focal.py` groups terms by optical frequency (GROUP_TOL = 1 kHz with a cluster-diameter cap): degenerate terms sum
coherently, distinct groups add in intensity (beat notes average out — Supplement
"interlaced vs simultaneous fading" logic). Per-group evaluation happens on a bounding patch
(±4 waists) and accumulates into the canvas.

### Examples as notebooks

`examples/` contains Jupyter notebooks, one per milestone, written as *narrative
documentation*: markdown cells state the physics being demonstrated (with the paper equation
and the closed-form prediction), code cells build the waveforms and run the simulation, and
figure/movie cells show the result next to the analytic expectation. They double as the user
guide for AMO labs. Conventions: committed with outputs cleared (figures/movies regenerate on
run), executed headlessly in CI via `pytest --nbmake` so they can never rot, and kept thin —
all reusable logic lives in `src/aodl/`, never in notebook cells.

### Movie (decision #3)

Default renderer: **focus-tracked planar view** — the XY plane is placed at the
intensity-weighted focal Z of the array each frame (per-spot sharpness preserved), each
spot tinted by its own Z via a diverging colormap (hue = Z, luminance = intensity), color
bar included. Side panel: **XZ slice** through a selectable row showing the out-of-plane
excursion directly. Optional bottom strip: per-channel spectrograms (like paper Figs. 3–4).
Alternate mode kept for realism checks: fixed-plane camera view (spots blur/dim off-plane).
Writer: matplotlib frames → `imageio`/`imageio-ffmpeg` mp4 (GIF fallback).

## 4. Dependencies

Runtime (deliberately minimal for lab deployability):

| Package | Why |
|---------|-----|
| `numpy` | all array math |
| `scipy>=1.8` | `special.wofz` (complex erf for aperture-edge/fill closed forms); `signal.czt` (the M6 checker's zoom transform — added in 1.8); `integrate` in tests |
| `matplotlib` | frame rendering, panels |
| `imageio` + `imageio-ffmpeg` | mp4 encoding without system ffmpeg |

Dev: `pytest`, `ruff` (lint + format), `mypy` (typed public API), `jupyterlab` + `nbmake`
(notebook examples and their CI execution). Python ≥ 3.11.
No JAX/torch/numba: term counts (~10²–10³/frame) and patch evaluation keep pure numpy fast
(target ≪ 1 s/frame for a 10×10 array at 512² rendering).

## 5. Decisions locked (2026-08-31)

1. Paper hardware defaults (DTSX-400: v = 650 m/s, D = 7.5 mm, f₀ = 100 MHz; F = 6.5 mm)
   at λ = 1030 nm; `paper_808` preset for figure reproduction.
2. Input beam: uncropped Gaussian, w_in = 2.0 mm default.
3. Movie: focus-tracked planar view, hue ↔ Z, XZ slice side panel.
4. Mixing: perturbative weak-drive expansion through IM3, calibrated drive strength C.
5. Waveform storage/export: generic NPZ containing the **parametric function
   representation** (segments + parameters); sample expansion is a separate render step.
6. Usable band: ±10 MHz on all four channels.
7. (2026-09-02, M6) **Checker pupil model:** the default is the full `exp(iCV)` crystal with
   the `+1` order selected in the aperture's spatial-frequency domain (`"bragg_band"`), on a
   grid pinned at `du = Λ/8` so the order comb's aliases land on order centres. The linear
   Eq. S3 model (`"weak"`) is kept as the cross-validation path against the simulator, not as
   the verdict path. Placement: `check_samples()` plus `MotionPlan.check()`, with
   `tests/test_check_flagship.py` as the CI gate.
