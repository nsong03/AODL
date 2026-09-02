# AODL user guide

**What this package does.** You describe a move — *"a 10×10 array, lift 10 µm out of the
focal plane, traverse 40 × 25 µm, drop"* — and it hands back (1) the RF waveforms to program
on the four AOD channels of a 3D acousto-optic deflector lens and (2) a closed-form optical
simulation and movie of the tweezers those waveforms make.

Physics reference throughout: Lu, Song, Xiang, Ho, Lee, Yan & Stamper-Kurn,
*Astigmatism-free 3D Optical Tweezer Control for Rapid Atom Rearrangement*
([arXiv:2510.11451](https://arxiv.org/abs/2510.11451)). Equation numbers `S#` refer to its
Supplement; `Eq. 1` is in the main text.

Contents: [install](#1-install) · [quickstart](#2-five-minute-quickstart) ·
[concepts](#3-concepts) · [parameters](#4-parameter-reference) ·
[outputs](#5-outputs) (incl. [checking a drive](#55-checking-a-rendered-drive)) ·
[fidelity and limits](#6-fidelity-and-limits) · [FAQ](#7-faq) ·
[numbers](#8-numbers-this-guide-quotes)

---

## 1. Install

Python 3.11+, and numpy / scipy / matplotlib / imageio (+ `imageio-ffmpeg`, for `.mp4`).

```bash
pip install -e ".[dev]"     # editable install with pytest, ruff, mypy, jupyterlab, nbmake
pytest                      # the full suite
pytest --nbmake examples/   # execute the seven notebooks (slower)
```

Nothing else is required: there is no compiled extension and no hardware in the loop.
Waveform files and movies are ordinary `.npz` and `.mp4`. (The *simulation* uses no FFT at
all; the independent checker of §5.5 uses `numpy.fft` and `scipy.signal.czt`, both of which
come with the dependencies above.)

---

## 2. Five-minute quickstart

```python
from aodl import ArraySpec, Lift, TrajectorySpec, Translate, plan_motion
from aodl.units import MHz, um, us

array = ArraySpec(10, 10, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz)  # 10.3 µm per MHz of pitch
story = TrajectorySpec(
    array=array,
    moves=(
        Lift(10 * um, 150 * us),                # up, out of the focal plane
        Translate(40 * um, 25 * um, 250 * us),  # across
        Lift(-10 * um, 150 * us),               # and back down
    ),
)

plan = plan_motion(story)      # Eq. S19, or the fading-Shepard ladders if the band refuses it
print(plan.report.summary())   # mode, band usage, axial budget, fade schedule, caveats
plan.save("move.npz")          # the deliverable: segment parameters, never samples
```

`plan_motion` returns a `MotionPlan`, which is the whole product surface:

| Call | What you get |
|------|--------------|
| `plan.report.summary()` | the block printed above: mode, per-channel band occupancy, axial budget, hand-over schedule, and the caveats that apply to *this* drive |
| `plan.report.figure()` | tone tracks (opacity = envelope) over the band, plus a per-channel occupancy bar |
| `plan.save(path)` | the parametric `.npz` (`docs/waveform_format.md`) |
| `plan.render_samples(rate)` | `{channel: float32 samples}` for the AWG, carrier included |
| `plan.simulate(times=None)` | a `SimResult`: one `SpotMetrics` per tweezer per frame |
| `plan.movie(path)` | simulate + render, hue ↔ Z, XZ panel, drive strip |
| `plan.check()` | a `CheckReport`: the tweezers rebuilt from the *rendered samples*, verdict included (§5.5) |

Nothing is simulated or rendered until you ask: `plan_motion` only synthesizes and reports.
`examples/06_product_tour.ipynb` runs exactly this path end to end.

---

## 3. Concepts

### 3.1 Four channels, three degrees of freedom

The device is four AODs with overlaid apertures, acoustic axes rotated by 90° in
counter-propagating pairs (`docs/PLAN.md` §1.1, `src/aodl/device/conventions.py`):

| Channel | Sound direction | Role |
|---------|-----------------|------|
| `Ax` | −x | first x deflector, single tone |
| `Bx` | +x | second x deflector, carries the column ladder |
| `Ay` | −y | first y deflector, single tone |
| `By` | +y | second y deflector, carries the row ladder |

Stacking multiplies the pupils (Eq. S7). Each AOD's retarded drive contributes a linear
phase (deflection) and a quadratic phase (a cylindrical lens of power ∝ `fdot`), and the
lens sign does *not* depend on the sound direction — which is why a counter-propagating
pair can move a trap sideways with no focal shift. Paper Table I, in code as properties of
`AODLParams`:

```text
X       = deflection_scale * (f_Bx - f_Ax)                            deflection_scale = λF/v
Y       = deflection_scale * (f_By - f_Ay)
Zbar    = 0.5 * lens_scale * (fdot_Ax + fdot_Bx + fdot_Ay + fdot_By)  lens_scale = λF²/v²
Delta_F = lens_scale * (fdot_Ax + fdot_Bx - fdot_Ay - fdot_By)        the astigmatic interval
```

Astigmatism-free 3D control is `Delta_F = 0`, leaving exactly three degrees of freedom for
(X, Y, Z). `synthesize` solves this with Eq. S19: the lateral term enters the two members of
a pair with opposite signs (so it *differs*, giving X) and cancels in the chirp sums, while
the axial term `f_Z` is common to all four (so it *adds*, giving Z, with `Delta_F ≡ 0`).

At the default hardware, **10.3 µm per MHz** of frequency difference, and a 10 µm axial
offset costs a co-chirp of **48.54 MHz/ms on every channel**.

### 3.2 The rotating frame

Every frequency inside the package — in a `ToneTrack`, in a saved `.npz`, in the ramps you
write yourself — is a **detuning from that channel's `f_center`** (Eq. S2). The carrier is
added back in exactly two places: `render_samples`, which is what a transducer sees, and the
report's `band_usage`, which quotes absolute RF so it compares directly against
`AODParams.band`.

### 3.3 Parametric waveforms

A waveform file holds **segment parameters, not samples** (`docs/PLAN.md` decision 5): per
channel, one row per polynomial segment of each tone's frequency law, one row per tone for
its phase and envelope, plus a JSON snapshot of the hardware it was designed for. The 550 µs,
four-channel, 93-tone Shepard drive of the quickstart is 97 kB; the same drive sampled at
625 MS/s in float32 is 5.7 MB, and it only grows with duration and rate while the parametric
file does not. The round trip is exact to float64, so what you export is what you simulated.

Rendering is a separate, explicit step (`plan.render_samples(rate)` /
`aodl.waveform.export.save_samples`), and sample files must be named `*_samples.npz` so the
two kinds can never be confused. Full schema: [`waveform_format.md`](waveform_format.md).

### 3.4 Retardation: the atom plane lags the drive by τ/2

The acoustic sample that illuminates the beam centre left the transducer half an aperture
transit earlier, so the device layer evaluates every frequency, chirp, envelope and phase at
`t_c = t - τ/2` (`docs/conventions.md` §7). With the default hardware **τ = 11.54 µs**, so
the lag is 5.77 µs — not cosmetic: at 50 MHz/ms it displaces a spot by about three waists.

Two consequences:

* compare a measurement at `t` against the *request* at `t − τ/2`, or pass
  `plan_motion(..., retard_compensate=True)`, which evaluates the trajectory at
  `min(t + τ/2, T)` so the atom plane reproduces the request at `t`;
* a pair-driven tweezer is **strictly dark before τ/2** and only fully lit at τ, because both
  counter-propagating wavefronts must reach a point before the pair diffracts there. That is
  physics, not a numerical artefact — see the FAQ.

### 3.5 The Eq. 1 axial budget, and `f_z_bias`

Holding Z off the focal plane costs a permanent chirp on all four channels, so the drive
frequency simply walks: `f_Z(t) = (1 / 2·lens_scale) ∫ Z dt'`. As Eq. S19 is written the walk
starts at the carrier and is therefore one-sided, giving

```
|∫ Z dt|  ≤  2 · lens_scale · (f_max − f_center)      # aodl.waveform.synthesis.max_z_integral
```

— at the default hardware **206 µs at 10 µm**, before an array ladder or a lateral move takes
its own share of the band. Exceed it and `synthesize` refuses by name, quoting the excursion,
the limit, the requested and feasible `|∫ Z dt|`, and the option that would buy more.

`plan_motion(..., f_z_bias="auto")` offsets the start frequency by −½·max f_Z, centring the
walk in the band: **412 µs at 10 µm**, exactly double. Being common to all four channels and
constant in time, the bias cancels in every Table I quantity — no trap moves. (Bisecting the
longest feasible `Hold` in a 10 µm lift-hold-drop with 2 µs ramps gives 204 µs → 410 µs; the
hold itself overshoots doubling by a hair — ×2.0098 — because the two ramps carry a *fixed*
share of the axial integral, so the hold gets 2·ceiling − ramps rather than 2·(ceiling −
ramps). Longer ramps overshoot further: 40 µs ones give 166 µs → 372 µs, ×2.241.)

### 3.6 Fading-Shepard ladders, and when `"auto"` engages

Beyond that budget the scheme of Eqs. S24–S28 replaces each channel's single tone by a
*ladder* spaced `Δf`, all co-chirping together, each rung switched on only while its fade
coordinate `g = f_Z + (n + ξ)Δf` sits inside a fixed window (`cos^p`, duty η, Eqs. S26/S27).
As `f_Z` walks past `Δf` the pattern of live rungs repeats one index lower: the drive is
periodic in `f_Z` although `f_Z` is periodic in nothing. The *live* excursion is therefore
bounded however long the hold (`shepard_band_bound`), which is the whole claim.

`plan_motion` defaults to `shepard="auto"`: it tries plain Eq. S19 first and falls back to
ladders **only** if the band check fails; `plan.report.description` records which happened
and, on a fallback, the refusal that caused it. Pass `shepard=None` to insist on Eq. S19 (and
get the error instead of a fallback), or a `ShepardConfig` to insist on ladders.

Two properties are worth knowing before you use a Shepard drive:

* `p_A + p_B = 1` keeps the co-located light constant through a hand-over — Table II splits
  it `(½, ½)` for a single tweezer and `(1, 0)` for an array, because the `B` ladder *is* the
  array ladder and must not be shaped;
* the same four lines make two **shadow tweezers** at `± deflection_scale · Δf`, each peaking
  at **half a trap's power** mid-fade (Eq. S31). `plan.report.fade_events` is the timetable:
  do not schedule a pick-up inside a fade zone.

---

## 4. Parameter reference

```python
from aodl import AODLParams, AODParams, OpticsParams, default_1030, paper_808
```

`AODParams` — one deflector:

| Field | Default (preset) | Meaning |
|-------|------------------|---------|
| `sound_speed` | 650 m/s | acoustic velocity `v` (TeO₂ slow shear) |
| `aperture` | 7.5 mm | active aperture `D`; `transit_time` = `D/v` is derived |
| `f_center` | 100 MHz | rotating-frame carrier |
| `band` | (90, 110) MHz | usable **absolute** RF band, checked by `synthesize` |
| `drive_strength` | 0.30 rad | `C·A` at unit envelope — the peak phase modulation of one tone |
| `mixing_order` | 3 | order of the `exp(iCV)` expansion: `1` = fundamentals only, `3` adds compression and the IM3 ghosts at `f_j + f_k − f_i` |

`OpticsParams` — illumination and objective: `wavelength` (1030 nm), `focal_length` (6.5 mm),
`w_in` (2.0 mm, uncropped Gaussian 1/e² *intensity* radius). Derived: `k`, `waist0`
(**w₀ = 1.0655 µm**), `rayleigh` (3.463 µm).

`AODLParams` — the stack: `optics` plus `channels`, a dict with exactly the keys
`("Ax", "Bx", "Ay", "By")`. Derived: `sound_speed` (raises if the four disagree — the Table I
scales assume one), `deflection_scale` = λF/v, `lens_scale` = λF²/v².

Presets: `default_1030()` is the product default (paper hardware at λ = 1030 nm);
`paper_808()` is the same hardware at the paper's 808 nm, for reproducing its figures.

**To model your own hardware**, copy a preset and replace what differs — the dataclasses are
frozen, so `dataclasses.replace` is the idiom:

```python
from dataclasses import replace
from aodl import default_1030
from aodl.units import MHz, mm

P = default_1030()
P = replace(P, optics=replace(P.optics, focal_length=10.0 * mm, w_in=3.0 * mm))
P = replace(P, channels={                      # e.g. a TPM-narrowed band on both A channels
    name: replace(aod, band=(95 * MHz, 105 * MHz)) if name.startswith("A") else aod
    for name, aod in P.channels.items()
})
plan = plan_motion(story, P)                   # every scale above follows automatically
```

Two knobs worth naming explicitly:

* **`mixing_order=1`** — the strictly linear weak-drive model, one tone → one beam. Much
  cheaper (the IM3 census is `O(M³)` in the tone count) and the right model for checking
  Eq. S19 *geometry*. `3` is the product default because a real crystal does compress and
  does make ghosts. Notebooks 04 §5 and 02 §3 quantify the difference.
* **`check_band=False`** — skip the Eq. 1 verification. For *plotting* an infeasible drive
  only: hardware outside its band simply does not diffract. The report says so in `notes`.

---

## 5. Outputs

### 5.1 The waveform file

`plan.save(path)` writes the parametric NPZ of [`waveform_format.md`](waveform_format.md): `meta`
(JSON: schema version, description, a full `AODLParams` snapshot, the driven channels),
`<ch>_segments` (one row per tone-segment: `tone_idx, t0, T, degree, c0…c9`, evaluated in
normalized local time), `<ch>_tones` (`tone_idx, phase0, env_kind, env_p0…p3`), and — schema
v2, fading drives only — `<ch>_env_polys`, the fade coordinate `g(t)`.

`WaveformSet.load(path)` round-trips it exactly (float64-identical coefficients).

**Known limitation.** A drive built with `switch_ramp > 0` carries a `SwitchRamped` envelope
that the v2 schema has no slot for, and `save()` refuses it by name with a `TypeError`
rather than silently dropping the ramp. Re-synthesize from `plan.spec`, or save the un-ramped
drive; a schema v3 slot is on the post-release backlog (`docs/ORCHESTRATION.md`).

### 5.2 Samples for the AWG

```python
window = plan.render_samples(rate=625e6, t_span=(0.0, 2e-6))   # {channel: float32}, ±1
# ... and plan.render_samples(rate=625e6) for the whole drive: 358 174 samples per channel
```

Rendering re-adds the carrier: `V(t) = Σ A_n(t) cos(2π f_center t + phase_n(t))`, divided by
one **global** peak across all four channels so the relative channel amplitudes — which set
the diffraction balance of the stack — survive. Pass `return_scale=True` to recover it, and
use `aodl.waveform.export.save_samples` to write a `*_samples.npz`.

### 5.3 Simulation

`plan.simulate(times=None)` expands the drive into pupil terms at each frame's retarded time,
groups them by optical frequency, and returns a `SimResult` — `times`, `metrics` (a list of
`SpotMetrics` per frame), and lazy `frame` / `slice_xz` / `spot_table` evaluators. `times=None`
uses ~40 frames over `[τ, T + τ/2]`: starting at a full transit skips the fill transient, and
ending half a transit past the trajectory is what lets the *last requested instant* be
observed.

`SpotMetrics`, one per optical-frequency group per frame — one tweezer, except under a fading
drive, where a group can hold a co-located pair on one node *or*, mid-hand-over, two array traps
two pitches apart, whose `x`/`y` is then a mean of two positions rather than either of them
(§6.7):

| Field | Meaning |
|-------|---------|
| `x`, `y` | lab position [m], power-weighted over the group's terms |
| `z_lab` | best-focus lab Z [m] — Table I's `Zbar` |
| `delta_f` | astigmatic interval `Z_x − Z_y` [m] — Table I's `ΔF`, zero for astigmatism-free control |
| `sigma_astig` | `delta_f / z_R`, the paper's dimensionless astigmatism |
| `wx`, `wy` | 1/e² intensity radii at `z_lab` [m] |
| `power` | **incoherent** sum `Σ|c_n|²∫∫|U_n|²`: the weight everything in this package uses |
| `power_coherent` | the exact Gram form `∫∫|Σ U_n|²`: what the group's own rendered frame integrates to |
| `df_opt` | optical-frequency tag of the group [Hz] |

The two powers agree except where *degenerate* terms actually overlap. That happens for the
Fig. S6 shadow-tweezer pair of a simultaneously-fading drive, where `power_coherent` swings
between ½ and 1½ of the incoherent reading under a path offset the experiment does not
control — and it is why Table II interlaces the two axes (`examples/05` §5).

`result.spot_table()` gives the same data in tidy long form,
`{column: array}` with one row per (frame, group), including both power columns.

### 5.4 Movies

```python
# not run by tests/test_docs.py - a render costs tens of seconds; examples/06 runs it
from aodl import auto_grid

run = plan.simulate()
plan.movie("move.mp4", times=run.times, grid=auto_grid(run, long_side=260),
           mode="tracked", fps=15, xz_shape=(104, 72), spectrogram_panel=True, dpi=100)
```

Default view: the XY plane follows the scene's own best focus (`mode="fixed"` pins it at the
lab focal plane instead), **hue carries each group's lab Z**, an XZ slice sits beside it, and
the four channel drives run underneath. `.mp4` needs `imageio-ffmpeg`; without it the writer
falls back to `.gif` and returns the path it actually wrote.

The XZ panel is the one part of a frame that cannot be patched — a spot sweeps *through*
focus along its Z axis — so it costs `nx·nz` per frequency group. Trading `xz_shape` and
frame count is how a hundred-trap scene stays inside a render budget.

### 5.5 Checking a rendered drive

Everything in §5.3 runs one way: trajectory → waveform → Taylor-expanded pupil terms →
closed-form Gaussians. A sign error shared by the synthesizer and the simulator cancels out of
every test that goes through both. `plan.check()` closes that loop from outside — it renders
the drive to **RF samples**, measures the drive back off them with an FFT, rebuilds the
aperture field point by point with no expansion at all, propagates it with a chirp-z transform
and fits the spots:

```python
# not run by tests/test_docs.py - a check renders the whole drive; examples/07 runs it
report = plan.check()                  # ~9 deterministic frames, the full exp(iCV) crystal
print(report.summary())                # verdict, worst residual per metric, blobs, caveats
assert report.passed
table = report.table                   # long format, one row per (frame, trap)
```

The rebuild imports nothing from `field/`, `device/aod*`, `engine` or the waveform IR — a
source scan enforces it (`tests/test_check_independence.py`) — so the only things the two
paths share are `params.py` and the sign table in `device/conventions.py`.

`check_samples(samples, expect, ...)` is the same thing one level down, for samples that came
from somewhere else: pass a `*_samples.npz` path (or a `SampleRecord`) and an `Expectation`.
`Expectation.from_table(times, x, y, z, array, params)` builds one from a *measured* trajectory
rather than a `TrajectorySpec`.

**What the report carries**

| Field | What it holds |
|-------|---------------|
| `passed`, `failures` | the verdict, and one line per violated gate — each naming its metric first |
| `table` | `{column: array}`, one row per (frame, trap): fitted `x`, `y`, `z_lab`, `delta_f`, `wx`, `wy`, `peak`, `power`, `beat_std`, the residuals `dx`/`dy`/`dz`, model-free profile moments, each trap's `uniformity_median` over the frames, and the flags saying which gates applied |
| `blobs` | every canvas maximum that is *not* a requested trap, with `on_lattice` |
| `worst()` | `{metric: (residual, tolerance, offender)}` — the numbers `summary()` prints |
| `median_uniformity()` | worst per-trap **time median** of the intensity deviation, and whose |
| `gated_fraction` | `{metric: gated rows / rows}` for `waist`, `uniformity`, `uniformity_median` — what fraction of the array each intensity gate actually judged |
| `notes` | exclusions applied, the blind spot, the drive-specific caveats |
| `out_of_band`, `sim_delta` | report-only diagnostics (splatter; a diff against `plan.simulate()`) |

**Tolerances** (`aodl.check.Tolerances`, all relative):

| Gate | Default | Measured against |
|------|---------|------------------|
| `lateral` | 0.05 | the focal waist `w₀` — 53 nm |
| `axial` | 0.05 | the Rayleigh range `z_R`; also gates \|ΔF\| |
| `waist` | 0.02 | `w₀`, on non-transient, non-fade frames |
| `uniformity` | 0.03 | each frame's median trap peak, non-fade frames |
| `uniformity_median` | `None` | the same deviation, medianed **over the frames**; `None` measures it without gating (see below) |
| `blob_off_lattice` / `blob_on_lattice` | 0.01 / 0.10 | the median trap peak |
| `blob_fading` | 1.2 | the median trap peak, for on-lattice light on a *fading* drive — whitelisted, but not above a real trap's depth |
| `missing_trap` | 0.25 | the median trap peak, below which a trap is "missing" |
| `require_coverage` | `False` | not a threshold: fail when an intensity gate judged **no row at all** |

**Two ways to read the intensity pattern.** `uniformity` asks, one frame at a time, whether a
trap is off its frame's median. `uniformity_median` asks whether it is off at *every* frame —
the median of that deviation over the frames a trap was gated at, `nan` for a trap gated on
fewer than two. The two have different blind spots: a one-frame excursion medians away, and a
standing offset does not. On an Eq. S19 drive the second is the sharper instrument — one tone
5 % down gives a median of 0.098 against a clean 0.012, an 8× separation
(`tests/test_check_verdict.py`) — so set `uniformity_median` there and get a second gate.

It is `None` by default because on a **fading-Shepard** drive it separates nothing. The ladder
slides through the array, so a faulted rung feeds a different column at every frame: on the
flagship, one rung at 80 % amplitude (a 36 % intensity fault) reads 0.28 at one frame and is
erased by the median exactly — clean and corrupted both measure 0.137, which is the *static*
Eqs. S20–S22 pattern rather than noise. That fault also passes the opened 0.30 per-frame gate,
so it is currently uncaught, and `tests/test_check_flagship.py` carries it as a strict `xfail`
rather than pretending otherwise.

**What a PASS certifies — and what it does not.**

* **Not the absolute intensity.** `render_samples` divides all four channels by one global
  peak, and a common gain only rescales the image, so a drive rendered at half amplitude is
  indistinguishable. Only the *pattern* of intensity is gated. This is the one blind spot, and
  the report says so in `notes`.
* **Not a drive whose two spacings are equal.** With `Δf_x == Δf_y` every anti-diagonal of the
  array shares one optical frequency, so those traps are mutually coherent and their beat note
  is exactly *zero* — no averaging window removes it, and what the checker measures is the
  interference, correctly (§6.8, `docs/conventions.md` §4). Give the two axes different
  spacings; the flagship's 1.0/1.3 MHz is why it has none.
* **Not the edges of a fading array.** On a fading-Shepard axis the `p_A + p_B = 1` identity
  holds every *interior* node exactly flat through a hand-over (measured to 1e-15), but the
  ladder slides: the outermost node trades its light with the extended grid (§6.6/§6.7), so
  the intensity gates skip the two edge lines and the report names them.
* **Not a small fading array's intensity at all — check `gated_fraction`.** That edge-line
  exemption costs the perimeter, which on a large array is a fringe (the 10×10 flagship keeps
  8×8 = 64 % of its rows) and on a small one is everything. A fading **2×2** has no interior
  trap: `waist`, `uniformity` and `uniformity_median` gate *zero* rows and the drive passes them
  by having nothing to say. The report never hides this — `gated_fraction` is `{…: 0.0}`,
  `summary()` prints `NONE GATED` and a note spells it out — and passing
  `Tolerances(require_coverage=True)` turns it into a failure. Small fading arrays also need
  their intensity gates *opened* per drive on physics grounds: a clean fading 3×3 on the
  flagship's own trajectory measures `waist` 0.080 and `uniformity` 0.069 against defaults of
  0.02/0.03, on the one interior trap (11 % of its rows) it has left.
* **Not a timing skew below a tenth of a microsecond.** Delay one channel's samples by `δ` and
  every trap it feeds moves by `|dX| = deflection_scale · |ḟ| · δ`, with `ḟ` that channel's own
  frequency slew — so the checker sees a skew only once that product clears the `lateral` gate.
  Measured on the flagship (its `Ax` buffer rolled by `δ`): 0.030 `w₀` at 0.05 µs, 0.149 at
  0.30 µs, 0.58 at 1.22 µs, i.e. the default 0.05 `w₀` gate is crossed at **≈ 0.1 µs**. A
  *common* delay on all four channels is a pure time shift of the whole drive and is harder
  still: it clears `lateral` only at ≈ 0.15 µs and `axial` only at ≈ 1.22 µs. Below those a
  skew is invisible, which matters because a skew is exactly what a mis-wired AWG produces.
* **Not the crystal's own nonlinearity.** The default `bragg_band` model *includes* compression
  and intermodulation, so those show up as real residuals rather than as errors: a 10×10
  Shepard array at `drive_strength = 0.30` renders with a normalization factor (peak over
  single-tone amplitude) of 4.59 and its per-trap intensity spreads by ~20 % from Eqs. S20–S22,
  with the spots widening up to 8 % mid-hand-over (`ρ = 0.30`, §6.4). Raise
  `uniformity`/`waist` for such a drive. Driving it more weakly removes only the Eqs. S20–S22
  part, which is most of it but not all: ~82 % of the spread scales as `C²`, while the
  fade-speed part does not depend on `C` at all and leaves a floor of ~3.8 % on the spread and
  the whole 8 % on the waist (measured on the flagship down to `drive_strength = 0.003`). Widen
  `Δf` — i.e. lower `ρ` — to move the floor.

Frames before `2τ`, or with an aperture still filling, are marked *transient* and leave the
waist and uniformity gates; while an aperture is genuinely filling the positions leave them
too. `mode="weak"` swaps the full crystal for the linear Eq. S3 model the simulator implements
— that is the cross-validation path, not the verdict path.

---

## 6. Fidelity and limits

Everything here is a *modelling* statement, with the number that quantifies it. None of it is
hidden inside the code: the same list, drive-specific, comes back in `plan.report.notes`.

1. **Weak-drive expansion (Eqs. S20–S22).** Diffraction is `exp(iCV)` expanded to
   `mixing_order` (default 3): fundamentals, compression (~`C²/8` per tone, ≈1 % at the
   default `drive_strength = 0.3`) and the IM3 ghosts at `f_j + f_k − f_i`. Full coupled-mode
   Bragg theory is deliberately out of scope, as is the measured efficiency ridge of Fig. S8
   (efficiency is flat across the band here — a per-channel calibration hook, later).
   Schroeder phases (Eq. S23/S28) suppress ghosts by 57–437× against random phases in M2's
   measurement.

2. **Quadratic pupil phase.** The retarded phase is Taylor-expanded to second order in the
   aperture coordinate: deflection + cylindrical lens, exact for linear chirps, with the cubic
   (coma) term dropped. `field/reference.py` — a direct quadrature of Eq. S11, tests only —
   bounds the error; the M0 check agrees with it to ~1e-15 relative on static tones.

3. **Uncropped Gaussian input.** The `|u| ≤ D/2` crop is deliberately not applied
   (`docs/PLAN.md` decision 2); the only aperture window in the field integrals is the
   *fill* edge of the travelling wavefront, which is exact.

4. **Fade speed, ρ.** The degree-2 amplitude expansion `(1, −sA′/v, A″/2v²)` describes an
   envelope only while it changes slowly compared with the beam transit. The measure is
   `ρ = (w_in/v) / T_fade` with `T_fade = η Δf / |ġ|`. Total co-located power then ripples as
   ρ², measured over a decade in `examples/05` §7: **1 % flatness at ρ ≈ 0.057**, i.e.
   Δf ≈ 5.25 MHz for a 10 µm hold, and 0.43 % at the 8 MHz ladder used there. It is physics,
   not error — a fade fast enough to apodize the pupil does change the trap's coupling — but
   it *is* the validity edge of the expansion. Wide ladders are both cheaper and slower to
   fade; `auto_config` maximizes Δf for that reason. An **array** axis has no such freedom —
   the Shepard ladder *is* the array ladder (Eq. S27), so the pitch fixes Δf — and its fades
   are correspondingly fast: the 10×10 story of `examples/06` runs at ρ = 0.30 with its
   interior traps rippling 7.6 % per hand-over. That is *better* than extrapolating the
   single-tweezer law (which would say ~28 %), because an array's `(p_A, p_B) = (1, 0)` pair
   has no divergent shoulder to clamp. Measure your own array rather than extrapolating.

5. **`α₁` tilt term at the fade edge.** Where the `cos^p` shoulder's log-derivative diverges
   (`p < 1`), `FadeZoneEnvelope` freezes the *shape* at `|A′/A| ≤ p·SLOPE_CLAMP·dθ/dt`. The
   residual weight that leaves behind is *estimated* by `S²(1+S²)^{−p}(p(π/2)ρ)²/4`; at the
   `examples/05` design point (p = ½, ρ = 0.0373) that estimate is 6.1e-4 against a measured
   8.6e-4 — an estimate, not a bound, and about 30 % optimistic.

6. **`p_B = 0` rectangles.** An array's `B` ladder must not be shaped (Table II), so its rungs
   switch on and off instantaneously and radiate roughly **−40 dB** of out-of-band splatter,
   with a switching period shorter than the aperture transit. `switch_ramp=<seconds>` replaces
   the step with a raised cosine, on **the rectangles only** — the `cos^p` `A` windows already
   reach zero smoothly, so ramping them bought no continuity and cost flatness. Scoped that way
   a ramp leaves the interior of the array *exactly* where it was: the only column that moves is
   the one the switching rung is arriving at (or leaving) — plus, on an **odd**-M ladder, whose
   rungs switch in step with an `A` hand-over, the array's own two **edge** columns, which dip by
   ~`(π|ġ|r/Δf)²/5` (**1.54 %** at `r = 3 µs` on a 1 MHz ladder holding 6 µm, 20 % at `r = τ`).
   An **even**-M ladder switches mid-plateau, where the `A` ladder is not handing over, and pays
   nothing at all. Such a drive does not round-trip through the NPZ (§5.1). The model also treats
   the step as instantaneous *across the whole aperture* rather than travelling through it — a
   fidelity item, tracked in the backlog.

7. **Extended grid, and pick-up scheduling.** A fading array is wider than the array you
   asked for: **`M + 2` columns during a hand-over for odd M, `M + 1` at every instant for
   even M**. Shadow tweezers sit at `± deflection_scale · Δf` and reach half a trap's power
   mid-fade. `plan.report.fade_events` lists every hand-over with its axis and shadow offset —
   schedule pick-ups on the plateaus between them.

   **A mid-fade group can hold two real traps.** A term's optical-frequency tag is the *sum* of
   its channel frequencies, while its position is the per-axis *difference*
   (`docs/conventions.md` §4–5): the rung pairs `(a, b)` and `(a−1, b+1)` therefore share a tag —
   `a + b` — but sit **two pitches apart**. Both `A` rungs are live during a hand-over, so the
   lit columns pair up two apart and `SpotMetrics.x`/`y`, being power-weighted means over a
   group, report the point halfway between two traps, where there is no light. (A different
   degeneracy from the equal-spacing one of §6.8, which merges whole anti-diagonals.) Read group
   positions on the plateaus, and track individual traps mid-fade **per term** — build the terms
   with `aodl.device.aodl.build_terms` and bin them by position, as the M3/M4 integration tests
   do — rather than by group.

8. **Interlacing is exact only for equal spacings.** `ξ = ½` tiles the x fade zones into the y
   plateaus when `Δf_x = Δf_y`. `auto_config` sizes each free axis against its own headroom
   and so can return `Δf_x ≠ Δf_y`, whose schedules beat: some hand-overs then light both axes
   at once (the sixteen-ray, phase-sensitive case). Force equal spacings with an explicit
   `ShepardConfig(Δf, Δf)` when that matters.

9. **Ideal geometry.** Perfectly overlaid apertures, ideal 4f with magnification −1, matched
   acoustic delays (`x_err = 0` in Eq. S29), scalar paraxial optics, ideal objective. Atom
   dynamics (Eq. S13) are not modelled: the output is the light, not the atoms.

---

## 7. FAQ

**Why is my spot dark before τ/2, and dim until τ?**
Because a counter-propagating pair only diffracts where *both* acoustic waves have arrived.
The aperture fills from the transducer side at `v`, so the two filled half-lines do not
intersect until `τ/2 = 5.77 µs` and the aperture is not full until `τ = 11.54 µs`
(`docs/conventions.md` §7). `plan.simulate()`'s default grid starts at τ for exactly this
reason. Frames before the drive starts, or past a tone's programmed span, are refused rather
than faked — extend the drive with `WaveformSet.with_hold_until` instead.

**Why did synthesis refuse my move?**
Almost always Eq. 1: a sustained Z costs `48.54 MHz/ms` at 10 µm on *every* channel, and an
array ladder plus the lateral term have already spent part of the band. The error names the
channel, the excursion, the band, the requested and feasible `|∫ Z dt|`, and it quotes the
doubled ceiling that `f_z_bias="auto"` would buy. Three ways out, in order of cost: shorten
the hold; pass `f_z_bias="auto"` (§3.5); let `shepard="auto"` fall back to the ladders, which
is what `plan_motion` does by default.

**Why does my 10-column array show 11 columns during a fade?**
Because a fading ladder always has a rung on the way in or out. For **even** M the extended
grid is `M + 1` wide at every instant; for **odd** M two extra columns switch on together
during a hand-over, giving `M + 2`. Those extra columns are real light at (for an array's
`p_B = 0` ladder) full depth — treat them as traps you did not ask for, not as artefacts, and
keep pick-ups off them via `plan.report.fade_events`.

**Should Δf_x and Δf_y be equal?**
For an *array* both are fixed by your `ArraySpec` — and there, unequal is usually what you
want: with `Δf_x == Δf_y` every anti-diagonal shares one optical frequency `f_x + f_y`, so
those traps become mutually coherent and the simulator reports one group per anti-diagonal
instead of one per trap (`docs/conventions.md` §4). For a *free* (single-tweezer) axis under
`shepard="auto"`, equal spacings are instead what makes interlacing exact (§6.8). The two
concerns never bind the same axis: an array axis is decided by its pitch (Eq. S27), and only
an axis without a ladder is left for `auto_config` to size.

**Which mode did I get, and how close to the edge am I?**
`plan.report.mode` (`"s19"` or `"shepard"`) and `plan.report.worst_margin`, both printed by
`summary()`. `plan.report.figure()` shows the same thing as a picture.

**How do I know the simulation is right, and not just self-consistent?**
Run `plan.check()` (§5.5). It renders the drive to RF samples and rebuilds the tweezers through
a completely separate path — one FFT per channel to measure the drive, the literal `exp(iCV)`
crystal with the `+1` order cut out in the aperture's spatial-frequency domain, a chirp-z
transform to the image plane — sharing only `params.py` and the sign table. It reports where
every trap actually is against where you asked for it, and it fails loudly when they disagree.
Two things it will tell you that the simulator cannot: the crystal's *nonlinear* per-trap
intensity spread (Eqs. S20–S22 to all orders, not just IM3), and the cubic pupil term Eqs.
S5–S6 drop — on a hurried 25/30/25 µs move that term moves the light **0.07 w₀** off Table I's
prediction, while the same move at 150/250/150 µs comes back inside 0.01 w₀.

**Where is the physics derived?** In the notebooks, each of which verifies its own claims
against closed form:

| Notebook | What it establishes |
|----------|---------------------|
| [01_single_aod_sweep](../examples/01_single_aod_sweep.ipynb) | one AOD: deflection tracks `f(t − τ/2)`, a chirp is a cylindrical lens, the fill transient lasts a beam transit |
| [02_crossed_pair_diagonal](../examples/02_crossed_pair_diagonal.ipynb) | two crossed AODs: arrays, diagonal moves → spherical defocus, IM3 ghosts and Schroeder suppression |
| [03_aodl_3d_motion](../examples/03_aodl_3d_motion.ipynb) | the full stack: pure-Z co-chirp, in-plane motion with zero focal shift, an "L" in 3D, and the Eq. 1 wall |
| [04_array_lift_traverse](../examples/04_array_lift_traverse.ipynb) | the user story at Eq. 1 pace: deliverables, tracking, `mixing_order=3` census |
| [05_fading_shepard](../examples/05_fading_shepard.ipynb) | ladders, windows, shadow tweezers, interlacing, the unhurried story, ρ and compression caveats |
| [06_product_tour](../examples/06_product_tour.ipynb) | the product path: `plan_motion` → report → NPZ + samples → simulation → movie |
| [07_fft_checker](../examples/07_fft_checker.ipynb) | the independent checker: what it rebuilds from the samples, the measured `2J₁(C)/C` compression, the blob audit, and a drive broken on purpose |

---

## 8. Numbers this guide quotes

Every row is checked against the code by `tests/test_docs.py`, with `P = default_1030()` and
the constants of `aodl.units`:

| Quantity | Value | Expression |
|----------|-------|------------|
| tweezer pitch per MHz | 10.3 µm | `P.deflection_scale * MHz / um` |
| focal waist `w₀` | 1.0655 µm | `P.optics.waist0 / um` |
| Rayleigh range `z_R` | 3.463 µm | `P.optics.rayleigh / um` |
| aperture transit `τ` | 11.54 µs | `P.channels["Ax"].transit_time / us` |
| half transit `τ/2` | 5.77 µs | `0.5 * P.channels["Ax"].transit_time / us` |
| beam transit `w_in/v` | 3.077 µs | `P.optics.w_in / P.sound_speed / us` |
| co-chirp for `Z = 10 µm` | 48.54 MHz/ms | `10 * um / (2 * P.lens_scale) / (MHz / ms)` |
| Eq. 1 budget at `Z = 10 µm` | 206 µs | `max_z_integral(P) / (10 * um) / us` |
| … with `f_z_bias="auto"` | 412 µs | `max_z_integral(P, biased=True) / (10 * um) / us` |
| checker aperture cell `Λ/8` | 0.8125 µm | `P.sound_speed / (8 * P.channels['Ax'].f_center) / um` |
| checker aperture half-span | 4.992 w_in | `24576 * P.sound_speed / (8 * P.channels['Ax'].f_center) / 2 / P.optics.w_in` |
| a frame's drive reach `half_span/v` | 15.36 µs | `24576 / (16 * P.channels['Ax'].f_center) / us` |
| lateral gate `0.05 w₀` | 0.0533 µm | `0.05 * P.optics.waist0 / um` |
| axial gate `0.05 z_R` | 0.1732 µm | `0.05 * P.optics.rayleigh / um` |

Measured by the checker and quoted in §5.5: the flagship's ~20 % per-trap intensity spread and
8 % waist swing at `drive_strength = 0.30`, its 0.137 time-median intensity pattern — the same
clean or with a rung at 80 % — and its 64 % intensity-gate coverage
(`tests/test_check_flagship.py`); the 0 % coverage of a fading 2×2 and the 0.012 → 0.098
time-median separation of a 5 % tone fault on the Eq. S19 3×3 (`tests/test_check_verdict.py`);
the 0.07 w₀ departure from Table I on a 25/30/25 µs move, and the 1e-9 field agreement with the
simulator at a min-jerk midpoint (`tests/test_check_weak_vs_sim.py`).

Measured in the WO-24 gate-policy pass and quoted in §5.5: the flagship's 4.59 render
normalization factor (peak over single-tone amplitude); the `waist` 0.080 / `uniformity` 0.069
/ 11 % coverage of a clean fading 3×3 on the flagship trajectory; the ≈ 0.1 µs one-channel and
≈ 0.15 µs (`lateral`) / ≈ 1.22 µs (`axial`) common timing-skew detection thresholds.

Measured elsewhere and quoted above: ρ ≈ 0.057 for 1 % flatness and 0.43 % at 8 MHz
(`examples/05` §7); ρ = 0.30 and a 7.6 % per-hand-over ripple for the array of
`examples/06` §5; the 8.6e-4 residual weight and the −40 dB splatter (M4 verification,
`docs/ORCHESTRATION.md`); 57–437× Schroeder ghost suppression (M2); the 204 µs → 410 µs
bisected hold under `f_z_bias` (M5); the 1.54 % and 20 % edge-column dips of a `switch_ramp`
at `r = 3 µs` and `r = τ`, checked against the closed form in
`tests/test_synthesis_options.py`.
