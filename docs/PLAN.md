# AODL Simulation & Waveform Synthesis — Project Plan

**Goal:** a deliverable software package for AMO labs that turns a high-level request —
*"move this 10×10 atom array from A to B, lifting 10 µm out of plane on the way"* — into
(1) the RF waveforms to program on each AOD channel of a 3D-AODL, and
(2) a physically grounded simulation + movie of the resulting tweezer motion.

Physics reference: Lu, Song, Xiang, Ho, Lee, Yan & Stamper-Kurn,
*"Astigmatism-free 3D Optical Tweezer Control for Rapid Atom Rearrangement"*
(arXiv:2510.11451). Equation numbers `S#` below refer to its Supplement.

---

## 1. Physics foundation

### 1.1 Device model

Four AODs with overlaid apertures, acoustic propagation directions rotated by 90°:

| Channel | Sound direction | Role |
|---------|----------------|------|
| `Ax` | −x | first x-AOD (single-tone) |
| `Bx` | +x | second x-AOD (multi-tone, defines columns) |
| `Ay` | −y | first y-AOD (single-tone) |
| `By` | +y | second y-AOD (multi-tone, defines rows) |

Per the design brief we assume: apertures perfectly overlaid (ideal 4f, magnification −1),
acoustic delays matched (`x_err = 0` in Eq. S29), scalar paraxial optics, ideal objective.

Each channel is driven by a real RF waveform

$$V_\mu(t) = \sum_n A_\mu^{(n)}(t)\cos\!\Big(2\pi\!\int_0^t f_\mu^{(n)}(t')\,dt' + \phi_\mu^{(n)}\Big),\qquad \mu \in \{Ax, Bx, Ay, By\}.$$

**The aperture window is the central object.** The acoustic field on AOD µ at time *t* is the
waveform segment $V_\mu(t \mp u/v)$ for aperture coordinate $u \in [-D/2, D/2]$ (sign set by
sound direction). All transit-time physics — chirp lensing, startup transients, irising —
comes from this retarded-time window.

### 1.2 Diffraction model (weak drive + controlled mixing)

Single AOD, +1 order, weak-drive limit (Eq. S1–S6). Writing the drive in the rotating frame
of the center frequency and Taylor-expanding the retarded phase around the beam center:

$$P^{(1)}(u,t) \propto \tilde A(t \mp u/v)\,
\exp\!\Big(\pm 2\pi i \tfrac{f(t)}{v}u \;-\; 2\pi i \tfrac{\dot f(t)}{2v^2}u^2 \;\pm\; 2\pi i \tfrac{\ddot f(t)}{3!\,v^3}u^3 - \cdots\Big)$$

- linear term → **deflection** (∝ instantaneous frequency at beam center),
- quadratic term → **cylindrical lens** with dioptric power $P = \lambda \dot f / v^2$
  (sign independent of sound direction — this is what makes counter-propagating pairs work),
- cubic term → coma (neglected in the fast path, checked by the reference integrator),
- amplitude Taylor terms → intensity control (0th), tilt (1st), **acoustic irising** (2nd).

Stacking all four AODs multiplies the pupils (Eq. S7) and yields (Eq. S8):

$$X\text{-deflection} \propto \frac{f_{Bx}-f_{Ax}}{v}x,\quad
Y\text{-deflection} \propto \frac{f_{By}-f_{Ay}}{v}y,$$
$$\text{spherical lens} \propto \frac{\dot f_{Ax}+\dot f_{Bx}+\dot f_{Ay}+\dot f_{By}}{4v^2}(x^2+y^2),\quad
\text{astigmatism} \propto \frac{\dot f_{Ax}+\dot f_{Bx}-\dot f_{Ay}-\dot f_{By}}{4v^2}(x^2-y^2).$$

Control mapping (paper Table I) — the heart of waveform synthesis:

| Quantity | Expression |
|----------|------------|
| X position | $X = \frac{\lambda F}{v}(f_{Bx} - f_{Ax})$ |
| Y position | $Y = \frac{\lambda F}{v}(f_{By} - f_{Ay})$ |
| Z position | $\bar Z = \frac{1}{2}\frac{\lambda F^2}{v^2}(\dot f_{Ax}+\dot f_{Bx}+\dot f_{Ay}+\dot f_{By})$ |
| Astigmatic interval | $\Delta F = \frac{\lambda F^2}{v^2}(\dot f_{Ax}+\dot f_{Bx}-\dot f_{Ay}-\dot f_{By})$ |

Astigmatism-free 3D control = keep $\Delta F = 0$; 3 remaining DOF ↔ (X, Y, Z).

**Frequency mixing.** Two distinct, both modeled:

1. **Inter-AOD products** (automatic): the pupil is a *product* of per-AOD tone sums, so every
   tone combination across the four AODs produces a beam. With 2 tones per AOD this is the
   16-ray picture of Fig. S6 — this is where *shadow tweezers* come from during fading.
2. **Intra-AOD intermodulation** (Eq. S20–S22): expanding $e^{iCV}$ beyond first order,
   IM2 at $f_i \pm f_j$ (out of band) remixes to in-band IM3 at $f_i + f_j - f_k$, producing
   ghost tweezers and per-trap intensity errors. We keep the expansion order configurable
   (default: through 3rd order, i.e. IM3), with drive strength $C$ calibrated from a target
   single-tone diffraction efficiency. The Schroeder phase (Eq. S23/S28) is implemented in
   synthesis and its IM3 suppression is verifiable in simulation.

Model simplifications (documented, revisitable): flat diffraction efficiency across the band
(the measured efficiency ridge of Fig. S8 becomes a per-channel calibration hook later);
weak-drive perturbative mixing rather than full coupled-mode Bragg theory.

### 1.3 Focal field by direct equations — no FFTs

Objective (focal length F) performs a Fourier transform; at defocus Z (Eq. S11):

$$U(X,Y,Z,t) \propto \iint U_{in}(x,y)\,P(x,y,t)\,
e^{-\frac{ik}{F}(xX+yY)}\;e^{-\frac{ikZ}{2F^2}(x^2+y^2)}\,dx\,dy$$

After term expansion, **every term** in P is (polynomial) × Gaussian × exp(linear + quadratic
phase), separable in x and y. With a Gaussian input beam each 1D factor is the closed form

$$\int (c_0 + c_1 u + c_2 u^2)\, e^{-a u^2 + b u}\,du
= \sqrt{\tfrac{\pi}{a}}\;e^{b^2/4a}\Big[c_0 + c_1\tfrac{b}{2a} + c_2\big(\tfrac{1}{2a}+\tfrac{b^2}{4a^2}\big)\Big],\qquad \mathrm{Re}(a)>0,$$

where $a$ packs the beam radius + chirp lens + defocus Z, and $b$ packs deflection + image
coordinate. So the field at any (X, Y, Z, t) is a **sum of closed-form astigmatic Gaussians**
— exact for quadratic phase (i.e. exact for linear chirps, excellent for smooth ramps), fast
enough to render movies on dense grids, and with defocus/astigmatism/irising built in.
Polynomial amplitude prefactors (Gaussian moments) capture the irising and tilt terms exactly.

**Interference done right:** each term carries an optical frequency offset (sum of its tone
frequencies with signs). Atoms and cameras see intensity averaged over MHz beat notes, so we
group terms by instantaneous frequency (within a tolerance) and compute
$I = \sum_g \big|\sum_{k\in g} U_k\big|^2$. Frequency-degenerate terms (e.g. the static
Mach–Zehnder-like shadow-tweezer pairs during simultaneous fading, Fig. S6) interfere;
non-degenerate ones add in intensity. This reproduces the paper's interlaced-fading rationale.

**Validation backend:** a deliberately dumb direct-quadrature evaluation of Eq. S11 (the
paper's own approach, ~100 aperture points) lives in `field/reference.py`, used only in tests
to bound the error of the Taylor truncation (coma) and of the uncropped-Gaussian assumption.

### 1.4 Waveform synthesis (trajectory → RF)

Rigid-array translation along $(X(t), Y(t), Z(t))$ (Eq. S19):

$$f_{Ax}(t) = f_0 - \tfrac{v}{2\lambda F}X(t) + f_Z(t),\qquad
f_{Bx}^{(n)}(t) = f_{x0}^{(n)} + \tfrac{v}{2\lambda F}X(t) + f_Z(t),$$

(similarly for y), with $f_Z(t) = \tfrac{v^2}{2\lambda F^2}\int_0^t Z\,dt'$ and array tones
$f_{x0}^{(n)} = f_0 + (n - \tfrac{M_x+1}{2})\Delta f$ carrying Schroeder phases. Time profiles
per segment: minimum-jerk (default), constant-jerk, constant-acceleration, SCJ, linear
(Eqs. S14–S17).

Sustained Z ≠ 0 costs bandwidth at rate $\dot f_Z = \tfrac{v^2}{2\lambda F^2}Z$ on all four
channels (Eq. 1) — the synthesizer checks per-channel band limits (±10 MHz default,
per-channel configurable) and either raises a clear error with the max feasible move, or (milestone 4)
switches to **fading-Shepard waveforms** (Eqs. S24–S28): tone ladders spaced Δf, cos^p fade
envelopes with duty η, x/y fading zones interlaced (ξ offset ½), p_A = 1 / p_B = 0 for arrays
so in-array intensity stays constant.

### 1.5 Default parameters (paper hardware, user-adjustable dataclasses)

| Parameter | Default | Source |
|-----------|---------|--------|
| Optical wavelength λ | **1030 nm** | product spec (paper used 808 nm — preset provided to reproduce its figures) |
| Acoustic velocity v | 650 m/s | TeO₂ slow shear, AA Opto DTSX(Y)-400 |
| Active aperture D | 7.5 mm | DTSXY-400 |
| Acoustic transit τ = D/v | 11.54 µs | derived |
| Center frequency f₀ | 100 MHz | paper |
| Usable band | ±10 MHz, all four channels | decision (TPM-narrowed A-band remains a config knob) |
| Objective focal length F | 6.5 mm | paper (effective F*) |
| Input beam 1/e² radius w_in | 2.0 mm (Gaussian, uncropped) | → w₀ ≈ 1.07 µm at 1030 nm (cf. measured 1.1 µm* at 808 nm) |
| Sample rate (AWG export) | 625 MS/s (configurable) | Spectrum M4i.6631-x8 |

Handy scales at these defaults: **1 MHz ↔ 10.3 µm lateral**; co-chirping all four channels at
β moves the focus to $\bar Z = 2\tfrac{\lambda F^2}{v^2}\beta$ → **10 µm requires
β ≈ 48.5 MHz/ms per channel**, so holding Z = 10 µm burns the ±10 MHz band in ≈ 400 µs —
short lift-move-lower sequences fit without Shepard tones; longer holds motivate milestone 4.
Rayleigh range z_R ≈ 3.5 µm; transit across the beam 2w_in/v ≈ 6 µs (startup transient time).

---

## 2. Architecture

```
User trajectory spec ("10×10 array, lift 10 µm, go to B, drop")
        │  trajectory/spec.py + ramps.py
        ▼
WaveformSet — per-channel ToneTracks (piecewise-poly freq, amp envelope, phase)
        │           │
        │           └─▶ waveform/export.py → sampled arrays / AWG files  ← the lab deliverable
        ▼
device/aod.py — aperture window V(t ∓ u/v); retarded f, ḟ, envelope at beam center
        ▼
device/aodl.py + mixing.py — 4-AOD pupil term expansion (tones × orders × channels)
        ▼
field/focal.py — closed-form Gaussian focal field per term; frequency-group intensities
        ▼
viz/movie.py + field/measure.py — 2D movie (z → color), XZ slice, spectrograms, metrics
```

```
src/aodl/
  params.py            # AODParams, OpticsParams, RFParams, presets (paper_808, default_1030)
  units.py             # MHz/µm/µs helpers, MHz↔µm and chirp↔Z calibration
  trajectory/
    spec.py            # ArraySpec + waypoint segments (lift / translate / lower / hold)
    ramps.py           # Eqs. S14–S17 time profiles
  waveform/
    tones.py           # ToneTrack: exact phase/freq/chirp evaluation + sample rendering
    synthesis.py       # Eq. S19 solver, band checking, Schroeder phases (S23/S28)
    shepard.py         # fading-Shepard ladders (S24–S28)          [M4]
    export.py          # NPZ/CSV/Spectrum-format export
  device/
    aod.py             # one AOD: orientation, sound sign, window extraction
    aodl.py            # the stack; per-term Taylor coefficients (amp, tilt, lens)
    mixing.py          # weak-drive expansion, IM3 product enumeration    [M2+]
  field/
    gaussian.py        # the closed-form 1D integrals (poly × Gaussian × quadratic phase)
    focal.py           # U(X,Y,Z,t) term sum; frequency grouping; intensity frames
    measure.py         # spot centroid, waists, σ_astig, per-group Z-focus
    reference.py       # direct-quadrature Eq. S11 (tests only)
  viz/
    movie.py           # XY movie @ fixed plane or focus-tracked; hue ↔ Z; panels
  api.py               # one-call front door: spec → (waveforms, movie, report)
tests/                 # pytest; analytic-vs-reference, formula regressions
examples/              # scripts matching milestones M1–M5
docs/                  # this plan, physics notes, user guide
```

Conventions: Python 3.11+, numpy/scipy/matplotlib only, SI units internally with explicit
helpers, dataclasses for configs, `src` layout with `pyproject.toml`, pytest + ruff, every
physics function documented with the equation it implements (`Eq. S8`, etc.).

---

## 3. Milestones (build & verify ladder)

Each milestone = code + example script + quantitative pytest against closed-form physics.

**M0 — Scaffold + field core.**
Package skeleton, params/units, `field/gaussian.py` + `field/reference.py`.
✓ Analytic Gaussian focal field matches direct quadrature to <10⁻⁶ (relative) for
static tones; waist/z_R match textbook formulas.

**M1 — One AOD, moving tweezer (astigmatic).**
Single `Ay` channel, static tone then linear chirp with min-jerk ramps; aperture-window
startup transient included.
✓ Position tracks $\tfrac{\lambda F}{v} f(t-\text{retard})$; ✓ fitted focal split matches
$\Delta F = \tfrac{\lambda F^2}{v^2}\dot f$; ✓ movie shows one astigmatized spot elongating
during the sweep; ✓ transient lasts ≈ beam transit time.

**M2 — Two crossed AODs (conventional 2D-AOD) + arrays + mixing.**
`Ax`+`Ay` (or equivalently one 2D-AOD): diagonal moves, multi-tone arrays, IM3.
✓ Diagonal chirp → equal x/y lensing = **spherical** defocus (out-of-plane excursion, no
astigmatism), reproducing Fig. 2 phenomenology; ✓ single-axis chirp → cylindrical;
✓ N-tone array spacing = $\tfrac{\lambda F}{v}\Delta f$; ✓ IM3 ghosts appear at
$f_i+f_j-f_k$ and Schroeder phases suppress them vs. random phases.

**M3 — Full 3D-AODL (4 AODs).**
Counter-propagating pairs, Table-I control mapping, Eq. S19 synthesis.
✓ Co-chirp all four → pure Z motion, laterally static, round spot (astigmatism-free);
✓ counter-chirp within a pair → in-plane motion with zero focal shift (the paper's key
result); ✓ σ_astig ≈ 0 during arbitrary short 3D moves; ✓ qualitative reproduction of the
Fig. 4 "L" trajectory for a 4×4 array; ✓ the user story: 10×10 array lift-traverse-lower.

**M4 — Fading-Shepard waveforms.**
Tone ladders, fade envelopes, interlaced x/y fading, Schroeder phases (S28).
✓ Sustained Z-offset beyond the single-chirp band limit (Eq. 1) with constant total
intensity; ✓ shadow tweezers appear at ±(2λF/v)Δf during fades and vanish outside fading
zones ((Mx+2)×My extended grid for arrays); ✓ interlaced vs simultaneous fading comparison.

**M5 — Product layer.**
`api.py` front door, trajectory DSL, parametric-NPZ waveform storage + sample rendering
for AWGs, report generation (band usage,
predicted ghosts/shadow schedule, astig metrics), docs + example gallery, packaging.
Stretch (post-v1): atom-motion Monte Carlo (Eq. S13), measured-efficiency calibration hooks,
misalignment/delay-mismatch knobs (Eq. S29), cropped-aperture erf corrections.

---

## 4. Decisions (resolved 2026-08-31)

1. **Hardware defaults** — paper hardware (DTSX-400 / 650 m/s / 7.5 mm / f₀ = 100 MHz /
   F = 6.5 mm) at λ = 1030 nm; `paper_808` preset kept for figure reproduction.
2. **Input beam** — uncropped Gaussian (w_in = 2 mm default); exact closed forms throughout.
3. **Movie default** — focus-tracked 2D planar view (plane follows the tweezers' focal spot),
   hue ↔ Z, with an XZ-slice side panel. Fixed-plane camera view kept as alternate mode.
4. **Mixing depth** — perturbative expansion through IM3 with calibrated drive strength C;
   full coupled-mode Bragg deferred.
5. **Waveform storage/export** — generic NPZ holding the *parametric function representation*
   (segments + parameters, no samples); expansion to samples (or to the analytic simulator's
   inputs) is a separate render step. AWG-specific binary formats deferred.
6. **Usable bandwidth** — ±10 MHz on all four channels (per-channel limits stay configurable).

See `ARCHITECTURE.md` for the resulting code structure.
