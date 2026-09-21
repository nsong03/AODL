# The AOD drive waveform: frequency mixing, phase optimization, and simulation fidelity

**Scope.** What we program into each AOD channel, what the crystal actually does with it, which
of that physics `aodl` implements, and — the part that matters for calling this an *accurate*
simulation environment — which of it we deliberately do not implement yet and what that costs.

Companion reading: `docs/PLAN.md` §1 (the model), `docs/conventions.md` (signs, retarded time),
`docs/guide.md` §5.5 (what a checker PASS certifies). Equation numbers `S#` are from
[arXiv:2510.11451](https://arxiv.org/abs/2510.11451); other citations are listed in §9.

---

## 1. The object we control

Each of the four channels carries one real RF voltage — a sum of enveloped, chirped tones:

$$V_\mu(t) \;=\; \sum_n A_n(t)\,\cos\!\Big(2\pi f_c t + 2\pi\!\!\int_0^t\! f_n(t')\,dt' \;+\; \varphi_n\Big),
\qquad \mu \in \{A_x, B_x, A_y, B_y\}$$

with `f_n` a **detuning** from the channel carrier `f_c` (the rotating frame of Eq. S2 — see
`docs/conventions.md` §3). Three independent knobs, with very different characters:

| Knob | Controls (to first order) | Cost if chosen badly |
|---|---|---|
| `f_n(t)` — frequency law | trap **position** (Table I: `X = λF(f_{Bx}−f_{Ax})/v`) and, through `ḟ`, **axial focus** | wrong trajectory; band overrun (Eq. 1) |
| `A_n(t)` — envelope | trap **intensity**; fade hand-overs (Eqs. S26–S27) | non-uniform array; irising (Eq. S5 `A''` term) |
| `φ_n` — tone phase | **nothing at first order** | crest factor, and the entire ghost budget |

That third row is the reason this document exists. The phases are free in the linear theory and
decisive in the nonlinear one — they are the cheapest lever we have on drive quality, and the
only one that costs nothing in bandwidth, power, or trajectory fidelity.

**What actually reaches the instrument.** `waveform/export.py:render_samples` evaluates the sum
above and divides *all* channels by one global peak, so the full-scale sample is `1.0` on exactly
one channel and relative channel amplitudes survive. The factor is returned (`return_scale=True`)
and stored in the samples NPZ, because the absolute radian scale is not recoverable from a
normalized buffer — and the nonlinear physics below depends on it. `check/record.py` insists on
it for exactly this reason.

---

## 2. From drive to light: the phase grating

The acoustic wave modulates the refractive index, imprinting on the optical field a phase delay
that is the *retarded* drive (Eq. S1):

$$\Psi(x,t) \;=\; C\,V\!\left(t - \frac{x}{v}\right), \qquad P(x,y,t) \;=\; e^{i\Psi(x,t)}$$

Two consequences set everything else up:

1. **The crystal holds a window, not an instant.** Only the last `τ = D/v = 11.54 µs` of drive is
   on the aperture. The beam centre sees the drive delayed by `τ/2`; a *chirp* becomes a
   quadratic phase across the aperture, i.e. a cylindrical lens (Eq. S6) — this is the whole
   3D-AODL mechanism, and it is why the simulator works on aperture windows rather than
   instantaneous frequencies (`device/aod.py`, `docs/conventions.md` §7).
2. **`exp(iΨ)` is not linear in `V`.** Everything in §3 follows from expanding it.

**Regime.** Our device is deep in the Bragg regime, but the paper (Supplement, "Theoretical
Model") adopts the weak-drive, linearized-phase treatment and justifies it by the tangential
phase-matching (TPM) broadening of the AAOptic DTSXY cells — the Bragg condition is satisfied
over a wide enough band that diffraction efficiency can be taken as flat to first order. `aodl`
inherits that simplification. §6 is about what it costs. The contrasting approach is
Mittenbühler *et al.* (2025), who keep coupled-wave theory and the `sinc²(ΔkL/2)` mismatch
explicitly; their Eq. (3) is the multi-tone saturation law we approximate perturbatively below.

---

## 3. Frequency mixing

### 3.1 The exact structure

Write `m_n = C A_n` (per-tone **modulation depth**, radians) and `Φ_n` for the tone's total
phase. Because the exponential factorizes over tones, each factor is a Jacobi–Anger series:

$$e^{iCV} \;=\; \prod_n \sum_{p=-\infty}^{\infty} i^{\,p} J_p(m_n)\, e^{\,i p \Phi_n}$$

A product term is labelled by integers `{p_n}`. Its optical frequency shift is
`−Σ p_n (f_c + f_n)`, so **the +1 diffraction order is exactly the set `Σ p_n = −1`** — this
selection rule is the spine of `device/mixing.py`, and it immediately explains the order
structure: `Σ p_n = −1` forces `Σ|p_n|` odd, so *even orders cannot appear in the useful band
at all*. Mixing is a strictly odd-order phenomenon here.

Keeping `Σ|p_n| ≤ 3` leaves three families:

| Family | Frequency | Amplitude | Physical meaning |
|---|---|---|---|
| fundamental `n` | `f_n` | `i J_1(m_n)\prod_{m≠n} J_0(m_m)` | the tweezer you asked for |
| IM3, `j<k`, `i` | `f_j + f_k − f_i` | `−(i/8)\,m_i m_j m_k` | ghost |
| IM3 degenerate | `2f_j − f_i` | `−(i/16)\,m_i m_j^2` | ghost |

Truncating the Bessel functions at third order turns the fundamental into

$$\frac{i}{2}m_n\Big[1 - \frac{m_n^2}{8} - \frac14\sum_{m\neq n} m_m^2\Big]$$

whose two correction terms are worth naming separately, because they are different physics:

- `−m_n²/8` — **self-compression**. The first bend of `J_1`; one tone saturating on its own.
- `−¼ Σ_{m≠n} m_m²` — **cross-compression**, i.e. *pump depletion*: the other tones sharing the
  crystal steal from this one. This is the perturbative face of Mittenbühler's Eq. (3), in which
  the global diffraction efficiency depends only on the **total** RF power and is then divided
  among tones by their power fraction. Our `Π_{m≠n} J_0(m_m)` is the same statement in the
  Raman–Nath limit. It is the reason a 10-tone array is dimmer per trap than a single tweezer at
  the same per-tone drive, and why per-tone amplitudes cannot be calibrated one at a time.

### 3.2 Where the ghosts land — the part that surprises people

IM2 products at `f_i ± f_j` land near DC or near `2f_c`: **out of band, never launched by the
transducer.** They matter only because they can remix to IM3, which lands squarely *inside* the
band. `device/mixing.py` enforces this with an explicit band-acceptance cut.

For an **equally spaced ladder** `f_n = f_0 + nΔf`, every IM3 frequency is
`f_0 + (j+k−i)Δf` — an integer multiple of the same spacing. So:

> **On a commensurate ladder, the ghosts land exactly on the tweezer lattice.**

This has a good face and a bad one, and both are load-bearing:

- **Good:** you see no spurious spots. The array looks clean.
- **Bad:** the ghost light is not gone, it is *hiding inside your traps*. IM3 contributions
  landing on an occupied site interfere with the real trap at that site and become **per-trap
  intensity error** — invisible as a ghost, fully visible as non-uniformity. Only at the array
  edges (`j+k−i` outside `[0, M−1]`) does it produce new spots, which is why a multi-tone array
  grows a halo one pitch out.

Mittenbühler's measurement is the clean experimental demonstration: driving 11 equally spaced
tones per axis with the *central* tone deliberately blanked, an IM3 peak appears in the gap at
**−23 dB**, and the array's uniformity degrades to 9.8 % rms. Our simulator reproduces the same
structure from first principles — notebook 02 measures an 8-tone ladder's edge ghosts at
`9.9×10⁻²` of a trap with aligned phases, against a re-derived prediction agreeing to four
significant figures.

**The 2D trap.** With two crossed AODs the optical frequency of the spot at `(n, m)` is
`f_{x,n} + f_{y,m}`. If `Δf_x = Δf_y`, every site on an **anti-diagonal** (`n + m` constant) has
the *same* optical frequency and therefore interferes statically instead of beating. Trypogeorgos
*et al.* flag this explicitly ("beams lying along diagonal lines have the same frequency and thus
interfere... easily avoided by using different frequency spacings"); our own M2 work rediscovered
it independently when a 5×5 array collapsed from 25 frequency groups to 9. **This is why every
gate and flagship drive in this repo uses `Δf_x = 1.0 MHz, Δf_y = 1.3 MHz`.**

### 3.3 Why beating is usually benign

Distinct tones produce distinct optical frequencies, so overlapping spots *beat* rather than
interfere statically. The beat runs at the tone spacing — hundreds of kHz to MHz — while atomic
trap frequencies are a few kHz. Trypogeorgos makes exactly this argument; the paper makes it
again for the fading-Shepard hand-overs ("MHz scale beatnote... the constituents simply sum up in
intensity"). `field/focal.py` implements precisely this rule: terms are grouped by optical
frequency, degenerate ones summed **coherently**, distinct groups added **in intensity**. It is
also why *interlaced* fading beats *simultaneous* fading — simultaneous fading creates
frequency-degenerate pairs whose interference depends on optical path length, which our M4 work
measured as a **100 % swing** in trap power under a phase perturbation, against `<10⁻¹²`
interlaced.

---

## 4. Phase optimization

### 4.1 Two jobs, one parameter

The tone phases `{φ_n}` do not move a trap or change its first-order brightness. They do two
other things, and both matter:

**Job 1 — crest factor.** For `M` equal-amplitude tones all starting in phase, the peak voltage
is `M·A` while the RMS is `A√(M/2)`: a crest factor `√(2M)` that grows without bound. Since the
optical phase depth is `Ψ = C·V`, a high crest means the *peak* modulation is deep even when the
average is modest — and deep modulation is exactly what generates the mixing products of §3. A
high crest factor also wastes the amplifier: at a fixed peak-power ceiling, lower crest buys more
RMS power, i.e. more light per trap. Mittenbühler put it plainly: *"a poorly optimized crest
factor strongly favors intermodulation."*

**Job 2 — IM3 phase scattering.** Each IM3 line carries the phase factor
`exp(−i(φ_j + φ_k − φ_i))`. Many index triples contribute to the *same* ghost frequency. With
all phases equal, those contributions add coherently — worst case. With a well-chosen phase set,
they scatter around the complex plane and largely cancel. The paper's Supplement makes the
sharper version of this argument: choose `{φ}` so that the **IM2** products cancel, and since
"without IM2, there's no IM3 or higher-order intermodulation," the whole cascade is suppressed at
its root (Fig. S5 shows the IM2 phasors distributed uniformly around the circle).

### 4.2 The phase families

Low-crest multitone phasing is a solved problem in RF engineering, and the AOD community borrows
from it. The main closed-form families:

| Scheme | Phase law | Notes |
|---|---|---|
| **Schroeder** (1970) | quadratic; group delays spread ∝ cumulative power | closed form, near-optimal for flat spectra |
| **Newman** | `θ_k = πk²/N` | crest factor provably below ~6 dB |
| **Narahashi–Nojima**, **Kitayoshi** | quadratic variants | slightly better crest than Schroeder |
| iterative / clipping | numerical | best results, but needs the pattern known in advance |

Shibasaki *et al.* (2020) showed these four closed-form schemes are essentially one family, and
unified their phase equations. The practical split: **iterative optimization** wins for *static*
patterns known ahead of time (several tweezer-array groups do this), while **direct formulas**
are what you use when the tone set changes in real time — Mittenbühler's GPU streaming system
uses Narahashi phases for exactly that reason.

### 4.3 What `aodl` implements

One implementation, `waveform/shepard.py:ladder_phases` (Eq. S28), re-exported as
`synthesis.schroeder_phases` (Eq. S23):

$$\varphi^{(n)} \;=\; \operatorname{mod}\!\Big(\frac{2\pi\,n(n-1)}{2M},\; 2\pi\Big)$$

This is the quadratic (Schroeder/Newman) family. Two implementation notes that are easy to get
wrong and are pinned by tests:

- The paper's Eq. S23 uses `2(M−1)` in the denominator for a counted ladder; Eq. S28 uses `2M`
  because a *fading* ladder has `M+1` live tones during hand-over. We implement the S28 form and
  index by the rung's own (possibly negative) integer, which is what a sliding Shepard ladder
  requires.
- The modular reduction is applied to `n(n−1)/2` **before** scaling by `2π/M`. Reducing afterwards
  leaves rungs at `2π − 7×10⁻¹⁵` instead of exactly `0` — physically identical, but it breaks the
  `M = 1 ⇒ all-zero` guarantee and puts results outside `[0, 2π)`. This was a real
  verification finding (`16a9fd6`).

**Measured effect, from our own suite** (notebook 02, 8-tone ladder, `Δf = 1 MHz`, `m = 0.3`):

| Phase set | per-trap intensity spread | total ghost light |
|---|---|---|
| zero | 0.2028 | 1.88×10⁻¹ |
| random | 0.049 – 0.176 (**seed-dependent**) | — |
| **Schroeder** | **0.0195** | **9.5×10⁻⁴** |

Random phases are a lottery, not a method: two independent verification passes measured 0.0489
and 0.1758 on different seeds. Schroeder beat random in **100 %** of 200 seeds tested, but beat
it by the ≥2× margin an early test asserted in only 93.5 % — which is why that assertion was
corrected to the claim the physics actually supports.

Individual ghosts are suppressed **57–437×**; total ghost light falls ~200×. On the crest side,
the flagship 30-rung Shepard ladder renders with a peak-to-single-tone-amplitude factor of
**4.59**; with ~10 rungs live at a time that is a crest factor of **≈2.0**, against `√(2·10) =
4.47` for zero phases and an ideal floor of `√2 ≈ 1.41`. So Schroeder recovers most, not all, of
the available headroom — consistent with the literature's verdict that closed-form phases are
"moderate" compared to iterative optimization.

---

## 5. What the simulator implements

| Physics | Module | Equation | Verified by |
|---|---|---|---|
| tone ladders, Schroeder phases | `waveform/synthesis.py`, `shepard.py` | S18, S23/S28 | `test_synthesis*`, `test_shepard` |
| trajectory → frequency laws | `waveform/synthesis.py` | S19, Table I | S19 inverted from waveforms at 1e-15 |
| fade envelopes, interlacing | `waveform/shepard.py` | S24–S27 | `test_shepard`, notebook 05 |
| AWG samples + normalization | `waveform/export.py` | — | `test_export` |
| retarded aperture window | `device/aod.py`, `device/conventions.py` | S1, S4 | τ/2 darkness, erf² fill law |
| **compression + IM3** | **`device/mixing.py`** | **S20–S22** | Jacobi–Anger + frozen-time projection |
| 4-channel product (16 rays) | `device/aodl.py` | S7–S8 | Table I to 5×10⁻²⁰ m |
| closed-form focal field | `field/` | S11 | vs quadrature, ~10⁻¹⁵ |
| frequency grouping | `field/focal.py` | Fig. S6 logic | 1000-trial fuzz |
| **independent audit** | **`check/`** | full `exp(iCV)`, no expansion | corruption tests FAIL correctly |

The last row is the useful cross-check on everything above it: `aodl.check` rebuilds the pupil
from the **literal rendered samples** with the full exponential and selects the +1 order by
angular-spectrum band-pass — no Taylor truncation anywhere. Comparing the two paths *measures*
the truncation error: at `C = 0.3` the order-3 simulator differs from the exact exponential by
`7.8×10⁻⁴` RMS in pupil amplitude, against `4.5×10⁻²` for the order-1 model — the IM3 terms buy a
factor **58**.

---

## 6. What the simulator does **not** model

This is the honest core of the document. `grep -rn "efficiency" src/aodl/` returns nothing: there
is no diffraction-efficiency model in the code at all. Ranked by fidelity impact:

### 6.1 Bragg phase mismatch / frequency-dependent efficiency — **the big one**

Real AODs have `η(ν) = η_c P_a \operatorname{sinc}^2\!\big(\tfrac{1}{2\pi}\sqrt{4η_cP_a + (Δk(ν)L)^2}\big)`
(Mittenbühler Eq. 1); the paper's own Figs. S1 and S8 map the measured efficiency ridge across
the 90–110 MHz band and show the usable A-pair band is *narrower* than the B-pair band because
the first AOD's output angle steers the second. We assume flat. Three consequences:

1. **Per-trap intensity is too flat in simulation.** A wide array rolls off at the band edges in
   reality; ours does not.
2. **Ghosts are over-estimated — possibly substantially.** An IM3 product is generated at an
   acoustic k-vector that generally does *not* satisfy the Bragg condition, so the crystal
   suppresses it. This is precisely Gazalet *et al.*'s result: a suitable anisotropic interaction
   reduces two-tone IM3 by **~16 dB**, with the residual set by *acoustic* nonlinearity. Our
   ghost predictions should therefore be read as **conservative upper bounds**.
3. The A-band/B-band asymmetry that constrains astigmatism-free trajectories (Supplement, "Design
   Caveats") is invisible to us.

*Proposed hook, cheap:* a per-channel `efficiency(f)` callable applied as a multiplicative weight
on each emission line — including IM3 lines, weighted at *their own* frequency. That single
change converts (2) from optimistic to realistic and gives (1) and (3) for free. It fits the
existing `Lines` structure without touching the field path.

### 6.2 Truncation at third order — adequate, not luxurious

We keep `Σ|p_n| ≤ 3`, i.e. the Bessel series to its first correction. At the nominal `C = 0.3`
this is excellent: `(m/2)(1 − m²/8) = 0.1483125` vs `J_1(0.3) = 0.1483188`, a relative error of
`4.3×10⁻⁵`. But a **crest-limited multi-tone drive is not weak**: our flagship reaches a peak
phase depth of `0.30 × 4.59 = 1.38 rad`, where `J_1(1.38) = 0.5377` against the third-order value
`0.5253` — a **2.3 % amplitude error at the crest**, and the series is alternating and slowly
converging there. Fine for current tolerances, thin if anyone drives harder.

(For contrast, the *checker's* `bragg_band` mode uses the full exponential and reproduces the
exact `2J₁(C)/C` compression to `1.1×10⁻⁸` at `C = 0.1, 0.3, 0.5` — so the cross-validation in §5
measures this truncation rather than sharing it.)

*Proposed hook:* an exact-Bessel mode for the fundamentals (`J_1(m_n)·Π J_0(m_m)` evaluated
directly rather than truncated). Nearly free, removes the worry entirely for the fundamentals,
and leaves only the ghost amplitudes perturbative.

### 6.3 RF-chain nonlinearity

`render_samples` normalizes to full scale and models **no clipping or amplifier compression**.
In the lab this is where crest factor hurts most: AM–AM/AM–PM compression in the amplifier
generates its own IM3 *before* the crystal sees anything, which is why Mittenbühler restricts
dual-tone drives to half power "to accommodate the crest factor." *Proposed hook:* an optional
clip/predistortion stage in `export.py` — and because `aodl.check` reads the **rendered samples**,
it would then audit the distorted drive automatically, with no changes to the checker.

### 6.4 Acoustic (elastic) nonlinearity

TeO₂ itself mixes sound waves, producing acoustic harmonics and IM products in the medium before
any optical interaction. Gazalet identifies this as the *floor* that remains after the optical
phase-grating IM is suppressed. Not modeled; it would enter as extra acoustic tones.

### 6.5 2D dimensional coupling

Two AODs in series are not two independent 1D systems: the first one's deflection changes the
incidence angle on the second, so `η^{(xy)}_{ij} = η^{(x)}_i η^{(y)}_j Ξ_{ij}` with an empirical
interaction term (Mittenbühler Eqs. 4–5). We assume perfectly overlaid apertures and matched
delays (a deliberate design-brief simplification; Eq. S29 misalignment is a PLAN post-v1 item).

### 6.6 Smaller omissions

Acoustic attenuation across the aperture, thermal drift, and — already quantified elsewhere —
inter-channel timing skew, to which the checker is blind below ≈0.1–1.2 µs depending on chirp
rate (`docs/guide.md` §5.5).

---

## 7. Practical recipe for an accurate simulation

1. **Spacing.** Pitch `= λFΔf/v` (10.3 µm/MHz at default hardware). **Never set `Δf_x = Δf_y`**
   for a square array (§3.2). Non-commensurate spacings additionally move IM3 ghosts *off* the
   lattice, where the checker's blob audit can see them.
2. **Drive strength.** `AODParams.drive_strength` is the peak phase modulation per unit-amplitude
   tone. Calibrate it from a **single-tone** diffraction-efficiency measurement — the multi-tone
   response then follows from §3.1 rather than needing per-pattern calibration (this is exactly
   the argument Mittenbühler make for their one-time single-tone calibration). Remember the
   ladder's crest multiplies it.
3. **Phases.** Keep `phases="schroeder"` unless you are reproducing a pathology; compare against
   `"zero"` and `"random"` to see the ghost budget move. For rapidly changing tone sets, the
   literature's answer is a direct formula (Narahashi), not iteration.
4. **Mixing order.** Leave `mixing_order=3`. Order 1 is a 4.5 % pupil-level lie at `C = 0.3`.
5. **Validate.** `plan.check()` — the FFT path shares none of the simulator's expansions. Read
   `out_of_band` (splatter), the blob audit, and `gated_fraction` before trusting a PASS.

## 8. Recommended fidelity roadmap

In priority order, by fidelity gained per line of code:

1. **`efficiency(f)` per channel** (§6.1) — the largest single gain; makes ghost predictions
   honest rather than conservative, and reproduces the band-edge roll-off and A/B asymmetry.
2. **Exact-Bessel fundamentals** (§6.2) — removes the crest-limited truncation error.
3. **Clip / predistortion in `export.py`** (§6.3) — brings the dominant lab IM source into the
   loop, audited for free by the existing checker.
4. **2D coupling term `Ξ`** (§6.5) — needs a measured map; matters for wide arrays.
5. **Acoustic nonlinearity** (§6.4) — research-grade; probably only worth it against a
   dedicated measurement.

Items 1–3 are self-contained and would not disturb the verified physics path.

---

## 9. Sources

- Lu, Song, Xiang, Ho, Lee, Yan & Stamper-Kurn, *Astigmatism-free 3D Optical Tweezer Control for
  Rapid Atom Rearrangement*, [arXiv:2510.11451](https://arxiv.org/abs/2510.11451) — the `S#`
  equations, Schroeder phases for IM3 suppression (S23/S28), fading-Shepard waveforms, and the
  measured efficiency maps of Figs. S1/S8.
- M. G. Gazalet, J. C. Kastelik, C. Bruneel, O. Bazzi & E. Bridoux, *Acousto-optic multifrequency
  modulators: reduction of the phase-grating intermodulation products*, Appl. Opt. **32**, 2455
  (1993) — [Optica](https://opg.optica.org/ao/abstract.cfm?uri=ao-32-13-2455) ·
  [PubMed](https://www.ncbi.nlm.nih.gov/pubmed/20820405). Two-tone IM3 reduced ~16 dB by
  anisotropic interaction; acoustic nonlinearity as the residual floor.
- M. Mittenbühler, L. Sturm, M. Schlosser & G. Birkl, *Model-Based Real-Time Synthesis of
  Acousto-Optically Generated Laser-Beam Patterns and Tweezer Arrays*, Phys. Rev. Applied **24**,
  064046 (2025) — [APS](https://journals.aps.org/prapplied/abstract/10.1103/d3tx-3tg8) ·
  [arXiv:2512.16774](https://arxiv.org/abs/2512.16774). Coupled-wave multi-tone saturation
  (Eqs. 1–3), 2D coupling `Ξ` (Eqs. 4–5), Narahashi phases for real-time crest control, and the
  −23 dB blanked-gap IM3 measurement.
- D. Trypogeorgos, T. Harte, A. Bonnin & C. Foot, *Precise shaping of laser light by an
  acousto-optic deflector*, Opt. Express **21**, 24837 (2013) —
  [arXiv:1307.6734](https://arxiv.org/abs/1307.6734). Composite-beam shaping; the equal-spacing
  diagonal-degeneracy warning; beat frequencies far above trap frequencies.
- M. R. Schroeder, *Synthesis of low-peak-factor signals and binary sequences with low
  autocorrelation*, IEEE Trans. Inf. Theory **16**, 85 (1970) — the quadratic phase law.
- Y. Shibasaki, K. Asami *et al.*, *Analysis and Design of Multi-Tone Signal Generation Algorithms
  for Reducing Crest Factor*, IEEE (2020) —
  [IEEE Xplore](https://ieeexplore.ieee.org/document/9301549/). Unifies Newman, Kitayoshi,
  Schroeder and Narahashi phase laws.
- M. Endres *et al.*, *Atom-by-atom assembly of defect-free one-dimensional cold atom arrays*,
  Science **354**, 1024 (2016) —
  [Science](https://www.science.org/doi/10.1126/science.aah3752). The multitone-AOD tweezer array
  this whole line of work builds on.
