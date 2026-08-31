# WO-03 — Device layer: conventions (sign authority), single-AOD physics, term builder

**Role:** implementation agent, Wave B (parallel with WO-02/WO-04). **No git.**
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.1–1.2, `docs/ARCHITECTURE.md` §3, then this.
WO-01 is merged (poly/params/gaussian/reference available). WO-02's `ToneTrack` API is
frozen in `docs/workorders/WO-02-waveform.md` §2 — code against that exact interface (it
may land in the tree after you start; write your tests constructing ToneTracks per that
spec, and they will pass at integration. If import fails at your runtime, still deliver;
mark affected tests `pytest.importorskip("aodl.waveform.tones")`).

## Owned files

```
src/aodl/device/conventions.py  src/aodl/device/aod.py  src/aodl/device/aodl.py
docs/conventions.md
tests/test_conventions.py  tests/test_device_single_aod.py
```

## 1. `device/conventions.py` — THE sign authority

```python
@dataclass(frozen=True) class ChannelGeometry:
    axis: int          # 0 = x, 1 = y
    sound_sign: int    # +1: sound toward +axis; −1: toward −axis

CHANNEL_GEOMETRY = {"Ax": (0, -1), "Bx": (0, +1), "Ay": (1, -1), "By": (1, +1)}  # Eq. S7
DIFFRACTION_ORDER = +1          # all channels
Z_LAB_SIGN = -1                 # see below
```

**Signs contract** (docstring + `docs/conventions.md`, with the derivation):

- Aperture coordinate u along the channel's axis, u = 0 at beam center.
- Retarded phase: channel with sound_sign s imprints optical phase −φ(t − s·u/v) + const
  (Eq. S4 generalized). Beam-center Taylor expansion (Eq. S5–S6) gives per-channel
  contributions **θ1 += s·2π·f(t_c)/v** and **θ2 += −2π·ḟ(t_c)/(2v²)** (θ2 independent of
  s — the reason counter-propagating pairs cancel deflection but add lensing), plus
  amplitude poly **α(u) = A − s·(A′/v)·u + (A″/(2v²))·u²** (Eq. S5; products of channels
  truncate at u²).
- Optical frequency bookkeeping: each +1 order shifts light by +f_center + f (absolute);
  store per-term `df_opt = Σ f(t_c)` over participating channels (common f_center shift
  is global — ignore).
- Lateral position: `X_spot = deflection_scale · Σ_x-channels s·f` — reproduces Table I:
  X = (λF/v)(f_Bx − f_Ax).
- **Axial sign** (`Z_LAB_SIGN`): pure S11 evaluation puts the sharpest focus at
  Z_S11 = −λF²·Σ_axis ḟ/v² (WO-01's `test_focal_geometry` pins θ2 = kZ/(2F²), and
  θ2_axis = −π·Σ_axis ḟ/v²). The paper's Table I reports Z̄ = +½(λF²/v²)·Σ_all ḟ
  ("above the static focal plane" for up-chirp). We adopt **lab Z ≡ Table I's sign**:
  `Z_lab = Z_LAB_SIGN · Z_S11`. All user-facing/trajectory Z is lab Z; `field/` evaluates
  at Z_S11 = Z_LAB_SIGN · Z_lab. State this in both docstring and docs/conventions.md.

- Acoustic timing: transducer at u = −s·D/2; the waveform sample emitted at drive time t′
  sits at position where t − (s·u + D/2)/v = t′. Beam-center retarded time
  **t_c = t − τ/2**, τ = D/v. Aperture leading-edge (fill) boundary: content exists where
  `s·u ≤ v·t − D/2` (drive starts at t = 0). Fully filled for t ≥ τ.

## 2. `device/aod.py`

```python
def channel_lines(cw: ChannelWaveform, aod: AODParams, t: float) -> Lines
    # M1 scope: fundamentals only (mixing lands in M2 as a drop-in producing more lines).
    # Lines: struct-of-arrays: amp (complex; = (i·C/2)·A_n(t_c)·exp(−i·phase_n(t_c)),
    #   Eq. S3/S6 rotating frame), f, fdot [Hz, Hz/s at t_c], dA, d2A (for α poly).
def aperture_window(cw, aod, geom, t, n=512) -> (u, V)   # diagnostic: literal V on crystal,
    # zero where unfilled; V = Σ A_n(t_ret(u))·cos(2π f_center·t_ret(u) + phase_n(t_ret(u)))
def fill_edge(aod, geom, t) -> float | None
    # aperture u-coordinate of the leading edge, None if fully filled (t ≥ τ);
    # plus which side is filled (return convention: document clearly for field/ to
    # select lower-edge vs upper-edge Gaussian moments: filled side is s·u ≤ v·t − D/2)
```

## 3. `device/aodl.py` — term builder

```python
@dataclass class TermArray:   # struct-of-arrays, one entry per pupil term
    c: complex[]            # product of line amps (incl. i^N factors already in line amps)
    theta1: float[ 2, N]    # rad/m, per axis (x=0, y=1)
    theta2: float[ 2, N]    # rad/m², per axis
    alpha:  complex[2, 3, N]# amplitude poly coeffs per axis (α0, α1, α2), product-truncated
    df_opt: float[N]        # Hz, for frequency grouping
    edge:   per-axis fill-edge info (u0 or None, side)   # M1: from the single channel

def build_terms(wfs: WaveformSet, t: float, channels: Sequence[str] | None = None) -> TermArray
    # channels=None → all present in wfs. Cartesian product of per-channel lines
    # (channels absent → identity factor). Accumulate θ1, θ2, α (α: polynomial product
    # per axis truncated at degree 2), c = Π amps, df_opt = Σ f.
```

M1 usage is a single channel (`channels=("Ay",)`) but implement the general product now —
it is ~20 lines and M2/M3 then need no rework.

## 4. Tests (use `field/reference.py` + `field/gaussian.py` from WO-01 — NOT `field/focal.py`)

`test_conventions.py`
- Geometry table matches Eq. S7 exactly; Z_LAB_SIGN documented value is −1.

`test_device_single_aod.py` (params `default_1030()`; build ToneTracks per WO-02 spec)
- **Static tone** f = +3 MHz detuning on Ay, t = 2τ (filled): one term;
  θ1y = −2π·3 MHz/v (sound_sign −1); position check by evaluating the analytic field
  (gauss_moments with the WO-01 §7 a,b mapping) on a line of Y: peak at
  Y = −deflection_scale·3 MHz within 1% of waist0. Cross-check same peak with
  `reference_field_separable` fed the *literal* windowed pupil (construct from the
  channel's phase at retarded times, not from Taylor coefficients) — this validates the
  Taylor step itself.
- **Linear chirp** ḟ = +50 MHz/ms on Ay: θ2y = −π·ḟ/v²; scan analytic on-axis intensity
  vs Z_S11: peak at −lens_scale·ḟ (rel 2%); restated in lab units:
  Z_lab = +lens_scale·ḟ (single channel ⇒ Table I's Z̄ + ΔF/2 combination — assert the
  algebra in a comment). x-axis focus unshifted (θ2x = 0).
- **Retardation**: chirping tone; positions extracted at t and t+δ move by
  deflection_scale·ḟ·δ; and the extracted f corresponds to t_c = t − τ/2 (compare against
  a deliberately wrong t_c = t to show the test has teeth: difference = ḟ·τ/2 offset).
- **Fill transient**: static tone, t = 0.6τ: `fill_edge` returns the correct u0 (hand
  computation in test); windowed analytic field (gauss_moments_lower/upper per side
  convention) matches `reference_field_separable` with the hard-edged literal pupil
  (rel 1e-3); intensity at t=0.6τ below the filled value.
- **Two-channel product** (Ax + Bx, static tones f₁, f₂): single term with
  θ1x = 2π(f₂ − f₁)/v, df_opt = f₁ + f₂, c = product of amps — pure algebra assertions.

## Definition of done

Own tests green; `ruff check` clean on owned files; no git; report per ORCHESTRATION.md
(explicitly flag any deviation you had to make from the WO-02 interface spec).
