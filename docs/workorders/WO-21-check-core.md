# WO-21 — M6 core: the FFT checker numerics (`src/aodl/check/`)

**Role:** implementation agent, Wave R (solo). **You are the wave closer**: commit and
push when done. **Read first:** `CLAUDE.md`, `docs/PLAN.md` §1 (physics + conventions),
`docs/conventions.md`, `src/aodl/waveform/export.py` (the input contract),
`src/aodl/device/conventions.py` (the sign authority you import), `tests/test_device_single_aod.py`
+ `tests/test_window.py` (the literal-pupil test patterns you will replicate *from
samples*), `src/aodl/field/reference.py`, then this. HEAD carries 334 green tests.

## Mission

M6 adds an independent verification path: take **rendered RF sample arrays** (the literal
AWG buffers — carrier included, globally normalized) and rebuild the tweezers with FFT
techniques, sharing nothing with the analytic simulator except parameters and the sign
conventions. This WO builds the numerics; WO-22 adds expectation/verdict/product wiring.

**Independence rule (hard, you enforce it with a test):** `src/aodl/check/` may import
ONLY `aodl.params`, `aodl.units`, `aodl.poly`, `aodl.trajectory.spec`,
`aodl.device.conventions`, constants from `aodl.waveform.export`
(`DEFAULT_SAMPLE_RATE`, the `_samples.npz` suffix), and numpy/scipy. Forbidden at
runtime: `aodl.field.*`, `aodl.device.aod|aodl|mixing`, `aodl.engine`,
`aodl.waveform.tones|synthesis|shepard|serialize`, `aodl.api`, `aodl.viz`. The
parametric IR is never evaluated — truth enters only via samples and (in WO-22) the
compiled spec polynomials.

## Owned files

```
src/aodl/check/__init__.py   src/aodl/check/record.py   src/aodl/check/demod.py
src/aodl/check/pupil.py      src/aodl/check/transform.py src/aodl/check/metrics.py
tests/test_check_demod.py    tests/test_check_pupil.py   tests/test_check_transform.py
tests/test_check_metrics.py  tests/test_check_independence.py
pyproject.toml               (edit: dependency line `scipy` → `scipy>=1.8` — czt)
```

(Tests MAY import anything — sim modules included — to build fixtures and cross-checks;
only `src/aodl/check/` is restricted.)

## 1. `record.py` — the input boundary

```python
@dataclass(frozen=True)
class SampleRecord:
    channels: dict[str, NDArray[np.float64]]   # equal-length 1D arrays
    sample_rate: float                         # [S/s]
    t_start: float                             # sample 0 time; t=0 is drive start (fill convention)
    params: AODLParams
    normalization: float = 1.0                 # samples × this = Eq. S1-unit drive V
    @property def t_span(self) -> tuple[float, float]

def from_arrays(samples, sample_rate, params, *, t_start=0.0, normalization=1.0) -> SampleRecord
def load_samples(path, params) -> SampleRecord
    # reads the *_samples.npz schema written by waveform/export.save_samples (schema 1):
    # sample_rate, t_start, normalization, per-channel f_center — raise loudly if a file
    # f_center mismatches params.channels[ch].f_center.
```

**Normalization is load-bearing** (docstring it): `render_samples` divides all channels
by one global peak; the phase modulation of a normalized buffer is
`drive_strength × normalization × sample`. Forgetting it silently mis-scales the
nonlinear (`bragg_band`) pupil by the crest factor (~4× for Shepard drives).

## 2. `demod.py` — one FFT per channel (Eq. S1/S2)

```python
@dataclass(frozen=True)
class Baseband:
    z: dict[str, NDArray[np.complex128]]; sample_rate: float; t_start: float; params: AODLParams
def demodulate(rec, *, oversample: int = 1) -> Baseband
    # z_mu[k] = 2·P+[V_mu][k]·exp(−i·2π·f_center_mu·t_k), normalization folded in.
    # P+ = positive-frequency projection: complex FFT, zero the negative half, IFFT.
    # oversample: FFT zero-padding factor (test/tightening knob).
def sample_baseband(bb, channel, t) -> complex array
    # vectorized cubic Hermite (Catmull–Rom) gather on I/Q; t < t_start → 0 (pre-drive);
    # t beyond the record end → ValueError (mirror engine's coverage-refusal style —
    # never clamp-hold a dead drive).
def out_of_band_fraction(rec) -> dict[str, float]
    # per channel: RF power outside AODParams.band from the record FFT (splatter diagnostic)
```

Error budget to state in the module docstring: cubic-Hermite on the ≤ ~15 MHz baseband
at 625 MS/s ≈ (2πf_bb/f_s)⁴/384 ≈ 1.5e-6 relative; analytic-signal edge ringing decays
as 1/(π f_c Δt) from the record ends (why WO-22 marks early frames "transient").

## 3. `pupil.py` — aperture rebuild + order selection (Eqs. S1–S4, literal)

```python
PupilMode = Literal["bragg_band", "weak"]
@dataclass(frozen=True)
class ApertureGrid:
    u: NDArray[np.float64]; du: float          # uniform, centered on u = 0
    @classmethod def design(cls, params, mode) -> ApertureGrid
        # bragg_band: du = v/(8·f_center) EXACTLY; n = 24576  (span ≈ 5·w_in at defaults;
        #   assert span ≥ 4.2·w_in, raise for exotic params with a clear message)
        # weak:       n = 4096 over the same span
def channel_pupil(bb, channel, t, grid, *, mode, band_margin=1.15, roll=0.25) -> complex (n_u,)
def axis_pupil(bb, axis, t, grid, *, mode, **kw) -> complex (n_u,)
    # product of the axis' channel pupils × exp(−u²/w_in²) (input Gaussian applied ONCE per axis)
def band_window(nu, center, half, roll) -> float array   # flat top ± half, raised-cosine roll·half
```

Per channel at frame time t: `t_ret = conventions.retarded_time(t, grid.u, geom, aod)`
(import the helper — never re-derive a sign), gather baseband `z(t_ret)`, zero where
`t_ret < 0` (unfilled).

- **weak**: `p = 0.5j·drive_strength·conj(z(t_ret))`. The demodulation at retarded time
  removes the carrier's spatial ramp automatically — this lands exactly in the sim's
  rotating frame (matches `theta1_contribution = s·2πf/v`; pin all four signs in tests).
- **bragg_band**: rebuild `V = Re[z(t_ret)·exp(+i·2π·f_center·t_ret)]` (carrier phase
  evaluated analytically at t_ret — never interpolate the carrier), transmission
  `T = where(filled, exp(1j·drive_strength·V), 1.0)` — **1, not 0, where unfilled** (no
  sound = clear crystal; its +1-band content is zero, which keeps band selection clean).
  Multiply by the Gaussian (via axis_pupil ordering: apply Gaussian before the FFT),
  FFT along u, apply `band_window` centered at signed `s·f_center/v` with flat
  half-width `band_margin·(band_hi−band_lo)/2/v`, IFFT, then multiply by
  `exp(−1j·s·2π·f_center/v·u)` to remove the carrier ramp exactly (do NOT roll the
  spectrum by bins — the post-IFFT phase multiply is exact for non-integer shifts).

**Why du = Λ/8 (document in `ApertureGrid.design`):** `exp(iCV)` carries harmonics at
`m·f_c/v`; sampling at 8/Λ makes aliases land on order *centers* (m mod 8), so the first
fold onto +1 is m = 9 — `J₉(C·V ≲ 1.2) ≈ 1.5e-7`. Any future change of du reopens this
analysis; pin it with a planted strong-two-tone alias test (no alias blob > 1e-5).

## 4. `transform.py` — zoom CZT + Eq. S11 defocus

```python
def zoom_field(pupil, grid, optics, coords, z_lab) -> complex array
    # multiply by exp(−i·k·z_s11_from_lab(z_lab)·u²/(2F²))  [conventions helper], then
    # du·Σ p[n]·exp(−i·k·u_n·X_m/F) evaluated on arbitrary uniform coords via
    # scipy.signal.czt: a = exp(+i·k·du·X0/F), w = exp(−i·k·du·dX/F), post-factor
    # exp(−i·k·u0·X_m/F). Same dropped prefactors as the sim (intensity-safe).
def subtimes(t, window, k) -> float array
    # golden-ratio low-discrepancy placement in [t−W/2, t+W/2]: t_j = t + W·(frac(j·φ)−0.5);
    # deterministic; never commensurate with any beat (uniform grids alias
    # n·δf_x+m·δf_y combos to DC — say so in the docstring)
```

Riemann sums here are exact by Poisson summation (spectrum aliased by 1/du falls on
order centers / Gaussian tails) — cite this in the module docstring instead of a
trapezoid-error claim.

## 5. `metrics.py` — averaged intensities → fits

```python
@dataclass(frozen=True) class TrapFit:  x, y, z_lab, delta_f, sigma_astig, wx, wy, peak, power, beat_std
@dataclass(frozen=True) class Blob:     time, x, y, rel_intensity, on_lattice
def fit_gaussian_1d(coords, profile) -> (center, radius_1e2, peak)
    # weighted quadratic fit of log I (weights I²): exact on a Gaussian, deterministic,
    # moment-initialized, no scipy.optimize; also return raw 2nd moments for reporting
def best_focus(z_planes, w2) -> float
    # vertex of the parabola w²(Z) = w0²(1+((Z−Zf)/z_R)²) — per-axis; chosen over
    # on-axis-intensity peaking because that peaks at the circle of least confusion
    # and cannot resolve ΔF (docstring this)
def find_blobs(canvas, xs, ys, floor) -> list[(x, y, rel_intensity)]
    # local maxima above floor, merged within 1 waist
def accumulate_intensity(...)  # your design: per sub-time |U_x|²⊗|U_y|² outer-product
    # accumulation (never average the factors separately — x/y beats are correlated),
    # per-z marginals for the Z-stack
```

`check/__init__.py`: re-export the public names built so far (`SampleRecord`,
`from_arrays`, `load_samples`, `Baseband`, `demodulate`, `ApertureGrid`,
`channel_pupil`, `axis_pupil`, `zoom_field`, `subtimes`, fits). WO-22 extends it.

## 6. Tests (tolerances are the acceptance criteria)

- `test_check_demod.py`: single in-band tone rendered float64 → interior `z` equals
  `A·e^{iφ}` to 1e-7; edge ringing decays ~1/(πf_cΔt); chirp: baseband phase matches
  the tone's quadratic law; `oversample` 1→4 shrinks interpolation error ~16×;
  `out_of_band_fraction` on a `switch_ramp=0` Shepard render detects the ≈ −40 dB
  splatter (ties to WO-19's measured number).
- `test_check_pupil.py`: weak pupil from rendered samples vs the closed-form literal
  pupil `(iC/2)A e^{−iφ(t_ret)}` (the `tests/test_device_single_aod.py` `_literal_pupil`
  pattern) ≤ 2e-6; bragg single tone: +1-band amplitude / weak = `2·J₁(C)/C` ≤ 1e-4
  (`scipy.special.j1` allowed in tests); band window keeps +1 and rejects DC/−1/+2 on
  a constructed two-order fixture; fill mask agrees with `conventions.is_filled`;
  static tone at detuning f → pupil phase slope `s·2πf/v` for **all four channels**
  (sign table pinned); the du=Λ/8 alias test (strong two-tone, no alias blob > 1e-5).
- `test_check_transform.py`: czt vs direct matrix DFT ≤ 1e-12; vs
  `reference_field_separable` on a literal pupil ≤ 1e-6; defocus sign: a quadratic
  pupil phase focuses at the `z_lab = +lens_scale·ḟ`-equivalent (pins `Z_LAB_SIGN`
  usage, the WO-01 geometry-test pattern).
- `test_check_metrics.py`: fits recover center/width/peak exactly on synthetic
  astigmatic Gaussians, with a neighbor at 1 pitch and 1% ghost contamination;
  parabola best-focus exact on synthetic w²(z); blob finder on planted ghosts;
  outer-product accumulation vs a brute 2D snapshot average on a 2-tone × 2-tone scene.
- `test_check_independence.py`: source-scan of `src/aodl/check/*.py` import lines
  against the allowlist (regex; a subprocess-import test can't work because
  `aodl/__init__` imports api→engine).

## Definition of done

Full `pytest` green (334 existing + yours); `ruff check src tests` clean;
`python -m mypy src/aodl` exit 0 (always `python -m mypy`; bare `mypy` on PATH is
broken); commit (`M6: FFT checker core — demod, pupil synthesis, zoom transform,
metrics`, footer per dispatch) and push. Report: files, pytest summary verbatim, the
measured weak-vs-literal-pupil and czt-vs-DFT and 2J₁(C)/C numbers, alias-test margin,
deviations/spec friction (or "none").
