# WO-02 — Waveform layer: ramps, tones, parametric serialization, sample export

**Role:** implementation agent, Wave B (runs in parallel with WO-03/WO-04). **No git.**
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.4, `docs/ARCHITECTURE.md` §3, then this.
Wave A (WO-01) is already merged: `aodl.poly.PiecewisePoly`, `aodl.params`, `aodl.units`
exist — code against them; if an API you need is missing/awkward, work around locally and
flag it in your report (do not modify WO-01 files).

## Owned files

```
src/aodl/trajectory/ramps.py
src/aodl/waveform/tones.py  src/aodl/waveform/serialize.py  src/aodl/waveform/export.py
docs/waveform_format.md
tests/test_ramps.py  tests/test_tones.py  tests/test_serialize.py  tests/test_export.py
```

## 1. `trajectory/ramps.py` (Eqs. S14–S17)

Each returns a `PiecewisePoly` on `[t0, t0+T]` from `y_i` to `y_f` (usable for positions
*or* frequencies; document that). Δ = y_f − y_i; τ = normalized local time.

- `min_jerk(t0, T, y_i, y_f)` — one segment, coeffs `y_i + Δ·[0,0,0,10,−15,6]` (S14)
- `constant_jerk(t0, T, y_i, y_f)` — one segment, `y_i + Δ·[0,0,3,−2]` (S15)
- `constant_accel(t0, T, y_i, y_f)` — two segments (S16):
  τ∈[0,½]: `y_i + Δ·2τ²`; τ∈[½,1]: `y_i + Δ·(1 − 2(1−τ)²)` (re-express each in its own
  normalized local time)
- `switching_constant_jerk(t0, T, y_i, y_f)` — three cubic segments (S17), boundaries at
  T/4 and 3T/4
- `linear(t0, T, y_i, y_f)`; `hold(t0, T, y)`

## 2. `waveform/tones.py`

Frequencies are **detunings from `AODParams.f_center`** (rotating frame, Eq. S2).

```python
class Envelope(Protocol):  # A(t), dA(t), d2A(t); vectorized; values in [0, 1]
@dataclass(frozen=True) class ConstantEnvelope:  amp: float = 1.0
@dataclass(frozen=True) class SmoothOnOff:
    # 0 before t_on; sin²(π(t−t_on)/(2 ramp)) rise over `ramp`; 1; symmetric sin² fall
    # ending at t_off; 0 after. dA, d2A analytic (d2A piecewise, endpoints defined a.e.).
    t_on: float; t_off: float; ramp: float

@dataclass(frozen=True) class ToneTrack:
    freq: PiecewisePoly            # detuning [Hz] vs time [s]
    env: Envelope = ConstantEnvelope()
    phase0: float = 0.0            # [rad]
    # cached: _phase = 2π · freq.antiderivative()  (build lazily, functools.cached_property)
    def f(self, t); def fdot(self, t); def phase(self, t)  # phase(t) = _phase(t) + phase0
    def with_hold_until(self, t_end) -> ToneTrack   # append hold segment at final freq

@dataclass(frozen=True) class ChannelWaveform:
    tones: tuple[ToneTrack, ...]
    def eval_table(self, t) -> dict of arrays  # per tone: f, fdot, A, dA, d2A, phase
                                               # (single vectorized pass; device layer input)

@dataclass(frozen=True) class WaveformSet:
    channels: dict[str, ChannelWaveform]       # keys ⊆ params.CHANNELS
    params: AODLParams
    description: str = ""
    @property def t_span(self) -> tuple[float, float]   # union of tone freq domains
    def save(self, path); @staticmethod def load(path)  # thin wrappers over serialize
```

Validation on construction: channel keys valid; tone freq domains all cover the same span
(use `with_hold_until` to extend; raise with a helpful message otherwise).

## 3. `waveform/serialize.py` + `docs/waveform_format.md`

NPZ, **parameters only** (product decision — never samples). Schema v1:

- `meta` — JSON string: `{schema_version: 1, description, params: {…all AODLParams fields…},
  channels: [names]}`
- per channel `<ch>_segments` — float64 array, one row per (tone, segment):
  `[tone_idx, t0, T, degree, c0…c9]` (pad to MAX_DEGREE+1 = 10)
- per channel `<ch>_tones` — one row per tone:
  `[tone_idx, phase0, env_kind, env_p0…env_p3]` (env_kind: 0 = constant(amp=p0),
  1 = smooth_on_off(t_on=p0, t_off=p1, ramp=p2))

`save(wfs, path)`, `load(path) -> WaveformSet`. Round-trip must be exact (float-identical
coefficients). Unknown `schema_version` or `env_kind` → clear error. Document the schema
with a worked example in `docs/waveform_format.md`.

## 4. `waveform/export.py`

`render_samples(wfs, sample_rate=625*MHz, t_span=None, dtype=np.float32, chunk=2**20)`
→ `dict[channel, array]`: `Σ_tones A_n(t)·cos(2π f_center t + phase_n(t))` — note the
carrier is added back here (samples are the *absolute* RF signal); chunked evaluation;
normalize the dict by the global max |value| (single common factor, preserving relative
channel amplitudes; record the factor). `save_samples(wfs, path, ...)` → NPZ with samples +
`meta` JSON echoing sample_rate, t_span, normalization; filename must end `_samples.npz`.

## 5. Tests

- `test_ramps.py`: endpoint values/derivatives per ramp family (min-jerk: ẏ=ÿ=0 at ends;
  const-accel: ÿ = ±4Δ/T² on halves; SCJ continuity of y, ẏ, ÿ at T/4, 3T/4); linear ramp
  antiderivative exactness.
- `test_tones.py`: phase continuity across segment breaks (chirp chain); `fdot` matches
  numerical derivative of `f` (rel 1e-6 interior); envelope derivative checks vs numerical;
  `with_hold_until` freezes frequency and keeps phase continuous.
- `test_serialize.py`: build a 3-tone, mixed-ramp, mixed-envelope WaveformSet on
  `default_1030()`; save/load; assert exact equality of breaks/coeffs/phases/env params
  and params round-trip. Assert no array in the NPZ is longer than ~10⁴ (no samples).
- `test_export.py`: single linear-chirp tone (f: 0→2 MHz detuning over 100 µs, f_center
  100 MHz), 625 MS/s: instantaneous frequency check — phase via `np.unwrap(np.angle(hilbert))`
  is noisy at band edges, so instead compare rendered samples at 64 random times against
  direct `A·cos(2π f_c t + phase(t))` (1e-6), plus a zero-crossing-count sanity check of
  mean frequency (±0.1%).

## Definition of done

Own tests green (`pytest tests/test_ramps.py tests/test_tones.py tests/test_serialize.py
tests/test_export.py`); `ruff check` clean on owned files; no git commands; report per
ORCHESTRATION.md §Rules.
