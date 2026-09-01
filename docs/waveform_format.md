# AODL waveform file format — schema v1

A `.npz` written by `aodl.waveform.serialize.save` (or `WaveformSet.save`) contains the
**parametric function representation** of a waveform set: segment breakpoints, polynomial
coefficients, envelope parameters, tone phases and a hardware snapshot.

> **It never contains samples.** Samples are a render target, produced on demand by
> `aodl.waveform.export.render_samples(wfs, sample_rate)` and — if they must be written to
> disk at all — stored separately in a file whose name ends `_samples.npz`
> (§6). A parametric file for a 100 µs, four-channel, ten-tone move is a few kilobytes;
> the same move sampled at 625 MS/s in float32 is 1 MB, and a 10 ms rearrangement
> sequence is 100 MB. Round-tripping the parameters is exact, so what you simulate is
> bit-for-bit what you export (`docs/ARCHITECTURE.md` §0.1, decision §5.5).

---

## 1. Entries in the archive

| key | dtype | shape | contents |
|-----|-------|-------|----------|
| `meta` | `<U…` (0-d) | — | JSON string, §2 |
| `<ch>_segments` | `float64` | `(n_rows, 14)` | one row per *(tone, polynomial segment)*, §3 |
| `<ch>_tones` | `float64` | `(n_tones, 7)` | one row per tone, §4 |

`<ch>` runs over the driven channels only (`meta["channels"]`), each a member of
`aodl.params.CHANNELS = ("Ax", "Bx", "Ay", "By")`. An undriven channel simply has no
arrays; the device layer treats it as an identity factor in the pupil product (Eq. S7).

**Frequencies in the file are detunings from `f_center`** (the Eq. S2 rotating frame), in
Hz, versus time in seconds. The carrier is re-added only by `render_samples`.

## 2. `meta` — JSON metadata

```json
{
  "schema_version": 1,
  "description": "demo: two tones, 100 us",
  "params": {
    "optics": {"wavelength": 1.03e-06, "focal_length": 0.006500000000000001, "w_in": 0.002},
    "channels": {
      "Ax": {"sound_speed": 650.0, "aperture": 0.0075, "f_center": 100000000.0,
             "band": [90000000.0, 110000000.0], "drive_strength": 0.3, "mixing_order": 3},
      "Bx": {"…": "…"}, "Ay": {"…": "…"}, "By": {"…": "…"}
    }
  },
  "channels": ["Bx"]
}
```

- `params` is a complete snapshot of the `AODLParams` the waveform was designed for — all
  four channels, whether driven or not — so a loaded file needs no external context.
  JSON round-trips float64 exactly (shortest repr), so `loaded.params == original.params`.
- `channels` lists the driven channels, in the order their arrays should be read.
- A `schema_version` this build does not know raises `ValueError` on load rather than
  guessing.
- Keys may be *added* to a channel's params block without bumping `schema_version`; a
  reader supplies its documented default for a key an older file lacks. So far that is
  `mixing_order` (added with M2, default **1** when absent — the package default at the
  time files without it could be written).

## 3. `<ch>_segments` — piecewise-polynomial frequency laws

One row per polynomial segment, grouped by tone, in time order:

| column | 0 | 1 | 2 | 3 | 4 … 13 |
|--------|---|---|---|---|--------|
| meaning | `tone_idx` | `t0` [s] | `T` [s] | `degree` | `c0 … c9` |

Segment `k` of a tone covers `[t0, t0 + T]` and evaluates in **normalized local time**
`tau = (t - t0) / T ∈ [0, 1]` as

```
f(t) = sum_j c_j * tau**j          (j = 0 … degree)
```

Coefficients beyond `degree` are padded with zeros to the fixed width
`MAX_DEGREE + 1 = 10` (`aodl.poly.MAX_DEGREE`), which is what keeps the table a plain
rectangular `float64` array. All segments of one tone share the same `degree`.
Breakpoints are reconstructed as `[t0_0, t0_1, …, t0_{K-1}, t0_{K-1} + T_{K-1}]`, so
interior breaks are stored, not accumulated.

Outside its programmed span a frequency law **clamp-holds** (`aodl.poly.PiecewisePoly`):
the tone keeps its terminal frequency, but its *phase* stops advancing. Use
`ToneTrack.with_hold_until(t_end)` to append an explicit hold segment whenever a tone must
stay coherent past its last ramp — that is also how all tones in a set are made to cover
the same span, which `WaveformSet` requires.

## 4. `<ch>_tones` — phases and envelopes

One row per tone:

| column | 0 | 1 | 2 | 3 … 6 |
|--------|---|---|---|-------|
| meaning | `tone_idx` | `phase0` [rad] | `env_kind` | `env_p0 … env_p3` |

`phase0` is the constant phase offset added to the exact integral of the frequency law:

```
phase(t) = 2*pi * ∫ f dt' + phase0        [rad, rotating frame]
```

Schroeder phases for tone ladders (Eq. S23/S28) live in this column.

| `env_kind` | envelope | parameters |
|-----------|----------|------------|
| `0` | `ConstantEnvelope` | `p0 = amp` (in `[0, 1]`) |
| `1` | `SmoothOnOff` | `p0 = t_on`, `p1 = t_off`, `p2 = ramp` [s] |

Unused parameter columns are zero. An `env_kind` this build does not know raises
`ValueError` on load; an envelope class the schema cannot represent raises `TypeError` on
save. `SmoothOnOff` is off before `t_on`, rises as `sin²(π (t − t_on) / (2 ramp))` over
`ramp`, holds at 1, falls symmetrically to zero at `t_off`, and is off after; it requires
`t_off − t_on ≥ 2 ramp`.

## 5. Worked example

```python
from aodl.params import default_1030
from aodl.trajectory import ramps
from aodl.units import MHz, us
from aodl.waveform.tones import (ChannelWaveform, ConstantEnvelope, SmoothOnOff,
                                 ToneTrack, WaveformSet)

tone0 = ToneTrack(ramps.min_jerk(0.0, 100 * us, 0.0, 2 * MHz))            # 0 -> +2 MHz
tone1 = ToneTrack(ramps.constant_accel(0.0, 100 * us, -1 * MHz, 1 * MHz), # -1 -> +1 MHz
                  SmoothOnOff(t_on=0.0, t_off=100 * us, ramp=20 * us),
                  phase0=2.0943951023931953)                              # 2*pi/3

wfs = WaveformSet({"Bx": ChannelWaveform((tone0, tone1))},
                  params=default_1030(),
                  description="demo: two tones, 100 us")
wfs.save("demo.npz")            # 4.1 kB
```

`demo.npz` then holds `meta`, `Bx_segments` and `Bx_tones`:

```
Bx_segments  (3, 14)
  tone t0        T         deg  c0      c1     c2     c3     c4     c5     c6..c9
  [ 0,  0,       1e-04,     5,   0,      0,     0,     2e+07, -3e+07, 1.2e+07, 0,0,0,0 ]
  [ 1,  0,       5e-05,     2,  -1e+06,  0,     1e+06, 0,      0,      0,      0,0,0,0 ]
  [ 1,  5e-05,   5e-05,     2,   0,      2e+06,-1e+06, 0,      0,      0,      0,0,0,0 ]

Bx_tones     (2, 7)
  tone phase0    kind  p0      p1      p2      p3
  [ 0,  0,        0,    1,      0,      0,      0    ]   # ConstantEnvelope(amp=1)
  [ 1,  2.0944,   1,    0,      1e-04,  2e-05,  0    ]   # SmoothOnOff(0, 100us, 20us)
```

Reading the first row: tone 0 has a single quintic segment on `[0, 100 µs]` with
`f(tau) = 0 + 20·tau³ − 30·tau⁴ + 12·tau⁵` MHz — the min-jerk profile of Eq. S14 scaled by
Δ = 2 MHz. Tone 1 has the two parabolic halves of the constant-acceleration profile
(Eq. S16): `−1 + 1·tau²` MHz on `[0, 50 µs]` and `0 + 2·tau − 1·tau²` MHz on
`[50 µs, 100 µs]`, each in *its own* normalized time.

```python
back = WaveformSet.load("demo.npz")
back.channels["Bx"].tones[1].freq.coeffs   # float-identical to the original
```

## 6. Rendered samples — a different file

`aodl.waveform.export.save_samples(wfs, "run_samples.npz", sample_rate=625e6)` writes the
*expanded* signal for an AWG. The filename **must** end in `_samples.npz`, so the two
kinds of file can never be confused. Its contents:

| key | contents |
|-----|----------|
| `meta` | JSON: `schema_version`, `kind: "samples"`, `description`, `sample_rate`, `t_span`, `n_samples`, `t_start`, `normalization`, `channels`, `f_center` per channel, `dtype` |
| `<ch>` | `float32` (by default) samples, `n_samples` long |

Sample `k` sits at `t_start + k / sample_rate`, and holds

```
V(t) = sum_n A_n(t) * cos(2*pi*f_center*t + phase_n(t))      / normalization
```

— note the carrier `f_center` is present here and nowhere else. All channels are divided
by the *same* `normalization` factor (the global peak over the whole render), so relative
channel amplitudes, which set the diffraction balance of the four AODs, survive intact;
multiply by `normalization` to recover the raw sum of tone amplitudes.
