# WO-14 — Fading-Shepard waveforms (Eqs. S24–S28)

**Role:** implementation agent, Wave K (solo). **You are the wave closer**: commit and
push when done. **Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.4 + §1.5 (Eq. 1 budget),
`docs/conventions.md`, `src/aodl/waveform/{tones,synthesis,serialize}.py`,
`src/aodl/device/aod.py` (how envelopes feed irising), then this. M3 is merged and
verified (255 tests at HEAD).

## Physics (transcribe exactly; frequencies are detunings, rotating frame)

Each channel becomes a tone *ladder*, n ∈ ℤ (Eq. S24/S25):

- `f_μ^(n)(t) = f_lat_μ(t) + f_Z(t) + (n + ξ_μ)·Δf_axis` where
  `f_lat = −(v/2λF)X` for Ax, `+(v/2λF)X` for Bx (Y for Ay/By), and
  `f_Z(t) = (v²/2λF²)∫₀ᵗ Z dτ` (unbounded — the point of the scheme).
- Define the tone's **fade coordinate** `g_n(t) = f_Z(t) + (n + ξ_μ)·Δf` (the S25
  `f_{μ,Z}^(n)`; lateral term excluded). Amplitude (Eq. S26, A channels):
  `A^(n) = 1` for `|g| ≤ (1−η)Δf/2`; `0` for `|g| ≥ (1+η)Δf/2`; else
  `cos^{p_μ}[ (π/2η)(|g|/Δf − 1/2) + π/4 ]`.
  B channels (Eq. S27): boundaries `(M∓η)Δf/2` and argument
  `(π/2η)(|g|/Δf − M/2) + π/4`, where M = Mx (Bx) or My (By).
- Configuration (paper Table II), η = 1/2 default:
  single tweezer — (M, p, ξ): Ax (1, 0.5, 0), Ay (1, 0.5, 0.5), Bx (1, 0.5, 0),
  By (1, 0.5, 0.5). Array Mx×My — Ax (1, 1, 0), Ay (1, 1, 0.5), Bx (Mx, 0, 0),
  By (My, 0, 0.5). Interlacing = the 0.5 ξ offset on the y pair; p_A + p_B = 1 per axis
  keeps the summed old/new tweezer power constant (cos² + sin² after the product —
  derive in the module docstring).
- B-ladder Schroeder phases (Eq. S28): `φ^(n) = mod(2π·n(n−1)/(2M), 2π)`; A-ladder
  phases 0.
- Only tones with nonzero amplitude somewhere in the run exist: n such that |g_n|
  enters the channel's outer fade boundary — compute the range from the f_Z span
  (finite ladder; assert its size is (f_Z span)/Δf + O(M)).
- **The Shepard claim** (assert in tests): max |f_μ^(n)| over active tones stays
  ≤ (M+η)Δf/2 + max|f_lat| + Δf for the entire run, however large ∫Z grows.

## Owned files

```
src/aodl/waveform/shepard.py       (new: FadeZoneEnvelope + ladder builder)
src/aodl/waveform/synthesis.py     (edit: shepard wiring into synthesize)
src/aodl/waveform/tones.py         (edit: only if the Envelope protocol needs a hook)
src/aodl/waveform/serialize.py     (edit: schema v2, see below)
docs/waveform_format.md            (edit: schema v2 section)
tests/test_shepard.py
```

## Implementation contract

- `FadeZoneEnvelope`: an `Envelope` whose `A(t)` is the S26/S27 window evaluated at
  `g(t)` (a `PiecewisePoly`), with `dA`, `d2A` by chain rule.
  **Trap (pre-empted):** for p < 1 the derivative diverges at the A→0 edge
  (d cos^p ~ cos^{p−1}); clamp the irising log-derivatives by evaluating them at
  `max(A, 1e-3)` and document that lines there are already below the amplitude prune —
  the truncation is invisible. |g| is piecewise in t: split segments at the (poly-root)
  zone-boundary and g=0 crossings so every branch is smooth; cache the crossing times.
- `ShepardConfig(delta_f_x, delta_f_y, eta=0.5, config="auto")` — "auto" picks Table II
  (M, p, ξ) from the ArraySpec (single vs array); explicit per-channel overrides allowed.
  For arrays the B ladder spacing must equal the array Δf (they are the same ladder —
  raise if a conflicting delta_f is passed).
- `synthesize(spec, params, *, shepard=None|"auto"|ShepardConfig, ...)`:
  `None` → current M3 behavior (band check may refuse). `"auto"` → use fading-Shepard
  iff the plain-S19 band check fails (report which in `WaveformSet.description`);
  a `ShepardConfig` → always fading-Shepard. Band check under Shepard verifies the
  *bounded* excursion above instead of Eq. 1.
- Serialization **schema v2** (architect-sanctioned): additive `<ch>_env_polys` table
  storing each fade envelope's g-poly segments (same 14-column layout as
  `<ch>_segments`, keyed by tone index), env_kind 2 with env_params
  (delta_f, eta, p, M) — ξ is inside the stored g-poly's constant term. v1 files load
  unchanged (write `schema_version: 2` only when a fade envelope is present). Update
  `docs/waveform_format.md` with a worked v2 example. Round-trip bit-exact.

## Tests (`tests/test_shepard.py`)

- Envelope algebra: plateau/zero/fade-zone values at hand-computed g; boundary values
  (1 at inner, 0 at outer, cos(π/4)^p at center); dA/d2A vs numerical derivative away
  from clamps; the p_A + p_B = 1 identity: `A_A(g)·A_B(g)` for the co-located pair
  equals `cos(θ)` and the partner pair `sin(θ)` with θ the S26 argument → sum of
  squares = 1 to 1e-12 across the fade (single-tweezer config).
- Ladder bookkeeping: a Z-hold spec with ∫Z dt = 5× the Eq. 1 ceiling: active-tone
  count matches the predicted range; **max |f| bounded** per the Shepard claim; plain
  S19 on the same spec raises.
- End-to-end (through `simulate`, mixing_order=1, probes on the plateau between fade
  zones AND mid-fade): single tweezer holding Z = +10 µm for 1 ms — X, Y static
  (< 1% waist), Z̄ tracks (< 2% z_R), |ΔF| < 0.02 z_R, and **total tweezer power
  (sum of the co-located frequency groups) constant to < 1%** through ≥ 4 complete
  fade cycles.
- Shadow tweezers (S31): mid-x-fade, two extra groups at exactly
  `±deflection_scale·Δf_x` in X with peak intensity `(cosθ·sinθ)²·…` — pin the
  measured peak ratio at fade center (derive the expected value; if your derivation
  disagrees with `(cos·sin)²` of the main trap, trust your derivation and report);
  absent outside fade zones. Interlacing: during an x-fade the y-pair is on its
  plateau (ξ offset works).
- Schema v2 round-trip; v1 file still loads (regression).

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `python -m mypy src/aodl` exit 0
(bare `mypy` on PATH is broken — always `python -m mypy`); commit (`M4: fading-Shepard
waveforms (Eqs. S24-S28)`, footer per dispatch) and push. Report: files, pytest summary
verbatim, the sustained-hold numbers (power flatness, tracking, max-|f| bound), shadow
ratio derivation vs measurement, active-tone counts, deviations (or "none").
