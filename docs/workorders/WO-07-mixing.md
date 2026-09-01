# WO-07 — Intra-AOD intermodulation (IM3) and compression

**Role:** implementation agent, Wave E (solo). **You are the wave closer**: commit your
files and push when done. **Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.2 (frequency
mixing), `docs/conventions.md`, `src/aodl/device/aod.py` (current fundamentals-only
`channel_lines`), then this. M1 is merged and verified (122 tests).

## Owned files

```
src/aodl/device/mixing.py
src/aodl/device/aod.py        (edit: mixing hook in channel_lines — keep the edit minimal)
src/aodl/device/aodl.py       (edit: term-level amplitude pruning only)
src/aodl/params.py            (edit: add optional field, see §1 — nothing else)
tests/test_mixing.py
```

## 1. Params amendment (architect-sanctioned)

`AODParams` gains `mixing_order: int = 3` (allowed values 1, 3). Existing presets pick up
the default. No other params change.

## 2. Physics (Eq. S20–S22, coefficients fixed by the architect — implement verbatim)

Per channel at evaluation time t_c, define per-tone modulation depth
**m_n = drive_strength · A_n(t_c)** and phase φ_n (the tone's `phase(t_c)`, rotating
frame). Expanding P = exp(iCV) and keeping the +1-order band (net exponent −iΦ):

| Line | Frequency (detuning) | Chirp | Complex amplitude | Phase factor |
|------|---------------------|-------|-------------------|--------------|
| fundamental n (order 1) | f_n | ḟ_n | (i/2)·m_n | e^{−iφ_n} |
| compression correction to n (order 3) | f_n | ḟ_n | −(i/16)·m_n³ − (i/8)·m_n·Σ_{m≠n} m_m² | e^{−iφ_n} |
| IM3, indices j<k, i∉{j,k} | f_j+f_k−f_i | ḟ_j+ḟ_k−ḟ_i | −(i/8)·m_i·m_j·m_k | e^{−i(φ_j+φ_k−φ_i)} |
| IM3 degenerate, j, i≠j | 2f_j−f_i | 2ḟ_j−ḟ_i | −(i/16)·m_i·m_j² | e^{−i(2φ_j−φ_i)} |

Sanity identity (make it a test): single tone at mixing_order=3 gives
(i/2)m(1 − m²/8) — the first two terms of i·J₁(m); relative error vs `scipy.special.j1`
must be < m⁴/100 for m ≤ 0.5.

**Envelope/irising polynomial for a mixed line**: the line's effective envelope is the
product of its constituents (with multiplicity: {i:1, j:1, k:1} or {i:1, j:2}). Using
per-tone log-derivatives l1_n = A′_n/A_n, l2_n = A″_n/A_n (guard A=0 → line amplitude is
0, skip): with L1 = Σ mult·l1 and L1′ = Σ mult·(l2 − l1²), the line's normalized α is
`(1, −s·L1/v, (L1² + L1′)/(2v²))` — same convention as fundamentals (α0 = 1, envelope
magnitude lives in the complex amplitude).

**Selection/pruning (all configurable kwargs with these defaults):**
- band acceptance: keep a line iff its absolute frequency `f_center + f_line` lies within
  the channel band widened by `band_margin = 0.2` (fraction of band width) on each side;
- amplitude prune: drop lines with |amp| < `line_prune = 1e-5` × max fundamental |amp|;
- term prune (in `build_terms`): after the channel product, drop terms with
  |c| < `term_prune = 1e-6` × max |c|. Document that pruning bounds relative intensity
  error by ~(prune)², and count pruned power (expose as a diagnostic return/attr).

## 3. API

```python
# device/mixing.py
@dataclass(frozen=True) class MixingConfig: order=3, band_margin=0.2, line_prune=1e-5
def expand_lines(fund, aod, cfg) -> Lines
    # fund: the fundamentals-only Lines (+ per-tone A, dA, d2A, phases) that
    # channel_lines already assembles; returns the full Lines including corrections,
    # IM3 lines, α polys; vectorized over tone triples (loops over O(M³) index sets are
    # fine for M ≤ ~32 if numpy-batched per signature class)
```

`channel_lines(cw, aod, t, mixing=None)`: `mixing=None` → use `aod.mixing_order` with
default MixingConfig; `mixing=MixingConfig(...)` overrides; order 1 must reproduce the
current M1 behavior bit-for-bit (regression-tested against a saved expectation, not
against the old code path).

## 4. Tests (`tests/test_mixing.py`)

- **Bessel identity** (above), plus: two equal tones m each — fundamental amp
  (i/2)m(1 − m²/8 − m²/4); assert against an independent frozen-time numeric check:
  build V(u) on a fine grid at fixed t, form exp(iC·V) literally, project onto
  e^{−iΦ_line(u)} by inner product over one acoustic period window, and compare each
  line's complex amplitude (rel 1e-3 at m = 0.2; this is a *reference* FFT-free
  projection, tests-only, in the spirit of field/reference.py).
- **Ghost bookkeeping**: 3 tones at detunings (−2, 0, +2) MHz, equal m: IM3 lines land at
  ±4, ±6 MHz *and* exactly on the fundamentals (degenerate f_j+f_k−f_i collisions);
  degenerate lines must MERGE with the fundamentals' df_opt (same frequency ⇒ coherent).
  Assert line count after merging, and the ±4 MHz ghost amp = −(i/16)m³ − (i/8)m³·(count).
  Careful: (j,k;i) = (0,+2;−2) gives +4 MHz with −(i/8)m³, and (j;i) = (+2... wait —
  (2f_{+2}−f_0) also gives +4 MHz with −(i/16)m³. Sum them: total = −(3i/16)m³.
  The test pins THIS total (the architect's number; if your enumeration disagrees,
  re-derive by the frozen-time projection and report).
- **Band acceptance**: tones near band edge produce IM3 outside the widened band → absent.
- **Pruning**: line_prune/term_prune drop what they should; diagnostic pruned-power
  counter consistent.
- **Order-1 regression**: `mixing_order=1` output identical to a frozen snapshot of the
  M1 fundamentals for a 2-tone chirping channel (amp, f, fdot, α, df_opt).
- **Envelope products**: SmoothOnOff on tone j only: IM3 line (j,k;i) α₁ scales with
  1·l1_j; degenerate (j,j;i) with 2·l1_j (probe mid-ramp).

## Definition of done

Full `pytest` green (including all M1 tests — your edits must not disturb them);
`ruff check src tests` clean; commit (`M2: intra-AOD intermodulation (IM3) and
compression lines`, footer lines per dispatch) and push. Report: line-count and timing
for an M=10 equal-amplitude channel (fundamentals+IM3, before/after pruning), test
summary, deviations.
