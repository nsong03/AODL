# WO-10 — Two-sided aperture window + term-product guard

**Role:** implementation agent, Wave H (parallel with WO-11 — it owns `trajectory/spec.py`
and `waveform/synthesis.py`; you must not touch those). **No git.**
**Read first:** `CLAUDE.md`, `docs/conventions.md` (timing/fill section),
`src/aodl/field/gaussian.py`, `src/aodl/device/aodl.py`, `src/aodl/field/focal.py`
(edge normalizer), then this. Context: M2 is merged and verified (188 tests at HEAD).

## Why

M3 drives counter-propagating pairs (Ax+Bx). For t < τ each is filled from its own
transducer side; today `device/aodl.py` raises `NotImplementedError` (WO-06 finding 4).
The physics: channel with sound_sign s has acoustic content where `s·u ≤ v·t − D/2`, so
Ax (s = −1) fills u ≥ D/2 − vt and Bx (s = +1) fills u ≤ vt − D/2. Their intersection
`[D/2 − vt, vt − D/2]` is **empty until t = τ/2** (both waves must reach a point), then
grows to the full aperture at t = τ. A pair-driven tweezer is therefore strictly dark for
t < τ/2 — a nice testable prediction.

Also (WO-09 finding 5): guard the Cartesian term product against non-commensurate
multi-tone blowup.

## Owned files

```
src/aodl/field/gaussian.py    (add window moments)
src/aodl/device/aodl.py       (interval intersection; remove NotImplementedError; guard)
src/aodl/field/focal.py       (edge normalizer: accept two-sided windows)
tests/test_window.py
```

## 1. `gauss_moments_window(a, b, u0, u1)` in `field/gaussian.py`

$W_n = \int_{u_0}^{u_1} u^n e^{-au^2+bu}\,du = E_n(a,b,u_0) - E_n(a,b,u_1)$, n = 0..2,
requiring u0 < u1 (raise otherwise). Reuse the stable `gauss_moments_lower`. Document the
cancellation caveat: when both edges sit on the same far tail the difference loses digits,
but the result is then negligible relative to full-aperture terms — quantify in a test
(agreement with quad degrades only where |W0| < 1e-12·|I0|). Vectorized like the rest.

## 2. `device/aodl.py`

Represent each axis's fill state as an interval `(lo, hi)` with `None` = unbounded:
each participating channel contributes its one-sided constraint (or none when filled);
intersect. Empty or negative-length interval → the term's amplitude is exactly 0 (keep
the term with c = 0 or drop it — dropping is fine, document). Both-sided interval →
edge info `(lo, hi)` passed through `TermArray.edge` for that axis; one-sided and full
cases unchanged. Remove the `NotImplementedError` path entirely.

**Product guard**: (a) pre-product per-channel amplitude cut at the same `term_prune`
threshold — a line with |amp| < term_prune·max|amp| of its channel can only produce
terms below the post-product cut (|c| = Π|amp|), so dropping it early is lossless
(document this bound); pruned power still accounted. (b) `max_terms = 200_000` kwarg on
`build_terms`: if the post-cut product would exceed it, raise a clear error naming the
per-channel line counts and suggesting `line_prune`/`term_prune`/`max_terms`.

## 3. `field/focal.py` edge normalizer

Accept the two-sided `(lo, hi)` form and route to `gauss_moments_window`; one-sided and
`None` behavior unchanged (regression-covered by the existing suite).

## 4. `tests/test_window.py`

- `gauss_moments_window` vs `scipy.integrate.quad` over random complex (a, b, u0, u1)
  (physical magnitudes; rel 1e-9 where |W0| > 1e-10·|I0|), and consistency
  `W_n(−∞..∞ limits via large window) → I_n`.
- **Pair fill physics** (Ax + Bx, static tones): intensity exactly 0 at t = 0.45τ;
  nonzero at 0.55τ; matches `reference_field_separable` with the literal two-edged pupil
  at t = 0.75τ (rel 1e-3, cut-cell edge weights as in WO-03's tests); full-aperture value
  and no edge info at t ≥ τ. Reuse the WO-03 test-file patterns for building literal
  pupils (read `tests/test_device_single_aod.py` first).
- Four-channel co-chirp scene at t = 0.8τ runs end-to-end through `simulate` +
  `intensity_frame` without error (metrics finite).
- Guard: 12 irrational-ratio tones on two channels with mixing_order=3 →
  raises the max_terms error by default settings? — compute what the post-cut count is
  and assert the documented behavior (raise, or pass if the pre-cut already tames it —
  pin whichever is true with the numbers in the assertion message); with
  `max_terms=10**7` explicitly, it must not raise. Pre-product cut losslessness: term set
  with cut on/off identical above the post-product threshold; pruned_power consistent.

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `mypy src/aodl` exit 0; no git.
Report: files, test summary verbatim, the t = 0.45τ/0.55τ/0.75τ numbers, guard behavior
numbers, deviations (or "none").
