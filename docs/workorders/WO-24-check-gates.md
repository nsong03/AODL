# WO-24 — M6 gate-policy reconciliation (WO-23 findings)

**Role:** implementation agent, Wave U (solo). **You are the wave closer**: commit and
push. **Read first:** `CLAUDE.md`, `docs/workorders/WO-23-verify-m6.md` and the Wave T
report content summarized in `docs/ORCHESTRATION.md` (the findings below reference it),
`src/aodl/check/report.py` + `expect.py`, `tests/test_check_flagship.py`,
`docs/guide.md` §5.5, notebook 07, then this. HEAD carries 438 green tests (`e7792c3`).
This wave closes the WO-23 verdict's "to be reconciled before the release tag" list.

## Owned files

```
src/aodl/check/report.py   src/aodl/check/expect.py   src/aodl/check/pupil.py
tests/test_check_flagship.py  tests/test_check_verdict.py
docs/guide.md   examples/07_fft_checker.ipynb
```

## 1. F-2 — restore fault-detection power on opened-tolerance drives (the substantive item)

Problem: the flagship's physics ripple (~22 % per-frame uniformity) forces the opened
0.30 gate, which also passes a persistent 36 % single-rung fault. Physics vs fault are
separable in TIME: hand-over ripple varies with fade phase across frames; a real rung
fault is a *persistent* per-trap offset at every frame.

Implement a second, verdict-bearing uniformity statistic: **per-trap time-median relative
intensity deviation** (median over the checked frames of each trap's relative-to-frame-
pattern intensity; gate its worst |deviation|). Add `Tolerances.uniformity_median`
(default 0.05 — but MEASURE first, see below). Procedure, in this order:

1. Measure the clean flagship's worst time-median deviation over the standard 7 frames
   (expect it well below the per-frame 22 % — hand-over phases differ per frame).
2. Measure the 0.80-scaled-interior-rung corrupted flagship's value (expect ≈ 0.28
   persistent).
3. Pick the default gate with **both-sided pins in the test**: clean < gate/2 and
   corrupted > 2×gate if the separation allows; otherwise the widest both-sided pin the
   data supports, stated numerically in the test.
4. `tests/test_check_flagship.py`: keep the opened per-frame gates (with their pins),
   add the median gate at its default, and add the 0.80-rung corrupted flagship as a
   MUST-FAIL case (it currently passes — that is the F-2 hole). Requires ≥ 2 checked
   frames for the median to mean anything; document that a single-frame check falls back
   to the per-frame gate only (note in report.notes).
5. If the measured separation is NOT clean (clean median within 2× of the corrupted),
   stop, do not force it: implement the statistic as report-only, keep the corrupted
   case as an xfail documenting the gap, and say so prominently in your report — the
   architect will re-rule.

## 2. F-6 — gate-coverage integrity on fading arrays

- `CheckReport` gains `gated_fraction` per intensity metric (rows gated / rows total).
  When an intensity metric has ZERO gated rows (e.g. a 2×2 fading array where the edge
  exemption is the whole perimeter), `passed` must not silently claim intensity health:
  add a loud entry to `failures` when `require_coverage=True` (new Tolerances field,
  default **False** to preserve behavior) and ALWAYS a note naming the coverage.
  `summary()` prints the coverage line.
- Bound the fading on-lattice blob whitelist: whitelisted on-lattice blobs on a fading
  drive are still gated at a new `blob_fading` tolerance (default 1.2 × median trap
  peak — WO-23 measured a legitimate 1.07× case; anything materially brighter than a
  real trap is wrong even mid-fade). Both-sided test: clean fading 3×3 passes; a planted
  2× blob (inject via a doctored expectation or a synthetic canvas test at the metrics
  level) fails.

## 3. Small items

- **F-3 remnants**: `tests/test_check_flagship.py` docstring + notebook 07 markdown
  still attribute the uniformity spread purely to "IM3 ∝ C²"; correct to "~82 % C²-
  scaled IM3 + a ~3.8 % C-independent floor" (guide already fixed by WO-23).
- **F-7 + F-12 (guide §5.5)**: add (a) a sentence that small fading arrays commonly need
  per-drive intensity openings (a clean fading 3×3 measures waist ≈ 0.11 — worse than
  the 10×10), with the coverage caveat from §2; (b) the timing-skew blind spot to the
  "what a PASS does not certify" list, with the measured law
  |dX| = deflection_scale·|ḟ|·δ and the ≈ 1.22 µs flagship detection threshold.
- **F-8**: read each trap's peak at the fitted center (quadratic interpolation of the
  patch maximum or evaluate the fitted Gaussian), not the nearest grid sample — removes
  the (1/FINE_PER_WAIST)² ≈ 0.35 % quantization floor from uniformity. Update any test
  numbers that legitimately shift (report which).
- **F-9**: fix the `pupil.py` refusal message direction ("too fast" → slow-sound /
  high-f_c wording that matches the actual half_span ∝ v/f_c scaling).
- **F-10**: rename "crest factor" → "normalization factor (peak over single-tone
  amplitude)" in guide §5.5/§8 rows, `record.py` docstring, flagship test docstring,
  notebook 07 markdown (the number 4.5912 is right; the term was wrong).
- **F-11**: when `_comb_window` refuses because `k_subtimes` is too small for the comb
  (bW > k/2) AND a commensurate comb exists, emit a prominent note ("k_subtimes=N too
  small for exact beat cancellation (needs ≥ M); using golden-ratio fallback — verdict
  noise increases") and surface it in `summary()`. Test pins the note's appearance at
  small k and absence at the default.

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `python -m mypy src/aodl` exit 0;
`pytest --nbmake examples/` green (all seven); outputs cleared; nothing binary staged;
commit (`M6 gate-policy reconciliation: median uniformity gate, coverage integrity,
doc corrections`, footer per dispatch) and push. Report: the §1 measured numbers (clean
vs corrupted time-median, chosen gate, both-sided pins) — this is the headline; §2
coverage behavior on the 2×2; every small item confirmed; pytest/nbmake summaries
verbatim; commit hash; deviations (or "none").
