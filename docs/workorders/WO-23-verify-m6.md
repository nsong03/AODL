# WO-23 — M6 verification pass (fresh eyes)

**Role:** verification agent, Wave T. Same charter and ground rules as WO-06/09/13/16/19
(read WO-06): adversarial, independent recomputation; small fixes (≤ ~15 lines, no
interface changes) committable as `M6 verification fixes: <list>`; larger defects
reported. Verdict line required: **M6 ACCEPTED** or **M6 REJECTED (blockers: …)**.
**Read first:** `CLAUDE.md`, `docs/PLAN.md` (incl. the new M6 block), `docs/guide.md`
§5.5, WO-21/WO-22, then execute. Use `python -m mypy src/aodl` (bare mypy broken).

## Checks

1. **Suite + hygiene**: clean venv install; `pytest`; `ruff check` + format; mypy;
   `pytest --nbmake examples/` (all seven).
2. **Numerics re-derived**: the du = Λ/8 alias arithmetic (which harmonic first folds
   onto the +1 band, at what Bessel level — your own m-mod-8 bookkeeping); the
   `(2J₁(C)/C)²` compression constant vs the code's bragg/weak ratio at C ∈
   {0.1, 0.3, 0.5}; the band-window geometry (centers at s·f_c/v per channel, clearance
   from DC/±2 orders); the cubic-Hermite interpolation error scaling (oversample sweep).
3. **Sign audit, adversarially**: build one drive per channel with a known detuning and
   chirp; confirm the checker's measured position and focus signs against Table I for
   all four channels; `Z_LAB_SIGN` in the defocus phase (up-chirp co-chirp → +z_lab);
   the t − τ/2 alignment BOTH retard modes — synthesize a `retard_compensate=True`
   plan, confirm `MotionPlan.check()` auto-wires it, then deliberately mis-set
   `Expectation.retard_compensated` and confirm the failure is legible (every position
   off by v_lat·τ/2, named in the failure strings).
4. **The founding story from scratch**: write the flagship check yourself in ≤ 10 lines
   (plan → check → summary), PASS; then invent **one new corruption not in the test
   suite** (e.g. a 2-sample timing skew on one channel, or a π phase flip on one ladder
   tone) and report whether and how the checker catches it — if it doesn't, quantify
   what it would take and file the finding.
5. **Independence audit**: re-run the source scan yourself; confirm no `check/` module
   imports simulation internals; confirm the SimResult comparison is structural
   (grep for engine imports); reason at source level that `check/` paths function
   standalone.
6. **Cross-validation honesty**: re-measure the weak-vs-sim gate margins (fields,
   positions, powers) — are the stated tolerances ≥ 10× above the observed residuals?
   Probe one Shepard fade frame and confirm it is excluded from tight gates rather
   than silently passing.
7. **Docs-number audit**: every number guide §5.5 and notebook 07 quote, recomputed;
   the doc code blocks execute; the scoped "No FFTs" wording is consistent everywhere
   (grep the census: CLAUDE.md, README, PLAN, ARCHITECTURE, guide, engine.py,
   field/gaussian.py).
8. **Performance**: flagship check wall time (target ≤ 10 s at k_subtimes=48, ~7
   frames); new-test suite delta (≤ 60 s); notebook 07 runtime (≤ 3 min); no regression
   > 1.5× on the M5-era benchmarks (re-benchmark `git archive` of the pre-M6 commit if
   needed).

## Output

Standard format: numbered findings (severity, file:line, repro); fixes with hash;
independent-vs-code table; performance table; explicit verdict line.
