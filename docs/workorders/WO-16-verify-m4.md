# WO-16 — M4 verification pass (fresh eyes)

**Role:** verification agent, Wave M. Same charter as WO-06/09/13 (read WO-06 for ground
rules): adversarial, independent recomputation; small fixes (≤ ~15 lines, no interface
changes) committable as `M4 verification fixes: <list>`; larger defects reported.
Verdict line required: **M4 ACCEPTED** or **M4 REJECTED (blockers: …)**.
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.4 + §3, `docs/conventions.md`,
WO-14/WO-15, the fading-Shepard equations in WO-14 §Physics — then verify them against
first principles, not against the code.

## Checks

1. **Suite + hygiene**: clean install; `pytest`; `ruff check`; `python -m mypy src/aodl`
   (bare `mypy` on PATH is broken); `pytest --nbmake examples/` (all five).
2. **Fade algebra from first principles**: re-derive the S26/S27 windows and the
   p_A + p_B = 1 constant-power identity yourself; check the implemented A(g) at
   hand-picked g values and the co-located product = cos θ / sin θ structure. Verify
   dA/d2A numerically *and* probe the p < 1 edge-divergence clamp: line amplitudes near
   the off edge must stay below the prune threshold so the clamp is invisible — quantify.
3. **The Shepard claim, adversarially**: your own long-hold spec (pick ∫Z dt ≥ 5× the
   Eq. 1 ceiling, plus a simultaneous lateral drift): max active-tone |f| bounded as
   WO-14 states, tracking at machine-ish precision on plateaus and within bounds
   mid-fade, total co-located power flat (< 1%) over ≥ 5 fade cycles — measured from
   *rendered frames*, not only `measure`.
4. **Shadow tweezers**: predict the offset and the mid-fade intensity ratio yourself
   from the fade product structure; compare against rendered frames; check they vanish
   on plateaus and that an array grows the extended grid exactly (Mx+2)×My during
   x-fades. If code and your prediction disagree, yours wins — finding.
5. **Interlaced vs simultaneous**: reproduce the static-interference sensitivity result
   with your own phase-offset sweep; confirm interlaced immunity; confirm
   `power_coherent`'s destructive-pair zero against a frame integral.
6. **Serialization**: schema v2 round-trip bit-exact including fade envelopes; a v1
   file still loads; re-simulation of a loaded long-hold WaveformSet reproduces spot
   metrics bitwise.
7. **Band occupancy**: over the unhurried user story, every channel's instantaneous
   tone frequencies (including fading tones) stay inside the ±10 MHz band — measure
   from the WaveformSet itself.
8. **Movies/notebooks**: read-back checks; notebook-quoted numbers re-derived; movie 05
   shows the full unhurried story with fades invisible in total brightness.
9. **Performance**: 1 ms hold simulate (24 probes) and one 512² frame at order 1 and 3;
   report numbers.

## Output

Same format as prior verification reports: numbered findings (severity, file:line,
repro); fixes with hash; independent-vs-code table; performance; explicit verdict line.
