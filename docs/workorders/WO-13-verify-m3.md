# WO-13 — M3 verification pass (fresh eyes)

**Role:** verification agent, Wave J. Same charter and ground rules as WO-06/WO-09 (read
WO-06 for them): adversarial, independent recomputation, small fixes (≤ ~15 lines, no
interface changes) committable as `M3 verification fixes: <list>`, larger defects
reported. Verdict line required: **M3 ACCEPTED** or **M3 REJECTED (blockers: …)**.
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.4 + §3, `docs/conventions.md`,
WO-10/11/12, then execute.

## Checks

1. **Suite + hygiene**: clean install; `pytest`; `ruff check`; `mypy src/aodl`;
   `pytest --nbmake examples/` (all four). Any red = finding.
2. **Eq. S19 inversion, independently**: pick a nontrivial spec (your own numbers, not
   the suite's), synthesize, then — without `measure` — recover the trajectory from the
   *waveforms alone*: X̂(t) = deflection_scale·(f_Bx − f_Ax)(t), Ŷ likewise,
   Ẑ(t) = 2·lens_scale·ḟ_Z extracted as the common chirp (e.g. ½(ḟ_Ax + ḟ_Bx) minus the
   differential part). Must equal the requested profiles to numerical precision. Then
   close the loop through the *rendered field* at 3 probe times (your own quadrature or
   the package's, but measure positions from rendered peaks): tracking within the stated
   test tolerances, at t − τ/2.
3. **Astigmatism-free claim, adversarially**: during a fast Translate (near band-limit
   speed), scan rendered waists vs z at several probe times: confirm |Z_x − Z_y| stays
   < 0.02 z_R *and* compare against the 2-AOD (Ax+Ay-only) control — the control must
   FAIL the same bound (if it doesn't, the test isn't probing anything; report).
4. **Fill physics**: pair-driven spot strictly dark before τ/2 (probe 0.3τ, 0.45τ);
   turn-on curve between τ/2 and τ against your own two-edged-pupil quadrature
   (rel 1e-3); four-channel co-chirp at 0.8τ finite and sensible.
5. **Band check** (Eq. 1): recompute the max feasible |∫Z dt| yourself from
   `lens_scale·(f_max − f_min)` bookkeeping; the synthesizer's error message must state
   the same number (±rounding) and trigger exactly at the boundary (bisect a Hold
   duration to find the code's threshold; compare to yours).
6. **Guard + serialization**: WO-10's max_terms guard behavior matches its documented
   numbers; a synthesized user-story WaveformSet round-trips through NPZ bit-exact and
   re-simulates identically (spot metrics equal).
7. **Notebooks + movies**: both new movies play (imageio read-back), the 04 movie shows
   the full array lift-translate-drop with hue tracking Z̄; notebook-quoted numbers match
   what the code actually produces on re-run.
8. **Performance**: time the 10×10 user-story simulate + one 512² frame at order 1 and
   order 3; report both.

## Output

Same format as WO-06/WO-09: numbered findings with severity/file:line/repro; fixes with
hash; independent-vs-code number table; performance numbers; explicit verdict line.
