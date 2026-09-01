# WO-19 — M5 verification + whole-product release audit (fresh eyes)

**Role:** verification agent, Wave P. Same charter as WO-06/09/13/16 (read WO-06 for
ground rules). Verdict line required: **M5 ACCEPTED — RELEASE CANDIDATE** or
**M5 REJECTED (blockers: …)**. This is the final gate: audit the *product*, not only
the milestone. **Read first:** `CLAUDE.md`, `docs/PLAN.md`, `docs/guide.md`,
`README.md`, WO-17/WO-18, then execute.

## Checks

1. **Suite + hygiene**: clean venv install; `pytest`; `ruff check`;
   `python -m mypy src/aodl`; `pytest --nbmake examples/` (all six).
2. **The founding user story, verbatim**: from the project brief — *"a 10×10 array of
   atoms moves from A to B by first going out of plane by 10 µm, then traveling in a
   straight line to B, then dropping onto the destination"*. Write it yourself with
   `plan_motion` in ≤ 10 lines, both hurried (plain S19) and unhurried (Shepard).
   Verify: report numbers against your own sweep of the tone laws; tracking at M3
   tolerances from `simulate`; the parametric NPZ round-trips and re-simulates bitwise;
   `render_samples` FFT stays in band for the A channels and quantify the B-channel
   splatter with `switch_ramp = 0` vs a τ-scale ramp (the WO-17 option) — the ramp must
   cut out-of-band power by ≥ 20 dB, else finding.
3. **New options, adversarially** (recompute expectations first): lattice coincidence
   S19 vs Shepard for M ∈ {2,3,4,5} (and that WO-16 F-2's half-pitch offset is gone);
   `f_z_bias="auto"` doubles the feasible hold (bisect both, ±2%); `retard_compensate`
   makes measured motion match at t exactly while default matches at t − τ/2; defaults
   reproduce pre-M5 behavior (synthesize a saved M4-era spec and compare bit-exact).
4. **Docs audit**: run every fenced code block in README + guide.md yourself; recompute
   every quoted number (scales, budgets, flatness, splatter, ρ law, shadow ratio,
   extended-grid parity); check the guide's limitation statements against the code's
   actual behavior. A wrong or unreproducible number is a finding (severity by impact).
5. **Repo hygiene sweep**: conventions (sign authority, no FFT in sim path, citations,
   parametric NPZ purity); no binaries tracked; notebook outputs cleared; work orders /
   ORCHESTRATION consistent with reality; `pip install aodl` metadata sane
   (name/version/deps); tests deterministic (run the suite twice, compare).
6. **Performance regression**: the standard numbers (10×10 story simulate/frame at
   order 1 and 3; 1 ms hold) vs the Wave M report — flag > 1.5× regressions.

## Output

Same format as prior verification reports; small fixes (≤ ~15 lines, no interface
changes) committable as `M5 verification fixes: <list>` with the dispatch footer;
independent-vs-code table; performance table; explicit verdict line.
