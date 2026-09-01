# WO-09 — M2 verification pass (fresh eyes)

**Role:** verification agent, Wave G. Same charter as WO-06 (read it for the ground
rules): you did not write this code; try to break it; small fixes (≤ ~15 lines, no
interface changes) may be committed and pushed as `M2 verification fixes: <list>`; larger
defects are reported. End with **M2 ACCEPTED** or **M2 REJECTED (blockers: …)**.
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.2 + §3 (M2), `docs/conventions.md`,
`docs/workorders/WO-07-mixing.md` §2 (the line table is the spec — but see check 2:
verify it against physics, not just against itself), WO-08.

## Checks

1. **Suite + hygiene**: clean `pip install -e ".[dev]"`; `pytest`; `ruff check`;
   `mypy src/aodl`; `pytest --nbmake examples/` (both notebooks). Any red = finding.
2. **Mixing amplitudes against first principles.** Independently (scratch script, not
   committed): build V(u) for 2–4 tones at a frozen time, form exp(iC·V(u)) literally on
   a fine grid, and extract each +1-band line's complex amplitude by projection. Compare
   against `device/mixing.py` for: single tone (must match i·J₁(m) to O(m⁵)), two tones
   (compression cross-term −(i/8)m·m′²), three equi-spaced tones (the −(3i/16)m³
   collision at ladder edge; degenerate lines landing ON fundamentals must be merged
   coherently). If the code and your projection disagree, the projection wins — that is
   a finding against the work-order table itself; quantify and report.
3. **Table I on two channels**: equal chirps on Ax+Ay → Z̄ = lens_scale·ḟ, ΔF ≈ 0
   (recompute expected values yourself); single-axis chirp → ΔF = lens_scale·ḟ. Verify
   both through rendered-field waist scans (not only through `measure`).
4. **Array geometry**: 5×5, Δf = 1 MHz → 10.3 µm pitch, positions to 1% waist; per-trap
   power spread; Schroeder vs random phase IM3 comparison reproduces (fixed seed).
5. **Grouping rule**: 1 kHz default + diameter cap behave per WO-08 §2 (ladder does not
   chain; exact degeneracies merge); check a randomized fuzz (1000 random df sets:
   every group diameter ≤ tol, union preserved, deterministic across repeats).
6. **Robustness**: mixing_order=1 vs 3 continuity (order 3 with m→0 tends to order 1);
   empty channel; M=1 array; ghost pruned-power diagnostic consistent; notebook 02
   movie file plays (imageio read-back) and its tracked plane keeps the array sharp
   during the Z excursion.
7. **Performance**: time `simulate` + one 512² frame, 5×5 array, mixing_order=3; compare
   against WO-08's 2 s guard; report the number.

## Output

Same format as WO-06: numbered findings with severity and repro; fixes committed (if
any) with hash; your independent numbers vs the code's; explicit final verdict line.
