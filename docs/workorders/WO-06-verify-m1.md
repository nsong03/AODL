# WO-06 — M0/M1 verification pass (fresh eyes)

**Role:** verification agent, Wave D. You did not write this code. Your job is to try to
break it, not to trust its tests. **You may commit small fixes and push**; larger defects
are reported, not fixed. **Read first:** `CLAUDE.md`, `docs/PLAN.md` §1+§3,
`docs/ARCHITECTURE.md`, `docs/conventions.md`, then this.

## Checks

1. **Suite + hygiene**: `pip install -e ".[dev]"` from clean; `pytest`; `ruff check`;
   `pytest --nbmake examples/`. Any red = finding.
2. **Sign audit against the paper** (the highest-risk area). Ground truth (arXiv:2510.11451):
   - Eq. S7 channel geometry: Ax sound −x, Bx +x, Ay −y, By +y; +1 order everywhere.
   - Table I: X = (λF/v)(f_Bx − f_Ax); Y = (λF/v)(f_By − f_Ay);
     Z̄ = ½(λF²/v²)(ḟ_Ax+ḟ_Bx+ḟ_Ay+ḟ_By); ΔF = (λF²/v²)(ḟ_Ax+ḟ_Bx−ḟ_Ay−ḟ_By).
   - Single-AOD (Ay, chirp β): dioptric power P = λβ/v²; ΔF_y = (λF²/v²)β; up-chirp on
     all channels moves the tweezer "above the static focal plane" (+lab Z).
   Verify the code + `docs/conventions.md` reproduce ALL of these, including the
   `Z_LAB_SIGN` relation between the S11 defocus parameter and lab Z. Write two
   *independent* spot checks in a scratch script (not committed): (a) static f on Bx →
   X > 0; (b) up-chirp on Ay only → measured z_lab of the y-focus = +lens_scale·β and
   x-focus unmoved.
3. **Physics spot-checks with independent numbers** (compute expected values yourself,
   don't reuse the suite's): waist0 at 1030 nm/F=6.5 mm/w_in=2 mm ≈ 1.066 µm;
   deflection_scale ≈ 10.3 µm/MHz; lens_scale·(50 MHz/ms) ≈ 5.15 µm. Simulation must
   agree within stated test tolerances.
4. **Robustness probes**: t before drive start; t far beyond waveform end (clamp-hold);
   zero-amplitude envelope; a term whose patch falls entirely off-grid; grid of 1×1 px;
   chirp of 0. No crashes, sane values.
5. **Movie sanity**: render 20 frames of the M1 sweep; confirm the file plays (probe with
   imageio read-back), tracked plane keeps the spot sharp, hue changes as z_lab grows,
   XZ panel shows the focal excursion.
6. **Convention drift**: grep that no file outside `device/conventions.py` hardcodes
   sound-direction/order signs; equation citations present on physics functions;
   parametric NPZ contains no sample arrays.

## Output

- Small fixes (≤ ~15 lines each, no interface changes): apply, note in report, include in
  one commit `M1 verification fixes: <list>` (plus the footer lines given in your dispatch
  instructions) and push.
- Larger findings: numbered report with severity (blocker / should-fix / nit), exact
  file:line, a minimal reproduction, and your suggested direction. Do not implement.
- End your report with an explicit verdict: **M1 ACCEPTED** or **M1 REJECTED (blockers: …)**.
