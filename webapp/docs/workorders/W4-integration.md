# W4 — integration, verification, CI

**Role:** verification agent (fresh eyes). **Authority:** this work order, then `../DESIGN.md`
§5, `../API.md`, and the three implementation reports in `../reports/`. You may edit **any**
file under `webapp/` and the W1 additions under `src/aodl` / `tests/` to fix defects, but
every change must be listed in your report with the reason. No git.

## 0. Context
The pre-implementation review is in `../reports/REVIEW.md`; its accepted corrections are in
DESIGN/API/W1–W3. Read the three implementation reports in `../reports/` first, then audit
the code against the documents — the reports are claims, not evidence.

## 1. Run everything
`make install` (root editable install + backend + `npm ci`), `make lint`, `make test`
(with `AODL_STUDIO_STRICT_ADVANCED=1`, which the Makefile sets), `make build`, root `pytest`,
`ruff`, `mypy` (the repo CI). Fix what fails; prefer the smallest correct fix; if a fix needs a
contract change, change API.md and both sides.

## 2. End-to-end checks (Playwright is pre-installed: `require('playwright')` works; if the
project pins its own, launch with `executablePath: '/opt/pw-browsers/chromium'`, `--headless`,
`--no-sandbox`). Write `webapp/scripts/e2e.mjs` that starts `uvicorn aodl_webapp.main:create_app
--factory` on :8000 and `vite preview` (built frontend) or the dev server, then:
- loads `/`, `/basic?example=pick_and_place`, `/basic?example=array_lift_hold`,
  `/basic?example=array_arbitrary_path`, `/advanced?example=pick_and_place`; asserts no console
  errors and no failed requests; captures one screenshot per page to
  `webapp/docs/screenshots/*.png` (landing; basic for `single_lift_hold` mid-fade with the
  shadow badge or the shadow spots visible; advanced efficiency);
- DESIGN §5 numeric checks: for `pick_and_place` at `t = 70 µs + τ/2` (mid-translate), the XY
  approximation's brightest pixel vs the exact frame's brightest pixel agree within 0.05 w0
  (read both from the page via `page.evaluate` on the exposed debug hooks — add
  `window.__aodl` hooks if missing); the aperture panel's envelope and `f_inst` equal
  `/api/aperture` to 1e-6; for `single_lift_hold` at the first x hand-over (`fade_events[0]
  .time_us + τ/2`) the simulate response holds a main spot of ≈ unit `power/power_ref` and two
  shadows at `±10.3·Δf µm` of ≈ 0.5 each.
- Playback runs at ≥ 30 fps on the 4×4 fading example (measure with `requestAnimationFrame`
  timestamps over 2 s; report the number).

## 3. Physics audit (read the code, do not trust the reports)
Check, and write the result into `webapp/docs/VERIFICATION.md` with the numbers:
1. sign of the `[t − τ, t]` aperture highlight and the `t − τ/2` marker against
   `docs/conventions.md` §7;
2. the Gaussian-approximation XY frame vs the closed-form frame for a static single tweezer at
   `t = 3τ` (should be visually identical; report the max relative pixel difference);
3. `Path` spline: a 24-point circle of radius 10 µm in XY on a 1×1 array planned at 300 µs —
   the simulated spot at `t + τ/2` follows the requested circle within 0.1 µm away from the
   two ends; the band check reports the right lateral span;
4. the advanced engine's `trajectory_efficiency` is 1.0·η_peak⁴ for a static tweezer and dips
   during the 40 µm traverse of `flagship_10x10` (report the min);
5. the `code_python` of every example runs under plain `aodl` (W1 tests claim it — rerun one
   by hand in a subprocess with `python -I`).

## 4. CI
Add `.github/workflows/webapp.yml`: Python 3.11 + Node 22; `pip install -e . -e webapp/backend[dev]`;
`ruff`/`mypy` on both backend packages; `AODL_STUDIO_STRICT_ADVANCED=1 pytest webapp/backend/tests`;
`npm ci && npm run lint && npm run typecheck && npm test -- --run && npm run build` in
`webapp/frontend`. Do not edit `ci.yml` (the root suite already covers the `Path` tests).

## 5. README
Write `webapp/README.md`: what it is (two paragraphs), screenshots, quick start (`make install`,
`make dev`, URLs), architecture summary with a link to DESIGN.md/API.md, the physics caveats
(DESIGN §1.1 verbatim in spirit), how to run the tests, and a "what the advanced engine
models / does not model" table.

## 6. Report → `webapp/docs/reports/W4.md`: every defect found (file, symptom, fix), the
measured numbers of §2–§3, open issues you did not fix (with your recommendation), and the
final green/red status of each check.
