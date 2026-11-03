# R2-backend — tutorial examples and the intensity cuts

**Read first:** `R2-DESIGN-ADDENDUM.md` §F3–F4, `webapp/docs/API.md` (§2 examples, §5.2 frame),
`webapp/backend/src/aodl_webapp/{examples,simulation,schemas}.py`, `routes/frame.py`,
`webapp/backend/tests/test_api_simulate.py`, `test_api_plan.py`. No git. Concurrent agents own
the other files (addendum ownership table); R2-core is changing `src/aodl`'s Shepard timing at
the same time — do not depend on exact fading numbers in your tests.

## 1. Examples (`examples.py`, API.md §2)
Add two ids (append to the list; keep the existing seven and their order):
- `tutorial_hold_plain`: 1×1, `Lift(+10 µm, 60 µs)`, `Hold(600 µs)`, `Lift(−10 µm, 60 µs)`,
  options `shepard = "never"`, `check_band = false`, tags `["tutorial", "no-shepard"]`, blurb
  saying the drive leaves the band after Eq. 1's 206 µs and that the closed-form engine does
  not know it (flat efficiency).
- `tutorial_hold_shepard`: the same trajectory with default options (auto → fading ladders),
  tags `["tutorial", "shepard"]`.
Both must plan: `tutorial_hold_plain` returns `mode == "s19"` with the `check_band=False` note
and `band_usage` margins negative on all four channels; the Shepard one `mode == "shepard"`.
Test both in `test_api_plan.py` (extend the "every example plans" test and assert the modes
and the negative margin). `PresetsResponse.examples` grows to nine; update API.md §2's id list.

## 2. `/api/frame` cuts (`simulation.py`, `schemas.py`, API.md §5.2)
Add `i_y` to `Cuts`: the closed-form intensity along `y` through the brightest spot's `x`
column at the frame plane (`intensity_frame` column nearest `x`), alongside the existing `i_x`
and `i_z`. Add `y_um` for it if the grid's `y` axis is not already in the response. Add
`gauss_peak` (the astigmatic-Gaussian peak `2P/(π w_x w_y)` summed over spots coincident with
the brightest one, same units as `peak`) and `exact_peak` (max of the XY image) to the
`FrameResponse` so the UI can show their ratio. Document in API.md with units. Tests: `i_y`
peaks at the brightest spot's `y`; `exact_peak / gauss_peak` within 2 % for a static single
tweezer at `t = 3τ`; the cuts' lengths match their axes.

## 3. Verification
```
cd /home/user/AODL/webapp/backend && AODL_STUDIO_STRICT_ADVANCED=1 python -m pytest tests -q
cd /home/user/AODL && ruff check webapp/backend && ruff format --check webapp/backend && python -m mypy webapp/backend/src/aodl_webapp
```
Report → `webapp/docs/reports/R2-backend.md` (what changed, tests, API.md diff summary);
final message ≤ 25 lines.
