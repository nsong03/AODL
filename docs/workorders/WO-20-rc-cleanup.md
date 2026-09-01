# WO-20 — Release-candidate cleanup (post-audit)

**Role:** implementation agent, Wave Q (solo). **You are the wave closer**: commit and
push. **Read first:** `CLAUDE.md`, `docs/workorders/WO-19-verify-m5.md` findings F-2,
F-6, F-9 (the audit that motivates every item here), `src/aodl/waveform/shepard.py`
(§switch_ramp), `docs/guide.md` §5.3/§6.7, then this. HEAD carries 329 green tests.

## Owned files

```
src/aodl/waveform/shepard.py       (edit: §1)
src/aodl/api.py                    (edit: §1 notes wording only)
tests/test_synthesis_options.py    (edit: §1 expectations)
docs/guide.md                      (edit: §1 dip-law wording + §2 clause)
.github/workflows/ci.yml           (new: §3)
README.md                          (edit: §3 wording only if needed)
```

## 1. `switch_ramp` scope — architect ruling on WO-19 F-2

Narrow the ramp to **`p == 0` rungs only** (the rectangles), per WO-17 §2.4's original
wording. The audit's numbers make this unambiguous: ramping only the rectangles gives
**identical B-channel splatter mitigation** (−41.4 → −103.4 dB at r = 3 µs, with or
without A ramps) while the interior-column dip drops from 4.19 % to **0.00 %** — the
cos^p A windows are already smooth and never needed ramping. Change the one condition
(`shepard.py:1199` area), keep `SwitchRamped` itself unchanged. Then:

- Update the docstrings that already state the p = 0 scope (they become true) and any
  place describing the interior-dip cost: the (πρ_r)² dip law now applies to the
  **edge/extended columns only**; interior columns are exactly flat for any ramp. Update
  `PlanReport.notes` wording in `api.py` accordingly ("a few µs of switch_ramp removes
  the splatter at no interior cost").
- Tests: adjust `test_synthesis_options.py` expectations — A-channel envelopes stay
  un-ramped (`FadeZoneEnvelope`), B rectangles ramped; add the decisive assertion:
  interior-column power with `switch_ramp = 3 µs` equals `switch_ramp = 0` to 1e-9 over
  an x hand-over, while the extended column still rises continuously over the ramp; and
  a B-channel-only splatter proxy (envelope discontinuity count = 0 when ramped).
- Do not change the default (0) or serialization behavior.

## 2. guide.md §6.7 clause — WO-19 F-9

Add one short paragraph: during a hand-over a fading frequency group can contain **two
real array traps two pitches apart** (indices (a, b) and (a−1, b+1) share a+b), so the
group-level `x`/`y` are power-weighted means, not trap positions — measure per *term*
(as the integration tests do) when tracking individual traps mid-fade. Cross-reference
from §5.3's existing hint.

## 3. CI workflow — WO-19 F-6

`.github/workflows/ci.yml`: on push + pull_request to any branch; ubuntu-latest,
Python 3.11; `pip install -e ".[dev]"`; steps: `ruff check src tests`,
`ruff format --check src tests`, `python -m mypy src/aodl`, `pytest`, and
`pytest --nbmake examples/` (six notebooks ≈ 5½ min — acceptable; give the job a 30 min
timeout). Movies render headlessly already (imageio-ffmpeg wheel). No caching games —
keep it a dozen lines and legible. This makes README's "executed in CI" literally true;
adjust the README sentence only if you must.

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `python -m mypy src/aodl` exit 0;
`pytest --nbmake examples/` green locally (CI itself will prove out on push); commit
(`RC cleanup: switch_ramp scoped to p=0 rungs; guide clause; CI workflow`, footer per
dispatch) and push. Report: the interior-flatness before/after numbers, envelope-type
table per channel at r = 3 µs, pytest/nbmake summaries verbatim, the workflow file
contents, deviations (or "none").
