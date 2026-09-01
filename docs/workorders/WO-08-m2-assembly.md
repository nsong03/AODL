# WO-08 — M2 assembly: arrays + Schroeder phases, crossed-pair notebook, cleanups

**Role:** implementation agent, Wave F. **You are the wave closer**: commit and push.
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §3 (M2), `docs/conventions.md`,
`docs/workorders/WO-07-mixing.md` (merged before you start), notebook
`examples/01_single_aod_sweep.ipynb` (house style), then this.

## Owned files

```
src/aodl/waveform/synthesis.py          (new: array helpers + Schroeder phases only —
                                         the full Eq. S19 trajectory solver is M3)
src/aodl/field/focal.py                 (edit: grouping rule, §3)
src/aodl/field/measure.py               (edit: hoist, §4.3)
src/aodl/field/gaussian.py  src/aodl/device/conventions.py  src/aodl/device/aod.py
                                        (edit: docstring citations only, §4.4)
src/aodl/engine.py                      (edit only if §2 needs a small hook)
src/aodl/params.py                      (edit: one line, §0b)
src/aodl/waveform/serialize.py          (edit: carry mixing_order, §0b)
tests/test_device_single_aod.py  tests/test_integration_m1.py  tests/test_serialize.py
                                        (edit ONLY per §0b)
pyproject.toml                          (edit: [tool.mypy] block, §4.1)
src/aodl/device/aodl.py + src/aodl/field/focal.py (edit: TermLike/mypy, §4.1)
src/aodl/viz/movie.py                   (edit: mypy narrowings only)
src/aodl/__init__.py                    (edit: export new synthesis helpers)
examples/02_crossed_pair_diagonal.ipynb
tests/test_synthesis.py  tests/test_grouping.py  tests/test_integration_m2.py
```

## 0b. Architect ruling — default `mixing_order` flips 1 → 3

WO-07 shipped `AODParams.mixing_order = 1` because the physical default (3) shifts
amplitudes by the m²/8 compression and breaks M1 tests pinned to exact weak-drive values
(`tests/test_device_single_aod.py` ~196, ~513 and `test_cartesian_product_over_tones`;
`tests/test_integration_m1.py` ~313 — locate by behavior, line numbers may have drifted).
Product realism wins: set the default to **3**, and update those M1 tests to construct
their params with an explicit `mixing_order=1` (they test the linear physics — that's
what they should say). Do not weaken any tolerance. Also: `waveform/serialize.py` must
round-trip `mixing_order` (additive JSON key in `meta`, read with a default of 1 for
older files so existing NPZ archives load unchanged; extend `test_serialize.py`
accordingly). Schema version stays 1.

## 1. `waveform/synthesis.py`

```python
def schroeder_phases(M: int) -> np.ndarray
    # Eq. S23/S28 generalized form: φ_n = mod(2π·n(n−1)/(2M), 2π), n = 0..M−1
def array_tones(M: int, delta_f: float, center: float = 0.0, amp: float = 1.0,
                phases: {"schroeder","zero","random"} | array = "schroeder",
                t0=0.0, t1=..., rng=None) -> ChannelWaveform
    # M constant-frequency tones at center + (n − (M−1)/2)·delta_f  (Eq. S18/S19 ladder),
    # equal amplitudes, chosen phases
def add_common_ramp(cw: ChannelWaveform, ramp: PiecewisePoly) -> ChannelWaveform
    # adds the same frequency ramp to every tone (PiecewisePoly '+'); used for diagonal
    # moves and later for Eq. S19 f_Z terms
```

## 2. Grouping ruling (verifier finding 3 — architect decision)

In `field/focal.py`: default `GROUP_TOL` drops **10 kHz → 1 kHz**, and after
neighbour-chaining, enforce a **cluster diameter cap ≤ tol** by splitting each oversized
cluster at its largest internal gaps (deterministic, largest gaps first) until all
diameters ≤ tol. Rationale (record in the docstring): physical cases are exact degeneracy
(IM3 landing on a fundamental, 0 Hz ⇒ coherent) vs tone spacings ≥ 100 kHz (⇒ incoherent);
nothing legitimate lives in between, and single-linkage chaining must not glue a ladder.
`tests/test_grouping.py`: the 40-terms × 9 kHz ladder → 40 groups; exact-degenerate pairs
merge; a 3-cluster scene with one oversized cluster splits at the right gaps; existing
suite stays green (some M1 tests may pin the old default — update only defaults, keep the
`tol` kwarg working).

## 3. `examples/02_crossed_pair_diagonal.ipynb`

House style of notebook 01 (markdown physics → code → measured vs closed form). Params
`default_1030()`, channels Ax + Ay (the "conventional 2D-AOD"). Sections:

1. **Static array**: 5×5 via `array_tones(5, 1 MHz)` on both channels; frame plot;
   measured spacing = deflection_scale·Δf (10.3 µm); note the corner-vs-center intensity
   pattern from the α product.
2. **Diagonal transport of a single tweezer**: equal min-jerk chirps on Ax and Ay
   (0→4 MHz over 120 µs): measured Z̄(t) = lens_scale·ḟ(t_c) (two-channel sum/2 ×2),
   ΔF ≈ 0 throughout (spherical lensing — PLAN M2 acceptance), spot stays round while
   defocusing; contrast single-axis chirp (cylindrical, σ_astig ≠ 0) in the same figure.
3. **IM3 ghosts & Schroeder**: 8-tone ladder on Ax (Δf = 1 MHz, m = 0.3): log-scale frame
   showing ghost spots at ladder edges; per-trap intensity spread (std/mean) for
   Schroeder vs zero vs random phases (fixed seed), Schroeder wins; table of the top
   ghost intensities vs the −(i/8)m³-class prediction.
4. **Movie**: 3×3 array diagonal transport with a mid-move Z excursion visible as hue
   (mode="tracked" — from two chirped axes the tracked plane now follows a *spherical*
   defocus so the array stays sharp; say so in markdown, referencing notebook 01's
   fixed-mode rationale). ≤ 90 s render, written to `examples/outputs/`.
5. Closing: what M3 adds (counter-propagating pairs → astigmatism-free in-plane motion).

## 4. Cleanups (verifier findings, architect-approved)

1. **mypy (finding 2)**: `TermLike` protocol members become read-only properties
   (sanctioned interface clarification; `TermArray` unchanged); add `[tool.mypy]` to
   pyproject: `python_version`, `ignore_missing_imports = true` scoped via
   `[[tool.mypy.overrides]]` for `scipy.*`, `imageio.*`, `matplotlib.*`; fix the
   `movie.py` narrowings. Gate: `mypy src/aodl` exit 0, and add that fact to your report
   (do NOT wire mypy into pytest).
2. *(covered by §2)*
3. **measure hoist (finding 6)**: compute `spot_params` once above the group loop.
4. **citations (finding 7)**: add the missing equation/derivation citations listed in the
   verifier report (gaussian edge moments, `intensity_slice_xz`, conventions timing
   helpers, `aperture_window`).

## 5. `tests/test_integration_m2.py` (M2 acceptance, end-to-end through `simulate`)

- Diagonal equal-chirp: |ΔF| < 0.02·z_R while Z̄ tracks lens_scale·ḟ(t_c) (2%);
  single-axis control shows |ΔF| = lens_scale·ḟ (2%).
- 5×5 array: 25 groups (fundamental products), positions on the Table-I grid (1% waist),
  per-trap power spread < 1% with order-1 mixing.
- With mixing_order=3, m = 0.3: ghost group appears at the predicted ladder-edge
  position with intensity within a factor 2 of the analytic estimate, and total pruned
  power < 1e-6 of total.
- Runtime guard: `simulate` + one 512² frame for the 5×5 array with mixing_order=3
  under 2 s on this machine (keeps notebooks viable).

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `mypy src/aodl` exit 0;
`pytest --nbmake examples/` green (both notebooks); nothing binary staged; commit
(`M2: arrays, Schroeder phases, crossed-pair notebook, grouping + typing cleanups`,
footer per dispatch) and push. Report: reconciliations, ghost-intensity comparison table
from the notebook, runtimes, deviations.
