# R2-core — even-M fading-Shepard arrays must keep every requested trap lit

**Read first:** `R2-DESIGN-ADDENDUM.md` §F2 (the measured defect, the mechanism, the fix), then
`src/aodl/waveform/shepard.py` in full (module docstring, `fade_window`, `FadeZoneEnvelope`,
`SwitchRamped` and its `Note`, `ChannelFade`, `table_ii`, `lattice_comb_offset`,
`ShepardConfig.comb_offsets`, `shepard_band_bound`, `active_indices`, `shepard_ladder`),
`src/aodl/waveform/synthesis.py` (`_synthesize_shepard`, `_check_shepard_bands`, `auto_config`),
`src/aodl/api.py` (`_notes`, `fade_schedule`), `docs/guide.md` §3.6, §6.4–6.8, §7 FAQ,
`docs/PLAN.md` §3 M4, `docs/conventions.md` §4–5, `tests/test_shepard.py`,
`tests/test_synthesis_options.py` (especially the even-M lattice tests around lines 130–260 and
the parity-parametrised `switch_ramp` test near line 679), `tests/test_integration_m4.py`
(`test_the_array_grid_grows_two_columns…`), `tests/test_docs.py`. Repo conventions:
`/home/user/AODL/CLAUDE.md` (SI, equation citations, no signs outside `conventions.py`). No git.

## 1. The change

1. `table_ii(array)`: for an array axis (`count > 1`) with **even** `count`, the B channel's
   offset is `xi + 0.5`; the A channel keeps `xi`. Odd counts and the single-tweezer row are
   unchanged. Rewrite the docstring with the phase argument of the addendum (why
   `(M + η)/2` vs `(1 + η)/2` decides it, with Eqs. S25–S27 cited), and state the invariant the
   tests pin: *both live A rungs of a hand-over see the same `M + 1` B rungs*.
2. `lattice_comb_offset(fade, delta_f)` returns `0.0` for every `fade` — the half pitch of an
   even-`M` lattice now comes from `ξ_B` — with a docstring saying so and why the frequency-only
   offset of WO-17 was not enough (it moved the lattice without moving the window timing).
   Keep the function and `ShepardConfig.comb_offset` (explicit mappings are still honoured; the
   `"auto"` path now yields zeros) so saved configurations and the API keep working; update
   `comb_offsets`' docstring, the module docstring paragraph "Where the traps land, and the comb
   offset", `shepard_band_bound`'s docstring (the comb term is now normally 0) and
   `_shepard_band_message`'s comb clause (it is emitted only when a non-zero explicit offset is
   given, so it needs no change in logic — check).
3. `src/aodl/api.py::_notes`: replace the parity sentences (`"{axis}: {count} + 2 … during a
   hand-over" if odd else "{count} + 1 … at every instant"`, and the "even-M ladders carry a
   delta_f/2 comb offset…" note) with one truth for both parities: `M + 2` columns during a
   hand-over, `M` otherwise, and for even `M` a one-line note that the B ladder is offset by
   half a rung so the array lands on Eq. S19's lattice and every requested site stays lit.
   The `switch_ramp` note's "odd M edge columns pay / even M pays nothing" text must follow
   whatever the re-measured law is (item 5).
4. `SwitchRamped` docstring `Note` and `tests/test_synthesis_options.py`'s parity-parametrised
   test: with the fix an even-`M` rung switches at an A hand-over like an odd one, so the
   even-`M` case now *pays* the same `(π ρ_r)²·SWITCH_FLATNESS_SHAPE` edge-column dip. Re-measure
   both parities, update the expected values and the prose (`docs/guide.md` §6.6 quotes 1.54 %
   and 20 % — re-measure and keep whatever the code gives, stating the parity).
5. Docs: `docs/guide.md` §6.7 ("`M + 2` columns during a hand-over for odd M, `M + 1` at every
   instant for even M" → one rule), FAQ "Why does my 10-column array show 11 columns?",
   `docs/PLAN.md` M4 bullet and its bracketed correction, `docs/guide.md` §8 if any quoted number
   moves; `docs/conventions.md` needs no change. Keep `tests/test_docs.py` green (it checks quoted
   numbers against code).

## 2. The load-bearing test (new, in `tests/test_shepard.py` or a new `tests/test_shepard_even_m.py`)

`test_even_m_array_sites_stay_lit_through_hand_overs`, parametrised over `(mx, my)` ∈
`{(2, 2), (4, 4), (1, 4), (4, 1)}` at `Δf = (1.0, 1.3) MHz` plus `(4, 4)` at `(1.0, 1.0)`, with
`Lift(8 µm, 80 µs), Hold(500 µs), Lift(−8 µm, 80 µs)` on `default_1030()` with
`mixing_order = 1` (`dataclasses.replace` on the channels), `plan_motion(spec, P)` (auto →
shepard): at ≥ 40 frames over `[2τ, 580 µs + τ/2]`, build the terms (`aodl.build_terms`), bin
every term's centre (`theta1·F/k`, `aodl.field.focal.spot_params`) onto the Eq. S19 lattice
`(n − (M−1)/2)·pitch` within `0.25·pitch`, sum `|c|²`-weighted term power per site using
`aodl.field.measure._term_power` (or `measure` on per-site subsets), and assert, relative to
the first frame's median site power: **every requested site ≥ 0.80 at every frame** (measured
0.857–0.935 with the fix; the residual is the ρ ripple), the brightest off-lattice spot ≤ 1.05,
and total power conserved to 5 % away from hand-overs. Add the **regression half**: the same
drive synthesised with the old timing (`ShepardConfig(..., config={B channels: ChannelFade(m, 0.0,
xi_A)}, comb_offset={"Bx": Δf_x/2, "By": Δf_y/2})`) drops a requested site below `0.05` —
so the test documents the defect it closes. Keep the test under ~20 s (use 40 frames, 2×2 and
4×4 only if time bites; measure).

Also keep/extend: the even-`M` lattice-coincidence test (traps on Eq. S19's half-integer sites,
now via `ξ_B`), `fade_schedule` still reports one event per hand-over per axis,
`test_integration_m4.py` stays green (add the 4×4 variant of the "grid grows two columns"
test if cheap), `shepard_band_bound` tests (comb term 0).

## 3. Verification you must run (all must be green)

```
cd /home/user/AODL && python -m pytest -q                      # root suite (≈2 min)
ruff check src tests && ruff format --check src tests
python -m mypy src/aodl                                        # only the 10 pre-existing numpy-2.5 overload errors may remain
cd webapp && make fixtures                                     # the IR of array_lift_hold changes → regenerate
make -C /home/user/AODL/webapp test-backend                    # 115 tests; fix any that pinned the old parity note
make -C /home/user/AODL/webapp test-frontend                   # 67 vitest tests read the regenerated fixtures
```
Then rerun the addendum's measurement through the web API (`POST /api/simulate` on the
`array_lift_hold` example, sum `power` per lattice site) and put the numbers in your report.

## 4. Report → `webapp/docs/reports/R2-core.md`
What changed and why (with the phase argument in two paragraphs), the before/after table,
every test whose expectation changed and the physical reason, timings, and anything in the
docs you were unsure how to reword. Your final message: ≤ 40 lines with the exact commands and
results.
