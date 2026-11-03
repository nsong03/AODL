# R2-core — even-M fading-Shepard arrays keep every requested trap lit

Work order: `webapp/docs/workorders/round2/R2-core.md`; defect and fix: `R2-DESIGN-ADDENDUM.md` §F2.
No git commands were run.

## 1. What changed and why

`table_ii(array)` now gives an **even**-count array axis `xi_B = xi_A + 1/2` (Bx: 0.5, By: 1.0).
`xi_A`, odd counts and the single-tweezer row are unchanged. `lattice_comb_offset` now returns
`0.0` for every input. It and `ShepardConfig.comb_offset` are kept: the `"auto"` path gives zeros,
and an explicit mapping is still used exactly as given.

**The phase argument.** Rung `n` of either ladder has fade coordinate `g_n = f_Z + (n + xi) Δf`
(Eq. S25). Take a rectangle of half-width `(m + η)Δf/2`. It holds `m + 1` rungs while
`frac(g/Δf + (m + η)/2) < η` and `m` rungs the rest of the time. With `m = 1` this rectangle is
the A window (Eq. S26), and the `m + 1` phase is the A hand-over. With `m = M` it is the B
rectangle (Eq. S27, `p_B = 0`). If A and B share one `xi`, the two phases are set by
`(1 + η)/2` and `(M + η)/2`:
- For odd M these have the same fractional part. B therefore holds `M + 1` rungs exactly while A
  hands over, and every requested site is fed by both live A rungs (`cos²θ + sin²θ = 1`).
- For even M they differ by exactly ½. B then holds only `M` rungs during the hand-over. The sites
  fed by the dying A rung are `{b − a}` and those fed by the rising rung are `{b − a} + 1`, so the
  lit block walks one pitch per fade. One end column of the requested array falls to
  `cos²θ → 0` and only comes back when B's next rung switches on a quarter cycle later.

**The fix.** `xi_B = xi_A + ½` brings back the odd-M phase relation, giving the invariant the tests
pin: both live A rungs of a hand-over see the same `M + 1` B rungs. Because the same half rung
sits inside `g`, it also puts the traps at `(b + xi_B) − (a + xi_A) = b − a + ½`. That is Eq.
S19's half-integer lattice, so the WO-17 frequency-only `Δf/2` comb has nothing left to do. That
comb had put the lattice in the right place but left the window timing where it was, and the
timing is exactly what decides which sites stay lit. A bonus: the half pitch no longer costs band.
The flagship's worst band margin improves from 1.31 to 1.96 MHz on By.

Docstrings rewritten: module paragraph "Where the traps land, and the comb offset", `table_ii`,
`lattice_comb_offset`, `ShepardConfig.comb_offset`/`comb_offsets`, `shepard_band_bound`,
`shepard_ladder`, the `SwitchRamped` Note and `SWITCH_FLATNESS_SHAPE`.

`api._notes` now gives one rule for both parities: `"x: M + 2 columns during a hand-over, M
otherwise"`. For even M it adds one line saying the B ladder is offset by half a rung, so the array
lands on Eq. S19's lattice and every requested site stays lit. The `switch_ramp` note now says the
edge columns of every ladder axis pay the dip.

I checked `_shepard_band_message` (in `synthesis.py`, which I do not own). Its comb clause is
logically fine: it is only emitted for a non-zero explicit offset. Its wording, "the comb offset
that puts an even-M ladder on Eq. S19's lattice", is now stale. I left it alone and flag it below.

## 2. Before / after (per requested lattice site, relative to the first frame's median site)

Measured through `POST /api/simulate` on `array_lift_hold` (Lift 8 µm/80 µs, Hold 500 µs,
Lift −8 µm/80 µs). Method: 120 frames over [2τ, 580 µs + τ/2], `power` summed within 0.25 pitch of
each Eq. S19 site. Script: `/tmp/claude-0/-home-user/ab020c98-5f5f-59f7-99aa-348034e0ade7/scratchpad/measure_api.py`.

| array (Δf MHz) | before, order 1 | after, order 1 | after, order 3 (API default) |
|---|---|---|---|
| 4×4 (1.0/1.3) | **0.0000** | 0.949 | 0.855 |
| 2×2 (1.0/1.3) | **0.0000** | 0.949 | 0.926 |
| 1×4 (1.0/1.3) | **0.0000** | 0.979 | 0.916 |
| 4×1 (1.0/1.3) | **0.0000** | 0.965 | 0.907 |
| 3×3 (1.0/1.3) | 0.949 | 0.949 (unchanged) | 0.915 |
| 4×4 (1.0/1.0) | **0.0000** | 0.965 | 0.935 |

Measured with the new test's metric (Parseval `_term_power` per site, mixing order 1, 40
frames):
- With the fix: worst site 0.966–0.980; brightest other spot ≤ 0.977.
- With the WO-17 timing: 0.0001–0.0054.

**A correction to the addendum.** Its "with the fix" column (0.857/0.928/0.910/0.902/0.917/0.935)
matches my **mixing-order-3** numbers, not order 1. The 7–14 % dip it attributes to the ρ ripple
is therefore mostly intermodulation (Eqs. S20–S22). The ρ ripple alone, at order 1, costs 2–5 %.

## 3. Tests whose expectation changed, and the physical reason

- **New:** `tests/test_shepard_even_m.py::test_even_m_array_sites_stay_lit_through_hand_overs`.
  Five cases (2×2, 4×4, 1×4, 4×1 at 1.0/1.3 MHz, and 4×4 at 1.0/1.0). It asserts:
  - every requested site is ≥ 0.80 at all 40 frames, and nothing else is above 1.05;
  - total power is conserved to 5 %. Each hand-over adds exactly one column or row of light, and
    nothing else does. This is checked on frames that are steady over the transit window and where
    no free axis is fading. Measured: within 0.1 %.
  - The regression half uses the explicit WO-17 `ShepardConfig` override and must drop a site
    below 0.05.

  Runtime: 10 s.

  On "away from hand-overs": with η = ½ and interlaced axes, some axis is almost always handing
  over, and at 1.0/1.0 MHz always. I therefore read the requirement as "conserved, counting the
  switched extra column".
- `test_shepard.py::test_table_ii_rows_and_the_s28_ladder_phases`: the even-M B row now has
  `xi_A + ½`. A 3×4 case was added (By xi = 1.0).
- `test_synthesis_options.py`:
  - The lattice-coincidence test now gets the half pitch from `xi_B`. It also shows that the WO-17
    comb had the lattice right too, and that `xi_B = xi_A` without a comb is off by half a pitch.
  - The comb-offset test became `test_the_half_rung_xi_b_carries_the_lattice_and_the_comb_offset_is_zero`:
    the frequency laws are identical to WO-17's, `g` is larger by Δf/2, auto comb = 0, and an
    explicit mapping is still used as given.
  - The flatness test became a fixed-vs-WO-17 comparison. Every requested node is flat to 1e-9 in
    `|c|²`, the grid is {M, M+2} wide, and WO-17 goes below 0.05.
  - The band-bound test now pins "comb term is 0 unless set by hand".
  - The ramp parity test changed from `(4, 0)` to `(4, 1)`. With the fix an even-M rung switches at
    an A hand-over, so its edge column pays.
  - The ρ-law test is now parametrised over M = 3 and M = 4. Both parities measured
    **0.1721 % at 1 µs, 1.537 % at 3 µs, 20.17 % at τ**, so the quoted 1.54 % and 20 % stand.
- `test_integration_m4.py`:
  - The "grid grows two columns" test is parametrised 3×3/4×4 and anchored on the first Eq. S19
    site. A 4×4 now reads {4, 6} columns, with all 16 sites flat to 1e-12.
  - The 10×10 frame test is now anchored on the Eq. S19 lattice instead of the extended grid's
    midpoint. That midpoint was only on a lattice node when the grid was always M + 1 wide.
- `tests/test_api.py` (outside my listed files, but it pinned the old note text): it now expects
  the "10 + 2 … during a hand-over, 10 otherwise" note, the half-rung note, no comb note, and the
  "x and y ladders' two edge columns dip" ramp note.
- `tests/test_check_flagship.py` (outside my listed files; the root suite would not be green
  otherwise):
  - The strict-xfail rung fault XPASSed because list index 19 is now a different rung.
    `FAULTED_RUNG` is now 20, which is again the rung live over the first four check frames, the
    file's own selection rule.
  - The pins were re-measured: one-frame dip 0.28 → 0.24, clean time median 0.137 → 0.102 (the
    faulted drive reads 0.098, so it is no worse), columns hit {0:[0], 1:[1], 2:[4]}.
  - Brightest extended node 0.974 → 0.729. "Ninety-three" tones → ninety-four.
  - The F-2 hole is still open: the fault still passes the 0.30 gate.

## 4. Docs

- `docs/guide.md` §6.6: the parity sentence was replaced. The edge columns pay for both parities,
  measured identically on M = 3 and M = 4.
- §6.7: one rule, `M + 2` during a hand-over and `M` otherwise, with a short note on the old
  M + 1 behaviour.
- The FAQ entry became "…show 12 columns during a fade?".
- §3.3: the flagship is now 94 tones and 98 kB (97.66 kB measured; it was 93 tones, 95.8 kB).
- §5.5 and §8: the flagship spread is ~17 % (was ~20 %) and the C²-share is ~78 % (was ~82 %).
  The floor is 3.8 % at C = 0.003 (unchanged), the waist 8 % (0.083), the normalization 4.59
  (unchanged), and the median 0.102/0.098.
- §6.4: re-measured examples/06's "7.6 %" interior ripple. It is 7.60 % both before and after.
- `docs/PLAN.md`: the M4 bullet now gives one rule, with a round-2 bracketed correction, and the
  ξ_B note was added to §1.4. The M6 line now reads "~17 % IM3 spread (~20 % before the fix)".

**Unsure.**
1. The 82 % → 78 % change. When I rebuild the old drive (WO-17 override) and measure it the same
   way (worst uniformity at C = 0.30 vs 0.003), I get 21 %/74 %/5.5 %, not the documented
   20 %/82 %/3.8 %. The original method evidently differed. I quote only what I measured on the
   current drive.
2. Not re-measured: the guide's one-channel skew thresholds (0.030/0.149/0.58 w₀). The Ax channel
   is bit-identical and those thresholds are dominated by ḟ·δ, but the clean lateral baseline
   moved from 0.0105 to 0.0087 w₀.
3. Left as written: guide §5.5's "outermost node trades its light with the extended grid". The
   checker's edge-line exemption may now be stricter than the physics needs for both parities.

## 5. Left for other owners (not touched)

- `examples/05` (cell title "11 × 11 nodes"), `examples/06` (cell 8 "11 x 11 extended grid", cell
  15 "M+1 columns at every instant for even M (11 here)", "97 kB") and `examples/07` ("M+1 columns
  at every instant for even M") are now wrong. examples/06's own numbers re-measured as 7.60 %
  ripple and a 24.7 % scene swing, so "about 8 %" and "a quarter" are still right. Groups per frame
  are now {100, 120, 144}.
- `webapp/backend/src/aodl_webapp/examples.py`:
  - the `array_lift_hold` blurb ("the lit grid grows by a column") should say two columns during
    a hand-over;
  - the flagship blurb's "93 fading-Shepard rungs" is now 94, and "483 frequency groups" was not
    re-measured.
- `src/aodl/api.py:71`: comment "A 93-tone fading drive…".
- `src/aodl/waveform/synthesis.py`: the `_shepard_band_message` comb clause wording.
- `webapp/frontend/tests/fixtures/examples.json` grew from 9.2 to 11.3 kB on regeneration. That
  growth comes from the other agents' in-progress example additions, not from this change.

## 6. Verification (all green)

| command | result |
|---|---|
| `python -m pytest -q` (root) | 464 passed, 1 xfailed, 145 s |
| `ruff check src tests && ruff format --check src tests` | clean |
| `python -m mypy src/aodl` | 10 errors in 5 files, all the pre-existing numpy-2.5 overloads (two at `api.py:114`, not in `_notes`) |
| `make -C webapp fixtures` | `array_lift_hold.json` and `examples.json` regenerated |
| `make -C webapp test-backend` | 122 passed (no backend test pinned the parity note) |
| `make -C webapp test-frontend` | 83 passed |
