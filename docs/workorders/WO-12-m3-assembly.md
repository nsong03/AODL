# WO-12 — M3 assembly: 3D-AODL notebooks, user story, integration

**Role:** implementation agent, Wave I. **You are the wave closer**: integrate Wave H
(WO-10 + WO-11, already committed when you start), make everything green, commit, push.
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §3 (M3), `docs/conventions.md`, work orders
WO-10/WO-11 (what just landed), notebooks 01–02 (house style), then this.

## Owned files

```
examples/03_aodl_3d_motion.ipynb
examples/04_array_lift_traverse.ipynb
tests/test_integration_m3.py
tests/test_integration_m2.py      (edit ONLY §cleanup below)
src/aodl/viz/movie.py             (edit only if §2 needs a small hook; keep minimal)
```

## 0. Integration duties

Full `pytest` first; reconcile any Wave-H friction with the smallest change, work orders
as truth; report every reconciliation. Then build.

## 1. `examples/03_aodl_3d_motion.ipynb` — the 3D-AODL physics notebook

House style; `default_1030()`; this is the paper's core result, write it that way:

1. **Counter-propagating pair fill**: single tweezer from Ax+Bx static tones; intensity
   vs t showing dark → turn-on at τ/2 → full at τ (WO-10 physics; markdown explains why).
2. **Pure Z motion** (Table I): `synthesize` a Lift-Hold-Lower single-tweezer spec;
   verify laterally static (< 1% waist drift), Z̄ tracks the profile, spot stays round
   (|ΔF| < 0.02 z_R) — plot Z̄(t), ΔF(t), wx/wy(t).
3. **Astigmatism-free in-plane motion**: Translate spec on the 3D-AODL vs the same
   motion done M2-style (chirp Ax+Ay only, no B compensation): side-by-side σ_astig(t)
   and Z̄(t) — the 4-AOD version holds both at ~0 while moving (reproduce the Fig. 2
   blue-vs-red phenomenology qualitatively; cite the figure in markdown).
4. **Omnidirectional**: a small helix or L-path (Translate+Lift compositions), 3D
   plot of measured (X, Y, Z̄)(t) colored by time.
5. Closing markdown: what M4 adds (fading-Shepard for sustained Z; Eq. 1 limit shown
   with the band-check error message as a teaser).

## 2. `examples/04_array_lift_traverse.ipynb` — the user story, end to end

The product demo: *"10×10 array, lift 10 µm out of plane, traverse to B, drop."*

1. Spec in ~5 lines: `ArraySpec(10, 10, ...)` (Δf_x 1.0 / Δf_y 1.3 MHz —
   note the degeneracy reason), `Lift(+10 µm, 150 µs) → Translate(+40 µm, +25 µm,
   250 µs) → Lift(−10 µm, 150 µs)`; `synthesize` → `WaveformSet`.
2. **The deliverables**: `wfs.save("array_move.npz")` (show the parametric NPZ contents
   + size vs a samples file); spectrogram-style figure of the four channel tone tracks
   (f vs t per tone, like paper Fig. 4b); band-usage figure (each channel's excursion
   vs its band).
3. Simulate + verify: tracking plots (X, Y, Z̄ vs request at t − τ/2), ΔF(t),
   per-trap Z̄ spread; state the achieved numbers in markdown.
4. **Movie**: the headline artifact — tracked mode, hue ↔ Z, XZ panel; whole 10×10 array
   lifting (whitening → red), translating, dropping. Budget ≤ 120 s render; grid sized
   to cover the trajectory (~250 µm span); target ~150 frames. Write to
   `examples/outputs/04_array_move.mp4`.
5. Closing: pointer to Eq. 1 limits and M4.

Keep total notebook runtime ≤ 3 min (use mixing_order=1 for the movie's simulate if
needed for budget — say so in markdown; keep one order-3 frame to show ghosts exist).

## 3. `tests/test_integration_m3.py` (M3 acceptance, PLAN §3)

- Pure-Z: co-chirp via Lift spec → lateral drift < 1% waist, Z̄ tracks (2% z_R),
  |ΔF| < 0.02 z_R.
- In-plane translate: motion tracks with |Z̄| and |ΔF| < 0.02 z_R throughout (the key
  4-AOD result).
- 2×2 lift-traverse-lower (reuse WO-11's mini story shape if convenient) at
  mixing_order=3: same tracking bounds; ghost/pruned power sane.
- 4-channel startup: frames at 0.45τ dark for a pair-driven spot; τ-filled frame matches
  static expectation.
- Runtime guard: the 10×10 spec's `simulate` (no frames) for 12 probe times < 5 s.

## 4. Cleanup (architect-sanctioned, WO-09 finding 4)

`tests/test_integration_m2.py` Schroeder-vs-random assertion: replace the seed-lucky
`schroeder < 0.5·random` with `schroeder < random` AND keep `schroeder < 0.25·zero`
(this corrects an over-claim, not a tolerance; cite WO-09 finding 4 in a comment).

## Definition of done

Full `pytest` green; `ruff check src tests` clean; `mypy src/aodl` exit 0;
`pytest --nbmake examples/` green (all four notebooks); nothing binary staged; notebook
outputs cleared; commit (`M3: 3D-AODL notebooks, user-story pipeline, integration`,
footer per dispatch) and push. Report: reconciliations, pytest/nbmake summaries verbatim,
notebook runtimes, movie sizes + one-line descriptions, the user-story tracking numbers,
deviations.
