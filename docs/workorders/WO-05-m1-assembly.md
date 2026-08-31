# WO-05 — M1 assembly: engine, movie renderer, notebook 01, integration

**Role:** implementation agent, Wave C. **You are the wave closer**: integrate Wave B,
make the full suite green, commit everything (including WO-02/03/04 files left in the
tree), push. **Read first:** `CLAUDE.md`, `docs/PLAN.md` §3 (M1 acceptance),
`docs/ARCHITECTURE.md`, the three Wave-B work orders (interfaces), then this.

## Owned files

```
src/aodl/engine.py  src/aodl/viz/style.py  src/aodl/viz/movie.py  src/aodl/__init__.py
examples/01_single_aod_sweep.ipynb
tests/test_engine.py  tests/test_integration_m1.py
```

## 0. Integration duties (before your own build)

Run the full `pytest`. Wave-B agents worked in parallel against frozen interface specs;
small mismatches are expected and yours to reconcile (choose the work-order spec as truth;
smallest fix wins; note every reconciliation in your report). Do not redesign interfaces.

## 1. `engine.py`

```python
@dataclass class FrameGrid: x0,x1,nx, y0,y1,ny        # meters (move here if WO-04 defined
                                                      # it locally; single source)
@dataclass class SimResult:
    times: float[]
    metrics: list[list[SpotMetrics]]     # per frame, per frequency group
    _wfs, _params                        # retained for lazy evaluation
    def frame(self, i, grid, z_lab=None) -> float[ny,nx]   # z_lab None → tracked plane
    def slice_xz(self, i, x_axis, z_axis, y0)
    def tracked_z(self) -> float[]       # measure.track_z per frame
    def spot_table(self) -> structured array/dict of arrays   # tidy per-frame metrics

def simulate(wfs: WaveformSet, times, channels=None) -> SimResult
    # per t: build_terms → measure; store metrics; lazy fields.
```

## 2. `viz/style.py` + `viz/movie.py` (read the repo's plotting conventions below)

- `style.py`: `Z_CMAP = matplotlib 'RdBu_r'`; `z_norm(z, z_max)` symmetric about 0;
  dark background style dict; γ = 0.7 intensity gamma.
- `movie.py`:

```python
def render_movie(result, path, grid=None, mode="tracked", fps=25,
                 xz_panel=True, xz_row_y=None, spectrogram_panel=False, dpi=110)
```

Layout: main axes = XY intensity at the tracked plane (`mode="tracked"`; `"fixed"` uses
z_lab=0). Each frequency group's patch is tinted `Z_CMAP(z_norm(group z_lab, z_max))`
scaled by (I/I_max)^γ — composite additively on dark background, single global I_max over
the movie for honest brightness. Z colorbar in µm; timestamp (µs) and tracked-Z annotation.
Side panel: XZ slice at `xz_row_y` (default: power-weighted mean Y), Z axis in µm,
horizontal line marking the tracked plane. Encode mp4 via imageio/imageio-ffmpeg;
`path.gif` fallback works. Auto grid default: cover all spot trajectories ± 8 waists,
512 px on the long side. z_max default: max |z_lab| over frames, floored at 1 µm.

## 3. `src/aodl/__init__.py` — public API wiring

Re-export: params presets, `PiecewisePoly`, ramps module, `ToneTrack`/`ChannelWaveform`/
`WaveformSet`/envelopes, `simulate`, `FrameGrid`, `render_movie`, `measure`. `__all__` set.

## 4. `examples/01_single_aod_sweep.ipynb`

Narrative notebook (markdown physics → code → result vs prediction), M1 story on
`default_1030()`, single channel Ay:

1. **Setup & static tone**: +3 MHz detuning; simulate; plot XY frame; assert-print
   position vs `−deflection_scale·f` (Table I) and waist vs `waist0`.
2. **Aperture window**: `device.aod.aperture_window` snapshot plot mid-chirp (the literal
   waveform on the crystal), annotated with beam Gaussian.
3. **Fill transient**: intensity vs t over the first 15 µs (drive starts at t=0):
   plateau after τ = 11.54 µs; annotate τ and 2w_in/v.
4. **Min-jerk sweep** (0 → 5 MHz over 100 µs): plot measured X_spot(t) vs prediction
   (retarded, t_c = t − τ/2); plot Z_x,lab(t) = lens_scale·ḟ(t_c) vs measured; plot
   wx, wy vs t (astigmatism during motion — the M1 headline); σ_astig(t).
5. **Movie**: `render_movie` (~120 frames), embedded via `IPython.display.Video`;
   file written to `examples/outputs/01_sweep.mp4` (gitignored).
6. Closing markdown: what M2 adds (crossed pair → spherical lensing).

Keep total notebook runtime < 2 min. Clear outputs before commit.

## 5. Tests

- `test_engine.py`: simulate a 2-frame static-tone run; metrics stable across frames;
  lazy frame == direct `intensity_frame`; spot_table shapes.
- `test_integration_m1.py` (the M1 acceptance from PLAN §3, end-to-end through
  `simulate`): min-jerk sweep on Ay; (a) X_spot(t) tracks
  `−deflection_scale·f(t−τ/2)` to 1% of waist0 for t > τ; (b) astigmatic interval
  matches `lens_scale·ḟ(t−τ/2)` to 2% at three probe times; (c) at the chirp peak,
  wx(z=0)/wy(z=0) > 1.05 (visible astigmatism); (d) fill transient: I(t=0.4τ)/I(t≥τ)
  ∈ (0.05, 0.95).

## Definition of done

Full `pytest` green (Wave B + yours); `ruff check src tests` clean;
`pytest --nbmake examples/01_single_aod_sweep.ipynb` passes; notebook committed with
outputs cleared; commit all current work (message: `M1: single-AOD moving tweezer —
engine, movie renderer, example notebook`, plus the footer lines given in your dispatch
instructions) and push with retries per repo instructions. Report: integration
reconciliations made, test summary, runtime of notebook, movie file size.
