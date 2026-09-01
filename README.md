# AODL — 3D Acousto-Optic Deflector Lens Simulator & Waveform Synthesizer

Describe a move — *"take this 10×10 atom array from A to B, lifting 10 µm out of the plane on
the way"* — and AODL writes the RF waveforms for the four AOD channels of a 3D-AODL and
simulates the tweezers they make, in closed form, with no FFT anywhere. It is one call from
the ask to two deliverables: a parametric waveform file for the AWG, and a movie of what the
atoms will see.

```python
from aodl import ArraySpec, Lift, TrajectorySpec, Translate, plan_motion
from aodl.units import MHz, um, us

array = ArraySpec(10, 10, delta_f_x=1.0 * MHz, delta_f_y=1.3 * MHz)  # 10.3 µm per MHz of pitch
moves = (Lift(10 * um, 150 * us), Translate(40 * um, 25 * um, 250 * us), Lift(-10 * um, 150 * us))

plan = plan_motion(TrajectorySpec(array=array, moves=moves))  # Eq. S19, or fading-Shepard
print(plan.report.summary())   # mode, band usage, axial budget, fade schedule, caveats
plan.save("move.npz")          # the AWG deliverable: segment parameters, never samples
```

`plan.render_samples(rate)` expands that file for the instrument, `plan.simulate()` returns
the per-trap metrics, and `plan.movie("move.mp4")` renders the scene (tens of seconds).

## Where to read next

- **[`docs/guide.md`](docs/guide.md)** — the lab-facing manual: quickstart, concepts,
  parameter reference, outputs, fidelity limits with their measured numbers, FAQ.
- **[`examples/06_product_tour.ipynb`](examples/06_product_tour.ipynb)** — the same path end
  to end in under a minute: report, waveform file, samples, simulation, movie.
- Notebooks **01**–**05** — the physics, one milestone each and each verifying its own
  claims: [one AOD](examples/01_single_aod_sweep.ipynb) ·
  [crossed pair, arrays, IM3](examples/02_crossed_pair_diagonal.ipynb) ·
  [the full 3D-AODL](examples/03_aodl_3d_motion.ipynb) ·
  [the user story](examples/04_array_lift_traverse.ipynb) ·
  [fading-Shepard](examples/05_fading_shepard.ipynb).
- **[`docs/PLAN.md`](docs/PLAN.md)** (physics model and milestone ladder) ·
  **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** (package layout) ·
  **[`docs/conventions.md`](docs/conventions.md)** (axes, signs, retarded time) ·
  **[`docs/waveform_format.md`](docs/waveform_format.md)** (the NPZ schema).

## Status

Physics follows Lu, Song, Xiang, Ho, Lee, Yan & Stamper-Kurn, *Astigmatism-free 3D Optical
Tweezer Control for Rapid Atom Rearrangement* ([arXiv:2510.11451](https://arxiv.org/abs/2510.11451));
`S#` equation numbers in the code refer to its Supplement.

**Built:** M1–M5 — single AOD → crossed pair → full 3D-AODL → fading-Shepard → product API.
**Checked:** **334 tests**, six notebooks executed in CI
([`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs lint, types, the suite and the
notebooks on every push), and an independent verification pass per milestone (M1–M5 accepted).

## Install

```bash
pip install -e ".[dev]"     # Python 3.11+; numpy, scipy, matplotlib, imageio
pytest                      # the suite
pytest --nbmake examples/   # execute the notebooks
```

## Key modeling choices

- **Aperture-window realism.** The simulator acts on the acoustic waveform segment actually
  present on each AOD at time *t* (retarded time), so chirp lensing, the fill transient and
  acoustic irising emerge rather than being added; the atom plane lags the drive by τ/2.
- **No FFTs.** Focal fields are closed-form astigmatic-Gaussian integrals of Eq. S11 — exact
  for chirped drives, evaluated at any (X, Y, Z, t). A direct-quadrature reference integrator
  lives in `field/reference.py` and is used only by the tests.
- **Frequency mixing.** Inter-AOD tone products (the 16-ray picture of Fig. S6) and intra-AOD
  intermodulation through IM3 (Eqs. S20–S22) are both modelled; Schroeder-phase suppression
  of the ghosts is measurable in simulation.
- **Interference done right.** Terms are grouped by optical frequency: degenerate ones
  interfere, the rest add in intensity — which is what makes shadow tweezers and the
  interlaced-fading rationale visible instead of assumed.
- **Parametric waveforms.** Files carry segments and parameters, never samples; rendering to
  an AWG buffer is a separate, explicit step.
- **Astigmatism-free 3D control.** Counter-propagating pairs give independent (X, Y, Z) with
  ΔF ≡ 0 (Table I, Eq. S19); fading-Shepard ladders (Eqs. S24–S28) hold Z out of plane past
  Eq. 1's bandwidth budget, at the price of shadow tweezers during hand-overs.

License: TBD
