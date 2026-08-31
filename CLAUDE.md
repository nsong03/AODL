# AODL — guidance for agents working in this repo

AODL turns a high-level atom-array motion request into (1) parametric RF waveforms for the
four channels of a 3D acousto-optic deflector lens (AODL) and (2) a closed-form optical
simulation + movie of the resulting tweezers. Physics reference: arXiv:2510.11451
(equation numbers `S#` cited throughout the code refer to its Supplement).

## Read first

1. `docs/PLAN.md` — physics model, milestones, acceptance formulas
2. `docs/ARCHITECTURE.md` — package layout, core types, dependencies
3. Your assigned work order in `docs/workorders/` — the authoritative task spec.
   If the work order conflicts with the docs above, the work order wins; note the
   conflict in your report.

## Commands

```bash
pip install -e ".[dev]"          # once per session
pytest                           # full suite
pytest tests/test_foo.py -x -q   # targeted
ruff check src tests && ruff format --check src tests
pytest --nbmake examples/        # execute notebooks (only when asked)
```

## Hard conventions

- SI units everywhere internally (Hz, m, s, rad). Use `aodl.units` constants
  (`MHz`, `um`, `us`, …) at boundaries. Never bake a unit conversion into physics code.
- Every function implementing paper physics cites its equation in the docstring
  (e.g. `"""Chirp lens quadratic phase. Eq. S6."""`).
- All orientation/sign conventions live in `src/aodl/device/conventions.py` only.
  Never hardcode a sound-direction or diffraction-order sign anywhere else.
- Frequencies in waveform IR are detunings from the channel center `f_center`
  (rotating frame), per Eq. S2.
- Parametric NPZ files contain segment parameters, never samples.
- Notebooks: thin cells (logic lives in `src/aodl/`), outputs cleared before commit.
- No FFTs in the simulation path. `field/reference.py` (quadrature) is tests-only.
- Type-annotate public APIs; dataclasses for configs; numpy-vectorized evaluation.

## Git

- Work only on branch `claude/optical-tweezer-simulation-ik9m8f`.
- Only the agent designated as "wave closer" in its work order runs git commands
  (stage/commit/push); other agents leave the working tree for the closer.
- Commit only files your work order owns plus files the closer role tells you to
  integrate. Never rewrite history. Commit-message footers: use exactly the footer lines
  given in your dispatch instructions; apart from that footer, never mention AI model
  names in commit messages or code.
