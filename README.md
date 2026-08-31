# AODL — 3D Acousto-Optic Deflector Lens Simulator & Waveform Synthesizer

Turn *"move this 10×10 atom array from A to B, lifting 10 µm out of plane"* into:

1. the RF waveforms to program on the four AOD channels of a 3D-AODL, and
2. a physically grounded simulation + movie of the resulting optical-tweezer motion.

Physics follows Lu, Song, Xiang, Ho, Lee, Yan & Stamper-Kurn, *Astigmatism-free 3D Optical
Tweezer Control for Rapid Atom Rearrangement* (arXiv:2510.11451): four AODs in two
counter-propagating pairs give independent, astigmatism-free control of (X, Y, Z), with
fading-Shepard waveforms for sustained out-of-plane displacement.

**Status:** planning. See [`docs/PLAN.md`](docs/PLAN.md) for the physics model, architecture,
and milestone ladder (single AOD → crossed pair → full 3D-AODL → fading-Shepard → product API).

Key modeling choices:

- Aperture-window realism: the simulator acts on the acoustic waveform segment actually
  present on each AOD at time *t* (retarded time), so chirp lensing, transients, and irising
  emerge naturally.
- Frequency mixing: inter-AOD tone products and intra-AOD intermodulation (IM3) are included;
  Schroeder-phase suppression is testable in simulation.
- No FFTs: focal fields are computed from closed-form astigmatic-Gaussian integrals, exact
  for chirped drives, evaluated at any (X, Y, Z, t).
