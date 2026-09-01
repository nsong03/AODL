r"""M3 acceptance, end to end: the full four-AOD 3D-AODL (``docs/PLAN.md`` §3).

Everything runs through the product path — ``TrajectorySpec`` -> :func:`aodl.waveform.
synthesis.synthesize` -> :func:`aodl.engine.simulate` -> metrics and rendered frames — and
every number is compared against a closed form derived here, not against the implementation:

1. **pure Z.**  A ``Lift`` co-chirps all four channels, so Table I gives
   ``Zbar = 2 lens_scale fdot_Z = Z(t)`` with ``Delta F = 0`` and no lateral term at all: the
   array rises out of the focal plane, laterally static and round.
2. **in-plane transport with no focal shift.**  A ``Translate`` counter-chirps *within* each
   pair, so the chirps cancel in both Table I axial sums: ``Zbar = Delta F = 0`` while the
   array moves.  The control is the same motion done M2-style on ``Ax``+``Ay`` alone, which
   cannot avoid either (``docs/PLAN.md`` §1.2, the paper's Fig. 2 blue-vs-red).
3. **the user story shape** — a 2x2 array lifted, traversed and lowered at the product
   default ``mixing_order=3``: the four real traps still track, and the IM3 ghosts stay a
   per-mille effect.
4. **four-channel startup.**  Both members of a pair fill their crystal from opposite sides,
   so a pair-driven tweezer is *strictly dark* until ``tau/2`` and full at ``tau``
   (``docs/conventions.md`` §7) — including the per-term ``power``, which the two-sided
   window makes exact (WO-12 §0).
5. **cost.**  The 10x10 user story has to stay interactive.

Positions are compared at the retarded time ``t_c = t - tau/2``: v1 does not pre-compensate
the aperture transit (``waveform/synthesis.py`` module docstring), so the atom plane replays
the requested trajectory half a transit late.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import erf

from aodl import ChannelWaveform, ToneTrack, WaveformSet, ramps, simulate
from aodl.field.focal import FrameGrid
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us
from aodl.waveform.synthesis import max_z_integral, synthesize

#: Array spacings.  Deliberately unequal: with ``delta_f_x == delta_f_y`` every anti-diagonal
#: of the array shares one optical frequency and groups into a single "spot"
#: (``docs/conventions.md`` §4), which would make every position assertion here ambiguous.
DELTA_FX, DELTA_FY = 1.0 * MHz, 1.3 * MHz

#: The 10x10 user story of ``examples/04``: lift 10 µm, traverse (40, 25) µm, drop.  The
#: durations are what Eq. 1 leaves once the ladders and the lateral term have taken their
#: share of the +-10 MHz band — see :func:`test_the_user_story_is_bandwidth_bound`.
STORY_MOVES = (
    Lift(10.0 * um, 25.0 * us),
    Translate(40.0 * um, 25.0 * um, 30.0 * us),
    Lift(-10.0 * um, 25.0 * us),
)


def _linear(params):
    """``params`` with every channel at ``mixing_order=1`` — one tone, one beam.

    The statements below are about Eq. S19's frequency algebra reaching the image plane;
    intermodulation is :mod:`aodl.device.mixing`'s business and gets its own test.
    """
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _tracking(params, spec, times, wfs=None):
    """Worst tracking errors over ``times``: ``(lateral, axial, |Delta F|, Z spread)`` [m].

    The array centre is the plain mean over the frame's groups (they are equal-power here),
    compared against the requested profile at ``t_c = t - tau/2``.
    """
    x, y, z = spec.compile()
    tau = params.channels["Ax"].transit_time
    result = simulate(synthesize(spec, params) if wfs is None else wfs, times)
    t_c = np.asarray(times, dtype=np.float64) - 0.5 * tau

    lateral = axial = astig = spread = 0.0
    for i, frame in enumerate(result.metrics):
        assert frame, f"frame {i} is dark"
        z_lab = np.array([m.z_lab for m in frame])
        lateral = max(
            lateral,
            abs(float(np.mean([m.x for m in frame])) - float(x(t_c[i]))),
            abs(float(np.mean([m.y for m in frame])) - float(y(t_c[i]))),
        )
        axial = max(axial, abs(float(z_lab.mean()) - float(z(t_c[i]))))
        astig = max(astig, float(np.max(np.abs([m.delta_f for m in frame]))))
        spread = max(spread, float(z_lab.max() - z_lab.min()))
    return lateral, axial, astig, spread


# ============================================================== 1. pure Z: the co-chirp


def test_pure_z_lift_is_laterally_static_and_astigmatism_free(params1030):
    """Lift 10 µm, hold, drop: ``Zbar`` tracks ``Z(t_c)``, ``X = Y = 0``, ``Delta F = 0``.

    Eq. S19 buys the axial offset with the *same* ``f_Z`` on all four channels, so Table I's
    lateral differences stay zero while its chirp sum quadruples: ``Zbar = 2 lens_scale
    fdot_Z = Z(t)``.  Nothing else in the drive moves, which is why this is the cleanest
    statement of the M3 claim — the spot changes depth without changing position or shape.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(
        array=ArraySpec(1, 1),
        moves=(Lift(10.0 * um, 60.0 * us), Hold(80.0 * us), Lift(-10.0 * um, 60.0 * us)),
    )
    times = np.linspace(tau, spec.duration + tau, 16)

    lateral, axial, astig, _ = _tracking(params, spec, times)
    assert lateral < 0.01 * optics.waist0
    assert axial < 0.02 * optics.rayleigh
    assert astig < 0.02 * optics.rayleigh

    # ... and the excursion is worth measuring: 10 µm is 2.9 Rayleigh ranges.
    _, _, z = spec.compile()
    assert float(np.max(z(times))) > 2.0 * optics.rayleigh

    # At its own best-focus plane the spot is round and diffraction-limited the whole way;
    # a camera parked at Z = 0 would see it swell by sqrt(1 + (Z/z_R)^2) instead.
    result = simulate(synthesize(spec, params), times)
    for frame in result.metrics:
        (spot,) = frame
        assert spot.wx == pytest.approx(optics.waist0, rel=1e-9)
        assert spot.wy == pytest.approx(spot.wx, rel=1e-12)


def test_a_ten_micron_lift_costs_what_eq_1_says(params1030):
    """The hold length is Eq. 1, not a modelling choice: 10 µm fits for ~206 µs alone.

    ``f_Z = int Z dt / (2 lens_scale)`` on every channel, so a single tweezer held 10 µm off
    the focal plane walks the whole +-10 MHz half-band in 206 µs (``docs/PLAN.md`` §1.5).
    Past that the synthesizer refuses, and says so with the numbers.
    """
    ceiling = max_z_integral(params1030)
    assert ceiling / (10.0 * um) == pytest.approx(206.0 * us, rel=1e-3)

    def hold_at_10um(hold: float) -> TrajectorySpec:
        return TrajectorySpec(
            array=ArraySpec(1, 1),
            moves=(Lift(10.0 * um, 40.0 * us), Hold(hold), Lift(-10.0 * um, 40.0 * us)),
        )

    # int Z dt = Z (T_lift/2 + T_hold + T_lower/2) for the symmetric min-jerk ramps.
    synthesize(hold_at_10um(150.0 * us), params1030)  # 190 µs of Z-time: fits
    with pytest.raises(ValueError, match="leaves its usable band"):
        synthesize(hold_at_10um(220.0 * us), params1030)  # 260 µs: does not


# ================================================= 2. in-plane transport, no focal shift


def test_in_plane_translate_holds_the_focal_plane_where_two_aods_cannot(params1030):
    """The paper's key result: move in x and y with ``Zbar = Delta F = 0`` throughout.

    Eq. S19 splits the lateral term antisymmetrically inside each counter-propagating pair,
    so the two chirps of a pair are equal and opposite: they *differ* in Table I's deflection
    and *cancel* in both of its axial sums.  The control is the same motion on ``Ax``+``Ay``
    alone (M2, notebook 02), where the deflecting chirp is also the lensing chirp —
    ``Zbar = (1/2) lens_scale (fdot_Ax + fdot_Ay)`` and ``Delta F = lens_scale (fdot_Ax -
    fdot_Ay)``, both large.  This is the Fig. 2 blue-vs-red comparison.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    dx, dy, move_time = 30.0 * um, 12.0 * um, 60.0 * us
    spec = TrajectorySpec(array=ArraySpec(1, 1), moves=(Translate(dx, dy, move_time),))
    times = np.linspace(tau, spec.duration + tau, 16)

    lateral, axial, astig, _ = _tracking(params, spec, times)
    assert lateral < 0.01 * optics.waist0
    assert axial < 0.02 * optics.rayleigh  # |Zbar| itself: the requested Z is 0 throughout
    assert astig < 0.02 * optics.rayleigh

    # The M2 control: one tone per crossed channel, X = -deflection_scale f_Ax (Table I).
    scale = params.deflection_scale

    def one_tone(displacement: float) -> ChannelWaveform:
        ramp = ramps.min_jerk(0.0, move_time, 0.0, -displacement / scale)
        return ChannelWaveform((ToneTrack(freq=ramp),))

    two_aod = WaveformSet({"Ax": one_tone(dx), "Ay": one_tone(dy)}, params).with_hold_until(
        spec.duration + tau
    )
    table = simulate(two_aod, times).spot_table()

    # Same path in the image plane ...
    x, y, _ = spec.compile()
    t_c = times - 0.5 * tau
    np.testing.assert_allclose(table["x"], x(t_c), atol=0.01 * optics.waist0)
    np.testing.assert_allclose(table["y"], y(t_c), atol=0.01 * optics.waist0)
    # ... at a wildly different depth, and astigmatic while it moves.
    fdot = ramps.min_jerk(0.0, move_time, 0.0, 1.0).derivative()
    peak_zbar = 0.5 * params.lens_scale * (-(dx + dy) / scale) * float(np.max(fdot(t_c)))
    peak_astig = params.lens_scale * (-(dx - dy) / scale) * float(np.max(fdot(t_c)))
    assert np.max(np.abs(table["z_lab"])) == pytest.approx(abs(peak_zbar), rel=0.05)
    assert np.max(np.abs(table["delta_f"])) == pytest.approx(abs(peak_astig), rel=0.05)
    assert np.max(np.abs(table["z_lab"])) > 50.0 * axial
    assert np.max(np.abs(table["delta_f"])) > 50.0 * astig
    assert np.max(np.abs(table["sigma_astig"])) > 1.0  # visibly astigmatic


# ================================================== 3. the user story shape, with mixing


def test_two_by_two_lift_traverse_lower_survives_intermodulation(params1030):
    """The story shape at the product default: four traps still track, ghosts stay small.

    IM3 (Eqs. S20-S22) adds ghost groups around the array and corrects the fundamentals
    coherently, but it changes neither the deflection nor the chirp of a real trap, so the
    Table I numbers must come through untouched.
    """
    optics = params1030.optics
    tau = params1030.channels["Ax"].transit_time
    spec = TrajectorySpec(
        array=ArraySpec(2, 2, DELTA_FX, DELTA_FY),
        moves=(
            Lift(5.0 * um, 60.0 * us),
            Translate(15.0 * um, 10.0 * um, 80.0 * us),
            Lift(-5.0 * um, 60.0 * us),
        ),
    )
    x, y, z = spec.compile()
    times = np.linspace(tau, spec.duration, 8)
    wfs = synthesize(spec, params1030)
    result = simulate(wfs, times)
    scale = params1030.deflection_scale
    dx, dy = spec.array.detunings()

    for i, metrics in enumerate(result.metrics):
        t_c = float(times[i]) - 0.5 * tau
        assert len(metrics) > spec.array.n_traps  # the ghosts are there
        traps = []
        for fx in dx:
            for fy in dy:
                target = (float(x(t_c)) + scale * fx, float(y(t_c)) + scale * fy)
                spot = min(metrics, key=lambda m: (m.x - target[0]) ** 2 + (m.y - target[1]) ** 2)
                assert np.hypot(spot.x - target[0], spot.y - target[1]) < 0.02 * optics.waist0
                assert abs(spot.z_lab - float(z(t_c))) < 0.02 * optics.rayleigh
                assert abs(spot.delta_f) < 0.02 * optics.rayleigh
                traps.append(spot)
        total = sum(m.power for m in metrics)
        assert sum(m.power for m in traps) / total > 0.99  # ghosts: a per-mille effect

    terms = result.terms(-1)
    kept = float(np.sum(np.abs(terms.c) ** 2))
    assert terms.pruned_power / kept < 1e-6  # nothing meaningful was thrown away


# ===================================================== 4. four-channel startup transient


def _static_pair_scene(params, mx=2, my=2):
    """A static, band-feasible array driven by all four channels (every axis is a pair)."""
    spec = TrajectorySpec(array=ArraySpec(mx, my, DELTA_FX, DELTA_FY), moves=(Hold(60.0 * us),))
    return spec, synthesize(spec, params)


def test_pair_driven_array_is_dark_before_half_a_transit_and_full_at_tau(params1030):
    """0.45 tau: nothing.  tau: exactly the static array (``docs/conventions.md`` §7).

    Each channel of a pair fills its own crystal from its own transducer, so the light that
    crosses both sees the intersection ``[D/2 - v t, v t - D/2]`` — empty until the two
    wavefronts meet at ``tau/2``, the whole aperture at ``tau``.  Nothing about the *drive*
    changes across that transient, so the frame at ``tau`` must equal the fully-filled frame
    at ``2 tau`` to round-off.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec, wfs = _static_pair_scene(params)

    dark = simulate(wfs, [0.45 * tau])
    assert dark.metrics == [[]]
    assert dark.terms(0).n_terms == 0
    assert dark.terms(0).pruned_power == 0.0  # darkness is physics, not pruning

    half = 20.0 * um
    grid = FrameGrid(-half, half, 201, -half, half, 201)
    assert np.all(dark.frame(0, grid) == 0.0)

    filled = simulate(wfs, [tau, 2.0 * tau])
    assert filled.terms(0).edge == (None, None)
    at_tau, later = (sorted(frame, key=lambda m: m.df_opt) for frame in filled.metrics)
    assert len(at_tau) == spec.array.n_traps == 4
    for spot, reference in zip(at_tau, later, strict=True):
        assert spot.x == pytest.approx(reference.x, rel=1e-12, abs=1e-18)
        assert spot.y == pytest.approx(reference.y, rel=1e-12, abs=1e-18)
        assert spot.power == pytest.approx(reference.power, rel=1e-12)
    np.testing.assert_allclose(filled.frame(0, grid), filled.frame(1, grid), rtol=1e-12, atol=0.0)

    # the static array is on the Table I grid, in the focal plane, round
    scale = params.deflection_scale
    dx, dy = spec.array.detunings()
    expected = {(round(scale * fx, 12), round(scale * fy, 12)) for fx in dx for fy in dy}
    for spot in at_tau:
        nearest = min(expected, key=lambda p: (p[0] - spot.x) ** 2 + (p[1] - spot.y) ** 2)
        assert abs(nearest[0] - spot.x) < 0.01 * optics.waist0
        assert abs(nearest[1] - spot.y) < 0.01 * optics.waist0
        assert abs(spot.z_lab) < 1e-12 * optics.rayleigh
        assert abs(spot.delta_f) < 1e-12 * optics.rayleigh


def test_power_during_the_two_sided_fill_matches_the_frame_and_the_erf_law(params1030):
    """WO-12 §0: ``SpotMetrics.power`` integrates the *window*, not the half-line.

    A single pair-driven tweezer with both axes mid-fill.  Two independent checks of the same
    number, at ``t = 0.55 tau`` and ``0.75 tau``:

    * against the rendered frame — ``power`` is by construction the integral of
      :func:`aodl.field.focal.intensity_frame` (Parseval, ``field/measure.py``), and a
      partially filled aperture must not change that;
    * against the closed form.  A constant-envelope drive leaves the pupil a pure Gaussian
      with a linear phase, so the two-sided window ``|u| <= h`` passes
      ``int_-h^h e^{-2u^2/w_in^2} du / int e^{-2u^2/w_in^2} du = erf(sqrt2 h / w_in)`` of the
      power per axis, i.e. its square across the two.

    Integrating such a term over ``u >= lo`` instead — the half-line view of the fill state —
    over-estimates the power by 4.89x at ``0.55 tau`` and 1.07x at ``0.75 tau``.
    """
    params = _linear(params1030)
    optics = params.optics
    aod = params.channels["Ax"]
    tau, v, aperture = aod.transit_time, aod.sound_speed, aod.aperture
    _, wfs = _static_pair_scene(params, mx=1, my=1)

    full = simulate(wfs, [2.0 * tau]).metrics[0][0].power
    # Wide and fine: a windowed pupil has hard-edge far-field tails that fall off only as
    # 1/dX^2, so the frame integral converges from below like 1/X_max (0.4% left outside).
    half, n = 200.0 * um, 2401
    grid = FrameGrid(-half, half, n, -half, half, n)

    for frac, over_estimate in ((0.55, 4.886), (0.75, 1.066)):
        t = frac * tau
        result = simulate(wfs, [t])
        (spot,) = result.metrics[0]
        integral = float(result.frame(0, grid).sum()) * grid.dx * grid.dy
        assert spot.power / integral == pytest.approx(1.0, rel=0.02)

        h = v * t - 0.5 * aperture
        passed = float(erf(np.sqrt(2.0) * h / optics.w_in)) ** 2
        assert spot.power / full == pytest.approx(passed, rel=1e-9)
        # ... and the half-line reading of the same window is the documented over-estimate.
        one_sided = float(erf(np.sqrt(2.0) * h / optics.w_in) + 1.0) ** 2 / 4.0
        assert one_sided / passed == pytest.approx(over_estimate, rel=1e-3)


# ================================================================== 5. the user story cost


def test_the_user_story_is_bandwidth_bound(params1030):
    """The 10x10 story fits only because it is *fast*: Eq. 1 sets the schedule.

    A 10-tone ladder at 1.0 MHz already occupies +-4.5 MHz of ``Bx`` and 1.3 MHz gives
    ``By`` +-5.85 MHz; the lateral term takes another 1.9 / 1.2 MHz at the far end of the
    traverse.  What is left for ``f_Z`` is ~2.9 MHz, i.e. ``|int Z dt| <= 6e-10 m.s`` — 60 µs
    of Z-time at 10 µm.  Ask for the same move over 550 µs and the synthesizer refuses.
    """
    spec = TrajectorySpec(array=ArraySpec(10, 10, DELTA_FX, DELTA_FY), moves=STORY_MOVES)
    wfs = synthesize(spec, _linear(params1030))
    assert wfs.n_tones == 22  # 1 + 10 + 1 + 10

    for name, cw in wfs.channels.items():
        aod = wfs.params.channels[name]
        lo, hi = aod.band
        t = np.linspace(*wfs.t_span, 2001)
        absolute = np.array([aod.f_center + np.asarray(tone.freq(t)) for tone in cw.tones])
        assert absolute.min() > lo and absolute.max() < hi

    slow = TrajectorySpec(
        array=spec.array,
        moves=(
            Lift(10.0 * um, 150.0 * us),
            Translate(40.0 * um, 25.0 * um, 250.0 * us),
            Lift(-10.0 * um, 150.0 * us),
        ),
    )
    with pytest.raises(ValueError, match="leaves its usable band"):
        synthesize(slow, params1030)


def test_ten_by_ten_probe_run_stays_fast(params1030):
    """``simulate`` (no frames) of the 100-trap story at 12 probe times, under 5 s.

    100 traps means 100 pupil terms in 100 frequency groups per frame — the notebook does
    this before it renders anything, and a lab sweeping a trajectory does it repeatedly.
    """
    params = _linear(params1030)
    optics = params.optics
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(array=ArraySpec(10, 10, DELTA_FX, DELTA_FY), moves=STORY_MOVES)
    wfs = synthesize(spec, params)
    times = np.linspace(tau, spec.duration + tau, 12)

    start = time.perf_counter()
    result = simulate(wfs, times)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"10x10 simulate over 12 probes took {elapsed:.2f} s"

    assert [len(frame) for frame in result.metrics] == [100] * 12
    lateral, axial, astig, spread = _tracking(params, spec, times, wfs=wfs)
    assert lateral < 0.01 * optics.waist0
    assert axial < 0.02 * optics.rayleigh
    assert astig < 0.02 * optics.rayleigh
    assert spread < 0.02 * optics.rayleigh
