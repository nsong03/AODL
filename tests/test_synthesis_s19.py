r"""M3 acceptance for waveform synthesis: the full Eq. S19 solver (``waveform/synthesis.py``).

:func:`~aodl.waveform.synthesis.synthesize` turns a
:class:`~aodl.trajectory.spec.TrajectorySpec` into the four channel drives

.. math::

    f_{Ax} = -\tfrac{v}{2\lambda F} X + f_Z, \quad
    f_{Bx}^{(n)} = f_{x0}^{(n)} + \tfrac{v}{2\lambda F} X + f_Z, \quad
    f_Z = \tfrac{v^2}{2\lambda F^2}\!\int_0^t\! Z\,dt'

(and likewise in y).  Three levels of check, in order:

1. **algebra** — the Table I inversions hold as *polynomial identities*, not as fits:
   ``deflection_scale (f_Bx - f_Ax) = X(t)`` and ``2 lens_scale fdot_Z = Z(t)`` compared
   coefficient by coefficient, and ``Delta F`` identically zero because the four chirps carry
   the same ``f_Z`` (the astigmatism-free claim, already visible in the waveform);
2. **geometry** — a static array lands on the Table-I grid through the real device path;
3. **the user story** — lift, traverse, lower a 2x2 array and watch the *simulated* tweezers
   track the requested ``(X, Y, Zbar)`` at the retarded time ``t - tau/2``, with
   ``|Delta F|`` staying under 2% of a Rayleigh range throughout.

Plus the guard rail that makes the synthesizer usable in a lab: a trajectory that would walk a
channel out of its RF band (Eq. 1) is refused with the numbers needed to fix it.

Everything is probed at ``t >= tau``, where the aperture is full: the fill transient is
``device/``'s business, pinned by ``tests/test_device_single_aod.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aodl import simulate
from aodl.params import CHANNELS
from aodl.poly import PiecewisePoly
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, ms, um, us
from aodl.waveform.synthesis import T_PAD_TRANSITS, f_z_ramp, max_z_integral, synthesize

#: The mini user story (``docs/workorders/WO-11`` §3): up 5 µm, across (15, 10) µm, down.
STORY = (
    Lift(5.0 * um, 60.0 * us),
    Translate(15.0 * um, 10.0 * um, 80.0 * us),
    Lift(-5.0 * um, 60.0 * us),
)

#: Deliberately unequal spacings: with ``delta_f_x == delta_f_y`` the anti-diagonals of the
#: array share one optical frequency and group into one "spot" each (WO-08's finding, pinned
#: by ``tests/test_integration_m2.py``), which would make every position assertion here
#: ambiguous.
DELTA_FX, DELTA_FY = 1.0 * MHz, 1.3 * MHz


def _linear(params):
    """``params`` with every channel at ``mixing_order=1`` — the strictly linear model.

    The product default (3) adds compression and IM3 ghosts, which are
    :mod:`aodl.device.mixing`'s business; the statements here are about Eq. S19's frequency
    algebra, so they use the model in which one tone means exactly one beam.
    """
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _padded(coeffs: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((coeffs.shape[0], width), dtype=np.float64)
    out[:, : coeffs.shape[1]] = coeffs
    return out


def _assert_same_poly(got: PiecewisePoly, want: PiecewisePoly, floor: float) -> None:
    """Coefficient-level equality of two piecewise polynomials (same breakpoints).

    ``floor`` is the absolute tolerance: the identities below are *exact* in real arithmetic,
    so the only error is float64 cancellation between terms of the drive's own magnitude.
    """
    np.testing.assert_allclose(got.breaks, want.breaks, rtol=1e-15, atol=1e-18)
    width = max(got.coeffs.shape[1], want.coeffs.shape[1])
    np.testing.assert_allclose(
        _padded(got.coeffs, width), _padded(want.coeffs, width), rtol=1e-12, atol=floor
    )


def _drive_scale(wfs, scale: float) -> float:
    """``1e-12`` of the largest drive coefficient, expressed in the compared quantity."""
    peak = max(
        float(np.max(np.abs(tone.freq.coeffs))) for cw in wfs.channels.values() for tone in cw.tones
    )
    return 1e-12 * scale * peak


def _story_spec(mx: int = 2, my: int = 2) -> TrajectorySpec:
    return TrajectorySpec(array=ArraySpec(mx, my, DELTA_FX, DELTA_FY), moves=STORY)


# ================================================================== 1. polynomial identities


def test_lateral_identity_holds_coefficient_by_coefficient(params1030):
    r"""``deflection_scale (f_Bx^{(n)} - f_Ax) = X(t) + pitch_x * n_offset`` — exactly.

    Table I inverted.  Eq. S19 puts ``-vX/2 lambda F`` on the A member of the pair and
    ``+vX/2 lambda F`` on the B member, so their *difference* is ``vX/lambda F`` and the
    common ``f_Z`` cancels identically — no sampling, no fitting: the two polynomials agree
    coefficient by coefficient, on every segment, for the whole 3D move.
    """
    spec = _story_spec()
    x, y, _ = spec.compile()
    wfs = synthesize(spec, params1030, t_pad=0.0)
    scale = params1030.deflection_scale
    floor = _drive_scale(wfs, scale)

    for axis, (a_name, b_name), position in (("x", ("Ax", "Bx"), x), ("y", ("Ay", "By"), y)):
        f_a = wfs.channels[a_name].tones[0].freq
        offsets = spec.array.detunings()[0 if axis == "x" else 1]
        for n, tone in enumerate(wfs.channels[b_name].tones):
            got = (tone.freq - f_a).scale(scale)
            want = position.offset(scale * offsets[n])
            _assert_same_poly(got, want, floor)


def test_axial_identity_holds_coefficient_by_coefficient(params1030):
    r"""``2 lens_scale fdot_Z = Z(t)`` — exactly, and ``f_Z`` is recoverable from the drive.

    ``f_Z`` is the *common* half of each pair, so ``(f_Ax + f_Bx^{(centre)})/2`` returns it
    from the channels alone; its derivative times ``2 lens_scale`` is Table I's ``Zbar``.
    """
    spec = TrajectorySpec(array=ArraySpec(3, 3, DELTA_FX, DELTA_FY), moves=STORY)
    _, _, z = spec.compile()
    wfs = synthesize(spec, params1030, t_pad=0.0)
    floor = _drive_scale(wfs, 2.0 * params1030.lens_scale)

    f_z = f_z_ramp(z, params1030)
    _assert_same_poly(f_z.derivative().scale(2.0 * params1030.lens_scale), z, floor)

    # the same object, read back out of the four channels (centre tone of each ladder)
    for a_name, b_name in (("Ax", "Bx"), ("Ay", "By")):
        common = (wfs.channels[a_name].tones[0].freq + wfs.channels[b_name].tones[1].freq).scale(
            0.5
        )
        _assert_same_poly(common.derivative().scale(2.0 * params1030.lens_scale), z, floor)


def test_the_drive_is_astigmatism_free_by_construction(params1030):
    """``Delta F = lens_scale (fdot_Ax + fdot_Bx - fdot_Ay - fdot_By) = 0`` identically.

    Every channel chirps at ``fdot_Z`` plus a lateral term that is equal and opposite within
    its pair, so the Table I astigmatism combination cancels twice over — before any
    simulation runs.  This is the paper's central claim, stated in the waveform.
    """
    wfs = synthesize(_story_spec(), params1030, t_pad=0.0)
    rates = {
        name: wfs.channels[name].tones[0].freq.derivative().scale(params1030.lens_scale)
        for name in CHANNELS
    }
    delta = rates["Ax"] + rates["Bx"] - rates["Ay"] - rates["By"]
    peak = float(np.max(np.abs(rates["Ax"].coeffs)))
    np.testing.assert_allclose(delta.coeffs, 0.0, atol=1e-12 * peak)


# ====================================================================== 2. the static array


def test_static_array_sits_on_the_table_i_grid(params1030):
    """A 3x3 spec with no moves at all: nine traps on the ``deflection_scale delta_f`` grid."""
    params = _linear(params1030)
    tau = params.channels["Ax"].transit_time
    spec = TrajectorySpec(array=ArraySpec(3, 3, DELTA_FX, DELTA_FY), moves=(Hold(50.0 * us),))
    wfs = synthesize(spec, params)
    metrics = simulate(wfs, [2.0 * tau]).metrics[0]
    optics = params.optics

    assert len(metrics) == 9
    dx, dy = spec.array.detunings()
    expected = {
        (round(params.deflection_scale * fx, 12), round(params.deflection_scale * fy, 12))
        for fx in dx
        for fy in dy
    }
    for spot in metrics:
        nearest = min(expected, key=lambda p: (p[0] - spot.x) ** 2 + (p[1] - spot.y) ** 2)
        assert abs(nearest[0] - spot.x) < 0.01 * optics.waist0
        assert abs(nearest[1] - spot.y) < 0.01 * optics.waist0

    measured = np.array([(m.x, m.y) for m in metrics])
    assert len({(round(x, 12), round(y, 12)) for x, y in measured}) == 9  # no trap doubled
    np.testing.assert_allclose(
        np.diff(np.unique(np.round(measured[:, 0], 12))),
        params.deflection_scale * DELTA_FX,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.diff(np.unique(np.round(measured[:, 1], 12))),
        params.deflection_scale * DELTA_FY,
        rtol=1e-9,
    )

    # a static array is in the focal plane, round, and evenly lit
    np.testing.assert_allclose([m.z_lab for m in metrics], 0.0, atol=1e-18)
    np.testing.assert_allclose([m.delta_f for m in metrics], 0.0, atol=1e-18)
    power = np.array([m.power for m in metrics])
    assert power.std() / power.mean() < 0.01


# ================================================================== 3. the mini user story


def _story_errors(params, spec, times, order_one: bool = True):
    """Tracking errors of the story: ``(lateral, axial, |Delta F|, per-trap Z spread)`` [m]."""
    x, y, z = spec.compile()
    tau = params.channels["Ax"].transit_time
    wfs = synthesize(spec, params)
    result = simulate(wfs, times)
    t_c = np.asarray(times) - 0.5 * tau  # docs/conventions.md §7

    lateral = axial = astig = spread = 0.0
    for i, frame in enumerate(result.metrics):
        if order_one:
            assert len(frame) == spec.array.n_traps
        z_lab = np.array([m.z_lab for m in frame])
        lateral = max(
            lateral,
            abs(float(np.mean([m.x for m in frame])) - x(t_c[i])),
            abs(float(np.mean([m.y for m in frame])) - y(t_c[i])),
        )
        axial = max(axial, abs(float(z_lab.mean()) - z(t_c[i])))
        astig = max(astig, float(np.max(np.abs([m.delta_f for m in frame]))))
        spread = max(spread, float(z_lab.max() - z_lab.min()))
    return lateral, axial, astig, spread


def test_mini_user_story_tracks_the_requested_trajectory(params1030):
    """The M3 core check: lift, traverse, lower a 2x2 array — and measure what came out.

    Twelve probes over ``[tau, T]`` (the aperture is full from ``tau`` on).  At each one the
    power-weighted array centre must sit where the spec asked at the *retarded* drive time
    ``t - tau/2``, within 1% of a waist laterally and 2% of a Rayleigh range axially, with
    ``|Delta F|`` under 2% ``z_R`` — the astigmatism-free claim — and every trap of the array
    at the same height to within 2% ``z_R``.
    """
    params = _linear(params1030)
    spec = _story_spec()
    tau = params.channels["Ax"].transit_time
    times = np.linspace(tau, spec.duration, 12)
    optics = params.optics

    lateral, axial, astig, spread = _story_errors(params, spec, times)
    assert lateral < 0.01 * optics.waist0
    assert axial < 0.02 * optics.rayleigh
    assert astig < 0.02 * optics.rayleigh
    assert spread < 0.02 * optics.rayleigh

    # the move is worth measuring: it is many waists lateral and more than a z_R axial
    x, _, z = spec.compile()
    assert x(spec.duration) > 10.0 * optics.waist0
    assert float(np.max(z(times))) > optics.rayleigh


def test_mini_user_story_spot_check_with_intermodulation(params1030):
    """One frame of the same story at the product default (``mixing_order=3``).

    IM3 adds ghost groups around the array (Eqs. S20-S22) but must not move the four real
    traps: each still sits on ``(X + pitch_x n, Y + pitch_y m)`` at the retarded time, still
    at ``Zbar = Z``, still with no astigmatism.
    """
    spec = _story_spec()
    x, y, z = spec.compile()
    tau = params1030.channels["Ax"].transit_time
    t = 120.0 * us
    t_c = t - 0.5 * tau
    optics = params1030.optics

    wfs = synthesize(spec, params1030)
    metrics = simulate(wfs, [t]).metrics[0]
    assert len(metrics) > spec.array.n_traps  # the ghosts are there

    scale = params1030.deflection_scale
    dx, dy = spec.array.detunings()
    traps = []
    for fx in dx:
        for fy in dy:
            target = (x(t_c) + scale * fx, y(t_c) + scale * fy)
            spot = min(metrics, key=lambda m: (m.x - target[0]) ** 2 + (m.y - target[1]) ** 2)
            assert np.hypot(spot.x - target[0], spot.y - target[1]) < 0.02 * optics.waist0
            assert abs(spot.z_lab - z(t_c)) < 0.05 * optics.rayleigh
            assert abs(spot.delta_f) < 0.05 * optics.rayleigh
            traps.append(spot)

    total = sum(m.power for m in metrics)
    assert sum(m.power for m in traps) / total > 0.99  # ghosts stay a per-mille effect


def test_pad_lets_a_run_probe_the_array_at_rest(params1030):
    """``t_pad`` (default ``2 tau``) is what makes the last frames simulatable.

    A frame at ``t`` needs drive time ``t - tau/2``, and :func:`aodl.engine.simulate` refuses
    to clamp-hold past a tone's programmed domain (the phase would stop advancing), so without
    the tail the array cannot be observed settling.
    """
    params = _linear(params1030)
    spec = _story_spec()
    tau = params.channels["Ax"].transit_time
    x, y, z = spec.compile()

    wfs = synthesize(spec, params)
    assert wfs.t_span == pytest.approx((0.0, spec.duration + T_PAD_TRANSITS * tau), rel=1e-12)

    late = spec.duration + tau
    frame = simulate(wfs, [late]).metrics[0]
    assert float(np.mean([m.x for m in frame])) == pytest.approx(x(spec.duration), abs=1e-12)
    assert float(np.mean([m.y for m in frame])) == pytest.approx(y(spec.duration), abs=1e-12)
    assert float(np.mean([m.z_lab for m in frame])) == pytest.approx(
        z(spec.duration), abs=1e-3 * params.optics.rayleigh
    )

    with pytest.raises(ValueError, match="Extend the waveform first"):
        simulate(synthesize(spec, params, t_pad=0.0), [late])


# =========================================================================== 4. band checks


def _hold_off_plane(hold: float, dz: float = 10.0 * um) -> TrajectorySpec:
    """Lift to ``dz``, sit there for ``hold``, come back — the bandwidth-hungry shape."""
    return TrajectorySpec(
        array=ArraySpec(2, 2, DELTA_FX, DELTA_FY),
        moves=(Lift(dz, 60.0 * us), Hold(hold), Lift(-dz, 60.0 * us)),
    )


def test_a_long_hold_off_the_focal_plane_is_refused_with_the_numbers_to_fix_it(params1030):
    """Eq. 1: sustained ``Z`` walks every channel out of its band, and synthesis says so.

    Holding ``Z = 10 µm`` costs ``fdot_Z = Z / (2 lens_scale) = 48.5 MHz/ms`` on all four
    channels (``docs/PLAN.md`` §1.5), so a 2 ms hold asks for ~100 MHz of a 20 MHz band.  The
    error has to name the channel, the excursion, the limit, and the ``|int Z dt|`` that
    *would* fit.
    """
    spec = _hold_off_plane(2.0 * ms)
    with pytest.raises(ValueError) as excinfo:
        synthesize(spec, params1030)
    message = str(excinfo.value)

    assert "leaves its usable band" in message
    assert "outside the limit [90.0000, 110.0000] MHz" in message
    assert f"{max_z_integral(params1030):.4g} m.s" in message  # what would fit
    assert "check_band=False" in message
    assert any(f"channel {name!r}" in message for name in CHANNELS)

    _, _, z = spec.compile()
    requested = 2.0 * params1030.lens_scale * abs(float(f_z_ramp(z, params1030)(spec.duration)))
    assert requested > max_z_integral(params1030)  # ... and it really is infeasible
    assert f"{requested:.4g} m.s" in message


def test_check_band_false_is_the_documented_escape_hatch(params1030):
    """Infeasible drives still synthesize for plotting — they just would not diffract."""
    spec = _hold_off_plane(2.0 * ms)
    wfs = synthesize(spec, params1030, check_band=False)
    assert wfs.n_tones == 6
    peak = max(
        float(np.max(np.abs(tone.freq(np.linspace(*wfs.t_span, 2001)))))
        for cw in wfs.channels.values()
        for tone in cw.tones
    )
    assert peak > 50.0 * MHz  # far outside the +/- 10 MHz band, as intended


def test_a_short_hold_off_the_focal_plane_passes_the_band_check(params1030):
    """The same shape, sized to the band: 10 µm for 50 µs fits with room to spare."""
    spec = _hold_off_plane(50.0 * us)
    wfs = synthesize(spec, params1030, check_band=True)

    for name, cw in wfs.channels.items():
        aod = wfs.params.channels[name]
        lo, hi = aod.band
        for tone in cw.tones:
            absolute = aod.f_center + np.asarray(tone.freq(np.linspace(*wfs.t_span, 4001)))
            assert absolute.min() > lo and absolute.max() < hi

    _, _, z = spec.compile()
    requested = 2.0 * params1030.lens_scale * abs(float(f_z_ramp(z, params1030)(spec.duration)))
    assert requested < max_z_integral(params1030)


def test_the_band_check_sees_the_array_ladder_too(params1030):
    """The ladder offsets ride on top of ``f_Z``: a wide array is what tips this one over.

    A 10 µm hold for 100 µs spends 7.8 of the 10 MHz headroom on ``f_Z`` alone; a nine-tone
    ladder at 1 MHz then hangs its outer tones up to 4 MHz further out, so the offender is a
    ``Bx`` tone even though the single-tone ``Ax``/``By`` drives fit.
    """
    moves = (Lift(10.0 * um, 60.0 * us), Hold(100.0 * us), Lift(-10.0 * um, 60.0 * us))
    narrow = TrajectorySpec(array=ArraySpec(1, 1), moves=moves)
    synthesize(narrow, params1030)  # fits on its own

    wide = TrajectorySpec(array=ArraySpec(9, 1, 1.0 * MHz), moves=moves)
    with pytest.raises(ValueError, match=r"channel 'Bx' tone [5-8] leaves its usable band"):
        synthesize(wide, params1030)


# ======================================================================== 5. the pure-Z case


def test_pure_z_lift_is_a_common_co_chirp(params1030):
    """Lift only: the pair *differences* stay constant and all four chirps are identical.

    This is the co-chirp of ``PLAN`` M3 — four equal chirps make a spherical lens, the
    lateral terms are absent, and the array simply rises: ``Zbar = Z``, ``Delta F = 0``,
    ``X = Y = 0``.
    """
    params = _linear(params1030)
    spec = TrajectorySpec(
        array=ArraySpec(3, 2, DELTA_FX, DELTA_FY),
        moves=(Lift(4.0 * um, 80.0 * us), Hold(30.0 * us), Lift(-4.0 * um, 80.0 * us)),
    )
    _, _, z = spec.compile()
    wfs = synthesize(spec, params, t_pad=0.0)

    for a_name, b_name, axis in (("Ax", "Bx", 0), ("Ay", "By", 1)):
        f_a = wfs.channels[a_name].tones[0].freq
        offsets = spec.array.detunings()[axis]
        for n, tone in enumerate(wfs.channels[b_name].tones):
            difference = tone.freq - f_a
            np.testing.assert_allclose(difference.coeffs[:, 1:], 0.0, atol=1e-6)  # µHz
            np.testing.assert_allclose(difference.coeffs[:, 0], offsets[n], rtol=1e-12, atol=1e-6)

    rates = [wfs.channels[name].tones[0].freq.derivative() for name in CHANNELS]
    peak = float(np.max(np.abs(rates[0].coeffs)))
    assert peak > 0.0
    for rate in rates[1:]:
        np.testing.assert_allclose(rate.breaks, rates[0].breaks, rtol=1e-15)
        np.testing.assert_allclose(rate.coeffs, rates[0].coeffs, rtol=1e-12, atol=1e-12 * peak)

    # ... and that is what the simulation sees
    tau = params.channels["Ax"].transit_time
    t = 90.0 * us
    frame = simulate(wfs.with_hold_until(spec.duration + tau), [t]).metrics[0]
    assert len(frame) == 6
    for spot in frame:
        assert spot.z_lab == pytest.approx(z(t - 0.5 * tau), abs=1e-3 * params.optics.rayleigh)
        assert abs(spot.delta_f) < 1e-3 * params.optics.rayleigh
    np.testing.assert_allclose(
        float(np.mean([m.x for m in frame])), 0.0, atol=1e-6 * params.optics.waist0
    )


# ============================================================================= 6. plumbing


def test_synthesize_builds_all_four_channels_with_the_requested_ladders(params1030):
    spec = _story_spec(mx=4, my=3)
    wfs = synthesize(spec, params1030, amp=0.5)

    assert set(wfs.channels) == set(CHANNELS)
    assert [wfs.channels[n].n_tones for n in CHANNELS] == [1, 4, 1, 3]
    assert wfs.params is params1030
    assert "2x2" not in wfs.description and "4x3 array" in wfs.description
    for cw in wfs.channels.values():
        for tone in cw.tones:
            assert tone.env.amp == pytest.approx(0.5)


@pytest.mark.parametrize("mode", ["schroeder", "zero", "random"])
def test_phase_conventions_reach_the_b_ladders_only(params1030, rng, mode):
    """The ``A`` channels carry a single tone, whose phase is a global factor (docstring)."""
    from aodl.waveform.synthesis import schroeder_phases

    wfs = synthesize(_story_spec(mx=4, my=4), params1030, phases=mode, rng=rng)
    assert [tone.phase0 for tone in wfs.channels["Ax"].tones] == [0.0]
    assert [tone.phase0 for tone in wfs.channels["Ay"].tones] == [0.0]

    phases = np.array([tone.phase0 for tone in wfs.channels["Bx"].tones])
    if mode == "schroeder":
        np.testing.assert_allclose(phases, schroeder_phases(4), rtol=1e-14)
    elif mode == "zero":
        np.testing.assert_allclose(phases, 0.0, atol=0.0)
    else:
        assert np.all((phases >= 0.0) & (phases < 2.0 * np.pi))
        assert len(set(np.round(phases, 12))) == 4


def test_phases_accept_explicit_arrays_and_a_per_channel_mapping(params1030):
    spec = _story_spec(mx=2, my=3)
    explicit = [0.25, 1.5]
    wfs = synthesize(spec, params1030, phases={"Bx": explicit, "By": "zero"})
    np.testing.assert_allclose([t.phase0 for t in wfs.channels["Bx"].tones], explicit, rtol=1e-15)
    np.testing.assert_allclose([t.phase0 for t in wfs.channels["By"].tones], 0.0, atol=0.0)

    with pytest.raises(ValueError, match="one entry per tone"):
        synthesize(spec, params1030, phases=explicit)  # 2 phases, but By has 3 tones
    with pytest.raises(ValueError, match="phases must be one of"):
        synthesize(spec, params1030, phases="quadratic")


def test_synthesize_validates_its_inputs(params1030):
    spec = _story_spec()
    with pytest.raises(TypeError, match="needs a TrajectorySpec"):
        synthesize(STORY, params1030)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="needs an AODLParams"):
        synthesize(spec, params1030.optics)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="t_pad must be finite and non-negative"):
        synthesize(spec, params1030, t_pad=-1.0 * us)

    mismatched = replace(
        params1030,
        channels={
            **params1030.channels,
            "By": replace(params1030.channels["By"], sound_speed=600.0),
        },
    )
    with pytest.raises(ValueError, match="one common sound speed"):
        synthesize(spec, mismatched)


def test_max_z_integral_is_the_eq_1_ceiling(params1030):
    r"""``|int Z dt| <= 2 lens_scale min(f_hi - f_c, f_c - f_lo)``: 10 µm for 206 µs.

    Eq. S19 starts the drive at the carrier, so only the headroom on *one side* of
    ``f_center`` is reachable — half of ``docs/PLAN.md`` §1.5's 412 µs, which assumes the
    chirp is free to sweep the whole band from edge to edge.
    """
    ceiling = max_z_integral(params1030)
    assert ceiling == pytest.approx(2.0 * params1030.lens_scale * 10.0 * MHz, rel=1e-15)
    assert ceiling == pytest.approx(params1030.lens_scale * 20.0 * MHz, rel=1e-15)  # centred
    assert ceiling / (10.0 * um) == pytest.approx(206.0 * us, rel=1e-3)

    # an off-centre carrier is limited by its *tightest* side
    lopsided = replace(
        params1030,
        channels={
            name: replace(aod, f_center=95.0 * MHz) for name, aod in params1030.channels.items()
        },
    )
    assert max_z_integral(lopsided) == pytest.approx(
        2.0 * params1030.lens_scale * 5.0 * MHz, rel=1e-15
    )
