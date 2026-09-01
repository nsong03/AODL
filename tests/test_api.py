r"""The product front door: :func:`aodl.api.plan_motion` and what it hands back (WO-17 §1).

Three asks, one call each, and the same four questions of every one of them:

* did it pick the right **mode** — plain Eq. S19 while the band allows it, fading-Shepard
  ladders (Eqs. S24-S28) when a sustained ``Z`` outgrows Eq. 1's budget;
* do the **report numbers** describe the waveform it actually built — band occupancy checked
  against an independent sweep of the tone laws, tone counts against the channels, the axial
  budget against the trajectory, the fade schedule against the ladders' own hand-overs;
* do the **outputs** work — the parametric NPZ round-trips, the sample render has the right
  shape, ``simulate`` puts the traps where the trajectory asked, and a movie renders;
* does the **summary** say the two things a lab reads first — which mode, and how close the
  tightest channel is to the edge of its band.

The three asks are ``docs/PLAN.md``'s own: a static 3x3, the *fast* 10x10 user story of M3
(``examples/04``, band-feasible at 25/30/25 µs) and the *unhurried* one of M4 (150/250/150 µs,
which Eq. S19 refuses).  ``mixing_order=1`` throughout, so the term census is one beam per tone
combination and the assertions are about the product path, not about intermodulation.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from aodl import plan_motion
from aodl.api import MotionPlan, PlanReport, band_usage, fade_schedule
from aodl.field.focal import FrameGrid
from aodl.params import CHANNELS
from aodl.trajectory.spec import ArraySpec, Hold, Lift, TrajectorySpec, Translate
from aodl.units import MHz, um, us
from aodl.waveform.shepard import FadeZoneEnvelope, ShepardConfig
from aodl.waveform.synthesis import max_z_integral, requested_z_integral
from aodl.waveform.tones import WaveformSet

#: (a) a static 3x3 array — nothing moves, so Eq. S19 has the whole band to itself.
STATIC = TrajectorySpec(array=ArraySpec(3, 3, 1.0 * MHz, 1.3 * MHz), moves=(Hold(60.0 * us),))

#: (b) the *fast* user story (``tests/test_integration_m3.py``): 10x10, lift, traverse, drop,
#: at the durations Eq. 1 leaves once the ladders and the lateral term have taken their share.
FAST_STORY = TrajectorySpec(
    array=ArraySpec(10, 10, 1.0 * MHz, 1.3 * MHz),
    moves=(
        Lift(10.0 * um, 25.0 * us),
        Translate(40.0 * um, 25.0 * um, 30.0 * us),
        Lift(-10.0 * um, 25.0 * us),
    ),
)

#: (c) the *unhurried* one (``tests/test_integration_m4.py``): the same move at a human pace,
#: which plain Eq. S19 refuses and the fading ladders carry.
SLOW_STORY = TrajectorySpec(
    array=ArraySpec(10, 10, 1.0 * MHz, 1.3 * MHz),
    moves=(
        Lift(10.0 * um, 150.0 * us),
        Translate(40.0 * um, 25.0 * um, 250.0 * us),
        Lift(-10.0 * um, 150.0 * us),
    ),
)


def _linear(params):
    """``params`` with every channel at ``mixing_order=1`` — one tone, one beam."""
    return replace(
        params,
        channels={name: replace(aod, mixing_order=1) for name, aod in params.channels.items()},
    )


def _swept_usage(wfs: WaveformSet, n: int = 20001) -> dict[str, tuple[float, float]]:
    """Absolute live-tone span per channel [Hz], from a brute-force sweep of the tone laws.

    Deliberately dumb — sample every tone on a dense grid, throw away the instants where its
    envelope is zero, take the extremes — so that :func:`aodl.api.band_usage`, which is sharp
    (it adds the exact switch instants and the frequency laws' own extrema), has something
    independent to be checked against.
    """
    grid = np.linspace(*wfs.t_span, n)
    out: dict[str, tuple[float, float]] = {}
    for name, cw in wfs.channels.items():
        f_center = wfs.params.channels[name].f_center
        lo, hi = math.inf, -math.inf
        for tone in cw.tones:
            live = np.asarray(tone.env.A(grid), dtype=np.float64) > 0.0
            if not np.any(live):
                continue
            f = f_center + np.asarray(tone.freq(grid), dtype=np.float64)[live]
            lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))
        out[name] = (lo, hi)
    return out


def _assert_report_describes(plan: MotionPlan) -> None:
    """The report's numbers are the waveform's numbers — checked, not assumed."""
    report = plan.report
    assert isinstance(report, PlanReport)
    assert report.tone_counts == {name: cw.n_tones for name, cw in plan.wfs.channels.items()}
    assert report.n_tones == plan.wfs.n_tones

    swept = _swept_usage(plan.wfs)
    for name in CHANNELS:
        lo, hi, margin = report.band_usage[name]
        sweep_lo, sweep_hi = swept[name]
        # the report is a superset of what a sweep can see (it lands on the switch instants
        # exactly), and by no more than the sweep's own time resolution can explain
        assert lo <= sweep_lo + 1.0 and hi >= sweep_hi - 1.0
        assert lo == pytest.approx(sweep_lo, abs=5.0 * 1e3)
        assert hi == pytest.approx(sweep_hi, abs=5.0 * 1e3)
        band_lo, band_hi = plan.params.channels[name].band
        assert margin == pytest.approx(min(band_hi - hi, lo - band_lo))
        assert margin > 0.0  # every one of these plans is meant to fit

    name, worst = report.worst_margin
    assert worst == min(usage[2] for usage in report.band_usage.values())
    assert report.band_usage[name][2] == worst

    requested, ceiling = report.z_budget
    assert requested == pytest.approx(requested_z_integral(plan.spec, plan.params))
    if report.mode == "shepard":
        assert ceiling == math.inf
        assert requested > max_z_integral(plan.params)  # ... which is why it is in that mode
    else:
        assert ceiling == max_z_integral(plan.params)
        assert requested <= ceiling


# ================================================================== the three asks


def test_a_static_array_is_planned_with_plain_eq_s19(params1030):
    """Nothing moves, so ``f_Z = 0``: one tone per A channel, the ladder on B, no fades."""
    plan = plan_motion(STATIC, _linear(params1030))

    assert plan.report.mode == "s19"
    assert plan.report.tone_counts == {"Ax": 1, "Bx": 3, "Ay": 1, "By": 3}
    assert plan.report.z_budget[0] == 0.0
    assert plan.report.fade_events == []
    _assert_report_describes(plan)

    # the ladder is where Eq. S18 puts it: +-delta_f about the carrier, nothing else lit
    lo, hi, _ = plan.report.band_usage["Bx"]
    f_center = plan.params.channels["Bx"].f_center
    assert hi - f_center == pytest.approx(1.0 * MHz, rel=1e-9)
    assert f_center - lo == pytest.approx(1.0 * MHz, rel=1e-9)


def test_the_fast_user_story_still_fits_the_band_as_eq_s19(params1030):
    """25/30/25 µs is what Eq. 1 leaves for a 10x10 array: ``shepard="auto"`` does not fire."""
    plan = plan_motion(FAST_STORY, _linear(params1030))

    assert plan.report.mode == "s19"
    assert "plain Eq. S19 fits the band" in plan.report.description
    assert plan.report.tone_counts == {"Ax": 1, "Bx": 10, "Ay": 1, "By": 10}
    assert not any(
        isinstance(tone.env, FadeZoneEnvelope)
        for cw in plan.wfs.channels.values()
        for tone in cw.tones
    )
    _assert_report_describes(plan)

    # It fits, but the *axial* budget is not what makes it tight: f_Z spends only a quarter of
    # Eq. 1's bare ceiling, while the 10-tone ladder (+-4.5 MHz) and the lateral term spend the
    # rest of the band — which is why the tightest channel has a fraction of a megahertz left,
    # and why the same move at a human pace has to become a Shepard drive.
    requested, ceiling = plan.report.z_budget
    assert 0.2 < requested / ceiling < 1.0
    _, worst = plan.report.worst_margin
    assert 0.0 < worst < 1.0 * MHz


def test_the_unhurried_user_story_switches_to_fading_shepard(params1030):
    """The same move at a human pace outgrows Eq. 1, and the front door notices by itself."""
    params = _linear(params1030)
    plan = plan_motion(SLOW_STORY, params)

    assert plan.report.mode == "shepard"
    assert "plain Eq. S19 refused" in plan.report.description
    assert all(count > 10 for count in plan.report.tone_counts.values())  # ladders, not tones
    _assert_report_describes(plan)

    # insisting on Eq. S19 gets the refusal instead of the fallback
    with pytest.raises(ValueError, match="leaves its usable band"):
        plan_motion(SLOW_STORY, params, shepard=None)

    # the hand-over schedule is the ladders' own, one entry per rung crossing
    assert plan.report.fade_events == fade_schedule(plan.wfs, params)
    assert plan.report.fade_events
    for event in plan.report.fade_events:
        assert event.axis in ("x", "y")
        assert plan.wfs.t_span[0] <= event.time <= plan.wfs.t_span[1]
    spacing = {"x": 1.0 * MHz, "y": 1.3 * MHz}
    for axis in ("x", "y"):
        shadows = {e.shadow for e in plan.report.fade_events if e.axis == axis}
        assert shadows == {params.deflection_scale * spacing[axis]}
        # one hand-over per delta_f of axial walk, which is the scheme's whole bookkeeping
        walk = requested_z_integral(SLOW_STORY, params) / (2.0 * params.lens_scale)
        count = sum(1 for e in plan.report.fade_events if e.axis == axis)
        assert count == pytest.approx(walk / spacing[axis], abs=1.5)


# ============================================================== the report's own surface


def test_the_summary_names_the_mode_and_the_worst_band_margin(params1030):
    """The two numbers a lab reads first are in the block, spelled the way it prints them."""
    for spec, mode in ((STATIC, "s19"), (SLOW_STORY, "shepard")):
        plan = plan_motion(spec, _linear(params1030))
        text = plan.report.summary()
        assert f"mode: {mode}" in text
        name, worst = plan.report.worst_margin
        assert f"worst band margin {worst / MHz:+.4f} MHz on {name!r}" in text
        assert "band usage" in text and "z budget" in text
        for channel in CHANNELS:
            assert channel in text
        assert plan.summary() == text
        assert str(plan.report) == text


def test_the_report_notes_the_caveats_that_apply_to_this_drive(params1030):
    """Notes are per-drive: the retardation lag always, the Shepard caveats only when fading."""
    params = _linear(params1030)
    plain = plan_motion(STATIC, params).report
    assert any("lags the drive by tau/2" in note for note in plain.notes)
    assert not any("shadow tweezers" in note for note in plain.notes)

    fading = plan_motion(SLOW_STORY, params).report
    assert any("shadow tweezers" in note for note in fading.notes)
    assert any("x: 10 + 1 columns at every instant" in note for note in fading.notes)
    assert any("y: 10 + 1 rows at every instant" in note for note in fading.notes)
    assert any("comb offset" in note for note in fading.notes)
    assert any("-40 dB" in note for note in fading.notes)

    ramped = plan_motion(SLOW_STORY, params, switch_ramp=3.0 * us).report
    assert not any("-40 dB" in note for note in ramped.notes)
    assert any("switch_ramp" in note for note in ramped.notes)

    compensated = plan_motion(STATIC, params, retard_compensate=True).report
    assert any("retard_compensate=True" in note for note in compensated.notes)


def test_the_report_figure_draws_both_panels(params1030):
    """A band-usage panel and a tone-track panel, on one figure, without touching pyplot."""
    plan = plan_motion(STATIC, _linear(params1030))
    assert plan.report.wfs is plan.wfs  # the report can draw itself, no arguments needed
    fig = plan.report.figure(samples=64)
    assert len(fig.axes) == len(plan.figure(samples=64).axes) == 2
    tracks, usage = fig.axes
    assert "tone" in tracks.get_title()
    assert "band" in usage.get_title()
    assert [label.get_text() for label in usage.get_yticklabels()] == list(CHANNELS)
    assert tracks.get_ylabel() == "tone frequency [MHz]"


# ===================================================================== the deliverables


def test_the_plan_saves_simulates_and_films(params1030, tmp_path):
    """save -> load -> simulate -> movie, on the small static array: the notebook's own path."""
    params = _linear(params1030)
    plan = plan_motion(STATIC, params)
    tau = params.channels["Ax"].transit_time

    path = plan.save(tmp_path / "static.npz")
    assert path.exists()
    loaded = WaveformSet.load(path)
    assert loaded.n_tones == plan.wfs.n_tones
    assert loaded.description == plan.wfs.description

    samples = plan.render_samples(rate=50.0 * MHz)
    span = plan.wfs.t_span[1] - plan.wfs.t_span[0]
    assert set(samples) == set(CHANNELS)
    assert all(a.shape == (int(round(span * 50.0 * MHz)) + 1,) for a in samples.values())

    # the default grid is ~40 frames over [tau, T + tau/2], every one of them lit
    result = plan.simulate()
    assert result.n_frames == 40
    assert result.times[0] == pytest.approx(tau)
    assert result.times[-1] == pytest.approx(STATIC.duration + 0.5 * tau)
    assert all(frame for frame in result.metrics)
    table = result.spot_table()
    assert "power_coherent" in table
    # a static 3x3 with unequal spacings: nine traps on the Eq. S18 lattice, standing still
    pitch_x, pitch_y = STATIC.array.pitch(params)
    assert len(result.metrics[0]) == 9
    xs = sorted(m.x for m in result.metrics[0])
    assert xs[0] == pytest.approx(-pitch_x, abs=0.01 * params.optics.waist0)
    assert xs[-1] == pytest.approx(pitch_x, abs=0.01 * params.optics.waist0)
    assert float(np.max(np.abs(table["z_lab"]))) < 0.01 * params.optics.rayleigh

    frames = np.linspace(tau, STATIC.duration, 8)
    grid = FrameGrid(-2.0 * pitch_x, 2.0 * pitch_x, 48, -2.0 * pitch_y, 2.0 * pitch_y, 48)
    movie = plan.movie(tmp_path / "static.mp4", times=frames, grid=grid, xz_shape=(32, 24), dpi=60)
    assert movie.exists() and movie.stat().st_size > 0


def test_plan_motion_defaults_to_the_product_hardware_and_to_auto(params1030):
    """No ``params`` means ``default_1030``; no ``shepard`` means "decide for me"."""
    plan = plan_motion(STATIC)
    assert plan.params.optics.wavelength == pytest.approx(1030e-9)
    assert plan.spec is STATIC
    assert plan.report.mode == "s19"

    # ... and an explicit config is honoured, fades and all
    forced = plan_motion(STATIC, _linear(params1030), shepard=ShepardConfig(1.0 * MHz, 1.3 * MHz))
    assert forced.report.mode == "shepard"
    assert forced.report.fade_events == []  # a static array never walks, so it never hands over


def test_plan_motion_says_what_it_will_not_do(params1030):
    """Argument errors surface as themselves, not as an AttributeError three layers down."""
    with pytest.raises(TypeError, match="needs a TrajectorySpec"):
        plan_motion(STATIC.array, params1030)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        plan_motion(STATIC, params1030, nonsense=1)


def test_band_usage_ignores_a_rung_that_is_never_driven(params1030):
    """A tone whose envelope is identically zero has no occupancy, however far its law runs.

    That is the whole reason the report measures *live* tones: a fading-Shepard ladder's outer
    rungs carry frequency laws tens of megahertz outside the band and are never launched.
    """
    params = _linear(params1030)
    plan = plan_motion(SLOW_STORY, params)
    usage = band_usage(plan.wfs)
    for name in CHANNELS:
        aod = params.channels[name]
        lo, hi, _ = usage[name]
        assert aod.band[0] < lo < hi < aod.band[1]
        # the *programmed* laws leave the band by far more than the live ones do
        grid = np.linspace(*plan.wfs.t_span, 2001)
        programmed = max(
            float(np.max(np.abs(np.asarray(tone.freq(grid), dtype=np.float64))))
            for tone in plan.wfs.channels[name].tones
        )
        assert programmed > 2.0 * max(hi - aod.f_center, aod.f_center - lo)
