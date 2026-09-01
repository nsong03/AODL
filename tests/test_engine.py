r"""Engine assembly: :func:`aodl.engine.simulate`, :class:`~aodl.engine.SimResult`, viz style.

These are the *wiring* tests — that the frame loop, the lazy field evaluators and the movie
renderer agree with the layers underneath them.  The physics itself is pinned by
``test_device_single_aod.py`` (device), ``test_focal.py`` / ``test_measure.py`` (field) and
``test_integration_m1.py`` (the end-to-end M1 acceptance).
"""

from __future__ import annotations

import numpy as np
import pytest

from aodl.device.aodl import build_terms
from aodl.engine import SPOT_TABLE_KEYS, simulate
from aodl.field.focal import FrameGrid, intensity_frame, intensity_slice_xz
from aodl.field.measure import measure, track_z
from aodl.poly import PiecewisePoly
from aodl.trajectory import ramps
from aodl.units import MHz, um, us
from aodl.viz import style
from aodl.viz.movie import auto_grid, render_movie
from aodl.waveform.tones import ChannelWaveform, ToneTrack, WaveformSet

DETUNING = 3.0 * MHz


def _static_run(params, t_end_factor: float = 8.0):
    """Single Ay tone at ``DETUNING``, programmed long enough for any frame below."""
    tau = params.channels["Ay"].transit_time
    tone = ToneTrack(freq=PiecewisePoly.constant(DETUNING, 0.0, t_end_factor * tau))
    return WaveformSet({"Ay": ChannelWaveform((tone,))}, params), tau


def _sweep_run(params, span: float = 5.0 * MHz, duration: float = 100.0 * us):
    """Min-jerk Ay chirp ``0 -> span`` over ``duration``, held out past the last frame."""
    tau = params.channels["Ay"].transit_time
    tone = ToneTrack(freq=ramps.min_jerk(0.0, duration, 0.0, span))
    wfs = WaveformSet({"Ay": ChannelWaveform((tone,))}, params).with_hold_until(duration + tau)
    return wfs, tau


# ================================================================= simulate / SimResult


def test_static_run_is_stationary_across_frames(params1030) -> None:
    """Two frames of an unchanging drive: identical metrics, one group, Table I position."""
    wfs, tau = _static_run(params1030)
    result = simulate(wfs, [2.0 * tau, 3.0 * tau])

    assert result.n_frames == len(result) == 2
    assert result.channels == ("Ay",)
    assert result.params is params1030
    assert [len(m) for m in result.metrics] == [1, 1]

    first, second = result.metrics[0][0], result.metrics[1][0]
    assert first == second  # frozen dataclass equality: nothing drifts once the aperture is full
    assert first.y == pytest.approx(-params1030.deflection_scale * DETUNING, rel=1e-12)
    assert first.x == 0.0
    assert first.z_lab == pytest.approx(0.0, abs=1e-18)
    assert first.delta_f == pytest.approx(0.0, abs=1e-18)
    assert first.wx == pytest.approx(params1030.optics.waist0, rel=1e-12)
    assert first.wy == pytest.approx(params1030.optics.waist0, rel=1e-12)
    np.testing.assert_allclose(result.tracked_z(), 0.0, atol=1e-18)


def test_frame_matches_direct_intensity_frame(params1030) -> None:
    """The lazy frame is exactly ``field.focal.intensity_frame`` on the same terms."""
    wfs, tau = _sweep_run(params1030)
    times = np.linspace(0.0, 100.0 * us + tau, 9)
    result = simulate(wfs, times)
    optics = params1030.optics
    i = 5

    grid = FrameGrid(
        x0=-6.0 * optics.waist0,
        x1=6.0 * optics.waist0,
        nx=41,
        y0=result.metrics[i][0].y - 6.0 * optics.waist0,
        y1=result.metrics[i][0].y + 6.0 * optics.waist0,
        ny=37,
    )
    terms = build_terms(wfs, float(times[i]), ("Ay",))
    plane = track_z(measure(terms, optics))

    np.testing.assert_array_equal(
        result.frame(i, grid), intensity_frame(terms, optics, grid, plane)
    )
    np.testing.assert_array_equal(
        result.frame(i, grid, z_lab=0.0), intensity_frame(terms, optics, grid, 0.0)
    )
    assert result.plane(i) == pytest.approx(plane, rel=1e-15)
    assert result.frame(i, grid).shape == (grid.ny, grid.nx)

    # ... and the group layers a movie tints separately add back up to it exactly.
    layers = result.group_frames(i, grid)
    assert len(layers) == len(result.metrics[i]) == 1
    np.testing.assert_array_equal(sum(layers), result.frame(i, grid))

    x_axis = np.linspace(grid.x0, grid.x1, 13)
    z_axis = np.linspace(-4.0 * um, 4.0 * um, 11)
    y_row = result.spot_row(i)
    np.testing.assert_array_equal(
        result.slice_xz(i, x_axis, z_axis),
        intensity_slice_xz(terms, optics, x_axis, z_axis, y_row),
    )
    assert y_row == pytest.approx(result.metrics[i][0].y, rel=1e-12)
    assert result.slice_xz(i, x_axis, z_axis).shape == (z_axis.size, x_axis.size)


def test_spot_table_is_tidy(params1030) -> None:
    """One row per (frame, group), columns in :data:`SPOT_TABLE_KEYS`, values matching metrics."""
    wfs, tau = _sweep_run(params1030)
    times = np.linspace(0.0, 100.0 * us + tau, 7)
    result = simulate(wfs, times)
    table = result.spot_table()

    n_rows = sum(len(m) for m in result.metrics)
    assert n_rows == times.size  # one tweezer per frame in M1
    assert tuple(table) == SPOT_TABLE_KEYS
    for key, column in table.items():
        assert column.shape == (n_rows,), key

    np.testing.assert_allclose(table["time"], times, rtol=1e-15)
    np.testing.assert_array_equal(table["frame"], np.arange(times.size))
    np.testing.assert_array_equal(table["group"], np.zeros(times.size))
    np.testing.assert_allclose(table["y"], [m[0].y for m in result.metrics], rtol=1e-15)
    np.testing.assert_allclose(
        table["sigma_astig"], table["delta_f"] / params1030.optics.rayleigh, rtol=1e-12
    )
    np.testing.assert_allclose(table["z_lab"], result.tracked_z(), rtol=1e-12, atol=1e-18)


def test_simulate_accepts_a_scalar_time_and_rejects_nonsense(params1030) -> None:
    wfs, tau = _static_run(params1030)
    assert simulate(wfs, 2.0 * tau).n_frames == 1

    with pytest.raises(ValueError, match="at least one frame time"):
        simulate(wfs, [])
    with pytest.raises(ValueError, match="finite"):
        simulate(wfs, [np.nan])
    with pytest.raises(KeyError, match="not present"):
        simulate(wfs, [2.0 * tau], channels=("Bx",))
    with pytest.raises(IndexError):
        simulate(wfs, [2.0 * tau]).terms(3)


# ================================================= clamp-hold guard (WO-02 deviation report)


def test_frames_past_the_drive_demand_with_hold_until(params1030) -> None:
    """Past its domain a tone freezes its phase, so ``simulate`` refuses instead of faking it."""
    duration = 100.0 * us
    tau = params1030.channels["Ay"].transit_time
    tone = ToneTrack(freq=ramps.min_jerk(0.0, duration, 0.0, 5.0 * MHz))
    wfs = WaveformSet({"Ay": ChannelWaveform((tone,))}, params1030)

    # The last frame the bare waveform supports is t = duration + tau/2 (retardation buys
    # exactly half a transit); anything past that must be held.
    simulate(wfs, [duration + 0.5 * tau])
    with pytest.raises(ValueError, match="with_hold_until"):
        simulate(wfs, np.linspace(0.0, duration + tau, 5))

    held = wfs.with_hold_until(duration + tau)
    result = simulate(held, np.linspace(0.0, duration + tau, 5))
    assert result.n_frames == 5

    # Teeth: without the guard the frozen phase would still *look* fine, because the frequency
    # clamp-holds at its terminal value - the spot simply stops where it is.
    assert held.channels["Ay"].tones[0].f(duration + 0.5 * tau) == pytest.approx(
        5.0 * MHz, rel=1e-12
    )
    assert result.metrics[-1][0].y == pytest.approx(
        -params1030.deflection_scale * 5.0 * MHz, rel=1e-12
    )

    # Frames before the drive exists are rejected too (with_hold_until cannot help there).
    with pytest.raises(ValueError, match="before channel"):
        simulate(held, [-1.0 * us, 0.0])


def test_fill_transient_frames_are_allowed(params1030) -> None:
    """``t < tau/2`` puts ``t_c`` before the drive start - legal: the aperture is still filling."""
    wfs, tau = _static_run(params1030)
    result = simulate(wfs, [0.0, 0.2 * tau, 0.4 * tau])
    assert result.n_frames == 3
    powers = [m[0].power for m in result.metrics]
    assert powers[0] < powers[1] < powers[2]  # light builds up as the column enters the beam


# ============================================================================== viz style


def test_z_norm_is_symmetric_and_saturates() -> None:
    z_max = 5.0 * um
    assert style.z_norm(0.0, z_max) == pytest.approx(0.5)
    assert style.z_norm(z_max, z_max) == pytest.approx(1.0)
    assert style.z_norm(-z_max, z_max) == pytest.approx(0.0)
    assert style.z_norm(0.5 * z_max, z_max) + style.z_norm(-0.5 * z_max, z_max) == pytest.approx(
        1.0
    )
    assert style.z_norm(9.0 * z_max, z_max) == pytest.approx(1.0)  # clipped, never wrapped

    assert style.z_max_from([0.0, 0.0]) == pytest.approx(style.Z_FLOOR)
    assert style.z_max_from([1.0 * um, -7.0 * um]) == pytest.approx(7.0 * um)

    # Hue carries Z, luminance carries intensity: every tint is full-brightness.
    for z in (-z_max, -0.4 * z_max, 0.0, 0.6 * z_max, z_max):
        assert float(np.max(style.z_color(z, z_max))) == pytest.approx(1.0)
    np.testing.assert_allclose(style.z_color(0.0, z_max), 1.0, atol=0.05)  # white in-plane


def test_composite_adds_layers_through_the_gamma() -> None:
    red, blue = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    a = np.array([[1.0, 0.0], [0.0, 0.25]])
    b = np.array([[0.0, 1.0], [0.0, 0.25]])
    rgb = style.composite([(a, red), (b, blue)], i_max=1.0, gamma=0.5)

    assert rgb.shape == (2, 2, 3)
    np.testing.assert_allclose(rgb[0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rgb[0, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rgb[1, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rgb[1, 1], [0.5, 0.0, 0.5])  # 0.25**0.5 per layer
    assert style.composite([(a, red)], i_max=0.0).max() == 0.0

    rows = style.tint_by_row(np.ones((3, 4)), style.z_color([-1.0, 0.0, 1.0], 1.0), 1.0)
    assert rows.shape == (3, 4, 3)
    assert rows[0, 0, 2] > rows[0, 0, 0]  # bottom row reads blue (below the focal plane)
    assert rows[2, 0, 0] > rows[2, 0, 2]  # top row reads red


# ============================================================================ movie glue


def test_auto_grid_covers_the_trajectory(params1030) -> None:
    wfs, tau = _sweep_run(params1030)
    result = simulate(wfs, np.linspace(0.0, 100.0 * us + tau, 11))
    table = result.spot_table()
    grid = auto_grid(result, waists=8.0, long_side=128)

    assert max(grid.nx, grid.ny) == 128
    assert grid.y0 < table["y"].min() and grid.y1 > table["y"].max()
    assert grid.x0 < 0.0 < grid.x1
    margin = table["y"].min() - grid.y0
    assert margin == pytest.approx(8.0 * max(table["wx"].max(), table["wy"].max()), rel=1e-12)
    assert grid.dx == pytest.approx(grid.dy, rel=0.02)  # near-square pixels


def test_render_movie_writes_a_playable_file(params1030, tmp_path) -> None:
    """End-to-end render, both panels on, read back frame by frame."""
    imageio = pytest.importorskip("imageio.v2")
    wfs, tau = _sweep_run(params1030)
    result = simulate(wfs, np.linspace(0.0, 100.0 * us + tau, 4))
    grid = auto_grid(result, long_side=64)

    path = render_movie(
        result,
        tmp_path / "sweep.mp4",
        grid=grid,
        mode="fixed",
        fps=5,
        spectrogram_panel=True,
        dpi=60,
    )
    assert path.exists() and path.stat().st_size > 1000

    reader = imageio.get_reader(path)
    try:
        assert reader.count_frames() == result.n_frames
        first = np.asarray(reader.get_data(0))
    finally:
        reader.close()
    height, width = first.shape[:2]
    assert height % 2 == 0 and width % 2 == 0  # libx264 needs even dimensions
    assert first.max() > 30  # something was actually drawn

    with pytest.raises(ValueError, match="mode must be"):
        render_movie(result, tmp_path / "bad.mp4", grid=grid, mode="sideways")


def test_render_movie_gif_fallback(params1030, tmp_path) -> None:
    """A ``.gif`` path uses the GIF writer (the no-ffmpeg fallback target)."""
    pytest.importorskip("imageio.v2")
    wfs, tau = _static_run(params1030)
    result = simulate(wfs, [1.5 * tau, 2.0 * tau])
    grid = auto_grid(result, long_side=48)

    path = render_movie(result, tmp_path / "static.gif", grid=grid, fps=4, xz_panel=False, dpi=50)
    assert path.suffix == ".gif"
    assert path.stat().st_size > 500
