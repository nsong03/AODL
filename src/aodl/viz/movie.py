r"""Movie renderer: a :class:`~aodl.engine.SimResult` becomes an mp4 (or GIF).

The default view is the one fixed by ``docs/ARCHITECTURE.md`` §3 (decision 3): a
**focus-tracked planar view**.  Each frame the XY plane is placed at the power-weighted
best-focus lab Z of the scene (``mode="tracked"``), every frequency group is tinted by *its
own* Z through :data:`aodl.viz.style.Z_CMAP`, and the tinted groups are composited additively
on black.  ``mode="fixed"`` instead pins the plane at the static lab focal plane ``Z = 0``,
the camera a real experiment would have — spots blur and dim as they leave it, which is
exactly how a single-AOD astigmat announces itself (M1).

Two panels frame the view: an **XZ slice** through the tweezers' row, tinted row by row on the
same hue scale so colour and vertical position agree, with the tracked plane marked; and an
optional **drive strip** showing, per channel, the tone frequency the beam centre sees at each
frame time — i.e. ``f_center + f(t - tau/2)``, retardation included — with a cursor on now.

Brightness is honest: one global intensity maximum is measured over the whole movie in a first
pass and reused for every frame (``docs/PLAN.md`` §1.3 — a fading or defocusing spot must
*look* dimmer).  Frames are never all held in memory; the physics pass simply runs twice.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, NamedTuple, cast

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..engine import SimResult
from ..field.focal import FrameGrid
from ..units import MHz, um, us
from .style import (
    CHANNEL_COLORS,
    DARK_STYLE,
    GAMMA,
    composite,
    tint_by_row,
    tint_cmap,
    z_color,
    z_max_from,
)

Float = NDArray[np.float64]

#: Grid margin around the spot trajectories, in intensity 1/e^2 radii.
GRID_WAISTS = 8.0

#: Pixels along the longer side of an auto-generated :class:`~aodl.field.focal.FrameGrid`.
GRID_LONG_SIDE = 512

#: Default samples along X and Z in the XZ side panel (see ``render_movie(xz_shape=...)``).
XZ_NX, XZ_NZ = 224, 160

#: Extra Z shown in the XZ panel beyond the light's own axial extent, in Rayleigh ranges.
XZ_Z_HEADROOM = 1.0

#: Longest side of the main axes [inch]; the shorter one follows the data aspect ratio.
AXES_LONG_SIDE = 4.7


def auto_grid(
    result: SimResult,
    waists: float = GRID_WAISTS,
    long_side: int = GRID_LONG_SIDE,
) -> FrameGrid:
    """A :class:`~aodl.field.focal.FrameGrid` covering every spot of the run.

    The bounding box of all spot centres is padded by ``waists`` times the largest 1/e^2 radius
    the run reaches (never less than the diffraction-limited ``waist0``), and sampled with
    ``long_side`` pixels along its longer side, the shorter one keeping square pixels.
    """
    table = result.spot_table()
    waist0 = result.optics.waist0
    if table["x"].size == 0:
        half = waists * waist0
        return FrameGrid(-half, half, long_side, -half, half, long_side)

    radius = max(float(np.max(table["wx"])), float(np.max(table["wy"])), waist0)
    margin = float(waists) * radius
    x0, x1 = float(np.min(table["x"])) - margin, float(np.max(table["x"])) + margin
    y0, y1 = float(np.min(table["y"])) - margin, float(np.max(table["y"])) + margin

    # Sample counts follow the *pitch*, which is span/(n - 1): matching n to the span ratio
    # directly would leave the pixels visibly non-square on a strongly elongated field.
    span_x, span_y = x1 - x0, y1 - y0
    long_px = max(int(long_side), 8)
    short_px = max(int(round(1 + (long_px - 1) * min(span_x, span_y) / max(span_x, span_y))), 8)
    nx, ny = (long_px, short_px) if span_x >= span_y else (short_px, long_px)
    return FrameGrid(x0=x0, x1=x1, nx=nx, y0=y0, y1=y1, ny=ny)


def _panel_half_range(result: SimResult, z_max: float) -> float:
    """Z half-range of the XZ panel [m]: the light's own axial extent plus some headroom.

    An astigmatic spot has no single focus — its two line foci sit at ``z_lab -+ delta_f/2``
    (Table I) — so the panel has to cover that whole interval, widened by a Rayleigh range,
    or the excursion the panel exists to show would run off the top of it.
    """
    table = result.spot_table()
    if table["z_lab"].size == 0:
        return max(z_max, result.optics.rayleigh)
    reach = float(np.max(np.abs(table["z_lab"]) + 0.5 * np.abs(table["delta_f"])))
    return max(reach + XZ_Z_HEADROOM * result.optics.rayleigh, z_max)


def _planes(result: SimResult, mode: str) -> Float:
    """Evaluation plane of every frame [m] for ``mode`` (``"tracked"`` or ``"fixed"``)."""
    if mode == "tracked":
        return result.tracked_z()
    if mode == "fixed":
        return np.zeros(result.n_frames, dtype=np.float64)
    raise ValueError(f"mode must be 'tracked' or 'fixed', got {mode!r}")


def _group_colors(result: SimResult, i: int, z_max: float) -> list[Float]:
    """Tint of each frequency group in frame ``i``, from its own best-focus lab Z."""
    return [z_color(m.z_lab, z_max) for m in result.metrics[i]]


def _xy_layers(
    result: SimResult, i: int, grid: FrameGrid, plane: float, z_max: float
) -> list[tuple[Float, Float]]:
    """``(intensity, rgb)`` layers of frame ``i`` — one per frequency group."""
    frames = result.group_frames(i, grid, plane)
    colors = _group_colors(result, i, z_max)
    return list(zip(frames, colors, strict=True))


class _XZPanel(NamedTuple):
    """The XZ side panel's sampling grids, row colours and live artists.

    Bundled because they exist together or not at all (``xz_panel=False`` drops the lot), so
    one ``panel is not None`` test says the panel is being drawn — and tells a type checker
    that all six are present.
    """

    x: Float
    z: Float
    row_colors: Float
    image: Any
    track: Any
    label: Any


def _blank(grid: FrameGrid) -> Float:
    return np.zeros((grid.ny, grid.nx), dtype=np.float64)


def _rows(result: SimResult, xz_row_y: float | None) -> Float:
    """Y row the XZ slice cuts through, per frame [m] (tracking the spots when unset)."""
    if xz_row_y is not None:
        return np.full(result.n_frames, float(xz_row_y))
    return np.array([result.spot_row(i) for i in range(result.n_frames)], dtype=np.float64)


def _peaks(
    result: SimResult,
    grid: FrameGrid,
    planes: Float,
    z_max: float,
    x_panel: Float | None,
    z_panel: Float | None,
    rows: Float,
) -> tuple[float, float]:
    """First pass: the global XY and XZ intensity maxima over the whole movie."""
    xy_peak = 0.0
    xz_peak = 0.0
    for i in range(result.n_frames):
        layers = _xy_layers(result, i, grid, float(planes[i]), z_max)
        total = sum((frame for frame, _ in layers), _blank(grid))
        xy_peak = max(xy_peak, float(total.max()))
        if x_panel is not None and z_panel is not None:
            panel = result.slice_xz(i, x_panel, z_panel, float(rows[i]))
            xz_peak = max(xz_peak, float(panel.max()))
    return xy_peak, xz_peak


# ------------------------------------------------------------------------------- layout


def _figure(
    grid: FrameGrid, dpi: int, want_xz: bool, want_strip: bool
) -> tuple[Figure, FigureCanvasAgg, dict[str, Any]]:
    """Build the dark figure, its Agg canvas and its axes; sizes follow the data aspect ratio.

    The canvas is returned rather than reached for through ``fig.canvas`` because only the
    Agg backend exposes ``buffer_rgba()``, which is how frames reach the movie writer.
    """
    span_x, span_y = grid.x1 - grid.x0, grid.y1 - grid.y0
    if span_y >= span_x:
        main_h = AXES_LONG_SIDE
        main_w = max(AXES_LONG_SIDE * span_x / span_y, 1.5)
    else:
        main_w = AXES_LONG_SIDE
        main_h = max(AXES_LONG_SIDE * span_y / span_x, 1.5)

    left, right, top, bottom = 0.62, 0.10, 0.40, 0.52
    gap, xz_w = 0.60, 1.70
    cbar_gap, cbar_w, cbar_label = 0.16, 0.15, 0.82
    strip_gap, strip_h = 0.62, 1.05

    side = (gap + xz_w) if want_xz else 0.0
    fig_w = left + main_w + side + cbar_gap + cbar_w + cbar_label + right
    fig_h = bottom + ((strip_h + strip_gap) if want_strip else 0.0) + main_h + top

    # Even pixel dimensions: libx264/yuv420p refuses odd frame sizes.
    fig_w = 2.0 * round(fig_w * dpi / 2.0) / dpi
    fig_h = 2.0 * round(fig_h * dpi / 2.0) / dpi

    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    def rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x / fig_w, y / fig_h, w / fig_w, h / fig_h)

    main_y = fig_h - top - main_h
    axes: dict[str, Any] = {"main": fig.add_axes(rect(left, main_y, main_w, main_h))}
    if want_xz:
        axes["xz"] = fig.add_axes(rect(left + main_w + gap, main_y, xz_w, main_h))
    axes["cbar"] = fig.add_axes(rect(left + main_w + side + cbar_gap, main_y, cbar_w, main_h))
    if want_strip:
        axes["strip"] = fig.add_axes(rect(left, bottom, main_w + side, strip_h))
    return fig, canvas, axes


def _drive_strip(ax: Any, result: SimResult) -> None:
    """Per-channel tone frequency seen at the beam centre, over the movie's time axis.

    Plots ``f_center + f(t - tau/2)`` against frame time ``t`` for every tone, with segment
    opacity following the envelope, so fades (M4's Shepard ladders) show up as fading lines.
    This is the paper's Fig. 3/4 spectrogram read analytically — no FFT (``CLAUDE.md``).

    A segment whose envelope is *identically zero* is not drawn at all, rather than drawn at
    the floor opacity, and the vertical range follows the *live* excursion only: such a tone
    is not launched by the transducer, and a fading-Shepard ladder spends most of its rungs'
    programmed span there — at 90-odd rungs the floor alone would paint a solid band across
    the panel and stretch it over three times the bandwidth the drive actually occupies.
    """
    t = np.linspace(float(result.times[0]), float(result.times[-1]), 400)
    live_lo, live_hi = np.inf, -np.inf
    for name in result.channels:
        aod = result.params.channels[name]
        t_c = t - 0.5 * aod.transit_time
        color = CHANNEL_COLORS.get(name, "#d7dde5")
        for tone in result.wfs.channels[name].tones:
            f = (aod.f_center + np.asarray(tone.f(t_c), dtype=np.float64)) / MHz
            env = np.clip(np.asarray(tone.env.A(t_c), dtype=np.float64), 0.0, 1.0)
            points = np.column_stack([t / us, f])
            segments = np.stack([points[:-1], points[1:]], axis=1)
            mean_env = 0.5 * (env[:-1] + env[1:])
            alpha = np.where(mean_env > 0.0, 0.25 + 0.75 * mean_env, 0.0)
            ax.add_collection(
                LineCollection(list(segments), colors=color, linewidths=1.6, alpha=alpha)
            )
            if np.any(env > 0.0):
                live_lo = min(live_lo, float(np.min(f[env > 0.0])))
                live_hi = max(live_hi, float(np.max(f[env > 0.0])))
        ax.plot([], [], color=color, lw=1.6, label=name)
    ax.set_xlim(t[0] / us, t[-1] / us)
    ax.set_ylabel("drive [MHz]")
    ax.set_xlabel("t [µs]")
    if live_hi > live_lo:
        margin = 0.08 * (live_hi - live_lo)
        ax.set_ylim(live_lo - margin, live_hi + margin)
    else:  # a single constant tone, or nothing live at all: let matplotlib decide
        ax.autoscale_view()
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncols=4, labelcolor="#d7dde5")


def _open_writer(path: Path, fps: int) -> tuple[Any, Path]:
    """Open an imageio writer for ``path``; fall back to GIF when ffmpeg is unavailable."""
    import imageio.v2 as imageio

    if path.suffix.lower() != ".gif":
        # imageio resolves `format` by name at run time; its annotation asks for a Format
        # object, so the options go through an untyped mapping rather than a `type: ignore`.
        ffmpeg: dict[str, Any] = {
            "format": "FFMPEG",
            "mode": "I",
            "fps": fps,
            "codec": "libx264",
            "quality": 8,
            "macro_block_size": 1,
            "pixelformat": "yuv420p",
        }
        try:
            writer = imageio.get_writer(path, **ffmpeg)
            return writer, path
        except Exception as exc:  # pragma: no cover - depends on the ffmpeg install
            path = path.with_suffix(".gif")
            warnings.warn(
                f"ffmpeg encoding unavailable ({exc}); writing {path.name} instead",
                RuntimeWarning,
                stacklevel=3,
            )
    for kwargs in ({"duration": 1000.0 / fps, "loop": 0}, {"fps": fps}, {}):
        try:
            return imageio.get_writer(path, mode="I", **kwargs), path
        except TypeError:  # pragma: no cover - imageio version differences
            continue
    raise RuntimeError(f"could not open an image writer for {path}")  # pragma: no cover


def render_movie(
    result: SimResult,
    path: str | Path,
    grid: FrameGrid | None = None,
    mode: str = "tracked",
    fps: int = 25,
    xz_panel: bool = True,
    xz_row_y: float | None = None,
    xz_shape: tuple[int, int] | None = None,
    spectrogram_panel: bool = False,
    dpi: int = 110,
    z_max: float | None = None,
    gamma: float = GAMMA,
) -> Path:
    """Render ``result`` to a movie and return the path actually written.

    Parameters
    ----------
    result:
        A simulated run (:func:`aodl.engine.simulate`).
    path:
        Output file.  ``.mp4`` is encoded with ffmpeg via ``imageio-ffmpeg``; ``.gif`` (or any
        path, if ffmpeg is missing) uses the GIF writer, in which case the returned path
        carries the ``.gif`` suffix.
    grid:
        Image-plane sampling grid; ``None`` calls :func:`auto_grid`.
    mode:
        ``"tracked"`` (default) puts the XY plane at each frame's power-weighted best focus;
        ``"fixed"`` pins it at the static lab focal plane ``Z = 0``.
    fps:
        Frames per second of the encoded movie.
    xz_panel:
        Draw the XZ slice side panel.
    xz_row_y:
        Y row the XZ slice cuts through [m].  ``None`` (default) tracks the frame's own
        brightest row (:meth:`aodl.engine.SimResult.spot_row`), so the panel follows the
        tweezers instead of watching an empty row when the array moves along y.
    xz_shape:
        ``(nx, nz)`` samples of the XZ panel; ``None`` uses ``(XZ_NX, XZ_NZ)``.  The panel is
        the one part of a frame that cannot be patched — a spot sweeps *through* focus along
        its Z axis — so its cost is ``nx * nz`` per frequency group, and a hundred-trap scene
        pays that a hundred times.  Trading panel resolution for frames is how a large array
        stays inside a render budget (``examples/04``).
    spectrogram_panel:
        Draw the per-channel drive strip under the view.
    dpi:
        Rendering resolution.
    z_max:
        Half-range of the Z colour scale [m]; ``None`` uses ``max |z_lab|`` over the run,
        floored at :data:`aodl.viz.style.Z_FLOOR`.
    gamma:
        Display gamma for intensity (:data:`aodl.viz.style.GAMMA`).

    Notes
    -----
    The scene is evaluated twice — once to find the movie-wide intensity maximum, once to
    render — so memory stays flat regardless of frame count, at the cost of repeating the
    (cheap) closed-form field evaluation.
    """
    if result.n_frames == 0:  # pragma: no cover - simulate() already refuses this
        raise ValueError("cannot render a movie of an empty run")
    planes = _planes(result, mode)  # validates `mode` before anything touches the disk
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grid = auto_grid(result) if grid is None else grid
    table = result.spot_table()
    z_max = z_max_from(table["z_lab"]) if z_max is None else float(z_max)
    rows = _rows(result, xz_row_y)

    shape = (XZ_NX, XZ_NZ) if xz_shape is None else xz_shape
    panel_nx, panel_nz = int(shape[0]), int(shape[1])
    if panel_nx < 2 or panel_nz < 2:
        raise ValueError(f"xz_shape needs at least 2 samples per axis, got {xz_shape!r}")
    x_panel = np.linspace(grid.x0, grid.x1, panel_nx) if xz_panel else None
    half = _panel_half_range(result, z_max)
    z_panel = np.linspace(-half, half, panel_nz) if xz_panel else None
    row_colors = z_color(z_panel, z_max) if z_panel is not None else None

    xy_peak, xz_peak = _peaks(result, grid, planes, z_max, x_panel, z_panel, rows)

    # DARK_STYLE is a plain {str: Any} rcParams patch; matplotlib types the argument with the
    # Literal set of every rcParam name, which no ordinary dict literal can satisfy.
    with mpl.rc_context(cast(Any, DARK_STYLE)):
        fig, canvas, axes = _figure(grid, dpi, xz_panel, spectrogram_panel)
        ax = axes["main"]
        extent = (grid.x0 / um, grid.x1 / um, grid.y0 / um, grid.y1 / um)
        im = ax.imshow(np.zeros((grid.ny, grid.nx, 3)), extent=extent, aspect="equal")
        ax.set_xlabel("X [µm]")
        ax.set_ylabel("Y [µm]")
        ax.set_title(
            "XY @ tracked focus" if mode == "tracked" else "XY @ fixed plane Z = 0", fontsize=9
        )
        clock = ax.text(
            0.04,
            0.985,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            color="#f2f5f8",
        )
        depth = ax.text(
            0.04,
            0.015,
            "",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8.5,
            color="#f2f5f8",
        )

        panel: _XZPanel | None = None
        if x_panel is not None and z_panel is not None and row_colors is not None:
            ax_xz = axes["xz"]
            im_xz = ax_xz.imshow(
                np.zeros((panel_nz, panel_nx, 3)),
                extent=(grid.x0 / um, grid.x1 / um, z_panel[0] / um, z_panel[-1] / um),
                aspect="auto",
            )
            (track_line,) = ax_xz.plot(
                [grid.x0 / um, grid.x1 / um],
                [0.0, 0.0],
                color="#f2f5f8",
                lw=0.8,
                ls="--",
                alpha=0.7,
            )
            row_text = ax_xz.text(
                0.04,
                0.985,
                "",
                transform=ax_xz.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                color="#f2f5f8",
            )
            ax_xz.set_xlabel("X [µm]")
            ax_xz.set_ylabel("Z lab [µm]")
            ax_xz.set_title("XZ slice (X, Z to different scales)", fontsize=9)
            panel = _XZPanel(x_panel, z_panel, row_colors, im_xz, track_line, row_text)

        bar = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(-z_max / um, z_max / um), cmap=tint_cmap()
            ),
            cax=axes["cbar"],
        )
        bar.set_label("spot Z lab [µm]", fontsize=9)
        bar.outline.set_edgecolor("#5a6472")

        cursor = None
        if spectrogram_panel:
            _drive_strip(axes["strip"], result)
            cursor = axes["strip"].axvline(
                float(result.times[0]) / us, color="#f2f5f8", lw=0.9, alpha=0.8
            )

        writer, out = _open_writer(out, fps)
        try:
            for i in range(result.n_frames):
                plane = float(planes[i])
                layers = _xy_layers(result, i, grid, plane, z_max)
                rgb = (
                    composite(layers, xy_peak, gamma) if layers else np.zeros((grid.ny, grid.nx, 3))
                )
                im.set_data(rgb)
                clock.set_text(f"t = {result.times[i] / us:.2f} µs")
                depth.set_text(f"plane Z = {plane / um:+.2f} µm")
                if panel is not None:
                    xz = result.slice_xz(i, panel.x, panel.z, float(rows[i]))
                    panel.image.set_data(tint_by_row(xz, panel.row_colors, xz_peak, gamma))
                    panel.track.set_ydata([plane / um, plane / um])
                    panel.label.set_text(f"y = {rows[i] / um:+.1f} µm")
                if cursor is not None:
                    cursor.set_xdata([result.times[i] / us, result.times[i] / us])
                canvas.draw()
                writer.append_data(np.asarray(canvas.buffer_rgba())[..., :3])
        finally:
            writer.close()
    return out


__all__ = ["GRID_LONG_SIDE", "GRID_WAISTS", "auto_grid", "render_movie"]
