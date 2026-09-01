r"""Rendering conventions shared by every AODL figure and movie.

The house style is one idea: **hue carries Z, luminance carries intensity**
(``docs/ARCHITECTURE.md`` §3, decision 3).  A tweezer is drawn in the colour its own
out-of-plane position earns on a diverging map — blue below the focal plane, red above,
white at Z = 0 — and brightened by its intensity, so a movie frame reads as a depth map
without ever leaving the 2D view.

Two details keep that honest:

* the colour normalization :func:`z_norm` is **symmetric about Z = 0**, so "no colour" always
  means "in the static focal plane" and never drifts with the data range;
* intensities are composited **additively on black** through a gamma
  (:data:`GAMMA` = 0.7, i.e. faint light is lifted), and every frame of a movie shares one
  global ``i_max`` — a spot that dims because it is defocused or because its aperture is still
  filling *looks* dimmer, instead of being silently re-normalized back to full brightness.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..units import um

Float = NDArray[np.float64]

#: Name of the diverging Z colormap (blue = below focus, red = above, white at Z = 0).
Z_CMAP_NAME = "RdBu_r"

#: The Z colormap itself.
Z_CMAP = mpl.colormaps[Z_CMAP_NAME]

#: Intensity gamma for display: ``(I / I_max) ** GAMMA``.  Below 1, so weak light (fill
#: transients, defocused spots, shadow tweezers) stays visible without washing out the peaks.
GAMMA = 0.7

#: Smallest half-range of the Z colour scale [m].  A run with no axial motion would otherwise
#: divide by zero and paint pure numerical noise across the whole colormap.
Z_FLOOR = 1.0 * um

#: One colour per AOD channel, for drive strips and spectrogram panels.  The counter-
#: propagating partners of a pair share a hue family (x = cyan/blue, y = green/amber).
CHANNEL_COLORS: dict[str, str] = {
    "Ax": "#4cc9f0",
    "Bx": "#4361ee",
    "Ay": "#8ac926",
    "By": "#ffb703",
}

#: Dark-background rcParams for movies and figures (use with ``matplotlib.rc_context``).
DARK_STYLE: dict[str, Any] = {
    "figure.facecolor": "#0b0d12",
    "savefig.facecolor": "#0b0d12",
    "axes.facecolor": "#000000",
    "axes.edgecolor": "#5a6472",
    "axes.labelcolor": "#d7dde5",
    "axes.titlecolor": "#f2f5f8",
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "text.color": "#d7dde5",
    "xtick.color": "#9aa5b3",
    "ytick.color": "#9aa5b3",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "grid.color": "#2a303a",
    "font.family": "sans-serif",
    "figure.dpi": 110.0,
    "image.interpolation": "nearest",
    "image.origin": "lower",
}


def z_max_from(z_values: ArrayLike, floor: float = Z_FLOOR) -> float:
    """Half-range of the Z colour scale: ``max |z|``, floored at ``floor`` [m]."""
    z = np.asarray(z_values, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(z))) if z.size else 0.0
    return max(peak, float(floor))


def z_norm(z: ArrayLike, z_max: float) -> Float | float:
    """Map lab Z [m] onto ``[0, 1]``, symmetric about zero: ``0.5 + z / (2 z_max)``.

    Values beyond ``+-z_max`` saturate rather than wrap, so the extremes of a clipped scale
    still read as "far above" / "far below" instead of jumping to the opposite hue.
    """
    scale = max(float(z_max), float(Z_FLOOR))
    out = np.clip(0.5 + np.asarray(z, dtype=np.float64) / (2.0 * scale), 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def z_color(z: ArrayLike, z_max: float) -> Float:
    """RGB tint(s) for lab Z [m] — :data:`Z_CMAP` at :func:`z_norm`, luminance-normalized.

    Every colour is rescaled so its brightest channel is 1.  Without that the scheme would
    lie: ``RdBu_r`` ends in *dark* red and *dark* blue, so a perfectly bright tweezer far from
    the focal plane would render dim purely because of its depth, and the viewer could not
    tell "far above" from "faint".  After normalization hue alone carries Z (white at Z = 0,
    saturating red/blue at the ends) and brightness alone carries intensity.
    """
    rgb = np.asarray(Z_CMAP(z_norm(z, z_max)), dtype=np.float64)[..., :3]
    peak = np.max(rgb, axis=-1, keepdims=True)
    return np.divide(rgb, peak, out=np.zeros_like(rgb), where=peak > 0.0)


def tint_cmap(n: int = 256) -> mpl.colors.ListedColormap:
    """The luminance-normalized :func:`z_color` scale as a colormap, for colorbars.

    Built from :func:`z_color` itself, so the legend on a movie shows exactly the colours the
    frames are painted with.
    """
    return mpl.colors.ListedColormap(z_color(np.linspace(-1.0, 1.0, n), 1.0), name="aodl_z")


def _scaled(intensity: ArrayLike, i_max: float, gamma: float) -> Float:
    """``(I / i_max) ** gamma``, clipped to ``[0, 1]`` (zero for a dead scene)."""
    peak = float(i_max)
    values = np.asarray(intensity, dtype=np.float64)
    if not peak > 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip(values / peak, 0.0, 1.0) ** float(gamma)


def composite(
    layers: Sequence[tuple[ArrayLike, ArrayLike]],
    i_max: float,
    gamma: float = GAMMA,
) -> Float:
    """Additively composite tinted intensity layers on black -> RGB image ``(ny, nx, 3)``.

    Each layer is ``(intensity, rgb)``: one frequency group's intensity frame and the colour
    its lab Z earns (:func:`z_color`).  Groups do not interfere, so adding their light is the
    physically right composite; overlapping spots at different depths therefore blend colours
    exactly as overlapping light would.
    """
    if not layers:
        raise ValueError("composite() needs at least one layer")
    first = np.asarray(layers[0][0], dtype=np.float64)
    out = np.zeros((*first.shape, 3), dtype=np.float64)
    for intensity, rgb in layers:
        out += _scaled(intensity, i_max, gamma)[..., None] * np.asarray(rgb, dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def tint_by_row(
    image: ArrayLike,
    row_colors: ArrayLike,
    i_max: float,
    gamma: float = GAMMA,
) -> Float:
    """Tint an ``(nz, nx)`` XZ slice row by row -> RGB ``(nz, nx, 3)``.

    Every row of an XZ panel *is* one Z, so colouring each row by its own depth puts the panel
    on exactly the same hue scale as the XY view: the vertical axis and the colour say the
    same thing, and a spot sweeping out of the focal plane changes colour as it climbs.
    """
    values = _scaled(image, i_max, gamma)
    colors = np.asarray(row_colors, dtype=np.float64)
    return np.clip(values[..., None] * colors[:, None, :], 0.0, 1.0)


__all__ = [
    "CHANNEL_COLORS",
    "DARK_STYLE",
    "GAMMA",
    "Z_CMAP",
    "Z_CMAP_NAME",
    "Z_FLOOR",
    "composite",
    "tint_by_row",
    "tint_cmap",
    "z_color",
    "z_max_from",
    "z_norm",
]
