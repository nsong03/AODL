r"""The four-AOD stack: emission lines -> pupil terms (Eq. S7/S8).

Stacking the AODs multiplies their pupils, so the pupil of the whole AODL is the
*Cartesian product* of the per-channel line sets: with ``n_mu`` lines on channel ``mu``
there are ``prod(n_mu)`` terms, each one beam (``docs/PLAN.md`` §1.2, the 16-ray picture of
Fig. S6 for two tones per AOD).  Each term is separable in x and y and carries

    c        complex amplitude, the product of its line amplitudes
    theta1   per-axis pupil tilt  [rad/m]     -> deflection,  X = theta1 F / k
    theta2   per-axis curvature   [rad/m^2]   -> defocus,     Z_S11 = 2 F^2 theta2 / k
    alpha    per-axis aperture amplitude polynomial (alpha0, alpha1, alpha2)
    df_opt   optical frequency offset [Hz], for frequency grouping in field/focal.py
    edge     per-axis aperture fill edge (or None once the aperture is full)

so ``field/`` can evaluate it in closed form.  All signs come from
:mod:`aodl.device.conventions`.

**Where the envelope lives.**  ``Lines.amp`` already carries ``A(t_c)`` (Eq. S3), so the
Eq. S5 amplitude polynomial enters here *normalized to* ``alpha0 = 1``: per line the shape
is ``(1, -s (A'/A) / v, (A''/A) / (2 v^2))``, and ``alpha`` is the degree-2 truncated
product of those over the channels sharing an axis.  Constant envelopes therefore give
``alpha = (1, 0, 0)`` and the field reduces to ``c * I0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import conventions
from .aod import Lines, channel_lines, fill_edge
from .conventions import N_AXES, ChannelGeometry, FillEdge

#: Number of aperture amplitude-polynomial coefficients kept per axis (Eq. S5, truncated
#: at ``u^2`` — the acoustic-irising term).
N_ALPHA = 3


@dataclass(frozen=True)
class TermArray:
    """Pupil terms of one frame — a struct of arrays (``docs/ARCHITECTURE.md`` §3, L3).

    Attributes
    ----------
    c:
        ``(N,)`` complex amplitude per term: the product of its line amplitudes (the
        ``i C / 2`` factors of Eq. S3 are already inside those).
    theta1:
        ``(2, N)`` pupil tilt per axis [rad/m]; axis 0 = x, axis 1 = y.
    theta2:
        ``(2, N)`` pupil curvature per axis [rad/m^2].
    alpha:
        ``(2, 3, N)`` aperture amplitude polynomial per axis, ``(alpha0, alpha1, alpha2)``,
        normalized so that a constant envelope gives ``(1, 0, 0)`` (see the module
        docstring); products across channels are truncated at degree 2.
    df_opt:
        ``(N,)`` optical frequency offset [Hz] = sum of the participating detunings.  Terms
        sharing a value (within tolerance) interfere; different values beat at MHz rates
        and add in intensity (``docs/PLAN.md`` §1.3).
    edge:
        Per-axis aperture fill edge, ``(edge_x, edge_y)``; an entry is ``None`` when that
        axis is fully filled.  The fill state depends only on the frame time and the
        channel geometry, so it is shared by every term of the frame.
    """

    c: NDArray[np.complex128]
    theta1: NDArray[np.float64]
    theta2: NDArray[np.float64]
    alpha: NDArray[np.complex128]
    df_opt: NDArray[np.float64]
    edge: tuple[FillEdge | None, FillEdge | None] = field(default=(None, None))

    @property
    def n_terms(self) -> int:
        """Number of pupil terms ``N``."""
        return int(self.c.size)

    def __len__(self) -> int:
        return self.n_terms


def _shaped(values: NDArray[Any], index: int, counts: Sequence[int]) -> NDArray[Any]:
    """Reshape a per-line array so that broadcasting realizes the Cartesian product."""
    shape = [1] * len(counts)
    shape[index] = int(counts[index])
    return values.reshape(shape)


def _shape_poly(
    lines: Lines, geom: ChannelGeometry, sound_speed: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalized (``alpha0 = 1``) Eq. S5 aperture amplitude polynomial of one channel.

    ``A(t_c)`` already lives in ``Lines.amp``, so only the polynomial's *shape* matters
    here.  Lines with ``A = 0`` contribute nothing (their amplitude vanishes), so their
    shape is set to the identity ``(1, 0, 0)`` rather than dividing by zero.
    """
    a0, a1, a2 = conventions.amplitude_poly(lines.A, lines.dA, lines.d2A, geom, sound_speed)
    nonzero = a0 != 0.0
    inverse = np.where(nonzero, 1.0 / np.where(nonzero, a0, 1.0), 0.0)
    return np.ones_like(a0), a1 * inverse, a2 * inverse


def _multiply_poly(acc: NDArray[Any], factor: Sequence[NDArray[Any]]) -> NDArray[np.complex128]:
    """Degree-2 truncated product of two aperture amplitude polynomials."""
    p0, p1, p2 = acc[0], acc[1], acc[2]
    q0, q1, q2 = factor
    product = np.stack([p0 * q0, p0 * q1 + p1 * q0, p0 * q2 + p1 * q1 + p2 * q0], axis=0)
    return np.asarray(product, dtype=np.complex128)


def _axis_edge(edges: Sequence[FillEdge]) -> FillEdge | None:
    """Combine the fill edges of the channels sharing one axis."""
    if not edges:
        return None
    if len(edges) == 1:
        return edges[0]
    sides = {edge.side for edge in edges}
    if len(sides) > 1:
        raise NotImplementedError(
            "counter-propagating channels on one axis are partially filled from opposite "
            "sides, which needs a two-sided aperture window; wait until the aperture is "
            "full (t >= tau) or drive one side at a time"
        )
    side = sides.pop()
    positions = [edge.u_edge for edge in edges]
    return FillEdge(u_edge=max(positions) if side == "lower" else min(positions), side=side)


def build_terms(wfs: Any, t: float, channels: Sequence[str] | None = None) -> TermArray:
    """Expand a :class:`~aodl.waveform.tones.WaveformSet` into pupil terms at time ``t``.

    Every combination of one line per participating channel becomes a term (Eq. S7):
    amplitudes multiply, ``theta1``/``theta2``/``df_opt`` accumulate, and the aperture
    amplitude polynomials multiply per axis (truncated at degree 2, Eq. S5).  Channels not
    listed contribute an identity factor — a bare AOD-less axis has ``theta1 = theta2 = 0``
    and ``alpha = (1, 0, 0)``, i.e. just the input Gaussian.

    Parameters
    ----------
    wfs:
        A :class:`~aodl.waveform.tones.WaveformSet` (duck-typed: needs ``.channels`` and
        ``.params``).
    t:
        Frame time [s].
    channels:
        Which channels to include; ``None`` means every channel present in ``wfs``.

    Returns
    -------
    :class:`TermArray` with ``N = prod(lines per channel)`` terms (``N = 1`` when no
    channel is driven).
    """
    available = dict(wfs.channels)
    names = tuple(available) if channels is None else tuple(channels)
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"channels {missing} are not present in the waveform set")

    geoms = [conventions.geometry(name) for name in names]
    aods = [wfs.params.channels[name] for name in names]
    lines = [channel_lines(available[name], aod, t) for name, aod in zip(names, aods, strict=True)]
    counts = [line.n_lines for line in lines]
    shape = tuple(counts)

    amplitude = np.ones(shape if shape else (1,), dtype=np.complex128)
    theta1 = np.zeros((N_AXES, *shape), dtype=np.float64)
    theta2 = np.zeros((N_AXES, *shape), dtype=np.float64)
    df_opt = np.zeros(shape, dtype=np.float64)
    alpha = np.zeros((N_AXES, N_ALPHA, *shape), dtype=np.complex128)
    alpha[:, 0] = 1.0
    axis_edges: list[list[FillEdge]] = [[] for _ in range(N_AXES)]

    for index, (geom, aod, line) in enumerate(zip(geoms, aods, lines, strict=True)):
        axis = geom.axis
        speed = aod.sound_speed
        amplitude = amplitude * _shaped(line.amp, index, counts)
        theta1[axis] += _shaped(conventions.theta1_contribution(line.f, geom, speed), index, counts)
        theta2[axis] += _shaped(conventions.theta2_contribution(line.fdot, speed), index, counts)
        df_opt = df_opt + _shaped(line.f, index, counts)
        factor = [_shaped(coeff, index, counts) for coeff in _shape_poly(line, geom, speed)]
        alpha[axis] = _multiply_poly(alpha[axis], factor)
        edge = fill_edge(aod, geom, t)
        if edge is not None:
            axis_edges[axis].append(edge)

    n_terms = int(np.prod(counts)) if counts else 1
    return TermArray(
        c=np.ascontiguousarray(amplitude).reshape(n_terms),
        theta1=theta1.reshape(N_AXES, n_terms),
        theta2=theta2.reshape(N_AXES, n_terms),
        alpha=alpha.reshape(N_AXES, N_ALPHA, n_terms),
        df_opt=np.asarray(df_opt, dtype=np.float64).reshape(n_terms),
        edge=(_axis_edge(axis_edges[0]), _axis_edge(axis_edges[1])),
    )


__all__ = ["N_ALPHA", "TermArray", "build_terms"]
