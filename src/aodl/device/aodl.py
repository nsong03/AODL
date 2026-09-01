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
    edge     per-axis aperture fill state: one wavefront, two, or None once it is full

so ``field/`` can evaluate it in closed form.  All signs come from
:mod:`aodl.device.conventions`.

**Where the envelope lives.**  ``Lines.amp`` already carries ``A(t_c)`` (Eq. S3), so the
Eq. S5 amplitude polynomial enters here *normalized to* ``alpha0 = 1``: per line the shape
is ``(1, -s (A'/A) / v, (A''/A) / (2 v^2))``, and ``alpha`` is the degree-2 truncated
product of those over the channels sharing an axis.  Constant envelopes therefore give
``alpha = (1, 0, 0)`` and the field reduces to ``c * I0``.

**Aperture fill.**  Each partially filled channel constrains its axis to one half-line
(``docs/conventions.md`` §7); the channels sharing an axis are *intersected*, so a
counter-propagating pair driven from ``t = 0`` leaves the two-sided window
``[D/2 - v t, v t - D/2]`` — empty until ``t = tau/2``, the whole aperture at ``t = tau``.
The result rides on :attr:`TermArray.edge` as ``None`` (full), a
:class:`~aodl.device.conventions.FillEdge` (one-sided) or a :class:`FillWindow`
(two-sided); see :func:`_axis_interval`.

**Pruning.**  With intra-AOD mixing on (:mod:`aodl.device.mixing`) a channel carries
``O(M^3)`` lines and the product carries the *product* of those counts, most of which are
negligible: a term whose ``|c|`` is a fraction ``eps`` of the strongest one contributes
``~eps^2`` of relative intensity, so dropping everything below
``term_prune = 1e-6`` bounds the relative intensity error at ``~1e-12`` while cutting the
term count sharply.  The discarded power is reported in :attr:`TermArray.pruned_power`.
The same cut is applied **before** the product as well (:func:`_pre_cut_lines`), which is
lossless and keeps the combinatorics from being paid for terms that are about to be thrown
away; :data:`MAX_TERMS` then bounds what is left.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import conventions
from .aod import Lines, channel_lines, fill_edge
from .conventions import N_AXES, ChannelGeometry, FillEdge

#: Number of aperture amplitude-polynomial coefficients kept per axis (Eq. S5, truncated
#: at ``u^2`` — the acoustic-irising term).
N_ALPHA = 3

#: Default term-level amplitude cut of :func:`build_terms`, as a fraction of the strongest
#: term's ``|c|``.  Relative intensity error scales as its square (~``1e-12``).
TERM_PRUNE = 1e-6

#: Default ceiling on the Cartesian term count of :func:`build_terms`.  Terms are cheap
#: individually (~130 bytes across ``c``, ``theta``, ``alpha`` and ``df_opt``) but multiply:
#: four channels carrying a few hundred lines each would ask for ``10^10`` of them.  The
#: limit turns that into a message naming the per-channel line counts instead of a
#: multi-minute allocation.
MAX_TERMS = 200_000


class FillWindow(NamedTuple):
    """Two-sided aperture fill window: acoustic content on ``lo <= u <= hi``.

    Emitted for an axis whose two counter-propagating channels are *both* still filling
    (``docs/conventions.md`` §7): ``sound_sign = -1`` holds content at ``u >= D/2 - v t``,
    ``+1`` at ``u <= v t - D/2``, and light has to cross both crystals, so it sees the
    intersection.  :mod:`aodl.field.focal` reads this structurally — by the ``lo`` / ``hi``
    fields — and routes it to :func:`aodl.field.gaussian.gauss_moments_window`.

    A one-sided axis keeps :class:`~aodl.device.conventions.FillEdge` and a fully filled one
    stays ``None``, both unchanged from M1.

    Attributes
    ----------
    lo, hi:
        Aperture coordinates [m] of the two wavefronts, ``lo`` from the ``sound_sign = -1``
        channel and ``hi`` from the ``+1`` one.  ``lo >= hi`` marks an *empty* window (the
        waves have not met yet): the frame is then dark and :func:`build_terms` drops every
        term, so the empty record is a diagnostic and is never integrated over.
    """

    lo: float
    hi: float


#: Fill state of one axis: ``None`` (full aperture), one wavefront, or two.
AxisFill = FillEdge | FillWindow | None


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
        Per-axis aperture fill state, ``(edge_x, edge_y)``: ``None`` when that axis is fully
        filled, a :class:`~aodl.device.conventions.FillEdge` when one wavefront bounds it,
        and a :class:`FillWindow` when the axis' counter-propagating pair bounds it from
        both sides.  The fill state depends only on the frame time and the channel geometry,
        so it is shared by every term of the frame.
    pruned_power:
        Diagnostic: the ``sum |c|^2`` that pruning removed from this frame — the terms cut
        here plus the per-channel lines cut by :mod:`aodl.device.mixing`, carried through
        the product (see :func:`build_terms`).  Compare against ``sum |c|^2`` of the terms
        that survived for a relative figure; ``0`` when nothing was dropped.
    """

    c: NDArray[np.complex128]
    theta1: NDArray[np.float64]
    theta2: NDArray[np.float64]
    alpha: NDArray[np.complex128]
    df_opt: NDArray[np.float64]
    edge: tuple[AxisFill, AxisFill] = field(default=(None, None))
    pruned_power: float = 0.0

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


def _axis_interval(edges: Sequence[FillEdge]) -> tuple[float, float]:
    """Intersect the filled half-lines of the channels sharing one axis (Eq. S4 timing).

    Every partially filled channel constrains its axis to the half-line containing its own
    transducer — ``u >= u_edge`` for ``side = "lower"`` (``sound_sign = -1``), ``u <= u_edge``
    for ``"upper"`` (``+1``) — and a channel that is already full contributes no constraint
    at all (:func:`aodl.device.aod.fill_edge` returns ``None`` and it never reaches
    ``edges``).  The light has to cross all of them, so the axis transmits on the
    intersection, returned as ``(lo, hi)`` with ``-inf`` / ``+inf`` for an unbounded side.

    For a counter-propagating pair driven from ``t = 0`` this is ``[D/2 - v t, v t - D/2]``:
    **empty until ``t = tau/2``** — both wavefronts have to reach a point before anything
    gets through it, so a pair-driven tweezer is strictly dark for the first half transit —
    then growing to the full aperture at ``t = tau``.
    """
    lo, hi = -math.inf, math.inf
    for edge in edges:
        if edge.side == "lower":
            lo = max(lo, edge.u_edge)
        else:
            hi = min(hi, edge.u_edge)
    return lo, hi


def _axis_fill(interval: tuple[float, float]) -> AxisFill:
    """Fill record for one axis' intersected interval (see :attr:`TermArray.edge`)."""
    lo, hi = interval
    if not math.isfinite(lo):
        return None if not math.isfinite(hi) else FillEdge(u_edge=hi, side="upper")
    if not math.isfinite(hi):
        return FillEdge(u_edge=lo, side="lower")
    return FillWindow(lo=lo, hi=hi)


def _pre_cut_lines(line: Lines, term_prune: float) -> Lines:
    """Drop the lines of one channel that cannot survive the post-product cut.

    **Why this is lossless.**  The channel product is Cartesian, so the strongest term is the
    product of the per-channel strongest lines, ``max|c| = prod_mu max_l |amp^mu_l|``, and a
    term using line ``l`` of channel ``mu`` obeys
    ``|c| <= |amp^mu_l| prod_{nu != mu} max|amp^nu|``.  Hence

        |amp^mu_l| < term_prune * max|amp^mu|   ==>   |c| < term_prune * max|c|

    for *every* term that line takes part in: they are all below the post-product cut of
    :func:`build_terms` and would be dropped there anyway.  Cutting the line here removes
    them from the combinatorics instead of from the result — same term set, at
    ``sum(n_mu)`` cost instead of ``prod(n_mu)``.

    The dropped ``sum |amp|^2`` is folded into the channel's own ``pruned_power``, so
    :func:`_lines_pruned_power` carries it through the product like any other line-level cut.
    ``term_prune = 0`` (and a channel whose lines are all zero) drops nothing.
    """
    if term_prune <= 0.0 or line.n_lines == 0:
        return line
    magnitude = np.abs(line.amp)
    keep = magnitude >= float(term_prune) * float(magnitude.max())
    if bool(keep.all()):
        return line
    return Lines(
        amp=line.amp[keep],
        f=line.f[keep],
        fdot=line.fdot[keep],
        A=line.A[keep],
        dA=line.dA[keep],
        d2A=line.d2A[keep],
        pruned_power=line.pruned_power + float(np.sum(magnitude[~keep] ** 2)),
    )


def _too_many_terms(names: Sequence[str], counts: Sequence[int], n: int, max_terms: int) -> str:
    """Message for a Cartesian product that would exceed ``max_terms``."""
    per_channel = ", ".join(f"{name}: {count}" for name, count in zip(names, counts, strict=True))
    return (
        f"the pupil term product would hold {n} terms, above max_terms={max_terms}.  "
        f"Lines per channel after the pre-product amplitude cut — {per_channel} — and the "
        f"product is Cartesian (Eq. S7), so the count is their product.  Non-commensurate "
        f"multi-tone drives blow up here because nothing merges: tighten "
        f"MixingConfig.line_prune (fewer intermodulation lines per channel) or term_prune "
        f"(a harder pre-product cut), drive fewer tones, or raise max_terms if this many "
        f"terms really is what you want."
    )


def _lines_pruned_power(lines: Sequence[Lines]) -> float:
    """Line-level pruned power carried through the channel product.

    ``sum_terms |c|^2 = prod_channels sum_lines |amp|^2`` exactly (the product is
    Cartesian), so the term power lost with channel ``mu``'s pruned lines is
    ``pruned_mu * prod_{nu != mu} kept_nu`` to first order in the (tiny) pruned fractions.
    """
    kept = [float(np.sum(np.abs(line.amp) ** 2)) for line in lines]
    total = 0.0
    for index, line in enumerate(lines):
        if line.pruned_power:
            others = 1.0
            for other in kept[:index] + kept[index + 1 :]:
                others *= other
            total += line.pruned_power * others
    return total


def build_terms(
    wfs: Any,
    t: float,
    channels: Sequence[str] | None = None,
    term_prune: float = TERM_PRUNE,
    max_terms: int = MAX_TERMS,
) -> TermArray:
    """Expand a :class:`~aodl.waveform.tones.WaveformSet` into pupil terms at time ``t``.

    Every combination of one line per participating channel becomes a term (Eq. S7):
    amplitudes multiply, ``theta1``/``theta2``/``df_opt`` accumulate, and the aperture
    amplitude polynomials multiply per axis (truncated at degree 2, Eq. S5).  Channels not
    listed contribute an identity factor — a bare AOD-less axis has ``theta1 = theta2 = 0``
    and ``alpha = (1, 0, 0)``, i.e. just the input Gaussian.

    The amplitude cut runs twice, to the same threshold and with the same result: once per
    channel *before* the product (:func:`_pre_cut_lines`, lossless — see its proof) and once
    on the terms afterwards.  With mixing on, the Cartesian product of ``O(M^3)`` line sets
    is dominated by combinations far too weak to see.

    Each channel's aperture fill state is intersected per axis (:func:`_axis_interval`), so a
    counter-propagating pair sees the two-sided window ``[D/2 - v t, v t - D/2]``.  While
    that window is **empty** — ``t < tau/2``, before the two wavefronts meet — no light
    reaches the image plane at all, and the frame is returned with *no terms* rather than
    with zero-amplitude ones (``field/`` short-circuits to a black frame).  Nothing is
    approximated away there, so ``pruned_power`` is ``0`` for such a frame.

    Parameters
    ----------
    wfs:
        A :class:`~aodl.waveform.tones.WaveformSet` (duck-typed: needs ``.channels`` and
        ``.params``).
    t:
        Frame time [s].
    channels:
        Which channels to include; ``None`` means every channel present in ``wfs``.
    term_prune:
        Drop terms with ``|c| < term_prune * max |c|`` (relative intensity error
        ``~term_prune^2``); ``0`` keeps every term.  Nothing is ever dropped when the
        strongest term is itself zero, so a frame whose drive is simply off still holds one
        term.
    max_terms:
        Ceiling on the Cartesian product (:data:`MAX_TERMS`).  Exceeding it raises
        ``ValueError`` naming the per-channel line counts, instead of allocating for
        minutes: non-commensurate multi-tone drives merge nothing, so their line sets
        multiply out unchecked.

    Returns
    -------
    :class:`TermArray` with at most ``N = prod(lines per channel)`` terms (``N = 1`` when no
    channel is driven, ``N = 0`` when an axis' fill window is empty), carrying the pruned
    power as a diagnostic.

    Raises
    ------
    ValueError
        If the post-cut product would exceed ``max_terms``.
    KeyError
        If a requested channel is absent from ``wfs``.
    """
    available = dict(wfs.channels)
    names = tuple(available) if channels is None else tuple(channels)
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"channels {missing} are not present in the waveform set")

    geoms = [conventions.geometry(name) for name in names]
    aods = [wfs.params.channels[name] for name in names]
    lines = [
        _pre_cut_lines(channel_lines(available[name], aod, t), term_prune)
        for name, aod in zip(names, aods, strict=True)
    ]
    counts = [line.n_lines for line in lines]
    product = int(np.prod(counts)) if counts else 1
    if product > max_terms:
        raise ValueError(_too_many_terms(names, counts, product, max_terms))
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

    n_terms = product
    c = np.ascontiguousarray(amplitude).reshape(n_terms)
    theta1 = theta1.reshape(N_AXES, n_terms)
    theta2 = theta2.reshape(N_AXES, n_terms)
    alpha = alpha.reshape(N_AXES, N_ALPHA, n_terms)
    df_opt = np.asarray(df_opt, dtype=np.float64).reshape(n_terms)

    intervals = [_axis_interval(axis_edges[axis]) for axis in range(N_AXES)]
    dark = any(lo >= hi for lo, hi in intervals)
    if dark:
        # No point of the aperture has been reached by every channel's sound yet, so the
        # frame carries no light: drop every term (their amplitude is exactly 0) and report
        # no pruned power — this is the physics of the fill transient, not an approximation.
        pruned = 0.0
        keep = np.zeros(n_terms, dtype=bool)
    else:
        pruned = _lines_pruned_power(lines)
        # `initial` keeps an undriven-but-listed channel (zero lines, so zero terms) working.
        keep = np.abs(c) >= float(term_prune) * float(np.abs(c).max(initial=0.0))
        if not keep.all():
            pruned += float(np.sum(np.abs(c[~keep]) ** 2))
    if not keep.all():
        c, theta1, theta2 = c[keep], theta1[:, keep], theta2[:, keep]
        alpha, df_opt = alpha[:, :, keep], df_opt[keep]

    return TermArray(
        c=c,
        theta1=theta1,
        theta2=theta2,
        alpha=alpha,
        df_opt=df_opt,
        edge=(_axis_fill(intervals[0]), _axis_fill(intervals[1])),
        pruned_power=pruned,
    )


__all__ = [
    "MAX_TERMS",
    "N_ALPHA",
    "TERM_PRUNE",
    "AxisFill",
    "FillWindow",
    "TermArray",
    "build_terms",
]
