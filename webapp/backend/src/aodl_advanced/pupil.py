r"""Misaligned-AOD pupils on a 2-D grid: retarded-time drive, J1 law, frequency lines, groups.

This is the FFT engine's device layer (DESIGN §1.7).  Nothing here Taylor-expands the drive:
the pupil of AOD ``mu`` is evaluated **pointwise** at each grid point's own retarded time,
through the channel's possibly misaligned aperture coordinate (:mod:`.misalignment`):

    u_mu  = (r - d_mu) . R(theta_mu) e_mu                         channel coordinate
    t_ret = t - tau/2 - s_mu u_mu / (m_mu v) - delta t_mu          Eq. S4 + Eq. S29
    fill  : s_mu u_mu <= m_mu [ v (t - delta t_mu) - D/2 ]        conventions.is_filled, scaled
    E_mu  = sum_n a_n A_n(t_ret) e^{i phase_n(t_ret)}              complex analytic drive
                                                                   (rotating frame, carrier dropped)
    T_mu  = i J1(C |E_mu|) E_mu^* / |E_mu|                         "j1": the +1 order of the thin
                                                                   phase grating exp(i C V)
          = (i C / 2) E_mu^*                                       "weak": Eq. S3, aodl's model

Both the retarded time and the fill mask are obtained from :mod:`aodl.device.conventions`
(``retarded_time``, ``is_filled``) evaluated at the magnification-scaled coordinate ``u / m``
and the delayed frame time ``t - delta t`` — so the sound-direction sign is never written
here, and a magnification mismatch scales the fill edge with the aperture (``D -> m D``).

**The J1 law.**  The real drive at a point is ``V = Re[E e^{i omega_c t}] = |E| cos(omega_c t
+ arg E)``; Jacobi–Anger gives ``exp(i C V) = sum_p i^p J_p(C|E|) e^{i p (omega_c t + arg E)}``
and the order that imprints ``-Phi`` and up-shifts the light (``p = -1``,
``conventions.DIFFRACTION_ORDER``) is ``i^{-1} J_{-1} e^{-i arg E} = i J1(C|E|) E^*/|E|``.
For one tone it is ``i J1(C A) e^{-i phase}``, which reduces to aodl's ``(i C/2) A e^{-i phase}``
as ``J1(x) -> x/2``; expanded to ``C^3`` it reproduces ``aodl.device.mixing``'s compression
``1 - S/4 + m_n^2/8`` and IM3 amplitudes ``-a_j a_k a_i^*``, ``-(1/2) a_j^2 a_i^*`` exactly.

**Lines instead of sub-time averaging (DESIGN §1.7).**  A camera or an atom averages the
MHz beats between tones away.  Each channel's transmission is therefore decomposed into
optical-frequency *lines*:

* ``"weak"`` — the lines are the tones themselves, one 2-D factor per live tone tagged
  ``f_n(t_c)``;
* ``"j1"`` — every drive ``plan_motion`` builds puts a channel's live tones on one comb
  ``f_ref + m Delta f`` (Eq. S19 ladders, Shepard ladders, single tones) and gives them one
  common chirp.  Writing ``b_n(u) = a_n A_n(t_ret) e^{i phase_n(t_ret)}`` the drive a period
  ``tau'`` later is, *after demodulating the reference tone's instantaneous frequency and the
  common chirp at that pixel* (both are common to every line of the channel and therefore
  cannot change any intensity),

      E~(u, tau') = sum_n b_n(u) e^{2 pi i m_n Delta f tau'}

  — exactly periodic in ``tau'`` with period ``1/Delta f``.  ``T~ = i J1(C|E~|) E~^*/|E~|`` is
  evaluated at the ``K`` sub-times ``tau'_k = k/(K Delta f)`` and a ``K``-point DFT along ``k``
  yields the lines ``L_m(u)`` at ``f_ref(t_c) + m Delta f`` (fundamentals, compression, IM3,
  ...), of which those above ``LINE_CUT`` of the strongest are kept.  ``K = 1`` is the frozen
  instantaneous pupil as a single line tagged with the amplitude^2-weighted mean frequency.
  Demodulating per pixel — rather than with the beam-centre frequency — matters: during a
  chirp the instantaneous frequency varies by ``fdot tau/2`` (~0.3 MHz at the defaults) across
  the aperture, which would otherwise leak ~2 rad of phase into the neighbouring bins.

The Cartesian product of the four channels' lines is then grouped by optical frequency sum
(``GROUP_TOL`` = 1 kHz, neighbour chaining plus a diameter cap — the rule of
``aodl.field.focal.group_terms``, re-implemented here because this package may not import
``aodl.field``); each group's pupil is the input Gaussian times the sum of its products times
the common pointing tilt, and the groups add in intensity.  When no channel is rotated every
line depends on one lab coordinate only and the product is built as outer products.

Two things are dropped as "alignment" and reported instead of rendered: the carrier's own
pupil tilt ``s 2 pi f_c u_mu / (m_mu v)`` — common to every term, it cancels between the
members of an aligned pair and otherwise is a static pointing offset of the whole image
(:func:`static_offset`) — and the constant carrier phase.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j1

from aodl.device.conventions import (
    AXIS_X,
    ChannelGeometry,
    beam_center_time,
    geometry,
    is_filled,
)
from aodl.device.conventions import (
    retarded_time as _conv_retarded_time,
)
from aodl.params import CHANNELS, AODLParams, AODParams
from aodl.units import kHz
from aodl.waveform.tones import ChannelWaveform

from .misalignment import AODMisalignment, Misalignment

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]
RelAmp = Callable[[ArrayLike], Any]

#: Optical-frequency tolerance of the interference grouping [Hz] (= ``aodl.field.focal.GROUP_TOL``).
GROUP_TOL: float = 1.0 * kHz
#: Comb residual [Hz] above which a channel's tones are declared off-comb (``"weak"`` fallback).
COMB_TOL: float = 1.0 * kHz
#: Keep comb lines whose pupil norm is at least this fraction of the strongest line's.
LINE_CUT: float = 1e-3
#: Keep product terms whose weight (product of line norms) is at least this fraction of the max.
TERM_CUT: float = 1e-5
#: Probe points of the coarse live-tone scan over the grid's retarded-time range.
LIVE_PROBE_POINTS: int = 64
#: Largest number of pixels evaluated per comb-DFT block (memory cap: ``K`` complex planes).
DFT_BLOCK_PIXELS: int = 1 << 18


@dataclass(frozen=True)
class PupilModel:
    """How a channel's transmission is turned into lines."""

    order_model: str = "j1"
    comb_dft_points: int = 16
    line_cut: float = LINE_CUT
    term_cut: float = TERM_CUT

    def __post_init__(self) -> None:
        if self.order_model not in ("j1", "weak"):
            raise ValueError(f"order_model must be 'j1' or 'weak', got {self.order_model!r}")
        if int(self.comb_dft_points) < 1:
            raise ValueError(f"comb_dft_points must be >= 1, got {self.comb_dft_points!r}")
        object.__setattr__(self, "comb_dft_points", int(self.comb_dft_points))


@dataclass(frozen=True)
class Line:
    """One optical-frequency line of one channel.

    Attributes
    ----------
    f_tag:
        Optical frequency tag [Hz]: the detuning the line adds to ``df_opt``, at ``t_c``.
    values:
        Complex pupil factor — ``(n,)`` along the channel's own lab axis when the channel is
        not rotated (``separable``), otherwise ``(ny, nx)`` on the full grid.
    axis:
        The channel's lab axis (0 = x, 1 = y); meaningful for separable lines.
    norm:
        ``sqrt(sum |values|^2 * step)`` — the line's brightness proxy.
    """

    f_tag: float
    values: Complex
    axis: int
    norm: float

    @property
    def separable(self) -> bool:
        """True when the line depends on one lab coordinate only."""
        return self.values.ndim == 1


@dataclass
class Group:
    """One optical-frequency group: the terms that interfere coherently.

    ``pupil()`` builds the group's 2-D pupil (Gaussian and tilt included) on demand, so a
    frame never holds every group's pupil at once.
    """

    f_opt: float
    power: float
    n_terms: int
    _build: Callable[[], Complex] = field(repr=False)

    def pupil(self) -> Complex:
        """The group's complex pupil, shape ``(ny, nx)``."""
        return self._build()


# ------------------------------------------------------------------------- geometry


def pupil_grid(w_in: float, n: int, span_w_in: float) -> tuple[Float, Float]:
    """Uniform centred pupil grid ``+- span_w_in * w_in`` with ``n`` points per axis."""
    if int(n) < 2:
        raise ValueError(f"pupil grid needs at least 2 points per axis, got {n!r}")
    if not span_w_in > 0.0:
        raise ValueError(f"span_w_in must be positive, got {span_w_in!r}")
    half = float(span_w_in) * float(w_in)
    axis = np.linspace(-half, half, int(n))
    return axis, axis.copy()


def axis_vectors(geom: ChannelGeometry) -> tuple[Float, Float]:
    """``(e_axis, e_perp)`` of a channel: ``e_perp = z x e_axis`` (see :mod:`.misalignment`)."""
    if geom.axis == AXIS_X:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    return np.array([0.0, 1.0]), np.array([-1.0, 0.0])


def channel_coordinate(
    X: ArrayLike, Y: ArrayLike, geom: ChannelGeometry, mis: AODMisalignment
) -> Float:
    """Channel aperture coordinate ``u = (r - d) . R(theta) e_axis`` at lab points ``(X, Y)``.

    ``d = offset_par e_axis + offset_perp e_perp``; ``R(theta)`` rotates counter-clockwise
    about the optical axis (:mod:`.misalignment`).  ``X`` and ``Y`` broadcast.
    """
    e_axis, e_perp = axis_vectors(geom)
    d = mis.offset_par * e_axis + mis.offset_perp * e_perp
    direction = math.cos(mis.rotation) * e_axis + math.sin(mis.rotation) * e_perp
    x = np.asarray(X, dtype=np.float64) - d[0]
    y = np.asarray(Y, dtype=np.float64) - d[1]
    return np.asarray(x * direction[0] + y * direction[1], dtype=np.float64)


def retarded_time(
    t: float, u: ArrayLike, geom: ChannelGeometry, aod: AODParams, mis: AODMisalignment
) -> Float:
    """``t_ret = t - tau/2 - s u / (m v) - delta t`` (Eq. S4 with the Eq. S29 delay).

    Obtained as ``conventions.retarded_time(t - delta t, u / m, ...)``: the magnified crystal
    carries sound at ``m v`` over an aperture ``m D`` (same transit time), and the delayed
    drive simply shifts the frame time.  The sound sign comes from ``conventions``.
    """
    u_scaled = np.asarray(u, dtype=np.float64) / mis.velocity_scale
    return _conv_retarded_time(float(t) - mis.delay, u_scaled, geom, aod)


def filled(
    u: ArrayLike, t: float, geom: ChannelGeometry, aod: AODParams, mis: AODMisalignment
) -> NDArray[np.bool_]:
    """Fill mask ``s u <= m [v (t - delta t) - D/2]`` — ``conventions.is_filled`` at ``u / m``,
    ``t - delta t``; the fill edge scales with the magnification like the aperture does."""
    u_scaled = np.asarray(u, dtype=np.float64) / mis.velocity_scale
    return is_filled(u_scaled, float(t) - mis.delay, geom, aod)


def static_offset(params: AODLParams, mis: Misalignment) -> dict[str, tuple[float, float]]:
    """Dropped common carrier-tilt pointing offset per channel, lab ``(x, y)`` [m].

    The carrier imprints the pupil tilt ``s 2 pi f_c u_mu / (m_mu v)``, i.e. an image offset
    ``deflection_scale s f_c R(theta) e_axis / m``; the aligned pair cancels it and ``aodl``
    drops it as the definition of the optical axis.  What the misalignment leaves over is
    ``deflection_scale * s * f_c * (R(theta) e_axis / m - e_axis)``, reported here per channel
    (it moves every spot of the frame alike and is not rendered).
    """
    out: dict[str, tuple[float, float]] = {}
    scale = params.deflection_scale
    for name in CHANNELS:
        geom = geometry(name)
        m = mis.channel(name)
        e_axis, e_perp = axis_vectors(geom)
        direction = math.cos(m.rotation) * e_axis + math.sin(m.rotation) * e_perp
        vector = (
            scale
            * geom.sound_sign
            * params.channels[name].f_center
            * (direction / m.velocity_scale - e_axis)
        )
        out[name] = (float(vector[0]), float(vector[1]))
    return out


# ------------------------------------------------------------------------ the drive


def live_tones(cw: ChannelWaveform, t_lo: float, t_hi: float) -> list[int]:
    """Indices of the tones whose envelope is not identically zero on ``[t_lo, t_hi]``.

    A ``FadeZoneEnvelope``'s ``is_active`` is consulted first; then the envelope is probed at
    :data:`LIVE_PROBE_POINTS` points plus both ends (every envelope of ``plan_motion`` is live
    for tens of µs at a time, far longer than the probe spacing over one aperture transit).
    """
    probe = np.linspace(float(t_lo), float(t_hi), LIVE_PROBE_POINTS)
    out: list[int] = []
    for i, tone in enumerate(cw.tones):
        active = getattr(tone.env, "is_active", None)
        if active is not None and not bool(active):
            continue
        if np.any(np.asarray(tone.env.A(probe), dtype=np.float64) != 0.0):
            out.append(i)
    return out


def tone_drives(
    cw: ChannelWaveform,
    indices: Sequence[int],
    t_ret: Float,
    amps: Sequence[float],
    mask: NDArray[np.bool_],
) -> list[Complex]:
    """``b_n(u) = a_n A_n(t_ret) e^{i phase_n(t_ret)}`` on the filled grid, one array per tone."""
    out: list[Complex] = []
    for a, i in zip(amps, indices, strict=True):
        tone = cw.tones[i]
        env = np.asarray(tone.env.A(t_ret), dtype=np.float64)
        phase = np.asarray(tone.phase(t_ret), dtype=np.float64)
        b = float(a) * env * np.exp(1j * phase)
        out.append(np.asarray(np.where(mask, b, 0.0), dtype=np.complex128))
    return out


def complex_drive(cw: ChannelWaveform, t_ret: ArrayLike, rel_amp: RelAmp | None) -> Complex:
    """``E(t_ret) = sum_n a_n A_n(t_ret) e^{i phase_n(t_ret)}`` over *all* tones (no fill mask).

    ``a_n = rel_amp(f_center + f_n)`` is left to the caller's closure; with ``rel_amp=None``
    every ``a_n = 1``.  ``rel_amp`` receives the tone's *detuning* here only through
    :func:`channel_lines`, which knows the carrier; this bare form takes ``a_n = 1``.
    """
    t = np.asarray(t_ret, dtype=np.float64)
    total = np.zeros(t.shape, dtype=np.complex128)
    for tone in cw.tones:
        env = np.asarray(tone.env.A(t), dtype=np.float64)
        phase = np.asarray(tone.phase(t), dtype=np.float64)
        total = total + env * np.exp(1j * phase)
    return total


def transmission(E: ArrayLike, C: float, order_model: str) -> Complex:
    """``"j1"``: ``i J1(C|E|) E^*/|E|`` (``0`` where ``|E| = 0``); ``"weak"``: ``(i C/2) E^*``."""
    e = np.asarray(E, dtype=np.complex128)
    if order_model == "weak":
        return np.asarray(0.5j * C * np.conj(e), dtype=np.complex128)
    if order_model != "j1":
        raise ValueError(f"order_model must be 'j1' or 'weak', got {order_model!r}")
    magnitude = np.abs(e)
    safe = np.where(magnitude > 0.0, magnitude, 1.0)
    unit = np.where(magnitude > 0.0, np.conj(e) / safe, 0.0)
    return np.asarray(1j * j1(C * magnitude) * unit, dtype=np.complex128)


def _norm(values: Complex, step: float) -> float:
    return float(math.sqrt(float(np.sum(np.abs(values) ** 2)) * step))


def _comb(f_live: Float) -> tuple[float, int, Float, float]:
    """``(Delta f, reference index, integer comb positions, residual)`` of the live tones."""
    order = np.argsort(f_live)
    gaps = np.diff(f_live[order])
    delta_f = float(np.min(gaps))
    centre = 0.5 * (float(np.min(f_live)) + float(np.max(f_live)))
    ref = int(np.argmin(np.abs(f_live - centre)))
    if delta_f <= COMB_TOL:
        return delta_f, ref, np.zeros(f_live.size), math.inf
    m = np.rint((f_live - f_live[ref]) / delta_f)
    residual = float(np.max(np.abs(f_live - f_live[ref] - m * delta_f)))
    return delta_f, ref, m, residual


def _comb_lines(
    drives: Sequence[Complex],
    m_pos: Float,
    K: int,
    C: float,
) -> tuple[Complex, NDArray[np.int64]]:
    """``K``-point DFT of the J1 transmission along the comb: ``(lines (K, *shape), m per bin)``.

    ``E~_k = sum_n b_n e^{2 pi i m_n k/K}``, ``T~_k = i J1(C|E~_k|) E~_k^*/|E~_k|``,
    ``L_m = (1/K) sum_k T~_k e^{+2 pi i m k/K}`` (``numpy.fft.ifft`` along ``k``); the bin
    order is that of ``numpy.fft.fftfreq`` (``m = 0 .. K/2-1, -K/2 .. -1``).  Evaluated in
    pixel blocks of at most :data:`DFT_BLOCK_PIXELS` to bound memory.
    """
    shape = drives[0].shape
    flat = [d.reshape(-1) for d in drives]
    n_pix = flat[0].size
    k = np.arange(K, dtype=np.float64)
    phases = np.exp(2j * np.pi * np.outer(m_pos, k) / K)  # (n_tones, K)
    out = np.empty((K, n_pix), dtype=np.complex128)
    block = max(1, DFT_BLOCK_PIXELS // K)
    for start in range(0, n_pix, block):
        stop = min(start + block, n_pix)
        e_tilde = np.zeros((K, stop - start), dtype=np.complex128)
        for b, ph in zip(flat, phases, strict=True):
            e_tilde += ph[:, None] * b[None, start:stop]
        out[:, start:stop] = np.fft.ifft(transmission(e_tilde, C, "j1"), axis=0)
    m_bins = np.rint(np.fft.fftfreq(K) * K).astype(np.int64)
    return out.reshape((K, *shape)), m_bins


def channel_lines(
    cw: ChannelWaveform,
    t: float,
    params: AODLParams,
    geom: ChannelGeometry,
    aod: AODParams,
    mis: AODMisalignment,
    model: PupilModel,
    rel_amp: RelAmp | None,
    u: Float,
    step: float,
) -> tuple[list[Line], list[str]]:
    """Optical-frequency lines of one channel at frame time ``t`` (module docstring).

    Parameters
    ----------
    cw, t, params, geom, aod, mis, model, rel_amp:
        The channel's drive, the frame time [s], hardware, the channel's orientation (from
        ``conventions``), its misalignment, the line model and the optional relative
        Bragg-efficiency amplitude ``a(f_abs)``.
    u:
        The channel coordinate on the grid — ``(n,)`` along the channel's lab axis when the
        channel is not rotated, ``(ny, nx)`` otherwise (:func:`channel_coordinate`).
    step:
        Grid pitch [m] for the line norms (``dx`` for 1-D lines, ``dx dy`` for 2-D ones).

    Returns the lines (empty when no tone is live) and human-readable notes about fallbacks.
    """
    notes: list[str] = []
    axis = geom.axis
    t_ret = retarded_time(t, u, geom, aod, mis)
    mask = filled(u, t, geom, aod, mis)
    if not np.any(mask):
        return [], notes
    lo = float(np.min(t_ret[mask]))
    hi = float(np.max(t_ret[mask]))
    live = live_tones(cw, lo, hi)
    if not live:
        return [], notes

    t_c = beam_center_time(t, aod)
    f_live = np.asarray([float(cw.tones[i].f(t_c)) for i in live], dtype=np.float64)
    a_live = np.asarray([float(cw.tones[i].A(t_c)) for i in live], dtype=np.float64)
    if rel_amp is None:
        amps = [1.0] * len(live)
    else:
        amps = [float(np.asarray(rel_amp(aod.f_center + f), dtype=np.float64)) for f in f_live]
    drives = tone_drives(cw, live, t_ret, amps, mask)
    C = float(aod.drive_strength)

    def make(values: Complex, tag: float) -> Line:
        return Line(f_tag=float(tag), values=values, axis=axis, norm=_norm(values, step))

    if model.order_model == "weak" or len(live) == 1 and model.order_model == "j1":
        if model.order_model == "weak":
            return [make(transmission(b, C, "weak"), f) for b, f in zip(drives, f_live)], notes
        return [make(transmission(drives[0], C, "j1"), float(f_live[0]))], notes

    # -- "j1" with several live tones: the comb decomposition.
    delta_f, ref, m_pos, residual = _comb(f_live)
    K = model.comb_dft_points
    if K == 1:
        total = np.zeros(drives[0].shape, dtype=np.complex128)
        for b in drives:
            total += b
        weights = (a_live * np.asarray(amps)) ** 2
        tag = (
            float(np.sum(weights * f_live) / np.sum(weights)) if weights.sum() > 0 else f_live[ref]
        )
        return [make(transmission(total, C, "j1"), tag)], notes
    span = int(np.max(m_pos) - np.min(m_pos)) if np.isfinite(residual) else K
    if residual > COMB_TOL or span >= K:
        why = (
            f"tones are not on one comb (residual {residual / kHz:.3g} kHz)"
            if residual > COMB_TOL
            else f"{len(live)} live tones span {span + 1} comb positions, more than the "
            f"{K} DFT points can hold"
        )
        notes.append(f"{why}: fell back to the 'weak' line model for this channel at this frame")
        return [make(transmission(b, C, "weak"), f) for b, f in zip(drives, f_live)], notes
    if 3 * int(np.max(np.abs(m_pos))) > (K - 1) // 2:
        notes.append(
            f"comb_dft_points = {K} cannot resolve every third-order product of {len(live)} "
            f"live tones (needs K >= {6 * int(np.max(np.abs(m_pos))) + 1}); the farthest IM3 "
            f"ghosts alias onto nearer bins"
        )
    lines_k, m_bins = _comb_lines(drives, m_pos, K, C)
    norms = np.asarray([_norm(lines_k[i], step) for i in range(K)])
    keep = norms >= model.line_cut * float(np.max(norms))
    f_ref = float(f_live[ref])
    return [
        Line(f_tag=f_ref + int(m) * delta_f, values=lines_k[i], axis=axis, norm=float(norms[i]))
        for i, m in enumerate(m_bins)
        if keep[i]
    ], notes


# ------------------------------------------------------------------------- grouping


def _cap_diameter(chunk: NDArray[np.intp], df: Float, tol: float) -> list[NDArray[np.intp]]:
    """Split a neighbour-chained cluster at its largest gaps until each piece spans <= tol."""
    out: list[NDArray[np.intp]] = []
    stack = [chunk]
    while stack:
        piece = stack.pop()
        values = df[piece]
        if piece.size < 2 or values[-1] - values[0] <= tol:
            out.append(piece)
            continue
        cut = int(np.argmax(np.diff(values))) + 1
        stack.append(piece[cut:])
        stack.append(piece[:cut])
    return out


def group_indices(df: ArrayLike, tol: float = GROUP_TOL) -> list[NDArray[np.intp]]:
    """Cluster tags by frequency — the ``aodl.field.focal.group_terms`` rule, re-implemented.

    Consecutive tags more than ``tol`` apart start a new group (neighbour chaining) and no
    group may span more than ``tol`` end to end (diameter cap, largest gap cut first).  Returns
    index arrays ordered by frequency, each sorted ascending.
    """
    values = np.asarray(df, dtype=np.float64).ravel()
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    splits = np.flatnonzero(np.diff(values[order]) > tol) + 1
    clusters = np.split(order, splits)
    return [np.sort(chunk) for cluster in clusters for chunk in _cap_diameter(cluster, values, tol)]


def _axis_product(
    lines_per_channel: Mapping[str, Sequence[Line]], axis: int, weight: Complex
) -> tuple[Complex, Float, Float]:
    """Product of the (separable) lines of the channels sharing one axis, times ``weight``.

    Returns ``(values (P, n), tags (P,), norms (P,))``; an axis with no driven channel yields
    the single identity line.
    """
    n = weight.size
    values = [np.ones(n, dtype=np.complex128)]
    tags = [0.0]
    for name in CHANNELS:
        lines = lines_per_channel.get(name)
        if not lines or geometry(name).axis != axis:
            continue
        values = [v * line.values for v in values for line in lines]
        tags = [t + line.f_tag for t in tags for line in lines]
    stacked = np.asarray(values, dtype=np.complex128) * weight[None, :]
    step = 1.0  # relative weights only: both axes share the grid pitch
    norms = np.sqrt(np.sum(np.abs(stacked) ** 2, axis=1) * step)
    return stacked, np.asarray(tags, dtype=np.float64), np.asarray(norms, dtype=np.float64)


def _separable_groups(
    lines_per_channel: Mapping[str, Sequence[Line]],
    gx: Complex,
    gy: Complex,
    model: PupilModel,
) -> list[Group]:
    lx, tx, nx = _axis_product(lines_per_channel, 0, gx)
    ly, ty, ny = _axis_product(lines_per_channel, 1, gy)
    tags = (tx[:, None] + ty[None, :]).ravel()
    weights = (nx[:, None] * ny[None, :]).ravel()
    keep = weights >= model.term_cut * float(np.max(weights))
    idx = np.flatnonzero(keep)
    ix, iy = np.divmod(idx, ty.size)
    tags, weights = tags[idx], weights[idx]
    groups: list[Group] = []
    for members in group_indices(tags):
        sel_x = ix[members]
        sel_y = iy[members]

        def build(sx: NDArray[np.intp] = sel_x, sy: NDArray[np.intp] = sel_y) -> Complex:
            return np.asarray(np.einsum("tn,tm->nm", ly[sy], lx[sx]), dtype=np.complex128)

        power = float(np.sum(weights[members] ** 2))
        f_opt = float(np.sum(tags[members] * weights[members] ** 2) / power) if power > 0 else 0.0
        groups.append(Group(f_opt=f_opt, power=power, n_terms=int(members.size), _build=build))
    return groups


def _as_2d(line: Line, shape: tuple[int, int]) -> Complex:
    if not line.separable:
        return line.values
    if line.axis == AXIS_X:
        return np.asarray(np.broadcast_to(line.values[None, :], shape), dtype=np.complex128)
    return np.asarray(np.broadcast_to(line.values[:, None], shape), dtype=np.complex128)


def _general_groups(
    lines_per_channel: Mapping[str, Sequence[Line]],
    g2d: Complex,
    model: PupilModel,
) -> list[Group]:
    names = [name for name in CHANNELS if lines_per_channel.get(name)]
    shape = g2d.shape
    planes = [[_as_2d(line, shape) for line in lines_per_channel[name]] for name in names]
    tag_lists = [[line.f_tag for line in lines_per_channel[name]] for name in names]
    norm_lists = [[line.norm for line in lines_per_channel[name]] for name in names]
    combos = list(itertools.product(*(range(len(p)) for p in planes)))
    tags = np.asarray(
        [sum(tag_lists[c][i] for c, i in enumerate(combo)) for combo in combos], dtype=np.float64
    )
    weights = np.asarray(
        [math.prod(norm_lists[c][i] for c, i in enumerate(combo)) for combo in combos],
        dtype=np.float64,
    )
    keep = weights >= model.term_cut * float(np.max(weights))
    idx = np.flatnonzero(keep)
    combos = [combos[i] for i in idx]
    tags, weights = tags[idx], weights[idx]
    groups: list[Group] = []
    for members in group_indices(tags):
        chosen = [combos[i] for i in members]

        def build(selected: list[tuple[int, ...]] = chosen) -> Complex:
            total = np.zeros(shape, dtype=np.complex128)
            for combo in selected:
                term = np.ones(shape, dtype=np.complex128)
                for c, i in enumerate(combo):
                    term = term * planes[c][i]
                total += term
            return np.asarray(total * g2d, dtype=np.complex128)

        power = float(np.sum(weights[members] ** 2))
        f_opt = float(np.sum(tags[members] * weights[members] ** 2) / power) if power > 0 else 0.0
        groups.append(Group(f_opt=f_opt, power=power, n_terms=int(members.size), _build=build))
    return groups


def pupil_groups(
    lines_per_channel: Mapping[str, Sequence[Line]],
    params: AODLParams,
    max_groups: int,
    x: Float,
    y: Float,
    tilt: tuple[float, float] = (0.0, 0.0),
    model: PupilModel | None = None,
) -> tuple[list[Group], int]:
    """Cartesian product of the channels' lines, grouped by optical frequency sum.

    Each group's pupil is ``exp(-(x^2+y^2)/w_in^2) * sum_{terms} prod_mu L_mu * exp(i k (alpha_x
    x + alpha_y y))`` with ``(alpha_x, alpha_y) = tilt`` the summed pointing tilt of the four
    channels (common to every term).  Groups are ranked by their incoherent pupil power and the
    brightest ``max_groups`` are returned (brightest first); the second value is how many were
    dropped.  A driven channel with no live line makes the frame dark (no groups).
    """
    cfg = PupilModel() if model is None else model
    present = {name: list(lines) for name, lines in lines_per_channel.items() if lines}
    if len(present) != len(lines_per_channel):
        return [], 0
    if not present:
        return [], 0
    k = params.optics.k
    w_in = params.optics.w_in
    gx = np.exp(-(x * x) / (w_in * w_in) + 1j * k * tilt[0] * x).astype(np.complex128)
    gy = np.exp(-(y * y) / (w_in * w_in) + 1j * k * tilt[1] * y).astype(np.complex128)
    if all(line.separable for lines in present.values() for line in lines):
        groups = _separable_groups(present, gx, gy, cfg)
    else:
        groups = _general_groups(present, gy[:, None] * gx[None, :], cfg)
    groups.sort(key=lambda g: g.power, reverse=True)
    dropped = max(0, len(groups) - int(max_groups))
    return groups[: int(max_groups)], dropped


__all__ = [
    "COMB_TOL",
    "GROUP_TOL",
    "LINE_CUT",
    "TERM_CUT",
    "Group",
    "Line",
    "PupilModel",
    "axis_vectors",
    "channel_coordinate",
    "channel_lines",
    "complex_drive",
    "filled",
    "group_indices",
    "live_tones",
    "pupil_grid",
    "pupil_groups",
    "retarded_time",
    "static_offset",
    "tone_drives",
    "transmission",
]
