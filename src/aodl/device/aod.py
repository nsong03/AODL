r"""One AOD channel: retarded-time evaluation, emission lines, aperture fill state.

This is layer L3's per-channel half (``docs/ARCHITECTURE.md`` §1): it turns a
:class:`~aodl.waveform.tones.ChannelWaveform` into the *lines* that
:mod:`aodl.device.aodl` multiplies into pupil terms, and reports how much of the aperture
currently holds drive content.

Every sign used here comes from :mod:`aodl.device.conventions`; nothing in this module
decides an orientation on its own.

**Weak drive (Eq. S1-S3).**  With drive ``V(t) = sum_n A_n(t) cos(Phi_n(t))`` and drive
strength ``C = AODParams.drive_strength``, the transmission ``exp(i C V)`` expands to
``1 + i C V + O(C^2)``.  The ``+1`` order keeps the ``exp(-i Phi_n)`` half of each cosine,
so tone ``n`` contributes the pupil factor

    amp_n * alpha_n(u) * exp(i (theta1_n u + theta2_n u^2 + ...))
    amp_n = (i C / 2) A_n(t_c) exp(-i phase_n(t_c))

with ``t_c = t - tau/2`` the beam-center retarded time (Eq. S6, rotating frame — the
carrier ``f_center`` is dropped, see :mod:`~aodl.device.conventions`).  Higher orders of
the same expansion — compression and intra-AOD intermodulation, Eqs. S20-S22 — live in
:mod:`aodl.device.mixing`, which :func:`channel_lines` calls to turn those fundamentals
into the full line set; ``mixing_order=1`` keeps this module's M1 behaviour exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..params import AODParams
from .conventions import (
    ChannelGeometry,
    FillEdge,
    beam_center_time,
    filled_side,
    is_filled,
    retarded_time,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle: mixing.py needs Lines from here
    from .mixing import MixingConfig

#: Quantities the device layer needs per tone at one retarded time.  Matches the keys of
#: ``ChannelWaveform.eval_table`` (WO-02 §2); the per-tone API is used as a fallback.
TABLE_KEYS: tuple[str, ...] = ("f", "fdot", "A", "dA", "d2A", "phase")


@dataclass(frozen=True)
class Lines:
    """Emission lines of one channel at one frame time — a struct of arrays.

    All arrays are 1-D with one entry per line: one line per tone at ``mixing_order=1``,
    plus the compression and IM3 products of :mod:`aodl.device.mixing` at order 3.  Values
    are taken at the **beam-center retarded time** ``t_c = t - tau/2``.

    Attributes
    ----------
    amp:
        Complex line amplitude; for a fundamental ``(i C / 2) A(t_c) exp(-i phase(t_c))``
        (Eq. S3/S6, rotating frame).  The envelope value is *already folded in here*, which
        is why :mod:`aodl.device.aodl` uses only the normalized *shape* of the aperture
        amplitude polynomial.
    f, fdot:
        Rotating-frame detuning [Hz] and chirp rate [Hz/s] at ``t_c``.
    A, dA, d2A:
        Envelope value and its first two time derivatives at ``t_c`` [1, 1/s, 1/s^2] —
        the raw ingredients of the Eq. S5 aperture amplitude polynomial.  A mixing product
        reports the *effective* envelope of its constituents (``A_i A_j A_k`` and its
        derivatives, :mod:`aodl.device.mixing`); only the polynomial's shape is used
        downstream, so the magnitude is never counted twice.
    pruned_power:
        Diagnostic: ``sum |amp|^2`` of the mixing products dropped by the amplitude cut
        (:attr:`aodl.device.mixing.MixingConfig.line_prune`); ``0`` when nothing was
        dropped.  Compare against ``sum |amp|^2`` of the surviving lines for a relative
        figure.
    """

    amp: NDArray[np.complex128]
    f: NDArray[np.float64]
    fdot: NDArray[np.float64]
    A: NDArray[np.float64]
    dA: NDArray[np.float64]
    d2A: NDArray[np.float64]
    pruned_power: float = 0.0

    @property
    def n_lines(self) -> int:
        """Number of emission lines."""
        return int(self.amp.size)

    def __len__(self) -> int:
        return self.n_lines


def _as_row(values: Any) -> NDArray[np.float64]:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def _table_from_eval_table(cw: Any, t: float) -> Mapping[str, Any] | None:
    """Preferred path: ``ChannelWaveform.eval_table(t)`` (WO-02 §2, one vectorized pass)."""
    evaluator = getattr(cw, "eval_table", None)
    if not callable(evaluator):
        return None
    try:
        table = evaluator(t)
    except (AttributeError, TypeError, KeyError, ValueError):
        return None
    if not isinstance(table, Mapping) or not set(TABLE_KEYS) <= set(table):
        return None
    return table


def _table_from_tones(cw: Any, t: float) -> Mapping[str, Any]:
    """Fallback path: per-tone ``ToneTrack`` / ``Envelope`` accessors (WO-02 §2)."""
    tones = tuple(getattr(cw, "tones", ()))
    return {
        "f": [tone.f(t) for tone in tones],
        "fdot": [tone.fdot(t) for tone in tones],
        "A": [tone.env.A(t) for tone in tones],
        "dA": [tone.env.dA(t) for tone in tones],
        "d2A": [tone.env.d2A(t) for tone in tones],
        "phase": [tone.phase(t) for tone in tones],
    }


def channel_table(cw: Any, t: float) -> dict[str, NDArray[np.float64]]:
    """Per-tone ``f, fdot, A, dA, d2A, phase`` of ``cw`` at scalar drive time ``t``.

    Uses :meth:`ChannelWaveform.eval_table` when it is available and complete, and falls
    back to the per-tone ``ToneTrack``/``Envelope`` accessors otherwise, so the device
    layer keeps working while the waveform layer is being built.
    """
    table = _table_from_eval_table(cw, t)
    if table is None:
        table = _table_from_tones(cw, t)
    return {key: _as_row(table[key]) for key in TABLE_KEYS}


def channel_lines(cw: Any, aod: AODParams, t: float, mixing: MixingConfig | None = None) -> Lines:
    """Emission lines of channel waveform ``cw`` at frame time ``t`` (Eq. S3/S6).

    Everything is evaluated at the beam-center retarded time ``t_c = t - tau/2``: the
    drive sample illuminating the aperture center was emitted half a transit earlier.  The
    fundamentals (one line per tone) are then handed to
    :func:`aodl.device.mixing.expand_lines`, which adds the compression and IM3 lines of
    Eqs. S20-S22 — or returns them untouched at expansion order 1.

    Parameters
    ----------
    cw:
        A :class:`~aodl.waveform.tones.ChannelWaveform` (duck-typed: see
        :func:`channel_table`).
    aod:
        Channel hardware parameters (supplies ``v``, ``D``, ``C``, the band and the default
        ``mixing_order``).
    t:
        Frame time [s] (the *observation* time, not a drive time).
    mixing:
        Expansion order and selection cuts.  ``None`` (the default) means
        ``MixingConfig(order=aod.mixing_order)``; pass a :class:`
        ~aodl.device.mixing.MixingConfig` to override the channel's own setting.
    """
    from .mixing import MixingConfig, expand_lines  # local: mixing.py imports Lines

    t_c = beam_center_time(t, aod)
    table = channel_table(cw, t_c)
    amp = (
        0.5j * aod.drive_strength * table["A"].astype(np.complex128) * np.exp(-1j * table["phase"])
    )
    fundamentals = Lines(
        amp=np.asarray(amp, dtype=np.complex128),
        f=table["f"],
        fdot=table["fdot"],
        A=table["A"],
        dA=table["dA"],
        d2A=table["d2A"],
    )
    cfg = MixingConfig(order=aod.mixing_order) if mixing is None else mixing
    return expand_lines(fundamentals, aod, cfg)


def fill_edge(aod: AODParams, geom: ChannelGeometry, t: float) -> FillEdge | None:
    """Leading edge of the acoustic column, or ``None`` once the aperture is full.

    The drive starts at ``t = 0``, so content exists exactly where ``s u <= v t - D/2``
    (:mod:`~aodl.device.conventions`).  The wavefront therefore sits at
    ``u_edge = s (v t - D/2)`` and the **filled side** is ``"upper"`` (``u <= u_edge``) for
    ``sound_sign = +1`` and ``"lower"`` (``u >= u_edge``) for ``sound_sign = -1`` — the
    half-line containing the transducer at ``u = -s D/2``.  ``field/`` selects
    :func:`~aodl.field.gaussian.gauss_moments_upper` / ``..._lower`` accordingly.

    Returns ``None`` for ``t >= tau = D / v`` (fully filled).

    Note
    ----
    The optical model uses an **uncropped** Gaussian input beam (``docs/PLAN.md`` §1.5
    decision 2), so this fill edge is the only aperture window in the field integrals; the
    physical ``|u| <= D/2`` crop is deliberately not applied.  Before the wavefront reaches
    the beam (``|u_edge|`` beyond a few ``w_in``) the edge simply suppresses everything,
    which is the intended behaviour.

    Returns
    -------
    A :class:`~aodl.device.conventions.FillEdge` — a ``(u_edge, side)`` named tuple — or
    ``None``.
    """
    # Compare times rather than positions: `t >= tau` is exact at the boundary, whereas
    # `v t - D/2 >= D/2` can round just short of it and crop the pupil by a hair.
    if float(t) >= aod.transit_time:
        return None
    reach = aod.sound_speed * float(t) - 0.5 * aod.aperture
    return FillEdge(u_edge=float(geom.sound_sign * reach), side=filled_side(geom))


def aperture_window(
    cw: Any, aod: AODParams, geom: ChannelGeometry, t: float, n: int = 512
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Diagnostic: the literal RF waveform sitting on the crystal at frame time ``t``.

    The drive of Eq. S1 sampled at Eq. S4's retarded time — i.e. the acoustic field itself,
    *before* the Eqs. S5-S6 beam-center Taylor expansion that the rest of the device layer
    works with.  Returns ``(u, V)`` on ``n`` points spanning the physical aperture
    ``u in [-D/2, +D/2]``, with

        V(u) = sum_n A_n(t_ret(u)) cos(2 pi f_center t_ret(u) + phase_n(t_ret(u)))

    and ``V = 0`` wherever the aperture is not yet filled (``t_ret(u) < 0``,
    ``docs/conventions.md`` §7).  This is the *absolute* RF signal (carrier included, matching
    ``waveform/export.py``), meant for plotting the startup transient — the field path never
    calls it, and the physical ``|u| <= D/2`` crop shown here is deliberately *not* applied to
    the field integrals (``docs/PLAN.md`` §1.5, decision 2).
    """
    half = 0.5 * aod.aperture
    u = np.linspace(-half, half, int(n))
    t_ret = retarded_time(t, u, geom, aod)
    signal = np.zeros_like(u)
    for tone in tuple(getattr(cw, "tones", ())):
        env = np.asarray(tone.env.A(t_ret), dtype=np.float64)
        phase = np.asarray(tone.phase(t_ret), dtype=np.float64)
        signal += env * np.cos(2.0 * math.pi * aod.f_center * t_ret + phase)
    return u, np.where(is_filled(u, t, geom, aod), signal, 0.0)


__all__ = ["TABLE_KEYS", "Lines", "aperture_window", "channel_lines", "channel_table", "fill_edge"]
