r"""Intra-AOD intermodulation: the weak-drive expansion past first order (Eqs. S20-S22).

``device/aod.py`` builds the *fundamentals*: the first term of ``exp(i C V) ~ 1 + i C V``,
one emission line per drive tone.  That truncation is only the leading edge of the real
transmission function.  Expanding the full exponential produces, inside the ``+1``
diffraction band, two families of extra lines (``docs/PLAN.md`` §1.2):

* **compression** of every fundamental (the drive saturates: ``J_1(m)`` bends over), and
* **third-order intermodulation (IM3)** at ``f_j + f_k - f_i``, which is *in band* whenever
  the tones are — the ghost tweezers of an atom-array drive.

**Derivation.**  Write the drive of one channel as ``V(t') = sum_n A_n(t') cos(Phi_n(t'))``
with ``Phi_n(t') = 2 pi f_center t' + phase_n(t')``, and let

    m_n = C A_n(t_c)          per-tone modulation depth (C = AODParams.drive_strength)
    phi_n = phase_n(t_c)      rotating-frame tone phase at the beam-center retarded time

The transmission factorizes over tones and each factor is a Jacobi-Anger series,

    exp(i C V) = prod_n sum_{p} i^p J_p(m_n) exp(i p Phi_n),

so a product term is labelled by integers ``p_n``.  Its optical frequency is
``-sum_n p_n (f_center + f_n)``: the ``+1`` diffraction order — the one that *up*-shifts the
light by one carrier and imprints ``-Phi`` (``docs/conventions.md`` §3-4) — is the band
``sum_n p_n = -1``.  Keeping ``sum_n |p_n| <= 3`` inside that band leaves exactly three
signature classes, and ``i^{-1} J_{-1}(m) = i J_1(m)``, ``i^{-2} J_{-2}(m) = -J_2(m)``:

===================== ================== ===================== =============================
class                 detuning           chirp                 complex amplitude
===================== ================== ===================== =============================
fundamental ``n``     ``f_n``            ``fdot_n``            ``i J_1(m_n) prod_{m!=n} J_0``
IM3 ``j<k``, ``i``    ``f_j+f_k-f_i``    ``fdot_j+fdot_k-``    ``-(i/8) m_i m_j m_k``
                                         ``fdot_i``
IM3 degenerate        ``2 f_j - f_i``    ``2 fdot_j - fdot_i``  ``-(i/16) m_i m_j^2``
===================== ================== ===================== =============================

each carrying the phase factor ``exp(-i sum_n p_n' phi_n)`` with the same coefficients
(``exp(-i phi_n)``, ``exp(-i(phi_j+phi_k-phi_i))``, ``exp(-i(2 phi_j - phi_i))``).
Truncating the Bessel functions at third order turns the fundamental into

    (i/2) m_n [1 - m_n^2/8 - (1/4) sum_{m!=n} m_m^2]

— the ``m_n^3`` piece is single-tone compression (the first two terms of ``i J_1``), the
cross piece is compression *by the other tones* sharing the crystal.  This module folds
that correction into the fundamental line itself, so a single tone at order 3 has exactly
one line of amplitude ``(i/2) m (1 - m^2/8)`` and the whole expansion stays a strict
superset of the order-1 line set.

**No phases are needed as input.**  Every amplitude above is a product of the fundamental
amplitudes ``a_n = (i/2) m_n exp(-i phi_n)`` that :func:`aodl.device.aod.channel_lines`
already computed:

    IM3 ``(i, j, k)``      amp = ``-a_j a_k conj(a_i)``
    IM3 degenerate         amp = ``-(1/2) a_j^2 conj(a_i)``
    fundamental ``n``      amp = ``a_n (1 - S/4 + m_n^2/8)``,  ``S = sum_m m_m^2``

(verify: ``a_j a_k conj(a_i) = (i/8) m_i m_j m_k exp(-i(phi_j+phi_k-phi_i))``).

**Envelopes.**  A mixed line's envelope is the product of its constituents *with
multiplicity* — ``A_i A_j A_k`` or ``A_i A_j^2`` — because the amplitude is a product of the
corresponding ``m``'s.  Products are handled through log-derivatives: with
``l1_n = A'_n/A_n`` and ``l2_n = A''_n/A_n``,

    L1 = sum mult * l1,     L1' = sum mult * (l2 - l1^2),     A''/A = L1^2 + L1'

so the line's Eq. S5 aperture polynomial keeps the fundamentals' normalized form
``(1, -s L1 / v, (L1^2 + L1')/(2 v^2))``.  The line therefore reports an *effective*
``A, dA, d2A`` triple and :mod:`aodl.device.aodl` normalizes it exactly as before, with the
envelope magnitude living (once) in the complex amplitude.

**Selection.**  Mixing products are subject to two cuts, both configurable through
:class:`MixingConfig`:

* *band acceptance* — a product whose absolute frequency ``f_center + f_line`` falls outside
  the channel band widened by ``band_margin`` is not launched by the transducer at all
  (this is what keeps IM2 at ``f_i +/- f_j`` out of the model: it lands near DC or near
  ``2 f_center``, and only its remix to IM3 returns in band);
* *amplitude pruning* — products weaker than ``line_prune`` times the strongest fundamental
  are dropped, after coherently merging frequency-degenerate lines.  Dropping an amplitude
  ``eps`` relative to a fundamental changes that spot's *intensity* by ~``eps^2``, so the
  default ``1e-5`` bounds the relative intensity error at ~``1e-10``; the dropped power
  ``sum |amp|^2`` is reported in :attr:`aodl.device.aod.Lines.pruned_power`.

Both cuts act on mixing products only: the programmed tones themselves are always kept, so
``mixing_order=3`` output is a superset of ``mixing_order=1`` output and an out-of-band or
faded drive tone never silently disappears from the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..params import AODParams
from .aod import Lines

#: Expansion orders this module implements: ``1`` = fundamentals only (Eq. S3, the M1
#: model), ``3`` = compression + IM3 (Eqs. S20-S22).  Even orders are absent by symmetry:
#: an order-``p`` product lands in the ``+1`` band only if ``sum p_n = -1``, which forces
#: ``sum |p_n|`` odd.
ALLOWED_ORDERS: tuple[int, ...] = (1, 3)

#: Relative tolerance used when deciding that two lines coincide (same frequency, chirp and
#: envelope shape) and must therefore be merged into one coherent line.  Quantization is
#: relative to the largest value present, with the absolute floors below for the all-zero
#: case.  Real drives separate their lines by kHz and their chirps by MHz/ms, i.e. by ~10^9
#: times these tolerances, while exact degeneracies (an IM3 landing on a fundamental) agree
#: to a few ULP.
MERGE_RTOL = 1e-9
#: Absolute floors of the merge quantizer, per quantity: frequency [Hz], chirp rate [Hz/s],
#: envelope log-derivative ``A'/A`` [1/s] and ``A''/A`` [1/s^2].
MERGE_ATOL: tuple[float, float, float, float] = (1e-3, 1.0, 1e-6, 1e-6)


@dataclass(frozen=True)
class MixingConfig:
    """How far to expand ``exp(i C V)`` and which product lines to keep.

    Attributes
    ----------
    order:
        Expansion order, one of :data:`ALLOWED_ORDERS`.  ``1`` returns the fundamentals
        untouched (bit-for-bit the M1 model); ``3`` adds compression and IM3.
    band_margin:
        Band acceptance window for *product* lines, as a fraction of the channel band
        width added on each side: a product at absolute frequency ``f_center + f_line`` is
        kept iff ``lo - band_margin * (hi - lo) <= f <= hi + band_margin * (hi - lo)``.
    line_prune:
        Drop product lines whose ``|amp|`` is below ``line_prune`` times the largest
        fundamental ``|amp|`` (relative intensity error ~ ``line_prune^2``).
    """

    order: int = 3
    band_margin: float = 0.2
    line_prune: float = 1e-5

    def __post_init__(self) -> None:
        if self.order not in ALLOWED_ORDERS:
            raise ValueError(f"mixing order must be one of {ALLOWED_ORDERS}, got {self.order!r}")
        if not np.isfinite(self.band_margin) or self.band_margin < 0.0:
            raise ValueError(f"band_margin must be finite and >= 0, got {self.band_margin!r}")
        if not np.isfinite(self.line_prune) or self.line_prune < 0.0:
            raise ValueError(f"line_prune must be finite and >= 0, got {self.line_prune!r}")


def _distinct_triples(n: int) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Index arrays ``(i, j, k)`` of the IM3 signature ``f_j + f_k - f_i``, ``j < k``, ``i``
    distinct from both.  ``n (n-1) (n-2) / 2`` entries, pair-major order."""
    j, k = np.triu_indices(n, k=1)
    i = np.arange(n)
    allowed = (i[None, :] != j[:, None]) & (i[None, :] != k[:, None])
    pair, i_idx = np.nonzero(allowed)
    return i_idx, j[pair], k[pair]


def _degenerate_pairs(n: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Index arrays ``(i, j)`` of the degenerate IM3 signature ``2 f_j - f_i``, ``i != j``."""
    j, i = np.nonzero(~np.eye(n, dtype=bool))
    return i, j


def _log_derivatives(
    line: Lines,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-tone envelope log-derivatives ``(A'/A, A''/A)``, zero where the envelope vanishes.

    A tone with ``A = 0`` contributes no amplitude to any line it takes part in, so the
    shape of its (absent) envelope is irrelevant; returning zeros keeps the division safe.
    """
    live = line.A != 0.0
    inverse = np.where(live, 1.0 / np.where(live, line.A, 1.0), 0.0)
    return line.dA * inverse, line.d2A * inverse


def _quantized_key(values: NDArray[np.float64], atol: float) -> NDArray[np.int64]:
    """Integer key that is equal exactly for values agreeing to :data:`MERGE_RTOL`."""
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    return np.rint(values / (atol + MERGE_RTOL * peak)).astype(np.int64)


def _merge_groups(
    f: NDArray[np.float64],
    fdot: NDArray[np.float64],
    l1: NDArray[np.float64],
    l2: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp], int]:
    """Group lines that are physically *the same* line: same ``f``, ``fdot`` and envelope shape.

    Frequency-degenerate lines (an IM3 landing exactly on a fundamental, say) interfere
    coherently and must be summed, not listed twice — and summing them is only meaningful
    when their aperture polynomials agree too, which is what the envelope keys enforce.

    Returns ``(representative_index, group_of_line, n_groups)``, where the representative of
    each group is its earliest member (so a fundamental always represents its own group).
    """
    keys = np.stack(
        [
            _quantized_key(f, MERGE_ATOL[0]),
            _quantized_key(fdot, MERGE_ATOL[1]),
            _quantized_key(l1, MERGE_ATOL[2]),
            _quantized_key(l2, MERGE_ATOL[3]),
        ]
    )
    order = np.lexsort(keys[::-1])  # primary key first -> reversed for lexsort
    ordered = keys[:, order]
    starts = np.ones(order.size, dtype=bool)
    starts[1:] = np.any(ordered[:, 1:] != ordered[:, :-1], axis=0)
    group = np.empty(order.size, dtype=np.intp)
    group[order] = np.cumsum(starts) - 1
    return order[starts], group, int(starts.sum())


def expand_lines(fund: Lines, aod: AODParams, cfg: MixingConfig = MixingConfig()) -> Lines:
    """Expand one channel's fundamentals into the full mixed line set (Eqs. S20-S22).

    Parameters
    ----------
    fund:
        The fundamentals-only :class:`~aodl.device.aod.Lines` of one channel at one frame
        time, exactly as :func:`aodl.device.aod.channel_lines` assembles them: ``amp`` must
        be ``(i C / 2) A_n(t_c) exp(-i phase_n(t_c))`` (Eq. S3), which is what carries the
        per-tone phases into the products.
    aod:
        Channel hardware parameters (supplies ``C`` and the acceptance band).
    cfg:
        Expansion order and selection cuts; see :class:`MixingConfig`.

    Returns
    -------
    A :class:`~aodl.device.aod.Lines` holding the compression-corrected fundamentals plus
    the surviving IM3 products, frequency-degenerate lines already merged coherently, with
    the amplitude-pruned power in ``pruned_power``.  ``cfg.order == 1`` returns ``fund``
    itself, unchanged.
    """
    n_tones = fund.n_lines
    if cfg.order == 1 or n_tones == 0:
        return fund

    amp = fund.amp
    depth = float(aod.drive_strength) * fund.A  # m_n = C A_n(t_c)
    total_depth = float(np.sum(depth**2))  # S = sum_m m_m^2
    l1, l2 = _log_derivatives(fund)

    # -- fundamentals, compressed by themselves (m_n^3/8) and by their neighbours (S/4).
    parts = [
        (
            amp * (1.0 - 0.25 * total_depth + 0.125 * depth**2),
            fund.f,
            fund.fdot,
            fund.A,
            l1,
            l2,
        )
    ]

    # -- IM3 with three distinct tones: f_j + f_k - f_i, amp = -a_j a_k conj(a_i).
    if n_tones >= 3:
        i, j, k = _distinct_triples(n_tones)
        big1 = l1[i] + l1[j] + l1[k]
        big2 = (l2[i] - l1[i] ** 2) + (l2[j] - l1[j] ** 2) + (l2[k] - l1[k] ** 2)
        parts.append(
            (
                -(amp[j] * amp[k] * np.conj(amp[i])),
                fund.f[j] + fund.f[k] - fund.f[i],
                fund.fdot[j] + fund.fdot[k] - fund.fdot[i],
                fund.A[i] * fund.A[j] * fund.A[k],
                big1,
                big1**2 + big2,
            )
        )

    # -- degenerate IM3, tone j counted twice: 2 f_j - f_i, amp = -(1/2) a_j^2 conj(a_i).
    if n_tones >= 2:
        i, j = _degenerate_pairs(n_tones)
        big1 = 2.0 * l1[j] + l1[i]
        big2 = 2.0 * (l2[j] - l1[j] ** 2) + (l2[i] - l1[i] ** 2)
        parts.append(
            (
                -0.5 * amp[j] ** 2 * np.conj(amp[i]),
                2.0 * fund.f[j] - fund.f[i],
                2.0 * fund.fdot[j] - fund.fdot[i],
                fund.A[j] ** 2 * fund.A[i],
                big1,
                big1**2 + big2,
            )
        )

    line_amp, line_f, line_fdot, line_a, line_l1, line_l2 = (
        np.concatenate(column) for column in zip(*parts, strict=True)
    )
    is_fundamental = np.zeros(line_amp.size, dtype=bool)
    is_fundamental[:n_tones] = True

    # -- band acceptance (products only): the transducer never launches them.
    lo, hi = aod.band
    margin = cfg.band_margin * (hi - lo)
    absolute = aod.f_center + line_f
    in_band = is_fundamental | ((absolute >= lo - margin) & (absolute <= hi + margin))
    if not in_band.all():
        line_amp, line_f, line_fdot, line_a, line_l1, line_l2, is_fundamental = (
            column[in_band]
            for column in (
                line_amp,
                line_f,
                line_fdot,
                line_a,
                line_l1,
                line_l2,
                is_fundamental,
            )
        )

    # -- coherent merge of degenerate lines, then the amplitude cut.
    first, group, n_groups = _merge_groups(line_f, line_fdot, line_l1, line_l2)
    merged_amp = np.bincount(group, weights=line_amp.real, minlength=n_groups) + 1j * np.bincount(
        group, weights=line_amp.imag, minlength=n_groups
    )
    merged_fundamental = (
        np.bincount(group, weights=is_fundamental.astype(np.float64), minlength=n_groups) > 0.0
    )
    strongest = float(np.max(np.abs(merged_amp[merged_fundamental])))
    dropped = ~merged_fundamental & (np.abs(merged_amp) < cfg.line_prune * strongest)
    keep = ~dropped
    pruned_power = float(np.sum(np.abs(merged_amp[dropped]) ** 2))

    index = first[keep]
    envelope = line_a[index]
    return Lines(
        amp=merged_amp[keep],
        f=line_f[index],
        fdot=line_fdot[index],
        A=envelope,
        dA=envelope * line_l1[index],
        d2A=envelope * line_l2[index],
        pruned_power=pruned_power,
    )


__all__ = ["ALLOWED_ORDERS", "MERGE_ATOL", "MERGE_RTOL", "MixingConfig", "expand_lines"]
