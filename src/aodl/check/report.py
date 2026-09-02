r"""The verdict: measured tweezers vs the requested ones, with the gaps named.

:func:`check_samples` is the whole checker in one call.  It takes the **rendered RF samples**
of a drive and an :class:`~aodl.check.expect.Expectation` of what that drive was asked to do,
rebuilds the tweezers with the FFT path of :mod:`aodl.check` — no Taylor expansion, no shared
code with the simulator — and returns a :class:`CheckReport` that either passes or names, in
words, every metric that did not.

Per frame the pipeline is::

    beat window W  ->  K golden-ratio sub-times  ->  per sub-time: 2 axis pupils (Eqs. S1-S4)
      ->  zoom transform onto a fine image grid at n_z defocus planes (Eq. S11)
      ->  correlated |U_x|^2 (x) |U_y|^2 accumulation
      ->  per-column / per-row Gaussian fits, w^2(Z) parabola -> Zbar and Delta F
      ->  blob audit of the full-field canvas against the expected lattice

Columns and rows are fitted rather than individual traps because the pupil is a *product*
(Eq. S7): every trap in a column shares one x profile exactly, so ``mx + my`` fits carry all
``mx * my`` traps and the cost of checking an array grows with its side, not its area.

**Why a window, and why that window.**  Atoms and cameras see the intensity averaged over the
MHz beat notes between pupil terms (``docs/PLAN.md`` §1.3), so a single instant is not what a
trap *is*.  The window is

.. math::  W = \min\Big(\frac{2}{\Delta f_\text{min}},\;
                        \frac{0.2\,w_0}{\max|v_\text{lat}|}\Big),

two full cycles of the slowest beat, capped so that the array cannot move more than a fifth of
a waist while the shutter is open, with the instants inside it the golden-ratio sequence of
:func:`aodl.check.transform.subtimes` — which never lines up with an array's regular beat comb.
That is the **fallback** (:func:`_beat_window`).  When the drive's same-node beats form a
commensurate comb — every fading array's do — :func:`_comb_window` takes precedence with one
full comb period and a *uniform* schedule, which annihilates each of those beats exactly; that
window is chosen for its beats alone and is **not** motion-capped (the ``docs/guide.md``
flagship's is 5 µs, over which the array travels 1.4 ``w_0``).  Either way the frame is
measured in the array's own moving frame (:func:`_desmear_shifts`), so what the smear cap
protects against — a fast traverse reported as a wide, dim, displaced spot — is removed
by construction rather than by shortening the window.

**What a PASS certifies, and what it does not.**  Verdict-bearing metrics are the ones a shared
model error cannot fake: lateral and axial position, the astigmatic interval, the fitted
waists, per-trap uniformity, missing traps and off-lattice light.  Deliberately report-only:

* **the absolute intensity scale.**  :func:`aodl.waveform.export.render_samples` divides every
  channel by one global peak, so a uniform gain on all four channels is invisible in the
  samples — and optically it *is* invisible, since it only rescales the whole image.  The
  checker therefore gates the *pattern* of intensity, never its overall level.  This is the
  known blind spot: a drive rendered at half amplitude checks out identically.
* the ``sim_delta`` comparison (:func:`aodl.check.expect.sim_delta`), the ``2 J_1(C)/C``
  compression the ``bragg_band`` model carries, ``beat_std``, the out-of-band splatter
  fraction, and everything measured on a transient or fade frame.

Frames before ``2 tau``, or with any channel's aperture grid still filling, are marked
*transient* and leave the waist and uniformity gates; while an aperture is genuinely still
filling the positions leave the gates too, because a half-lit pupil does not point anywhere in
particular.  On a fading-Shepard drive the array's two **edge** lines on each fading axis leave
the intensity gates as well: the ``p_A + p_B = 1`` identity holds every *interior* node exactly
flat through a hand-over, while the outermost node trades its light with the extended grid
(``docs/guide.md`` §6.6-§6.7) — documented physics, not a fault.

Every one of those exemptions narrows what a PASS is *about*, so the report counts them:
:attr:`CheckReport.gated_fraction` says what fraction of the rows each intensity gate actually
judged, and a metric that judged **none** — a 2x2 fading array, whose edge lines are its whole
perimeter — gets a note saying so and, under :attr:`Tolerances.require_coverage`, a failure.
The intensity pattern is read twice over: once per frame, and once as each trap's *time median*
(:meth:`CheckReport.median_uniformity`), which is what tells a fault that holds still apart from
hand-over ripple that does not.  The median is measured on every check and gated only when
:attr:`Tolerances.uniformity_median` is set, for the reason that method documents.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..params import AODLParams
from ..units import um, us
from .demod import Baseband, demodulate, out_of_band_fraction
from .expect import Expectation, ExpectedTraps, SimResultLike, sim_delta
from .metrics import (
    Blob,
    accumulate_marginals,
    best_focus,
    find_blobs,
    fit_gaussian_1d,
    profile_moments,
)
from .pupil import ApertureGrid, PupilMode, axis_pupil
from .record import SampleRecord, from_arrays, load_samples
from .transform import subtimes, zoom_field

Complex = NDArray[np.complex128]
Float = NDArray[np.float64]
Bool = NDArray[np.bool_]

#: Samples per waist on the fine image grid the per-trap fits are made on.
FINE_PER_WAIST = 16

#: Samples per waist on the full-field canvas the blob audit runs on.  The canvas is the fine
#: grid strided down, so no second transform is paid for it.
COARSE_PER_WAIST = 3

#: Half-width of a fit window, in waists.  Beyond ``e^{-2 x 3.5^2} = 3e-11`` there is nothing
#: left of a Gaussian, and the fit itself only uses the run above ``e^{-2}``.
FIT_HALF_WAISTS = 3.5

#: Largest fine grid the checker builds per axis.  A wider array is sampled more coarsely (the
#: report says so) rather than silently costing minutes.
MAX_FINE_SAMPLES = 8192

#: Beat-window cap: the array may not move more than this fraction of a waist while the
#: averaging window is open (module docstring).
MOTION_FRACTION = 0.2

#: Cycles of the slowest beat the window covers when nothing is moving.
BEAT_CYCLES = 2.0

#: Guard factor on the decimation bandwidth check (:func:`_decimation`).
DECIMATION_SAFETY = 1.5

#: Columns of :attr:`CheckReport.table`, in order — one row per (frame, requested trap).
CHECK_TABLE_KEYS: tuple[str, ...] = (
    "frame",
    "time",
    "ix",
    "iy",
    "x",
    "y",
    "z_lab",
    "delta_f",
    "sigma_astig",
    "wx",
    "wy",
    "peak",
    "power",
    "beat_std",
    "dx",
    "dy",
    "dz",
    "x_expect",
    "y_expect",
    "z_expect",
    "x_centroid",
    "y_centroid",
    "x_rms",
    "y_rms",
    "uniformity",
    "present",
    "transient",
    "filling",
    "in_fade",
    "gated",
    "verdict_trap",
    "verdict_frame",
    "uniformity_median",
)

#: The columns one frame can fill on its own.  ``uniformity_median`` is the one that cannot:
#: it is a statistic *over* frames, so :func:`check_samples` writes it once the frames are
#: joined (:func:`_time_median_uniformity`).
_FRAME_TABLE_KEYS: tuple[str, ...] = CHECK_TABLE_KEYS[:-1]

#: Fewest frames a per-trap time median means anything over.  With one frame it is the
#: per-frame deviation again, and the report says so rather than quoting it twice.
MIN_MEDIAN_FRAMES = 2

#: The metrics :attr:`CheckReport.gated_fraction` reports coverage for — the ones an edge-line
#: or fade exemption can empty out (``docs/guide.md`` §5.5).
COVERAGE_METRICS: tuple[str, ...] = ("waist", "uniformity", "uniformity_median")

__all__ = [
    "BEAT_CYCLES",
    "CHECK_TABLE_KEYS",
    "COARSE_PER_WAIST",
    "COVERAGE_METRICS",
    "FINE_PER_WAIST",
    "FIT_HALF_WAISTS",
    "MAX_FINE_SAMPLES",
    "MIN_MEDIAN_FRAMES",
    "MOTION_FRACTION",
    "CheckReport",
    "Tolerances",
    "averaging_window",
    "check_samples",
    "frame_reach",
]


# ------------------------------------------------------------------------------ tolerances


@dataclass(frozen=True)
class Tolerances:
    """The verdict thresholds.  Every one of them is a *relative* quantity.

    Attributes
    ----------
    lateral:
        Largest ``|dx|`` / ``|dy|`` in units of the focal waist ``w_0``.  0.05 ``w_0`` is 53 nm
        at the product optics — a twentieth of a spot, well under anything an atom notices, and
        far above the checker's own ~1e-3 ``w_0`` reconstruction floor.
    axial:
        Largest ``|dz|`` **and** ``|Delta F|`` in units of the Rayleigh range ``z_R``.  The
        astigmatic interval shares this gate because it is the same kind of length and the
        paper's claim is that it stays at zero.
    waist:
        Largest relative departure of a fitted ``wx``/``wy`` from ``w_0``, on non-transient,
        non-fade frames.
    uniformity:
        Largest relative departure of a trap's peak intensity from its frame's median, on
        non-fade frames.  This is the *pattern* gate, read one frame at a time; the absolute
        scale is not checked (module docstring).
    uniformity_median:
        Largest tolerated **time-median** of that same per-trap deviation, taken over the
        frames a trap is gated at (:meth:`CheckReport.median_uniformity`).  ``None`` — the
        default — measures and reports it without gating on it, which is the WO-24 ruling for
        the drives this package ships: see :meth:`CheckReport.median_uniformity` for the
        measured flagship numbers and why the statistic separates a *persistent* fault (an
        Eq. S19 tone) but not a *sliding* one (a fading-Shepard rung).  Set it to a number on a
        drive whose faults hold still and it becomes a second, verdict-bearing gate.
    blob_off_lattice:
        Brightest tolerated blob sitting on no expected lattice node, relative to the median
        trap peak.  Light there is steered somewhere the drive never asked for.
    blob_on_lattice:
        Brightest tolerated blob on an extended-lattice node that is not a requested trap, for
        an Eq. S19 drive — a commensurate IM3 product is the realistic occupant.
    blob_fading:
        The same, for a **fading** drive, whose extended lattice is lit on purpose
        (``docs/guide.md`` §6.7) at something close to full trap depth.  Whitelisted, but not
        unboundedly: 1.2 median trap peaks leaves room for the 1.07x mid-hand-over node WO-23
        measured while still calling light that is materially *brighter than a real trap* what
        it is.
    missing_trap:
        Faintest tolerated trap peak relative to the frame median, below which a trap counts as
        missing.
    require_coverage:
        Fail when an intensity metric ends up with **no gated rows at all**
        (:attr:`CheckReport.gated_fraction`).  A 2x2 fading array is the case that matters: its
        edge-line exemption is the whole perimeter, i.e. every trap, so ``passed`` would
        otherwise certify an intensity pattern nothing was measured against.  ``False``
        (default) keeps that a note; ``True`` makes it a failure.
    """

    lateral: float = 0.05
    axial: float = 0.05
    waist: float = 0.02
    uniformity: float = 0.03
    uniformity_median: float | None = None
    blob_off_lattice: float = 0.01
    blob_on_lattice: float = 0.10
    blob_fading: float = 1.2
    missing_trap: float = 0.25
    require_coverage: bool = False

    def __post_init__(self) -> None:
        for name in (
            "lateral",
            "axial",
            "waist",
            "uniformity",
            "blob_off_lattice",
            "blob_on_lattice",
            "blob_fading",
            "missing_trap",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Tolerances.{name} must be positive and finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.uniformity_median is not None:
            median = float(self.uniformity_median)
            if not math.isfinite(median) or median <= 0.0:
                raise ValueError(
                    "Tolerances.uniformity_median must be positive and finite, or None to "
                    f"measure it without gating on it, got {self.uniformity_median!r}"
                )
            object.__setattr__(self, "uniformity_median", median)
        object.__setattr__(self, "require_coverage", bool(self.require_coverage))


# ---------------------------------------------------------------------------------- report


@dataclass(frozen=True)
class CheckReport:
    """What the checker measured, whether it passes, and why not when it does not.

    Attributes
    ----------
    passed:
        ``True`` when ``failures`` is empty.
    mode:
        Which pupil model was rebuilt (:data:`aodl.check.pupil.PupilMode`).
    times:
        Frame observation times [s].
    table:
        Long-format measurements, ``{column: array}`` with one row per (frame, requested trap)
        and the columns of :data:`CHECK_TABLE_KEYS` — fitted position, best-focus ``z_lab`` and
        astigmatic ``delta_f``, radii, peak, power, beat depth, the residuals against the
        expectation, the model-free profile moments, the flags saying which gates applied, and
        each trap's ``uniformity_median`` over the frames (:meth:`median_uniformity`).
    blobs:
        Every local maximum of the full-field canvas that is not a requested trap
        (:class:`aodl.check.metrics.Blob`), brightest first, frame by frame.
    failures:
        One line per violated gate, each naming its metric first.  Empty on a pass.
    notes:
        What a reader needs in order to read the numbers: exclusions applied, the known blind
        spot, drive-specific caveats.
    gated_fraction:
        ``{metric: gated rows / rows}`` for the intensity metrics of :data:`COVERAGE_METRICS`.
        A **zero** here is the finding: the metric measured nothing, so a ``passed`` that
        includes it certifies nothing about it (:attr:`Tolerances.require_coverage`).
    out_of_band:
        Per-channel splatter fraction (:func:`aodl.check.demod.out_of_band_fraction`).
        Report-only: the band is a hardware limit, not a physics prediction.
    sim_delta:
        The optional simulator comparison (:func:`aodl.check.expect.sim_delta`), or ``None``.
    tolerances:
        The thresholds this verdict used.
    params:
        The hardware the rebuild used — ``waist0`` and ``rayleigh`` are what the residuals are
        quoted in.
    """

    passed: bool
    mode: PupilMode
    times: Float
    table: dict[str, Float]
    blobs: tuple[Blob, ...]
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    gated_fraction: dict[str, float]
    out_of_band: dict[str, float]
    sim_delta: dict[str, float] | None
    tolerances: Tolerances
    params: AODLParams

    @property
    def n_rows(self) -> int:
        """Rows in :attr:`table` — frames times requested traps."""
        return int(self.table["time"].size)

    def worst(self) -> dict[str, tuple[float, float, str]]:
        """``{metric: (residual, tolerance, offender)}`` — the number to read per gate.

        Residuals are in the same relative units as :class:`Tolerances`, so a metric passes
        exactly when its residual is at or below its tolerance.  ``offender`` names the frame
        and trap the residual came from.  Only *gated* rows are considered, which is what makes
        the table comparable against the thresholds.
        """
        table = self.table
        waist0 = self.params.optics.waist0
        rayleigh = self.params.optics.rayleigh
        tol = self.tolerances
        if not self.n_rows:
            return {}
        gated = table["gated"] > 0.5
        present = table["present"] > 0.5
        quiet = table["in_fade"] < 0.5
        shaped = gated & present & (table["transient"] < 0.5) & quiet
        placed = present & (table["filling"] < 0.5)
        out: dict[str, tuple[float, float, str]] = {}
        candidates: tuple[tuple[str, Float, float, Bool], ...] = (
            (
                "lateral",
                np.maximum(np.abs(table["dx"]), np.abs(table["dy"])) / waist0,
                tol.lateral,
                placed,
            ),
            ("axial", np.abs(table["dz"]) / rayleigh, tol.axial, placed),
            ("astigmatism", np.abs(table["delta_f"]) / rayleigh, tol.axial, placed),
            (
                "waist",
                np.maximum(np.abs(table["wx"] / waist0 - 1.0), np.abs(table["wy"] / waist0 - 1.0)),
                tol.waist,
                shaped,
            ),
            ("uniformity", np.abs(table["uniformity"]), tol.uniformity, gated & present & quiet),
        )
        if tol.uniformity_median is not None:
            candidates += (
                (
                    "uniformity_median",
                    np.abs(table["uniformity_median"]),
                    tol.uniformity_median,
                    gated & present & quiet,
                ),
            )
        for name, residual, limit, mask in candidates:
            if not np.any(mask):
                out[name] = (0.0, limit, "no gated rows")
                continue
            values = np.where(mask & np.isfinite(residual), residual, -np.inf)
            i = int(np.argmax(values))
            # A statistic taken *over* time has no one frame to blame, so it names the trap.
            where = self._which(i) if name == "uniformity_median" else self._where(i)
            out[name] = (float(values[i]), limit, where)
        return out

    def _where(self, i: int) -> str:
        """``"trap (ix,iy) at t = ... us"`` for row ``i``."""
        table = self.table
        return f"{self._which(i)} at t = {table['time'][i] / us:.4g} us"

    def _which(self, i: int) -> str:
        """``"trap (ix,iy)"`` for row ``i`` — no time, for statistics taken *over* time."""
        table = self.table
        return f"trap ({int(table['ix'][i])},{int(table['iy'][i])})"

    def median_uniformity(self) -> tuple[float, str]:
        r"""Worst per-trap **time-median** relative intensity deviation, and whose it is.

        Each gated trap's ``uniformity`` is medianed over the frames it was gated at (at least
        :data:`MIN_MEDIAN_FRAMES` of them), and this is the largest ``|median|``.  The point of
        the median is to separate two things the per-frame gate cannot tell apart: the
        hand-over ripple of a fading drive is *fade-phase dependent*, so it moves from frame to
        frame and medians away, while a fault that holds still survives every median.

        **Report-only by default**, and here is why (WO-24 §1, measured on the ``docs/guide.md``
        flagship at ``k_subtimes = 48`` over its seven :meth:`aodl.api.MotionPlan.check_times`):

        =========================================  ==========  ============
        drive                                      per-frame   time-median
        =========================================  ==========  ============
        clean flagship                             0.213       0.137
        one ``Bx`` rung scaled 0.80 (36 % of it)   0.280       0.137
        =========================================  ==========  ============

        — no separation whatsoever, because a fading-Shepard rung fault does **not** hold
        still.  The ladder slides through the array (that is what Shepard means), so the
        corrupted rung feeds column 0 at the first frame, column 1 at the second and column 4
        at the third, then leaves the band: a −0.3 excursion at one frame in seven, which any
        median over seven frames erases.  What the clean 0.137 *is*, meanwhile, is real and
        persistent — the Eqs. S20-S22 intermodulation products land on lattice nodes at those
        nodes' own optical frequency and interfere permanently (:func:`_beat_comb`), so the
        IM3 pattern is exactly the kind of thing a time median keeps.

        The statistic does separate a fault that holds still: on the Eq. S19 3x3 of
        ``tests/test_check_verdict.py``, one tone 5 % down is a fixed column at every frame and
        the median reads 0.098 against a clean 0.012 — an 8x separation, gated there at 0.04
        with both sides pinned.  Set :attr:`Tolerances.uniformity_median` on such a drive.

        Returns
        -------
        ``(worst |median|, "trap (ix,iy)")``; ``(0.0, reason)`` when no trap qualified.
        """
        table = self.table
        if not self.n_rows:
            return 0.0, "no rows"
        values = np.abs(table["uniformity_median"])
        finite = np.isfinite(values)
        if not np.any(finite):
            return 0.0, f"no trap gated on {MIN_MEDIAN_FRAMES} or more frames"
        i = int(np.argmax(np.where(finite, values, -np.inf)))
        return float(values[i]), self._which(i)

    def summary(self) -> str:
        """A short human-readable block: verdict, worst residual per metric, blobs, exclusions."""
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"AODL sample check - {verdict}   (pupil model: {self.mode})"]
        times = self.times
        if times.size:
            lines.append(
                f"  frames       {times.size} over [{times[0] / us:.4g}, {times[-1] / us:.4g}] "
                f"us; {self.n_rows} (frame, trap) rows"
            )
        lines.append(f"  residuals    {'metric':<17} {'worst':>9}  {'tolerance':>9}   where")
        for name, (residual, limit, where) in self.worst().items():
            flag = "" if residual <= limit else "   <-- FAIL"
            lines.append(f"               {name:<17} {residual:9.4g}  {limit:9.4g}   {where}{flag}")
        if self.tolerances.uniformity_median is None:
            median, whose = self.median_uniformity()
            lines.append(
                f"               {'uniformity_median':<17} {median:9.4g}  {'-':>9}   {whose}"
                "   (report-only)"
            )
        if self.gated_fraction:
            lines.append(
                "  coverage     "
                + ", ".join(
                    f"{name} {fraction:.0%}" + ("  <-- NONE GATED" if fraction <= 0.0 else "")
                    for name, fraction in self.gated_fraction.items()
                )
            )
        off = [b for b in self.blobs if not b.on_lattice]
        on = [b for b in self.blobs if b.on_lattice]
        lines.append(
            f"  blobs        {len(off)} off-lattice"
            + (f" (brightest {max(b.rel_intensity for b in off):.3g})" if off else "")
            + f", {len(on)} on-lattice"
            + (f" (brightest {max(b.rel_intensity for b in on):.3g})" if on else "")
        )
        if self.out_of_band:
            name, value = max(self.out_of_band.items(), key=lambda kv: kv[1])
            lines.append(
                f"  splatter     worst out-of-band power {value:.3g} on {name!r} (report-only)"
            )
        if self.sim_delta is not None:
            delta = self.sim_delta
            lines.append(
                f"  vs simulator {int(delta['n_matched'])}/{int(delta['n_rows'])} matched; "
                f"max |dx| {delta['max_dx'] / um:.3g} um, max |dz| {delta['max_dz'] / um:.3g} um,"
                f" max relative power gap {delta['max_dpower']:.3g}  (report-only)"
            )
        if self.failures:
            lines.append(f"  failures     {len(self.failures)}")
            lines.extend(f"    - {failure}" for failure in self.failures[:12])
            if len(self.failures) > 12:
                lines.append(f"    ... and {len(self.failures) - 12} more")
        if self.notes:
            lines.append("  notes")
            lines.extend(f"    - {note}" for note in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


# ------------------------------------------------------------------------------- machinery


def frame_reach(grid: ApertureGrid, params: AODLParams) -> float:
    """How far past its own retarded time a frame reads the drive: ``grid.half_span / v`` [s].

    A frame at observation time ``t`` gathers drive times ``t - tau/2 - s u / v`` over the whole
    aperture grid, so the latest one it needs is ``t - tau/2 + half_span / v`` and a record must
    reach that far.  The grid is deliberately *wider than the crystal* (±4.99 ``w_in`` against a
    7.5 mm aperture at the product defaults, :class:`aodl.check.pupil.ApertureGrid`), which is
    what makes this **15.36 µs** rather than the beam-transit figure ``4 w_in / v`` — 3.1 µs too
    small, and a record trimmed to that trips
    :func:`aodl.check.demod.sample_baseband`'s coverage refusal.
    """
    return grid.half_span / params.sound_speed


def _as_record(
    samples: SampleRecord | Mapping[str, ArrayLike] | str | Path,
    params: AODLParams,
    sample_rate: float | None,
    normalization: float | None,
) -> SampleRecord:
    """Coerce the ``samples`` argument of :func:`check_samples` into a :class:`SampleRecord`."""
    if isinstance(samples, SampleRecord):
        return samples
    if isinstance(samples, str | Path):
        return load_samples(samples, params)
    if isinstance(samples, Mapping):
        if sample_rate is None:
            raise ValueError(
                "check_samples needs sample_rate when it is handed raw arrays — the rate is not "
                "recoverable from the buffers.  Pass a SampleRecord (aodl.check.from_arrays) or "
                "a '*_samples.npz' path to carry it along with the normalization."
            )
        if normalization is None:
            raise ValueError(
                "check_samples needs normalization when it is handed raw arrays: render_samples "
                "divides every channel by one global peak, so the physical drive is "
                "samples * normalization and the nonlinear pupil is mis-scaled by that "
                "normalization factor without it (aodl.check.record).  Pass normalization=1.0 "
                "explicitly if the arrays really are in Eq. S1 units."
            )
        return from_arrays(samples, sample_rate, params, normalization=normalization)
    raise TypeError(
        f"check_samples needs a SampleRecord, a mapping of arrays or a path, got {type(samples)!r}"
    )


def _decimation(grid: ApertureGrid, params: AODLParams, mode: PupilMode, shift: float) -> int:
    """How far a rebuilt pupil may be down-sampled before the Eq. S11 transform.

    The aperture grid is pinned at ``du = Lambda / 8`` so that the ``exp(i C V)`` order comb
    aliases onto order *centres* (:mod:`aodl.check.pupil`).  Once the ``+1`` order has been cut
    out and its carrier ramp removed, though, what is left is band-limited to the drive band:
    each channel contributes at most ``band_margin (1 + roll) (f_hi - f_lo) / 2 / v`` of spatial
    frequency and an axis stacks two of them — ~44 mm^-1 at the product defaults, against the
    grid's 615 mm^-1 Nyquist.  Decimating by 8 keeps a 1.5x guard and makes the chirp-z
    transform ~18x cheaper, which is the difference between a checker one runs and one one does
    not.  ``weak`` mode's grid is already six times coarser and is left alone.

    The band is read from the **widest** of the four channels, not from one of them: per-channel
    limits are a documented knob (``docs/PLAN.md`` decision 6), and a channel wider than the one
    sampled would be decimated below its own Nyquist and alias into the pupil silently.

    ``shift`` is the largest co-moving de-smear displacement the frame will apply
    (:func:`_check_frame`).  That is a pupil phase ramp of spatial frequency ``shift / (lambda
    F)`` — 112 mm^-1 for three quarters of a micron — so it has to fit inside the decimated
    Nyquist too, and a fast traverse simply buys a finer transform grid.
    """
    if mode == "weak":
        return 1
    widest = max(hi - lo for lo, hi in (aod.band for aod in params.channels.values()))
    optics = params.optics
    content = 2.0 * 1.15 * 1.25 * 0.5 * widest / params.sound_speed
    content += abs(shift) / (optics.wavelength * optics.focal_length)
    limit = 1.0 / (2.0 * DECIMATION_SAFETY * content)
    factor = 1
    while (
        factor * 2 <= grid.n // 64 and grid.du * factor * 2 <= limit and grid.n % (factor * 2) == 0
    ):
        factor *= 2
    return factor


def _decimate(pupil: Complex, grid: ApertureGrid, factor: int) -> tuple[Complex, ApertureGrid]:
    """Band-limited down-sampling of a rebuilt pupil by ``factor`` (spectral truncation).

    Exact for the band-limited pupil of :func:`_decimation`: the low ``n / factor`` spatial
    frequencies are kept and the rest dropped, which is the same operation the ``+1``-band
    window already performed once.  Nothing is re-windowed, so no light moves — and the
    surviving samples are the fine grid's own, at ``u_0 + j factor du``.
    """
    if factor <= 1:
        return pupil, grid
    n = grid.n
    m = n // factor
    spectrum = np.fft.fft(pupil, axis=-1)
    half = m // 2
    kept = np.zeros(pupil.shape[:-1] + (m,), dtype=np.complex128)
    kept[..., :half] = spectrum[..., :half]
    kept[..., m - half :] = spectrum[..., n - half :]
    if m % 2:
        kept[..., half] = spectrum[..., half]
    coarse = np.fft.ifft(kept, axis=-1) * (float(m) / float(n))
    du = grid.du * factor
    u = grid.u[0] + du * np.arange(m, dtype=np.float64)
    return np.asarray(coarse, dtype=np.complex128), ApertureGrid(u=u, du=du)


def _fine_axis(lo: float, hi: float, waist0: float) -> Float:
    """Uniform image grid covering ``[lo, hi]`` at ``waist0 / 16``, capped in size."""
    step = waist0 / FINE_PER_WAIST
    count = int(math.ceil((hi - lo) / step)) + 1
    if count > MAX_FINE_SAMPLES:
        count = MAX_FINE_SAMPLES
        step = (hi - lo) / (count - 1)
    return lo + step * np.arange(count, dtype=np.float64)


def _window(coords: Float, center: float, half: float) -> slice:
    """Index slice of ``coords`` within ``half`` of ``center`` (at least three samples)."""
    lo = int(np.searchsorted(coords, center - half, side="left"))
    hi = int(np.searchsorted(coords, center + half, side="right"))
    if hi - lo < 3:
        middle = int(np.clip(np.searchsorted(coords, center), 1, max(coords.size - 2, 1)))
        lo, hi = middle - 1, middle + 2
    return slice(max(lo, 0), min(hi, coords.size))


def _parabola_waist(z_planes: Float, w2: Float) -> tuple[float, float]:
    """``(best-focus lab Z, the waist there)`` from the through-focus ``w^2`` parabola.

    The vertex comes from :func:`aodl.check.metrics.best_focus`, the sole authority for it; the
    *value* at the vertex is the same least-squares parabola evaluated there, which is the waist
    the beam actually has rather than the width measured at whichever plane happened to be
    sampled.  Keeping the two apart is what stops an axial error from being reported as a waist
    error as well.
    """
    focus = best_focus(z_planes, w2)
    mean = float(z_planes.mean())
    scale = float(np.max(np.abs(z_planes - mean)))
    s = (z_planes - mean) / scale
    coeffs = np.polynomial.polynomial.polyfit(s, w2, 2)
    at = (focus - mean) / scale
    value = float(coeffs[0] + coeffs[1] * at + coeffs[2] * at * at)
    return focus, math.sqrt(max(value, 0.0))


@dataclass(frozen=True)
class _AxisFit:
    """Per-column (or per-row) fit results, reduced over the Z stack."""

    center: Float
    focus: Float
    waist: Float
    centroid: Float
    rms: Float
    ok: Bool


def _fit_lines(
    coords: Float, marginals: Float, centers: Float, z_planes: Float, waist0: float
) -> _AxisFit:
    """Fit one Gaussian per expected line at every Z plane, then reduce to a focus and a waist.

    ``marginals`` is ``(n_z, n_coords)`` — the correlated marginal of
    :func:`aodl.check.metrics.accumulate_marginals` — and ``centers`` holds the expected line
    positions.  A line whose profile is not a fittable peak (a dark column, say) is flagged in
    ``ok`` rather than raising: that is a *finding*, and the missing-trap gate is where it is
    reported.
    """
    n_lines = int(centers.size)
    half = FIT_HALF_WAISTS * waist0
    middle = marginals.shape[0] // 2
    center = np.zeros(n_lines)
    focus = np.zeros(n_lines)
    waist = np.zeros(n_lines)
    centroid = np.zeros(n_lines)
    rms = np.zeros(n_lines)
    ok = np.ones(n_lines, dtype=bool)
    for line in range(n_lines):
        window = _window(coords, float(centers[line]), half)
        xs = coords[window]
        w2 = np.zeros(marginals.shape[0])
        try:
            for plane in range(marginals.shape[0]):
                fitted, radius, _ = fit_gaussian_1d(xs, marginals[plane, window])
                w2[plane] = radius * radius
                if plane == middle:
                    center[line] = fitted
            centroid[line], rms[line] = profile_moments(xs, marginals[middle, window])
            focus[line], waist[line] = _parabola_waist(z_planes, w2)
        except ValueError:
            ok[line] = False
            center[line] = float(centers[line])
            focus[line] = float(z_planes[middle])
            waist[line] = math.nan
            centroid[line] = float(centers[line])
            rms[line] = math.nan
    return _AxisFit(center=center, focus=focus, waist=waist, centroid=centroid, rms=rms, ok=ok)


def _beat_window(expect: Expectation, traps: ExpectedTraps) -> float:
    """The fallback averaging window ``W`` [s] of the module docstring."""
    spacing = expect.min_spacing()
    window = math.inf if not math.isfinite(spacing) else BEAT_CYCLES / spacing
    speed = traps.speed
    if speed > 0.0:
        window = min(window, MOTION_FRACTION * expect.params.optics.waist0 / speed)
    return 0.0 if not math.isfinite(window) else window


def _beat_comb(expect: Expectation) -> tuple[float, ...]:
    """The frequencies at which two terms landing on the *same* lattice node beat [Hz].

    A tweezer's position depends on the *difference* of its A and B ladder indices while its
    optical frequency depends on their *sum* (``docs/conventions.md`` §4-§5), so one node can
    be fed by several index pairs at once — and those add coherently only in the instant, not
    in the average an atom sees.  Under a fading-Shepard drive exactly two adjacent A rungs are
    ever live together (the hand-over), so the pairs feeding node ``d`` differ by
    ``(a - a', b - b') = (n, m)`` with ``|n|, |m| <= 1`` and their optical frequencies differ by

        ``2 n delta_f_x + 2 m delta_f_y``.

    That finite comb — ``{0.6, 2.0, 2.6, 4.6} MHz`` for the guide's 1.0/1.3 MHz flagship — is
    what the frame average has to annihilate.  Under a plain Eq. S19 drive every node has one
    term and the comb is empty: an IM3 product landing on a node carries that node's own
    optical frequency exactly, so it interferes *permanently* (that is a real, static intensity
    error, and the uniformity gate is where it belongs) rather than beating.
    """
    if not expect.fading:
        return ()
    array = expect.spec.array
    scale = expect.params.deflection_scale
    spacing = {
        "x": abs(array.delta_f_x) if array.mx > 1 else 0.0,
        "y": abs(array.delta_f_y) if array.my > 1 else 0.0,
    }
    for _, axis, offset in expect.shadows:  # the fading ladder's own spacing, via Eq. S31
        if offset:
            spacing[axis] = abs(offset) / scale
    beats = {
        abs(2.0 * n * spacing["x"] + 2.0 * m * spacing["y"]) for n in (-1, 0, 1) for m in (-1, 0, 1)
    }
    return tuple(sorted(beat for beat in beats if beat > 0.0))


def _comb_window(beats: tuple[float, ...], k: int) -> float | None:
    """The shortest window over which a **uniform** ``k``-point average kills every beat exactly.

    A uniform schedule of ``k`` instants spanning ``W`` annihilates a beat ``b`` exactly when
    ``b W`` is a non-zero integer that is not a multiple of ``k``; the golden-ratio schedule of
    :func:`aodl.check.transform.subtimes` cannot do better than its own discrepancy, and its
    Fibonacci weak spots land squarely on a regular comb (``b W = 13`` leaves 19 % of the swing
    at ``k = 48``).  When the comb of :func:`_beat_comb` is commensurate — which an atom array's
    always is — the exact schedule is available and is worth taking.

    ``W = 1 / gcd(beats)`` is that window, computed on integer hertz (:func:`_comb_period`);
    ``None`` means there is no usable one, either because the comb is empty or not
    commensurate, or because ``k`` is too small to resolve the fastest beat over it.  Aliasing
    onto DC needs ``b W = k``; the test below refuses at the Nyquist half of that
    (``b W > k / 2``), and then the golden-ratio fallback is the honest answer — a *noisier*
    one, which is why :func:`_comb_shortfall` puts the ``k`` that would have worked in the
    report's notes.  The threshold is why the choice of schedule depends on ``k``: the
    flagship's ``b W`` reaches 23, so it takes the uniform comb at ``k >= 46`` and the fallback
    below that.
    """
    period = _comb_period(beats)
    if period is None:
        return None
    window, harmonic = period
    return None if 2 * harmonic > k else window


def _comb_period(beats: tuple[float, ...]) -> tuple[float, int] | None:
    """``(1 / gcd(beats) [s], fastest beat in cycles per that window)``, or ``None``.

    The harmonic is returned as an **integer** rather than left as ``max(beats) * window``:
    the comb is commensurate by the test just above, so the product is a whole number
    mathematically, and in floating point it is a whole number plus 4e-15 — enough to push the
    ``k`` threshold of :func:`_comb_window` up by one and drop the exact schedule for no reason.
    """
    if not beats:
        return None
    step = 0
    for beat in beats:
        step = math.gcd(step, int(round(beat)))
    if step <= 0:
        return None
    window = 1.0 / step
    if any(abs(beat * window - round(beat * window)) > 1e-6 for beat in beats):
        return None
    return window, int(round(max(beats) * window))


def _comb_shortfall(beats: tuple[float, ...], k: int) -> int | None:
    """Smallest ``k_subtimes`` that would take the exact comb schedule, when ``k`` does not.

    ``None`` when there is no commensurate comb to take (nothing is being given up) or when
    ``k`` is already large enough.  Otherwise the caller is on the golden-ratio fallback purely
    because it asked for too few sub-times, which costs verdict noise for no reason — the
    report says so (:func:`_notes`).
    """
    period = _comb_period(beats)
    if period is None:
        return None
    needed = 2 * period[1]
    return None if needed <= k else needed


def _schedule(expect: Expectation, traps: ExpectedTraps, k: int) -> tuple[Float, float, bool]:
    """``(sub-times, window, exact)``: the beat-commensurate grid, or the fallback."""
    window = _comb_window(_beat_comb(expect), k)
    if window is not None:
        offsets = (np.arange(k, dtype=np.float64) + 0.5) / k - 0.5
        return traps.time + window * offsets, window, True
    fallback = _beat_window(expect, traps)
    count = 1 if fallback == 0.0 else max(int(k), 1)
    return subtimes(traps.time, fallback, count), fallback, False


def averaging_window(expect: Expectation, k_subtimes: int = 64) -> float:
    """The widest beat-averaging window any frame of this drive can use [s].

    A frame does not read the drive at one instant: it averages over ``W``, so its sub-times
    reach ``W/2`` either side of it and the record has to cover *that*.  The motion cap of
    :func:`_beat_window` only ever narrows the window, so the bound is the beat-commensurate
    window when there is one (:func:`_comb_window`) and ``BEAT_CYCLES / delta_f_min`` otherwise
    — zero for a drive with nothing to beat.

    This is what :func:`check_samples` and :meth:`aodl.api.MotionPlan.check` subtract from the
    latest checkable frame time, on top of ``grid.half_span / v`` (:func:`frame_reach`).
    """
    window = _comb_window(_beat_comb(expect), max(int(k_subtimes), 1))
    if window is not None:
        return window
    spacing = expect.min_spacing()
    return 0.0 if not math.isfinite(spacing) else BEAT_CYCLES / spacing


def _axis_pupils(
    bb: Baseband, axis: str, times: Float, grid: ApertureGrid, mode: PupilMode
) -> Complex:
    """``(K, grid.n)`` axis pupils, one per sub-time (Eqs. S1-S4, :mod:`aodl.check.pupil`)."""
    return np.stack([axis_pupil(bb, axis, float(t), grid, mode=mode) for t in times])


def _is_filled(t: float, grid: ApertureGrid, params: AODLParams) -> bool:
    """Is every cell of the aperture grid carrying drive, on every channel, at frame time ``t``?

    ``s u <= v t - D/2`` (``docs/conventions.md`` §7) must hold at both grid ends, i.e.
    ``v t - D/2 >= half_span`` — 1.83 ``tau`` on the pinned grid, not ``tau``, because the grid
    deliberately runs wider than the crystal.
    """
    return all(
        aod.sound_speed * t - 0.5 * aod.aperture >= grid.half_span
        for aod in params.channels.values()
    )


def _line_intensities(
    field: Complex, coords: Float, centers: Float, waist0: float
) -> tuple[Float, Float]:
    """``(peak, integral)`` per sub-time and line, ``(K, n_lines)`` each.

    ``peak`` is the intensity **at** the line's fitted centre, ``integral`` its intensity summed
    over the fit window.  Both are kept **per sub-time** so that the outer product with the
    other axis is formed before the average, which is what keeps the two axes' beats correlated
    (:mod:`aodl.check.metrics`).

    The centre almost never falls on a grid sample, so the value there is interpolated by the
    parabola through the three samples bracketing it rather than read off the nearest one.
    Reading the nearest sample costs a Gaussian ``exp(-2 d^2 / w_0^2)`` with ``d`` up to half a
    cell, i.e. a *quantization floor* of ``(1/FINE_PER_WAIST)^2 / 2 = 0.2 %`` on every peak —
    which is noise in the uniformity gate, and noise that changes when a trap drifts across a
    cell boundary.  Three samples of a ``w_0/16`` grid resolve the curvature to ~1e-5, so the
    interpolation removes the floor outright rather than trading it for another.
    """
    intensity = np.abs(field) ** 2
    step = float(coords[1] - coords[0]) if coords.size > 1 else 1.0
    half = FIT_HALF_WAISTS * waist0
    peaks = np.empty((intensity.shape[0], int(centers.size)))
    sums = np.empty_like(peaks)
    for line, center in enumerate(centers):
        window = _window(coords, float(center), half)
        node = int(np.clip(np.searchsorted(coords, center), 1, max(coords.size - 2, 1)))
        if coords.size < 3:
            peaks[:, line] = intensity[:, node]
        else:
            low, mid, high = intensity[:, node - 1], intensity[:, node], intensity[:, node + 1]
            offset = (float(center) - float(coords[node])) / step
            peaks[:, line] = np.maximum(
                mid
                + 0.5 * offset * (high - low)
                + 0.5 * offset * offset * (low - 2.0 * mid + high),
                0.0,
            )
        sums[:, line] = intensity[:, window].sum(axis=1) * step
    return peaks, sums


def _gate(
    failures: list[str],
    verdict: Bool,
    mask: Bool,
    residual: Float,
    tol: float,
    traps: ExpectedTraps,
    metric: str,
    describe: Callable[[int], str],
) -> None:
    """Record one failure line per trap that violates ``metric`` where ``mask`` applies."""
    bad = mask & np.isfinite(residual) & (residual > tol)
    for index in np.nonzero(bad)[0]:
        i = int(index)
        verdict[i] = False
        failures.append(
            f"{metric}: trap ({int(traps.ix[i])},{int(traps.iy[i])}) at "
            f"t = {traps.time / us:.4g} us is {describe(i)}, tolerance {tol:.4g}"
        )


def _intensity_masks(table: dict[str, Float]) -> dict[str, Bool]:
    """The row mask each :data:`COVERAGE_METRICS` gate actually applied, from the flag columns.

    One place to say what "gated" means per metric, shared by :meth:`CheckReport.worst`'s
    bookkeeping, the coverage fractions and the time median, so the three cannot drift apart.
    """
    gated = table["gated"] > 0.5
    present = table["present"] > 0.5
    quiet = table["in_fade"] < 0.5
    intensity = gated & present & quiet
    return {
        "waist": intensity & (table["transient"] < 0.5),
        "uniformity": intensity,
        "uniformity_median": intensity,
    }


def _time_median_uniformity(table: dict[str, Float]) -> Float:
    """Each trap's time-median ``uniformity``, broadcast back over that trap's own rows.

    The median runs over the frames where the trap was actually gated (``_intensity_masks``),
    and a trap gated on fewer than :data:`MIN_MEDIAN_FRAMES` frames gets ``nan`` rather than a
    number that would only be its per-frame deviation wearing a different name.  Every row of a
    trap carries the same value, so ``nanmax(abs(...))`` over rows is the worst over traps.
    """
    rows = int(table["time"].size)
    out = np.full(rows, np.nan)
    if not rows:
        return out
    usable = _intensity_masks(table)["uniformity_median"] & np.isfinite(table["uniformity"])
    ix, iy = table["ix"], table["iy"]
    keys = np.stack([ix, iy], axis=1)
    for key in np.unique(keys, axis=0):
        here = (ix == key[0]) & (iy == key[1])
        sample = table["uniformity"][here & usable]
        if sample.size >= MIN_MEDIAN_FRAMES:
            out[here] = float(np.median(sample))
    return out


def _coverage(table: dict[str, Float]) -> dict[str, float]:
    """``{metric: gated rows / rows}`` for :data:`COVERAGE_METRICS`."""
    rows = int(table["time"].size)
    if not rows:
        return dict.fromkeys(COVERAGE_METRICS, 0.0)
    masks = _intensity_masks(table)
    fractions = {name: float(np.count_nonzero(masks[name])) / rows for name in COVERAGE_METRICS}
    fractions["uniformity_median"] = (
        float(np.count_nonzero(np.isfinite(table["uniformity_median"]))) / rows
    )
    return fractions


def _match_tolerance(expect: Expectation) -> float:
    """Largest lateral distance at which two measurements count as the same tweezer [m]."""
    array = expect.spec.array
    scale = expect.params.deflection_scale
    pitches = [
        abs(scale * spacing)
        for count, spacing in ((array.mx, array.delta_f_x), (array.my, array.delta_f_y))
        if count > 1 and spacing != 0.0
    ]
    return 0.4 * min(pitches) if pitches else 6.0 * expect.params.optics.waist0


# ------------------------------------------------------------------------------ the driver


def check_samples(
    samples: SampleRecord | Mapping[str, ArrayLike] | str | Path,
    expect: Expectation,
    *,
    times: ArrayLike | None = None,
    mode: PupilMode = "bragg_band",
    tolerances: Tolerances | None = None,
    sim: SimResultLike | None = None,
    k_subtimes: int = 64,
    n_z: int = 7,
    z_half_range: float | None = None,
    sample_rate: float | None = None,
    params: AODLParams | None = None,
    normalization: float | None = None,
    oversample: int = 1,
) -> CheckReport:
    """Rebuild the tweezers from ``samples`` and judge them against ``expect``.

    Parameters
    ----------
    samples:
        A :class:`~aodl.check.record.SampleRecord`, a ``{channel: array}`` mapping (then
        ``sample_rate`` and ``normalization`` are required), or the path of a ``*_samples.npz``
        written by :func:`aodl.waveform.export.save_samples`.
    expect:
        What the drive was asked for (:class:`aodl.check.expect.Expectation`).
    times:
        Frame observation times [s].  ``None`` spreads nine frames from ``2 tau`` to the last
        time the record can be gathered for (:func:`frame_reach`).
    mode:
        ``"bragg_band"`` (default) rebuilds the full ``exp(i C V)`` crystal and cuts the ``+1``
        order out; ``"weak"`` rebuilds the linear Eq. S3 model the simulator implements, which
        is the cross-validation path rather than the verdict path.
    tolerances:
        Thresholds; ``None`` uses :class:`Tolerances`' defaults.
    sim:
        Optional simulator run to diff against — report-only
        (:func:`aodl.check.expect.sim_delta`).
    k_subtimes:
        Instants averaged per frame (:func:`aodl.check.transform.subtimes`).
    n_z:
        Planes in the through-focus stack; three is the minimum the ``w^2`` parabola needs.
    z_half_range:
        Half-height [m] of that stack about the expected ``Z``; ``None`` uses one Rayleigh
        range, where a Gaussian has widened by ``sqrt(2)`` — plenty of curvature to fit.
    sample_rate, params, normalization:
        Used only when ``samples`` is raw arrays or a path; ``params`` defaults to
        ``expect.params``.
    oversample:
        Spectral zero-padding of the demodulation (:func:`aodl.check.demod.demodulate`).  ``1``
        is right for the verdict path, whose tolerances are ``0.05 w_0``-scale against a
        ``0.016 theta^3`` interpolation error (5.5e-5 at the band edge); the tight
        cross-validation gates use ``2``.

    Returns
    -------
    A :class:`CheckReport`.
    """
    hardware = expect.params if params is None else params
    rec = _as_record(samples, hardware, sample_rate, normalization)
    hardware = rec.params
    tol = Tolerances() if tolerances is None else tolerances
    optics = hardware.optics
    grid = ApertureGrid.design(hardware, mode)
    reach = frame_reach(grid, hardware)
    tau = expect.transit_time
    # A frame reads drive over [t - W/2 - tau/2 - reach, t + W/2 - tau/2 + reach]: the aperture
    # grid runs wider than the crystal (frame_reach) *and* the frame averages over a window
    # (averaging_window).  Both ends of the record have to cover that.
    half_window = 0.5 * averaging_window(expect, k_subtimes)
    limit = rec.t_span[1] + 0.5 * tau - reach - half_window
    floor = rec.t_start + 0.5 * tau + reach + half_window if rec.t_start > 0.0 else -math.inf

    frames = _resolve_times(times, rec, tau, reach, limit, floor)
    bb = demodulate(rec, oversample=oversample)
    z_half = float(optics.rayleigh if z_half_range is None else z_half_range)
    planes = max(int(n_z), 3)

    chunks: list[dict[str, Float]] = []
    blobs: list[Blob] = []
    failures: list[str] = []
    windows: list[float] = []
    factors: list[int] = []
    exact = True
    counts = [0, 0, 0]

    for index, t in enumerate(frames):
        traps = expect.traps(float(t))
        sub, window, commensurate = _schedule(expect, traps, max(int(k_subtimes), 1))
        windows.append(window)
        exact &= commensurate
        filling = not _is_filled(float(t), grid, hardware)
        transient = filling or float(t) < 2.0 * tau
        fade = expect.in_fade(float(t))
        counts[0] += int(transient)
        counts[1] += int(filling)
        counts[2] += int(fade)

        rows, frame_blobs, frame_failures, factor = _check_frame(
            bb=bb,
            grid=grid,
            expect=expect,
            traps=traps,
            sub=sub,
            mode=mode,
            planes=planes,
            z_half=z_half,
            tol=tol,
            flags=(transient, filling, fade),
            index=index,
        )
        chunks.append(rows)
        blobs.extend(frame_blobs)
        failures.extend(frame_failures)
        factors.append(factor)

    table = {
        key: np.concatenate([chunk[key] for chunk in chunks]) if chunks else np.zeros(0)
        for key in _FRAME_TABLE_KEYS
    }
    table["uniformity_median"] = _time_median_uniformity(table)
    coverage = _coverage(table)
    failures.extend(_time_median_failures(table, tol))
    failures.extend(_coverage_failures(coverage, tol, frames.size))
    delta = None if sim is None else sim_delta(table, sim, _match_tolerance(expect))
    shortfall = _comb_shortfall(_beat_comb(expect), max(int(k_subtimes), 1))
    notes = _notes(
        expect, table, tol, mode, factors, windows, exact, counts, coverage, shortfall, k_subtimes
    )
    return CheckReport(
        passed=not failures,
        mode=mode,
        times=frames,
        table=table,
        blobs=tuple(blobs),
        failures=tuple(failures),
        notes=notes,
        gated_fraction=coverage,
        out_of_band=out_of_band_fraction(rec),
        sim_delta=delta,
        tolerances=tol,
        params=hardware,
    )


def _time_median_failures(table: dict[str, Float], tol: Tolerances) -> list[str]:
    """The ``uniformity_median`` gate, when :attr:`Tolerances.uniformity_median` sets one.

    One line per offending trap — a *time* statistic, so the line names the trap and the frames
    it was medianed over rather than one instant — and the trap's ``verdict_trap`` rows are
    cleared with it.  ``None`` (the default) measures without gating; see
    :meth:`CheckReport.median_uniformity` for what that ruling rests on.
    """
    limit = tol.uniformity_median
    if limit is None or not table["time"].size:
        return []
    residual = np.abs(table["uniformity_median"])
    bad = np.isfinite(residual) & (residual > limit)
    failures: list[str] = []
    for key in np.unique(np.stack([table["ix"][bad], table["iy"][bad]], axis=1), axis=0):
        here = (table["ix"] == key[0]) & (table["iy"] == key[1])
        table["verdict_trap"][here] = 0.0
        value = float(table["uniformity_median"][here][0])
        frames = int(np.count_nonzero(here & _intensity_masks(table)["uniformity_median"]))
        failures.append(
            f"uniformity_median: trap ({int(key[0])},{int(key[1])}) sits {value * 100:+.3g} % off "
            f"its frames' median peak at the median of {frames} gated frame(s) — a persistent "
            f"offset, not hand-over ripple, tolerance {limit:.4g}"
        )
    return failures


def _coverage_failures(coverage: dict[str, float], tol: Tolerances, n_frames: int) -> list[str]:
    """The :attr:`Tolerances.require_coverage` gate: an intensity metric that gated nothing.

    Always a note (:func:`_notes`); a failure only when the caller asked for one, so that the
    default verdict of every drive already in the suite is unchanged.
    """
    if not tol.require_coverage:
        return []
    return [
        f"coverage: the {name!r} gate applied to no row at all over {n_frames} frame(s), so "
        "this PASS certifies nothing about it — every requested trap is exempt (a fading array "
        "whose edge lines are its whole perimeter, a wholly transient or wholly mid-fade set of "
        "frames).  Check an interior-bearing array, or more frames, or drop require_coverage."
        for name, fraction in coverage.items()
        if fraction <= 0.0
    ]


def _resolve_times(
    times: ArrayLike | None,
    rec: SampleRecord,
    tau: float,
    reach: float,
    limit: float,
    floor: float,
) -> Float:
    """Frame times, defaulted and checked against both ends of the record's aperture reach.

    ``limit`` is the latest frame the record can be gathered for and ``floor`` the earliest
    (``-inf`` for a record that starts at the drive's own ``t = 0``, where a retarded time
    before the start is *physics* — the aperture holds no sound yet — rather than missing data).
    """
    if times is None:
        start = max(2.0 * tau, floor if math.isfinite(floor) else 0.0)
        if limit <= start:
            raise ValueError(
                f"this record is too short to check: the earliest frame it supports is "
                f"{start / us:.4g} us (the fill transient ends at 2 tau = {2.0 * tau / us:.4g} us"
                + (
                    ""
                    if not math.isfinite(floor)
                    else f", and the record itself starts at {rec.t_start / us:.4g} us"
                )
                + f"), but a frame reads drive over t -+ (W/2 + tau/2 + grid.half_span/v), so the "
                f"record only supports frames up to {limit / us:.4g} us.  Render the samples over "
                f"a longer span."
            )
        return np.linspace(start, limit, 9)
    frames = np.atleast_1d(np.asarray(times, dtype=np.float64)).ravel()
    if frames.size == 0:
        raise ValueError("check_samples() needs at least one frame time")
    if not np.all(np.isfinite(frames)):
        raise ValueError("frame times must all be finite")
    t0, t1 = rec.t_span
    worst = float(np.max(frames))
    if worst > limit:
        raise ValueError(
            f"frame time t = {worst / us:.4g} us needs drive out to "
            f"t + W/2 - tau/2 + grid.half_span/v = {(worst + t1 - limit) / us:.4g} us, "
            f"but the record covers [{t0 / us:.4g}, {t1 / us:.4g}] us — frames are gatherable "
            f"only up to {limit / us:.4g} us.  The aperture grid runs wider than the crystal "
            f"(grid.half_span/v = {reach / us:.4g} us, against tau/2 = {0.5 * tau / us:.4g} us) "
            f"and each frame averages over a beat window on top of that, so render the samples "
            f"over a longer span (or drop the late frames)."
        )
    earliest = float(np.min(frames))
    if earliest < floor:
        raise ValueError(
            f"frame time t = {earliest / us:.4g} us reaches drive earlier than "
            f"{t0 / us:.4g} us, where this record starts — and a record that starts after the "
            f"drive did has no content there, so part of the aperture would be silently dark.  "
            f"Frames are gatherable from {floor / us:.4g} us; render the samples from an earlier "
            f"time, or ask for later frames."
        )
    return frames


def _check_frame(
    *,
    bb: Baseband,
    grid: ApertureGrid,
    expect: Expectation,
    traps: ExpectedTraps,
    sub: Float,
    mode: PupilMode,
    planes: int,
    z_half: float,
    tol: Tolerances,
    flags: tuple[bool, bool, bool],
    index: int,
) -> tuple[dict[str, Float], list[Blob], list[str], int]:
    """Measure and judge one frame.  Returns its rows, blobs, failures and decimation factor."""
    transient, filling, fade = flags
    params = bb.params
    optics = params.optics
    waist0, rayleigh = optics.waist0, optics.rayleigh
    t = traps.time

    # -- fields: two axis pupils per sub-time, decimated, transformed onto one fine grid per axis
    pad = FIT_HALF_WAISTS * waist0
    coords = [
        _fine_axis(float(nodes.min()) - pad, float(nodes.max()) + pad, waist0)
        for nodes in expect.lattice(t, extend=1)
    ]
    stride = max(1, int(round(FINE_PER_WAIST / COARSE_PER_WAIST)))
    z_planes = float(traps.z) + np.linspace(-z_half, z_half, planes)
    middle = planes // 2
    shifts = _desmear_shifts(expect, sub, t)
    factor = _decimation(grid, params, mode, float(np.max(np.abs(shifts))))

    fields: list[Complex] = []
    for axis, axis_coords, offsets in zip(("x", "y"), coords, shifts, strict=True):
        pupils, coarse = _decimate(_axis_pupils(bb, axis, sub, grid, mode), grid, factor)
        pupils = pupils * np.exp(
            -1j * (optics.k / optics.focal_length) * np.outer(offsets, coarse.u)
        )
        fields.append(
            np.stack([zoom_field(pupils, coarse, optics, axis_coords, float(z)) for z in z_planes])
        )
    ux, uy = fields  # each (n_z, K, n_coords)

    marg_x, marg_y = accumulate_marginals(np.moveaxis(ux, 1, 0), np.moveaxis(uy, 1, 0))
    fit_x = _fit_lines(coords[0], marg_x, traps.columns, z_planes, waist0)
    fit_y = _fit_lines(coords[1], marg_y, traps.rows, z_planes, waist0)

    # -- per-trap peak, power and beat depth from the *joint* intensity at the expected plane
    ix, iy = traps.ix, traps.iy
    peak_x, sum_x = _line_intensities(ux[middle], coords[0], fit_x.center, waist0)
    peak_y, sum_y = _line_intensities(uy[middle], coords[1], fit_y.center, waist0)
    joint = peak_x[:, ix] * peak_y[:, iy]
    peak = joint.mean(axis=0)
    beat_std = np.divide(joint.std(axis=0), peak, out=np.zeros_like(peak), where=peak > 0.0)
    power = (sum_x[:, ix] * sum_y[:, iy]).mean(axis=0)

    reference = float(np.median(peak)) if peak.size else 0.0
    if reference > 0.0:
        present = (peak >= tol.missing_trap * reference) & fit_x.ok[ix] & fit_y.ok[iy]
        uniformity = peak / reference - 1.0
    else:
        present = np.zeros(peak.size, dtype=bool)
        uniformity = np.zeros(peak.size)

    # -- Table I quantities, per trap
    x = fit_x.center[ix]
    y = fit_y.center[iy]
    z_x, z_y = fit_x.focus[ix], fit_y.focus[iy]
    z_lab = 0.5 * (z_x + z_y)
    delta_f = z_x - z_y
    wx, wy = fit_x.waist[ix], fit_y.waist[iy]

    edge_cols, edge_rows = expect.edge_lines()
    gated = ~(np.isin(ix, edge_cols) | np.isin(iy, edge_rows))

    # -- the gates
    failures: list[str] = []
    verdict = np.ones(peak.size, dtype=bool)
    placed = present & (not filling)
    shaped = present & gated & (not transient) & (not fade)
    lateral = np.maximum(np.abs(x - traps.x), np.abs(y - traps.y)) / waist0
    _gate(
        failures,
        verdict,
        placed,
        lateral,
        tol.lateral,
        traps,
        "lateral",
        lambda i: (
            f"{lateral[i]:.4g} w0 off (dx = {(x[i] - traps.x[i]) / um:+.4g} um, "
            f"dy = {(y[i] - traps.y[i]) / um:+.4g} um)"
        ),
    )
    axial = np.abs(z_lab - traps.z) / rayleigh
    _gate(
        failures,
        verdict,
        placed,
        axial,
        tol.axial,
        traps,
        "axial",
        lambda i: f"{axial[i]:.4g} z_R off (dz = {(z_lab[i] - traps.z) / um:+.4g} um)",
    )
    astig = np.abs(delta_f) / rayleigh
    _gate(
        failures,
        verdict,
        placed,
        astig,
        tol.axial,
        traps,
        "astigmatism",
        lambda i: f"at |Delta F| = {astig[i]:.4g} z_R ({delta_f[i] / um:+.4g} um), Table I says 0",
    )
    waists = np.maximum(np.abs(wx / waist0 - 1.0), np.abs(wy / waist0 - 1.0))
    _gate(
        failures,
        verdict,
        shaped,
        waists,
        tol.waist,
        traps,
        "waist",
        lambda i: f"fitted to wx/w0 = {wx[i] / waist0:.4g}, wy/w0 = {wy[i] / waist0:.4g}",
    )
    _gate(
        failures,
        verdict,
        present & gated & (not fade),
        np.abs(uniformity),
        tol.uniformity,
        traps,
        "uniformity",
        lambda i: f"{uniformity[i] * 100:+.3g} % off the frame's median peak",
    )
    for index_missing in np.nonzero(gated & ~present & (not transient))[0]:
        i = int(index_missing)
        verdict[i] = False
        level = peak[i] / reference if reference > 0.0 else 0.0
        failures.append(
            f"missing trap: trap ({int(ix[i])},{int(iy[i])}) at t = {t / us:.4g} us peaks at "
            f"{level:.4g} of the frame median (threshold {tol.missing_trap:.4g})"
        )

    # -- the blob audit, on the canvas the fine grid strides down to
    frame_blobs, blob_failures = _audit_blobs(
        ux[middle][:, ::stride],
        uy[middle][:, ::stride],
        coords[0][::stride],
        coords[1][::stride],
        expect=expect,
        traps=traps,
        reference=reference,
        tol=tol,
    )
    failures.extend(blob_failures)

    ones = np.ones(peak.size)
    rows: dict[str, Float] = {
        "frame": ones * index,
        "time": ones * t,
        "ix": ix.astype(np.float64),
        "iy": iy.astype(np.float64),
        "x": x,
        "y": y,
        "z_lab": z_lab,
        "delta_f": delta_f,
        "sigma_astig": delta_f / rayleigh,
        "wx": wx,
        "wy": wy,
        "peak": peak,
        "power": power,
        "beat_std": beat_std,
        "dx": x - traps.x,
        "dy": y - traps.y,
        "dz": z_lab - traps.z,
        "x_expect": traps.x,
        "y_expect": traps.y,
        "z_expect": ones * traps.z,
        "x_centroid": fit_x.centroid[ix],
        "y_centroid": fit_y.centroid[iy],
        "x_rms": fit_x.rms[ix],
        "y_rms": fit_y.rms[iy],
        "uniformity": uniformity,
        "present": present.astype(np.float64),
        "transient": ones * float(transient),
        "filling": ones * float(filling),
        "in_fade": ones * float(fade),
        "gated": gated.astype(np.float64),
        "verdict_trap": verdict.astype(np.float64),
        "verdict_frame": ones * float(bool(verdict.all()) and not blob_failures),
    }
    return rows, frame_blobs, failures, factor


def _desmear_shifts(expect: Expectation, sub: Float, t: float) -> Float:
    """``(2, K)`` co-moving displacements: where the array *should* be at each sub-time.

    The averaging window has to be long enough to kill the beat notes (:func:`_comb_window`),
    and a 5 µs window is three waists of travel at the guide flagship's traverse speed — enough
    to report a wide, dim, displaced spot for a perfectly good drive.  So the frame is measured
    in the array's **own frame**: each sub-time's pupil is multiplied by ``exp(-i k u d_j / F)``,
    which is exactly a lateral image shift of ``d_j`` (:func:`aodl.check.transform.zoom_field`
    evaluates ``int p(u) e^{-i k u X / F} du``), with ``d_j`` the *expected* displacement
    between that sub-time and the frame time.

    This does not hide a motion error, it isolates one: what survives is
    ``x_measured(t_j) - d_j``, so a drive moving at the right speed comes out sharp and one
    moving at the wrong speed comes out both displaced *and* smeared, in proportion to the
    velocity error rather than to the velocity.
    """
    here = expect.center(t)
    shifts = np.zeros((2, sub.size), dtype=np.float64)
    for j, tj in enumerate(sub):
        there = expect.center(float(tj))
        shifts[0, j] = there[0] - here[0]
        shifts[1, j] = there[1] - here[1]
    return shifts


def _audit_blobs(
    ux: Complex,
    uy: Complex,
    xs: Float,
    ys: Float,
    *,
    expect: Expectation,
    traps: ExpectedTraps,
    reference: float,
    tol: Tolerances,
) -> tuple[list[Blob], list[str]]:
    """Find every maximum on the frame's canvas and say whether it belongs there."""
    if reference <= 0.0:
        return [], []
    canvas = np.einsum("ki,kj->ij", np.abs(ux) ** 2, np.abs(uy) ** 2) / ux.shape[0]
    floor = min(tol.blob_off_lattice, tol.blob_on_lattice) * reference
    found = find_blobs(
        canvas, xs, ys, floor, merge_radius=expect.params.optics.waist0, reference=reference
    )
    match = _match_tolerance(expect)
    lattice_x, lattice_y = expect.lattice(traps.time, extend=1)
    nodes_x, nodes_y = [lattice_x], [lattice_y]
    for axis, offset in expect.shadow_offsets(traps.time):
        target, base = (nodes_x, lattice_x) if axis == "x" else (nodes_y, lattice_y)
        target.extend([base + offset, base - offset])
    on_x = np.unique(np.concatenate(nodes_x))
    on_y = np.unique(np.concatenate(nodes_y))

    blobs: list[Blob] = []
    failures: list[str] = []
    for bx, by, level in found:
        if np.min(np.abs(traps.x - bx)) <= match and np.min(np.abs(traps.y - by)) <= match:
            continue  # one of the requested tweezers
        on_lattice = bool(np.min(np.abs(on_x - bx)) <= match and np.min(np.abs(on_y - by)) <= match)
        blobs.append(
            Blob(
                time=traps.time,
                x=float(bx),
                y=float(by),
                rel_intensity=float(level),
                on_lattice=on_lattice,
            )
        )
        if on_lattice and expect.fading:
            # The Shepard extended grid and its shadow tweezers are documented light, so they
            # are whitelisted - but only up to a real trap's depth.  A node several times
            # brighter than the traps is not a hand-over, whatever lattice it sits on.
            limit, where = tol.blob_fading, "on the fading drive's extended lattice"
        else:
            limit = tol.blob_on_lattice if on_lattice else tol.blob_off_lattice
            where = "on an extended-lattice node" if on_lattice else "off lattice"
        if level > limit:
            failures.append(
                f"blob: light {where} at ({bx / um:+.4g}, {by / um:+.4g}) um, "
                f"t = {traps.time / us:.4g} us, carries {level:.4g} of the median trap peak "
                f"(tolerance {limit:.4g})"
            )
    return blobs, failures


def _notes(
    expect: Expectation,
    table: dict[str, Float],
    tol: Tolerances,
    mode: PupilMode,
    factors: list[int],
    windows: list[float],
    exact: bool,
    counts: list[int],
    coverage: dict[str, float],
    shortfall: int | None,
    k_subtimes: int,
) -> tuple[str, ...]:
    """The caveats that apply to *this* check — the honest part of the report."""
    n_transient, n_filling, n_fade = counts
    array = expect.spec.array
    factor = max(factors, default=1)
    n_frames = len(windows)
    notes = [
        f"drive: {expect.describe()}; pupil model {mode!r}"
        + (f", transform grid decimated {min(factors, default=1)}-{factor}x" if factor > 1 else ""),
    ]
    if shortfall is not None:
        notes.append(
            f"k_subtimes={int(k_subtimes)} too small for exact beat cancellation (needs >= "
            f"{shortfall}); using golden-ratio fallback - verdict noise increases.  This drive's "
            "same-node beats form a commensurate comb that a uniform schedule would annihilate "
            "exactly; the fallback only averages them down."
        )
    notes.append(
        "blind spot: a uniform gain on all four channels is divided out by render_samples' "
        "global normalization, and is optically invisible anyway, so only the intensity "
        "*pattern* is gated - never its absolute scale."
    )
    empty = [name for name, fraction in coverage.items() if fraction <= 0.0]
    notes.append(
        "intensity-gate coverage: "
        + ", ".join(
            f"{name} {fraction:.0%} of {int(table['time'].size)} row(s)"
            for name, fraction in coverage.items()
        )
        + (
            ""
            if not empty
            else f".  {', '.join(empty)} gated NO row: this verdict says nothing about the "
            "intensity pattern (every trap exempt), and Tolerances(require_coverage=True) turns "
            "that into a failure."
        )
    )
    if tol.uniformity_median is None and coverage["uniformity_median"] > 0.0:
        notes.append(
            "uniformity_median (per-trap median over the frames) is measured but not gated: on "
            "the fading-Shepard drives this package ships the ladder slides, so a rung fault is "
            "a one-frame excursion that any median erases (WO-24 §1; clean and 0.80-rung "
            "flagships both median 0.137).  Set Tolerances(uniformity_median=...) on a drive "
            "whose faults hold still - an Eq. S19 tone fault does."
        )
    elif coverage["uniformity_median"] <= 0.0 and n_frames >= MIN_MEDIAN_FRAMES:
        notes.append(
            "uniformity_median: no trap was gated on "
            f"{MIN_MEDIAN_FRAMES} or more of the {n_frames} frames, so there is no time median "
            "to read; the per-frame uniformity gate is the whole intensity verdict here."
        )
    elif n_frames < MIN_MEDIAN_FRAMES:
        notes.append(
            f"uniformity_median: a {n_frames}-frame check has no time axis to median over "
            f"(< {MIN_MEDIAN_FRAMES} frames), so the per-frame uniformity gate is the whole "
            "intensity verdict; the column is nan."
        )
    if windows:
        how = (
            "uniform sub-times over one full period of the drive's beat comb, which annihilates "
            "every same-node beat exactly"
            if exact
            else "golden-ratio sub-times (never commensurate with an array's beat comb)"
        )
        notes.append(
            f"beat-averaging window {min(windows) / us:.4g}-{max(windows) / us:.4g} us over "
            f"{len(windows)} frame(s), measured in the array's own moving frame; {how}."
        )
    if n_transient:
        notes.append(
            f"{n_transient} frame(s) marked transient (before 2 tau, or with the aperture grid "
            "still filling): excluded from the waist and uniformity gates"
            + (
                f"; {n_filling} of them still filling, so positions are not gated there either."
                if n_filling
                else "."
            )
        )
    if n_fade:
        notes.append(
            f"{n_fade} frame(s) inside a hand-over window (fade_pad = "
            f"{expect.fade_pad / us:.4g} us): excluded from the waist and uniformity gates, with "
            "the Eq. S31 shadow tweezers whitelisted there."
        )
    edge_cols, edge_rows = expect.edge_lines()
    if edge_cols or edge_rows:
        notes.append(
            f"fading drive: the intensity gates skip the array's edge lines (columns "
            f"{list(edge_cols)}, rows {list(edge_rows)}).  The p_A + p_B = 1 identity holds every "
            "interior node exactly flat through a hand-over, but the ladder slides, so the "
            "outermost node trades its light with the extended grid (docs/guide.md §6.6)."
        )
        notes.append(
            "fading drive: light on the extended lattice is whitelisted up to "
            f"{tol.blob_fading:.4g} of the median trap peak rather than gated at "
            f"{tol.blob_on_lattice:.4g} - an M + 1 (even M) or M + 2 (odd M) wide array is what "
            "a fading ladder makes, at close to full trap depth; brighter than a real trap it "
            "is still a finding."
        )
    if array.mx > 1 and array.my > 1 and array.delta_f_x == array.delta_f_y:
        notes.append(
            "delta_f_x == delta_f_y: every anti-diagonal of this array shares one optical "
            "frequency f_x + f_y, so those traps are mutually coherent and their beat note is "
            "exactly zero - no averaging window removes it.  The interference is real "
            "(docs/conventions.md §4); give the two axes different spacings to avoid it."
        )
    if table["time"].size and np.any(table["present"] < 0.5):
        dark = int(np.sum(table["present"] < 0.5))
        notes.append(
            f"{dark} (frame, trap) row(s) carried less than {tol.missing_trap:.4g} of their "
            "frame's median peak and were not fitted; their position and waist columns are the "
            "expectation, not a measurement."
        )
    return tuple(notes)
