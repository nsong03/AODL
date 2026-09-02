r"""The one-call front door: a trajectory in, a programmable, simulatable plan out.

Everything else in the package is a layer of the pipeline (``docs/ARCHITECTURE.md`` §1); this
module is the *product*.  :func:`plan_motion` takes the same
:class:`~aodl.trajectory.spec.TrajectorySpec` the synthesizer takes, decides for itself whether
the request fits Eq. 1's bandwidth budget or needs the fading-Shepard ladders of Eqs. S24-S28,
and returns a :class:`MotionPlan` that can be saved for the AWG, simulated, filmed, and — the
part a lab reads first — *explained*, by a :class:`PlanReport` carrying the band occupancy, the
tone counts, the axial budget, the hand-over schedule and the caveats that apply to this
particular drive.

No new physics lives here.  Synthesis is :func:`aodl.waveform.synthesis.synthesize`, simulation
is :func:`aodl.engine.simulate`, rendering is :func:`aodl.viz.movie.render_movie`; this is the
composition of the three, plus the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from .check.expect import Expectation, SimResultLike
from .check.pupil import ApertureGrid, PupilMode
from .check.record import from_arrays
from .check.report import (
    CheckReport,
    Tolerances,
    averaging_window,
    check_samples,
    frame_reach,
)
from .engine import SimResult, simulate
from .params import CHANNELS, AODLParams, default_1030
from .trajectory.spec import TrajectorySpec
from .units import MHz, um, us
from .viz.movie import render_movie
from .viz.style import CHANNEL_COLORS, DARK_STYLE
from .waveform.export import DEFAULT_SAMPLE_RATE, render_samples
from .waveform.shepard import FadeZoneEnvelope, SwitchRamped
from .waveform.synthesis import (
    _poly_extrema,
    max_z_integral,
    requested_z_integral,
    synthesize,
)
from .waveform.tones import ToneTrack, WaveformSet

Float = NDArray[np.float64]

#: Frames in the default :meth:`MotionPlan.simulate` grid — enough to watch a move without
#: paying for a movie.
DEFAULT_FRAMES = 40

#: Times per tone sampled by :func:`band_usage`, on top of the exact candidates (the frequency
#: law's own extrema and every envelope switch instant) that make the answer sharp.
BAND_SAMPLES = 1201

#: Time offset [s] used to decide whether a tone is live *just* inside a switch instant.  A
#: picosecond is :data:`aodl.waveform.tones.TIME_TOL`, far below anything the hardware
#: resolves and far above float64 round-off on microsecond breakpoints.
_LIVE_EPS = 1e-12

#: Samples per :func:`aodl.waveform.export.render_samples` pass inside :meth:`MotionPlan.check`.
#: A 93-tone fading drive renders ~25 % faster in 16 k blocks than in the 1 M default — one
#: block's per-tone evaluation then stays in cache.
_RENDER_CHUNK = 1 << 14

#: Extra drive [s] rendered on each side of what the requested frames strictly need
#: (:meth:`MotionPlan.check`).  The analytic-signal edge ringing of
#: :func:`aodl.check.demod.demodulate` decays as ``1 / (pi f_s dt)`` from a record end, so five
#: microseconds of slack puts it under 1e-4 before it reaches an aperture cell carrying any
#: light at all.
_RENDER_PAD = 5.0 * us

#: Default sub-times per frame, mirroring :func:`aodl.check.report.check_samples`.  It is
#: named here because the *coverage* bound depends on it: a frame averages over a window
#: whose width the beat comb and this count together decide.
DEFAULT_SUBTIMES = 64


# ------------------------------------------------------------------------ band occupancy


def _live_extrema(tone: ToneTrack, grid: Float) -> tuple[float, float] | None:
    """``(min, max)`` detuning [Hz] over the times ``tone`` is actually driven, or ``None``.

    A fading-Shepard rung spends most of its programmed span switched off, and its frequency
    law is unbounded there by design (:func:`aodl.waveform.shepard.shepard_band_bound`) — so
    the occupancy question is only ever about the times the envelope is non-zero.  The
    candidate set is the sample ``grid`` plus the two *exact* extrema of the frequency law and
    every branch boundary the envelope publishes (``zone_times``); a boundary counts as live
    when the envelope is non-zero a picosecond to either side of it, which is what makes the
    edge of a fade window — where the excursion is largest — land in the answer exactly.
    """
    times = [grid, np.asarray([t for _, t in _poly_extrema(tone.freq)], dtype=np.float64)]
    zones = getattr(tone.env, "zone_times", None)
    if zones is not None:
        times.append(np.asarray(zones, dtype=np.float64))
    t = np.unique(np.concatenate(times))
    amplitude = np.asarray(tone.env.A(t), dtype=np.float64)
    for shift in (-_LIVE_EPS, _LIVE_EPS):
        amplitude = np.maximum(amplitude, np.asarray(tone.env.A(t + shift), dtype=np.float64))
    live = amplitude > 0.0
    if not np.any(live):
        return None
    f = np.asarray(tone.freq(t), dtype=np.float64)[live]
    return float(f.min()), float(f.max())


def band_usage(
    wfs: WaveformSet, samples: int = BAND_SAMPLES
) -> dict[str, tuple[float, float, float]]:
    """``{channel: (f_min, f_max, margin)}`` [Hz] over the drive's **live** tones.

    ``f_min``/``f_max`` are *absolute* RF frequencies — ``f_center`` plus the IR's detuning
    (``docs/conventions.md`` §1) — so they compare directly against
    :attr:`aodl.params.AODParams.band`, and ``margin`` is the distance to the nearer band edge:
    positive means the drive fits, with that much room to spare, and negative is by how much it
    does not.  A channel whose tones are all silent reports its carrier and the full headroom.
    """
    t0, t1 = wfs.t_span
    grid = np.linspace(t0, t1, max(int(samples), 2))
    out: dict[str, tuple[float, float, float]] = {}
    for name, cw in wfs.channels.items():
        aod = wfs.params.channels[name]
        spans = [span for span in (_live_extrema(tone, grid) for tone in cw.tones) if span]
        lo = aod.f_center + min((span[0] for span in spans), default=0.0)
        hi = aod.f_center + max((span[1] for span in spans), default=0.0)
        band_lo, band_hi = aod.band
        out[name] = (lo, hi, min(band_hi - hi, lo - band_lo))
    return out


# ------------------------------------------------------------------------- fade schedule


@dataclass(frozen=True)
class FadeEvent:
    """One fading-Shepard hand-over: when it happens, on which axis, and what it lights up.

    Attributes
    ----------
    time:
        Drive time [s] of the fade *centre* — the instant ``|g| = M delta_f / 2`` where the
        dying and rising rungs of that axis are equally loud (Eqs. S26/S27).  The atom plane
        sees it half an aperture transit later (``docs/conventions.md`` §7).
    axis:
        ``"x"`` or ``"y"``; Table II interlaces the two by ``xi = 1/2``, so with equal
        spacings they alternate.
    shadow:
        Distance [m] of the Eq. S31 shadow tweezers from each trap at that instant,
        ``deflection_scale * delta_f`` on that axis.  Each shadow peaks at half a trap's
        power, and a *pick-up must not be scheduled here* — that is what the list is for.
    """

    time: float
    axis: str
    shadow: float


def fade_schedule(wfs: WaveformSet, params: AODLParams) -> list[FadeEvent]:
    """Every hand-over instant of a fading-Shepard drive, in time order (empty for Eq. S19).

    Read off the ``A`` channel of each axis, whose window is one ladder step wide on both
    Table II rows: rung ``n`` reaching ``+g_centre`` is the same instant as rung ``n-1``
    reaching ``-g_centre``, so the union over the ladder has exactly one entry per hand-over.
    """
    t0, t1 = wfs.t_span
    events: list[FadeEvent] = []
    for axis, name in (("x", "Ax"), ("y", "Ay")):
        cw = wfs.channels.get(name)
        if cw is None:
            continue
        found: list[Float] = []
        spacing = 0.0
        for tone in cw.tones:
            env = tone.env
            if not isinstance(env, FadeZoneEnvelope | SwitchRamped):
                continue
            spacing = env.base.delta_f if isinstance(env, SwitchRamped) else env.delta_f
            found.append(np.asarray(env.crossing_times(env.g_centre), dtype=np.float64))
        if not found:
            continue
        times = np.unique(np.concatenate(found))
        times = times[(times >= t0) & (times <= t1)]
        shadow = params.deflection_scale * spacing
        events.extend(FadeEvent(time=float(t), axis=axis, shadow=shadow) for t in times)
    return sorted(events, key=lambda event: event.time)


# ------------------------------------------------------------------------------- report


@dataclass(frozen=True)
class PlanReport:
    """What :func:`plan_motion` did, and what a lab has to know about the result.

    Attributes
    ----------
    mode:
        ``"s19"`` — the plain Eq. S19 drive of ``docs/PLAN.md`` §1.4 — or ``"shepard"``, the
        fading ladders of Eqs. S24-S28 that hold ``Z`` past Eq. 1's budget.
    band_usage:
        ``{channel: (f_min, f_max, margin)}`` [Hz] of the live tones (:func:`band_usage`).
    tone_counts:
        ``{channel: number of programmed tones}``.  Under ``"shepard"`` this is the *ladder*
        length; only a handful of rungs are audible at any instant.
    z_budget:
        ``(requested, ceiling)`` [m.s]: the ``max |int Z dt|`` the trajectory asks for, and
        what a plain Eq. S19 drive can buy (:func:`aodl.waveform.synthesis.max_z_integral`,
        doubled when ``f_z_bias`` is in force).  The ceiling is ``inf`` under ``"shepard"`` —
        the whole point of the scheme.
    fade_events:
        The hand-over schedule (:func:`fade_schedule`); empty under ``"s19"``.
    notes:
        The caveats that apply to *this* drive, in the order they matter.
    description:
        The synthesizer's own account of what it built — including, under ``shepard="auto"``,
        the band refusal that made it switch modes.
    wfs:
        The drive these numbers describe, kept so :meth:`figure` can draw it.
    """

    mode: str
    band_usage: dict[str, tuple[float, float, float]]
    tone_counts: dict[str, int]
    z_budget: tuple[float, float]
    fade_events: list[FadeEvent]
    notes: tuple[str, ...]
    description: str = ""
    wfs: WaveformSet | None = None

    # -- derived numbers

    @property
    def worst_margin(self) -> tuple[str, float]:
        """``(channel, margin)`` of the tightest channel [Hz] — the number to read first."""
        if not self.band_usage:
            return "", math.inf
        name = min(self.band_usage, key=lambda key: self.band_usage[key][2])
        return name, self.band_usage[name][2]

    @property
    def n_tones(self) -> int:
        """Total programmed tones across all four channels."""
        return sum(self.tone_counts.values())

    def summary(self) -> str:
        """A short human-readable block: mode, tones, band usage, budget, schedule, caveats."""
        worst_name, worst = self.worst_margin
        lines = [f"AODL motion plan - mode: {self.mode}"]
        if self.description:
            head = self.description.split(";")[0]
            lines.append(f"  drive        {head}{' ...' if head != self.description else ''}")
        counts = "  ".join(f"{name}:{self.tone_counts.get(name, 0)}" for name in CHANNELS)
        lines.append(f"  tones        {counts}   ({self.n_tones} total)")
        lines.append("  band usage   channel    live span [MHz]        margin [MHz]")
        for name in CHANNELS:
            if name not in self.band_usage:
                continue
            lo, hi, margin = self.band_usage[name]
            lines.append(
                f"               {name:<10} [{lo / MHz:8.4f}, {hi / MHz:8.4f}]   "
                f"{margin / MHz:+9.4f}"
            )
        lines.append(
            f"               worst band margin {worst / MHz:+.4f} MHz on {worst_name!r}"
            + ("" if worst >= 0.0 else "  <-- out of band")
        )
        requested, ceiling = self.z_budget
        budget = "unbounded (the ladder slides, the live window does not)"
        if math.isfinite(ceiling):
            budget = f"{ceiling:.4g} m.s ({100.0 * requested / ceiling:.1f} % used)"
        lines.append(f"  z budget     |int Z dt| = {requested:.4g} m.s of {budget}")
        if self.fade_events:
            per_axis = {axis: sum(1 for e in self.fade_events if e.axis == axis) for axis in "xy"}
            shadows = {e.axis: e.shadow for e in self.fade_events}
            offsets = ", ".join(
                f"{axis}: +-{shadows[axis] / um:.2f} um" for axis in sorted(shadows)
            )
            lines.append(
                f"  fades        {len(self.fade_events)} hand-over(s) "
                f"(x: {per_axis['x']}, y: {per_axis['y']}); shadows at {offsets}"
            )
        if self.notes:
            lines.append("  notes")
            lines.extend(f"    - {note}" for note in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()

    # -- the picture

    def figure(self, wfs: WaveformSet | None = None, samples: int = 401) -> Figure:
        """Band-usage and tone-track overview: what the four channels do, and where in the band.

        Top panel — every tone's beam-centre frequency ``f_center + f(t)`` against time, with
        segment opacity following its envelope, so a fading-Shepard ladder shows its live rungs
        and nothing else (the same analytic "spectrogram" the movie's drive strip draws, no FFT:
        ``CLAUDE.md``).  Bottom panel — the live span of each channel inside its band, so the
        margin the summary quotes is visible as a gap.

        ``wfs`` defaults to the drive the report was built from (:attr:`wfs`).
        """
        drive = self.wfs if wfs is None else wfs
        if drive is None:
            raise ValueError(
                "this PlanReport carries no waveform set, so there is nothing to draw; pass "
                "the drive explicitly: report.figure(wfs)"
            )
        with mpl.rc_context(cast(Any, DARK_STYLE)):
            fig = Figure(figsize=(8.2, 5.4), dpi=110)
            top, bottom = fig.subplots(2, 1, height_ratios=(2.0, 1.0))
            _plot_tone_tracks(top, drive, samples)
            _plot_band_usage(bottom, drive, self.band_usage)
            fig.suptitle(f"AODL plan - {self.mode}", fontsize=11)
            fig.tight_layout()
        return fig


def _plot_tone_tracks(ax: Any, wfs: WaveformSet, samples: int) -> None:
    """Per-channel tone frequencies over time, opacity following the envelope."""
    from matplotlib.collections import LineCollection

    t0, t1 = wfs.t_span
    t = np.linspace(t0, t1, max(int(samples), 2))
    live_lo, live_hi = math.inf, -math.inf
    for name in CHANNELS:
        cw = wfs.channels.get(name)
        if cw is None:
            continue
        aod = wfs.params.channels[name]
        color = CHANNEL_COLORS.get(name, "#d7dde5")
        segments: list[Float] = []
        alphas: list[Float] = []
        for tone in cw.tones:
            f = (aod.f_center + np.asarray(tone.freq(t), dtype=np.float64)) / MHz
            env = np.clip(np.asarray(tone.env.A(t), dtype=np.float64), 0.0, 1.0)
            points = np.column_stack([t / us, f])
            segments.append(np.stack([points[:-1], points[1:]], axis=1))
            mean_env = 0.5 * (env[:-1] + env[1:])
            alphas.append(np.where(mean_env > 0.0, 0.25 + 0.75 * mean_env, 0.0))
            if np.any(env > 0.0):
                live_lo = min(live_lo, float(np.min(f[env > 0.0])))
                live_hi = max(live_hi, float(np.max(f[env > 0.0])))
        if segments:
            ax.add_collection(
                LineCollection(
                    list(np.concatenate(segments)),
                    colors=color,
                    linewidths=1.4,
                    alpha=np.concatenate(alphas),
                )
            )
        ax.plot([], [], color=color, lw=1.6, label=name)
        for edge in aod.band:
            ax.axhline(edge / MHz, color="#5a6472", lw=0.8, ls=":")
    ax.set_xlim(t0 / us, t1 / us)
    if live_hi > live_lo:
        margin = 0.12 * (live_hi - live_lo)
        ax.set_ylim(live_lo - margin, live_hi + margin)
    ax.set_xlabel("drive time [µs]")
    ax.set_ylabel("tone frequency [MHz]")
    ax.set_title("live tone tracks (opacity = envelope)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncols=4, labelcolor="#d7dde5")


def _plot_band_usage(
    ax: Any, wfs: WaveformSet, usage: dict[str, tuple[float, float, float]]
) -> None:
    """One bar per channel: the band, the live span inside it, and the carrier."""
    names = [name for name in CHANNELS if name in usage]
    for row, name in enumerate(names):
        aod = wfs.params.channels[name]
        lo, hi, _ = usage[name]
        ax.barh(
            row,
            (aod.band[1] - aod.band[0]) / MHz,
            left=aod.band[0] / MHz,
            height=0.68,
            color="#20252e",
            edgecolor="#5a6472",
            linewidth=0.8,
        )
        ax.barh(
            row,
            max(hi - lo, 1e-9) / MHz,
            left=lo / MHz,
            height=0.42,
            color=CHANNEL_COLORS.get(name, "#d7dde5"),
        )
        ax.plot([aod.f_center / MHz], [row], marker="|", color="#f2f5f8", ms=9, mew=1.0)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("RF frequency [MHz]")
    ax.set_title("band occupancy (bar = usable band, fill = live tones)", fontsize=9)


# --------------------------------------------------------------------------------- plan


@dataclass
class MotionPlan:
    """A synthesized move: the drive, the hardware it was written for, and the report.

    The object :func:`plan_motion` returns and the one a lab keeps: :meth:`save` writes the
    parametric NPZ that goes into version control, :meth:`render_samples` expands it for the
    AWG, :meth:`simulate` and :meth:`movie` show what the tweezers will do, :meth:`check`
    verifies the *rendered samples* against the request from outside the simulator, and
    :attr:`report` says whether any of it is close to a limit.

    Attributes
    ----------
    spec, params, wfs, report:
        The request, the hardware, the drive, and what :func:`plan_motion` made of them.
    options:
        The synthesis options this plan was built with — ``retard_compensate``, ``amp``,
        ``f_z_bias``, ``switch_ramp`` and the resolved ``shepard`` mode.  They are not
        recoverable from the waveforms (a compensated and an uncompensated drive are both
        perfectly good drives, they simply encode different instants), and :meth:`check` needs
        them to know what the drive was *asked* for.  Keyword-constructed and defaulted, so a
        :class:`MotionPlan` built the pre-M6 way still works.
    """

    spec: TrajectorySpec
    params: AODLParams
    wfs: WaveformSet
    report: PlanReport
    options: dict[str, Any] = field(default_factory=dict)

    # -- outputs

    def save(self, path: str | Path) -> Path:
        """Write the **parametric** NPZ (segments and parameters, never samples).

        See ``docs/waveform_format.md``.  A drive built with ``switch_ramp > 0`` carries an
        envelope the v2 schema has no slot for and is refused by name — re-synthesize from
        :attr:`spec`, or save the un-ramped drive.
        """
        return self.wfs.save(path)

    def render_samples(
        self, rate: float = DEFAULT_SAMPLE_RATE, **kwargs: Any
    ) -> dict[str, NDArray[Any]] | tuple[dict[str, NDArray[Any]], float]:
        """Expand the drive into normalized AWG samples at ``rate`` [S/s].

        Thin wrapper over :func:`aodl.waveform.export.render_samples` (which is where the
        carrier is put back); keyword arguments pass straight through.
        """
        return render_samples(self.wfs, rate, **kwargs)

    def default_times(self, n_frames: int = DEFAULT_FRAMES) -> Float:
        """The default frame grid: ``n_frames`` times over ``[tau, T + tau/2]`` [s].

        Starting at one full aperture transit skips the fill transient (before ``tau/2`` a
        pair-driven tweezer is strictly dark, ``docs/conventions.md`` §7), and ending half a
        transit past the trajectory is what lets the last requested instant be observed —
        the drive at ``t - tau/2`` is then exactly its final state.
        """
        tau = max(aod.transit_time for aod in self.params.channels.values())
        return np.linspace(tau, self.spec.duration + 0.5 * tau, max(int(n_frames), 1))

    def simulate(self, times: ArrayLike | None = None, **kwargs: Any) -> SimResult:
        """Simulate the drive at ``times`` [s]; ``None`` uses :meth:`default_times`.

        Keyword arguments (``channels``, ``tol``) pass through to
        :func:`aodl.engine.simulate`.
        """
        frames = self.default_times() if times is None else times
        return simulate(self.wfs, frames, **kwargs)

    def movie(self, path: str | Path, times: ArrayLike | None = None, **kwargs: Any) -> Path:
        """Simulate and render a movie; returns the path actually written.

        Keyword arguments pass through to :func:`aodl.viz.movie.render_movie` (``grid``,
        ``mode``, ``fps``, ``xz_panel``, ``dpi``, ...).
        """
        return render_movie(self.simulate(times), path, **kwargs)

    def figure(self, **kwargs: Any) -> Figure:
        """The report's overview figure for this drive (:meth:`PlanReport.figure`)."""
        return self.report.figure(self.wfs, **kwargs)

    def summary(self) -> str:
        """Shorthand for ``self.report.summary()``."""
        return self.report.summary()

    # -- the independent check

    def expectation(self, fade_pad: float = 0.0) -> Expectation:
        """What this plan *asked* for, as the checker's :class:`~aodl.check.expect.Expectation`.

        Built from :attr:`spec`, :attr:`params` and :attr:`options` — never from the waveforms —
        plus the hand-over schedule of :attr:`PlanReport.fade_events`, converted from drive
        times to **observation** times by adding ``tau/2`` (``docs/conventions.md`` §7): a fade
        centred on the drive at ``t`` reaches the atom plane half an aperture transit later,
        whether or not the drive was retard-compensated.
        """
        tau = self._tau()
        return Expectation(
            spec=self.spec,
            params=self.params,
            retard_compensated=bool(self.options.get("retard_compensate", False)),
            amp=float(self.options.get("amp", 1.0)),
            shadows=tuple(
                (event.time + 0.5 * tau, event.axis, event.shadow)
                for event in self.report.fade_events
            ),
            fade_pad=float(fade_pad),
        )

    def check_times(
        self, mode: PupilMode = "bragg_band", k_subtimes: int = DEFAULT_SUBTIMES
    ) -> Float:
        """The default frame grid of :meth:`check` — nine or fewer deterministic instants.

        ``2 tau`` (the first frame with the aperture grid fully lit), then every move's midpoint
        and every seam, and finally the end of the trajectory — each shifted by ``tau/2`` unless
        the drive was retard-compensated, so that what is *observed* at the frame is what was
        *requested* at that waypoint.  Frames the record could not reach are dropped: a frame
        gathers drive over ``t -+ (W/2 + tau/2 + grid.half_span/v)``, the aperture grid running
        wider than the crystal (:func:`aodl.check.report.frame_reach`) and the frame averaging
        over a beat window on top of that (:func:`aodl.check.report.averaging_window`).
        """
        tau = self._tau()
        shift = 0.0 if self.options.get("retard_compensate", False) else 0.5 * tau
        marks = [2.0 * tau]
        elapsed = 0.0
        for move in self.spec.moves:
            marks.append(elapsed + 0.5 * move.duration + shift)
            elapsed += move.duration
            marks.append(elapsed + shift)
        frames = np.unique(np.asarray(marks, dtype=np.float64))
        limit = self._frame_limit(mode, k_subtimes)
        kept = frames[frames <= limit]
        if kept.size == 0:
            raise ValueError(
                f"none of this plan's default check frames fit inside the rendered drive: the "
                f"earliest is {frames[0] / us:.4g} us and the record supports frames only up to "
                f"{limit / us:.4g} us.  Synthesize with a longer t_pad."
            )
        return kept

    def check(
        self,
        *,
        times: ArrayLike | None = None,
        rate: float = DEFAULT_SAMPLE_RATE,
        mode: PupilMode = "bragg_band",
        tolerances: Tolerances | None = None,
        sim: SimResultLike | None = None,
        fade_pad: float = 0.0,
        **kwargs: Any,
    ) -> CheckReport:
        """Render this drive to samples and check the tweezers they make.  The M6 gate.

        The samples are rendered in ``float64`` with their normalization kept
        (``return_scale=True``), wrapped in a :class:`~aodl.check.record.SampleRecord`, and
        handed with :meth:`expectation` to :func:`aodl.check.report.check_samples`.  Nothing of
        the simulator is involved: the verdict comes from an FFT rebuild of the literal RF
        buffers (:mod:`aodl.check`).

        Only the span the requested frames actually need is rendered — ``[t - tau/2 -
        half_span/v, t - tau/2 + half_span/v]`` over the frames, plus a few microseconds of
        slack for the demodulation's edge ringing — so checking a handful of frames of a long
        drive costs a short render.  The normalization is the peak of *that* window rather than
        of the whole buffer, which changes nothing physical: ``samples * normalization`` is the
        same Eq. S1 drive either way.

        Parameters
        ----------
        times:
            Frame observation times [s]; ``None`` uses :meth:`check_times`.
        rate:
            Sample rate [S/s] to render at (default 625 MS/s, the product AWG).
        mode:
            Pupil model — ``"bragg_band"`` (default, the full crystal) or ``"weak"``.
        tolerances:
            :class:`aodl.check.report.Tolerances`; ``None`` uses the defaults.
        sim:
            Optional :class:`aodl.engine.SimResult` to diff against, report-only.
        fade_pad:
            Half-width [s] of the hand-over exclusion (:attr:`aodl.check.expect.
            Expectation.fade_pad`).  ``0`` — the default — gates every frame: on an array drive
            the hand-over disturbance is confined to the *edge lines* of a fading axis, which
            the checker excludes per trap instead, so dropping whole frames would only weaken
            the gate.  Set it to a fraction of ``tau`` for a single-tweezer fading drive, whose
            shadow tweezers are then whitelisted while they exist.
        **kwargs:
            Forwarded to :func:`aodl.check.report.check_samples` (``k_subtimes``, ``n_z``,
            ``z_half_range``, ``oversample``).

        Returns
        -------
        A :class:`aodl.check.report.CheckReport`.
        """
        k_subtimes = int(kwargs.get("k_subtimes", DEFAULT_SUBTIMES))
        frames = (
            self.check_times(mode, k_subtimes)
            if times is None
            else np.atleast_1d(np.asarray(times, dtype=np.float64)).ravel()
        )
        if frames.size == 0:
            raise ValueError("MotionPlan.check() needs at least one frame time")
        limit = self._frame_limit(mode, k_subtimes)
        worst = float(np.max(frames))
        if worst > limit:
            raise ValueError(
                f"frame time t = {worst / us:.4g} us cannot be checked: a frame gathers drive "
                f"over t -+ (W/2 + tau/2 + grid.half_span/v), and this drive is only programmed "
                f"to {self.wfs.t_span[1] / us:.4g} us, which supports frames up to "
                f"{limit / us:.4g} us.  Synthesize with a longer t_pad, or ask for earlier "
                f"frames."
            )

        tau = self._tau()
        reach = frame_reach(ApertureGrid.design(self.params, mode), self.params)
        # The render has to cover every sub-time's aperture, not just the frame's own.
        edge = 0.5 * averaging_window(self.expectation(), k_subtimes) + reach + _RENDER_PAD
        t0, t1 = self.wfs.t_span
        span = (
            max(t0, float(np.min(frames)) - 0.5 * tau - edge),
            min(t1, worst - 0.5 * tau + edge),
        )
        arrays, scale = cast(
            tuple[dict[str, NDArray[Any]], float],
            render_samples(
                self.wfs,
                rate,
                span,
                dtype=np.float64,
                chunk=_RENDER_CHUNK,
                return_scale=True,
            ),
        )
        record = from_arrays(arrays, rate, self.params, t_start=span[0], normalization=scale)
        return check_samples(
            record,
            self.expectation(fade_pad),
            times=frames,
            mode=mode,
            tolerances=tolerances,
            sim=sim,
            **kwargs,
        )

    def _tau(self) -> float:
        """The stack's aperture transit time [s] — the longest of the four channels."""
        return max(aod.transit_time for aod in self.params.channels.values())

    def _frame_limit(self, mode: PupilMode, k_subtimes: int = DEFAULT_SUBTIMES) -> float:
        """Latest checkable frame, ``t_end + tau/2 - half_span/v - W/2``."""
        reach = frame_reach(ApertureGrid.design(self.params, mode), self.params)
        half_window = 0.5 * averaging_window(self.expectation(), k_subtimes)
        return self.wfs.t_span[1] + 0.5 * self._tau() - reach - half_window


# ------------------------------------------------------------------------------ the door


def _is_fading(wfs: WaveformSet) -> bool:
    """Does any tone carry a fading-Shepard window?  (That is what "shepard mode" means.)"""
    return any(
        isinstance(tone.env, FadeZoneEnvelope | SwitchRamped)
        for cw in wfs.channels.values()
        for tone in cw.tones
    )


def _switch_ramp_of(wfs: WaveformSet) -> float:
    """The ``switch_ramp`` [s] the rectangular rungs carry, or ``0`` when they carry none."""
    for cw in wfs.channels.values():
        for tone in cw.tones:
            if isinstance(tone.env, SwitchRamped):
                return tone.env.ramp
    return 0.0


def _notes(
    spec: TrajectorySpec, wfs: WaveformSet, mode: str, options: dict[str, Any]
) -> tuple[str, ...]:
    """The caveats that apply to *this* drive — the honest part of the report."""
    array = spec.array
    tau = max(aod.transit_time for aod in wfs.params.channels.values())
    ramp = _switch_ramp_of(wfs)
    notes: list[str] = []

    if options.get("retard_compensate"):
        notes.append(
            f"retard_compensate=True: the trajectory is read tau/2 = {0.5 * tau / us:.2f} us "
            f"ahead, so the atom plane matches the request at t (not t - tau/2).  The array "
            f"starts half a transit into its first move, and the fill transient of the first "
            f"tau is unchanged - it belongs to the aperture, not to the schedule."
        )
    else:
        notes.append(
            f"the atom plane lags the drive by tau/2 = {0.5 * tau / us:.2f} us: compare "
            f"measurements at t against the request at t - tau/2 (or pass "
            f"retard_compensate=True)."
        )
    bias = options.get("f_z_bias", 0.0)
    if mode == "s19" and not (isinstance(bias, float | int) and float(bias) == 0.0):
        notes.append(
            "f_z_bias is in force: the f_Z walk is centred in the band, which doubles Eq. 1's "
            "budget.  Being common to all four channels and constant in time it cancels in "
            "every Table I quantity - no trap moves."
        )
    if mode == "shepard":
        if array.mx > 1 or array.my > 1:
            parity = []
            for axis, count, what in (("x", array.mx, "columns"), ("y", array.my, "rows")):
                if count == 1:
                    continue
                parity.append(
                    f"{axis}: {count} + 2 {what} during a hand-over"
                    if count % 2
                    else f"{axis}: {count} + 1 {what} at every instant"
                )
            notes.append(
                "the fading ladder lights extra columns beyond the array you asked for ("
                + "; ".join(parity)
                + ") - do not schedule a pick-up inside a fade zone (see fade_events)."
            )
            if any(count % 2 == 0 for count in (array.mx, array.my)):
                notes.append(
                    "even-M ladders carry a delta_f/2 comb offset so their traps land on the "
                    "same lattice as Eq. S19 (WO-17 §2.1); it costs delta_f/2 of band."
                )
            if ramp == 0.0:
                notes.append(
                    "the array's B rungs are Table II rectangles (p_B = 0): they switch on and "
                    "off instantaneously, radiating roughly -40 dB of out-of-band splatter.  "
                    "A few us of switch_ramp removes it at no interior cost - only the "
                    "rectangles are ramped, and the cos^p A windows already switch smoothly."
                )
            else:
                odd = [
                    axis
                    for axis, count in (("x", array.mx), ("y", array.my))
                    if count > 1 and count % 2
                ]
                axes = " and ".join(odd)
                whose = f"{axes} ladders'" if len(odd) > 1 else f"{axes} ladder's"
                cost = (
                    f"the interior columns stay exactly flat, and only the {whose} two "
                    f"edge columns dip, by ~(pi |gdot| r / delta_f)^2 / 5 during a hand-over"
                    if odd
                    else "every column of the array stays exactly flat (an even-M rung "
                    "switches mid-plateau, where the A ladder is not handing over)"
                )
                notes.append(
                    f"switch_ramp = {ramp / us:.3g} us softens the p_B = 0 rectangles and "
                    f"nothing else: {cost}.  Such a drive does not round-trip through the "
                    f"parametric NPZ."
                )
        notes.append(
            "shadow tweezers at +- deflection_scale * delta_f carry half a trap's power "
            "mid-fade (Eq. S31); Table II's xi = 1/2 keeps them to one axis at a time."
        )
    if not options.get("check_band", True):
        notes.append(
            "check_band=False: the band was NOT verified.  A drive outside the AOD's band "
            "simply does not diffract - this waveform is for plotting."
        )
    return tuple(notes)


def plan_motion(
    spec: TrajectorySpec,
    params: AODLParams | None = None,
    *,
    shepard: str | Any = "auto",
    **synth_opts: Any,
) -> MotionPlan:
    r"""Plan a move: array + waypoints in, waveforms + simulation + report out.

    ::

        from aodl import ArraySpec, Lift, Translate, TrajectorySpec, plan_motion
        from aodl.units import MHz, um, us

        array = ArraySpec(10, 10, 1.0 * MHz, 1.3 * MHz)          # 10x10 traps, 10.3/13.4 um
        moves = (Lift(10 * um, 150 * us),                        # up, out of the plane
                 Translate(40 * um, 25 * um, 250 * us),          # across
                 Lift(-10 * um, 150 * us))                       # and down
        plan = plan_motion(TrajectorySpec(array=array, moves=moves))
        print(plan.report.summary())                             # band usage, budget, caveats
        plan.save("move.npz")                                    # parameters, not samples
        plan.movie("move.mp4")                                   # what the atoms will see

    Parameters
    ----------
    spec:
        The array and its waypoints.
    params:
        Hardware; ``None`` uses :func:`aodl.params.default_1030`.
    shepard:
        Passed to :func:`aodl.waveform.synthesis.synthesize`, but defaulting to ``"auto"``
        rather than ``None``: the product front door tries the plain Eq. S19 drive first and
        falls back to the fading-Shepard ladders of Eqs. S24-S28 only when the band refuses it.
        Pass ``None`` to insist on Eq. S19 (and get its error instead of a fallback), or a
        :class:`~aodl.waveform.shepard.ShepardConfig` to insist on ladders.
    **synth_opts:
        Everything else :func:`~aodl.waveform.synthesis.synthesize` takes — ``amp``,
        ``phases``, ``t_pad``, ``rng``, ``check_band``, and the M5 options
        ``retard_compensate``, ``f_z_bias``, ``switch_ramp``.

    Returns
    -------
    A :class:`MotionPlan`.  Nothing is simulated or rendered yet: synthesis is cheap, and the
    report is derived from the waveform set alone.
    """
    hardware = default_1030() if params is None else params
    if not isinstance(spec, TrajectorySpec):
        raise TypeError(f"plan_motion() needs a TrajectorySpec, got {type(spec)!r}")
    wfs = synthesize(spec, hardware, shepard=shepard, **synth_opts)

    mode = "shepard" if _is_fading(wfs) else "s19"
    biased = mode == "s19" and bool(synth_opts.get("f_z_bias", 0.0))
    ceiling = math.inf if mode == "shepard" else max_z_integral(hardware, biased=biased)
    report = PlanReport(
        mode=mode,
        band_usage=band_usage(wfs),
        tone_counts={name: cw.n_tones for name, cw in wfs.channels.items()},
        z_budget=(requested_z_integral(spec, hardware), ceiling),
        fade_events=fade_schedule(wfs, hardware),
        notes=_notes(spec, wfs, mode, synth_opts),
        description=wfs.description,
        wfs=wfs,
    )
    # The synthesis options are not recoverable from the waveforms, and M6's checker needs
    # them to know what the drive was *asked* for (MotionPlan.options).
    options: dict[str, Any] = {
        "retard_compensate": bool(synth_opts.get("retard_compensate", False)),
        "amp": float(synth_opts.get("amp", 1.0)),
        "f_z_bias": synth_opts.get("f_z_bias", 0.0),
        "switch_ramp": _switch_ramp_of(wfs),
        "shepard": mode,
    }
    return MotionPlan(spec=spec, params=hardware, wfs=wfs, report=report, options=options)


__all__ = [
    "BAND_SAMPLES",
    "DEFAULT_FRAMES",
    "FadeEvent",
    "MotionPlan",
    "PlanReport",
    "band_usage",
    "fade_schedule",
    "plan_motion",
]
