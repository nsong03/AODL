r"""Bragg / tangential-phase-matching diffraction efficiency of the AODL's AOD pairs.

**Single AOD (Kogelnik coupled-wave, plane waves, one interaction length).**  The
diffraction efficiency of a thick phase grating is

    eta(Gamma, delta) = Gamma^2 sin^2( sqrt(Gamma^2 + delta^2) ) / (Gamma^2 + delta^2)

with ``Gamma`` the coupling strength (``eta_peak = sin^2 Gamma`` at perfect matching) and
``delta = Delta k L / 2`` the phase-mismatch parameter over the interaction length ``L``.

**Tangential phase matching (TPM).**  For an anisotropic (TeO2 slow-shear) deflector the
Bragg incidence angle as a function of the acoustic frequency ``f`` is

    theta_i(f) = lambda f / (2 n v) + v (n_i^2 - n_d^2) / (2 n lambda f)

which is *stationary* at the TPM frequency ``f_tpm = (v / lambda) sqrt(n_i^2 - n_d^2)`` (a
crystal-cut design parameter; commercial cuts put it near band centre).  At fixed incidence
``theta_i(f_tpm)`` the mismatch is ``Delta k ~ K [theta_i(f) - theta_i(f_tpm)]`` with
``K = 2 pi f / v``, and since ``theta_i'(f_tpm) = 0`` the leading term is quadratic,
``theta_i''(f_tpm) = lambda / (n v f_tpm)``, hence

    delta(f) = Delta k L / 2 = (pi lambda L / (2 n v^2)) (f - f_tpm)^2 = C2 (f - f_tpm)^2

(:func:`tpm_mismatch`).  At ``n = 2.26``, ``v = 650 m/s``, ``lambda = 1030 nm``, ``L = 4 mm``,
``eta_peak = 0.85`` this gives a 3 dB full bandwidth of 27.9 MHz (:func:`bandwidth_3db`), the
right order for a DTSXY-400.  Only ``delta^2`` enters ``eta``, so the overall sign of ``delta``
is immaterial here and below.

**The second AOD of a pair sees the first one's deflection as an additive mismatch.**  The
B crystal of a counter-propagating pair is illuminated by light the A crystal has already
deflected by the *internal* angle ``Delta theta_A = lambda (f_A - f_tpm) / (n v)``.  Through
the 4f relay a signed fraction ``gamma`` of that angle reaches B's Bragg plane, so B's
incidence is ``theta_in = theta_i(f_tpm) + gamma lambda (f_A - f_tpm) / (n v)``.  At TPM the
Bragg curve is flat to first order (``theta_i'(f_tpm) = 0``), so **nothing about f_B can
compensate an incidence change to first order**; with ``K_B = 2 pi f_B / v ~ 2 pi f_tpm / v``:

    delta_B(f_B; f_A) = (L/2) K_B [theta_in - theta_i(f_B)]
                      = C2 [ (f_B - f_tpm)^2 - 2 gamma f_tpm (f_A - f_tpm) ],
    C2 = pi lambda L / (2 n v^2)

(:func:`second_aod_efficiency`; the documents write the bracket with this sign, the literal
derivation gives its negative, and ``eta`` depends on ``delta^2`` only).  Consequences — the
paper's "Design caveats": at fixed ``f_B`` the pair efficiency falls off in ``f_A`` on a scale
``~ 1/(L gamma)`` while in ``f_B`` it keeps the TPM-broadened width, so **the first AOD's
usable band is narrow because its deflection detunes the second**.  The map is asymmetric in
``f_A``: where ``gamma (f_A - f_tpm) > 0`` B re-matches on the parabolic ridge
``(f_B - f_tpm)^2 = 2 gamma f_tpm (f_A - f_tpm)`` (the two matched frequencies of anisotropic
diffraction); on the other side it is simply cut off.  ``|gamma| = 1`` is an ideal relay of
magnification -1 with plane acoustic waves; smaller values stand in for the angular-acceptance
widening by acoustic divergence that a single-``L`` plane-wave model lacks.  The default
``gamma = 0.2`` reproduces the paper's "a few times narrower" first-AOD band at ``L = 4 mm``.

**Caveats of this model** (for the UI as well): plane waves and a single ``L`` (no acoustic
walk-off or divergence), no optical activity near [001], no TeO2 mid-band degeneracy dip, no
measured calibration — the paper's Fig. S1c / S8 maps are the calibration target.  Do *not*
replace ``C2`` by the ``n_o^3 / (v^4 (n_e^2 - n_o^2))`` form found in some AODL simulators: it
is off by ~10^4 and predicts GHz bandwidths.

All quantities are SI (Hz, m, rad).  Signs that concern the pupil (sound direction,
diffraction order) are not used here: efficiency is a scalar weight per tone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from aodl.params import CHANNELS, AODLParams
from aodl.units import MHz
from aodl.waveform.tones import ChannelWaveform, WaveformSet

Float = NDArray[np.float64]

#: Product default interaction length [m] (DESIGN §1.6).
DEFAULT_INTERACTION_LENGTH = 4.0e-3
#: TeO2 slow-shear refractive index used by the default model.
DEFAULT_REFRACTIVE_INDEX = 2.26
#: Default peak single-AOD efficiency ``sin^2 Gamma``.
DEFAULT_PEAK_EFFICIENCY = 0.85
#: Default signed angular coupling ``gamma`` of the A deflection into B's Bragg plane.
DEFAULT_ANGULAR_COUPLING = 0.2

#: Points per axis of the internal map :func:`first_aod_usable_width` evaluates.
_USABLE_GRID = 1601
#: Points of the in-band ``f_B`` scan behind the ridge.
_RIDGE_GRID = 401


@dataclass(frozen=True)
class EfficiencyModel:
    """Parameters of the TPM Bragg model (module docstring).

    Attributes
    ----------
    interaction_length:
        Acousto-optic interaction length ``L`` [m], ``> 0``.
    f_tpm:
        Tangential-phase-matching frequency [Hz] (absolute RF), ``> 0``.  Build with
        :func:`default_model` to default it to the hardware ``f_center``.
    refractive_index:
        ``n``, ``> 1``.
    gamma_peak:
        Coupling strength ``Gamma = asin(sqrt(eta_peak))`` [rad], in ``(0, pi/2]``.
    angular_coupling:
        Signed fraction ``gamma`` in ``[-1, 1]`` of the first AOD's deflection that reaches the
        second AOD's Bragg plane.
    """

    interaction_length: float
    f_tpm: float
    refractive_index: float
    gamma_peak: float
    angular_coupling: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.interaction_length) and self.interaction_length > 0.0):
            raise ValueError(
                f"interaction_length must be positive, got {self.interaction_length!r}"
            )
        if not (math.isfinite(self.f_tpm) and self.f_tpm > 0.0):
            raise ValueError(f"f_tpm must be positive, got {self.f_tpm!r}")
        if not (math.isfinite(self.refractive_index) and self.refractive_index > 1.0):
            raise ValueError(f"refractive_index must exceed 1, got {self.refractive_index!r}")
        if not (0.0 < self.gamma_peak <= 0.5 * math.pi):
            raise ValueError(f"gamma_peak must lie in (0, pi/2], got {self.gamma_peak!r}")
        if not (-1.0 <= self.angular_coupling <= 1.0):
            raise ValueError(f"angular_coupling must lie in [-1, 1], got {self.angular_coupling!r}")

    @property
    def peak_efficiency(self) -> float:
        """``eta_peak = sin^2 Gamma``."""
        return math.sin(self.gamma_peak) ** 2


def default_model(
    hw: AODLParams,
    interaction_length: float = DEFAULT_INTERACTION_LENGTH,
    f_tpm: float | None = None,
    refractive_index: float = DEFAULT_REFRACTIVE_INDEX,
    peak_efficiency: float = DEFAULT_PEAK_EFFICIENCY,
    angular_coupling: float = DEFAULT_ANGULAR_COUPLING,
) -> EfficiencyModel:
    """Build an :class:`EfficiencyModel`; ``f_tpm = None`` means the hardware ``f_center``.

    ``peak_efficiency`` in ``(0, 1]`` sets ``Gamma = asin(sqrt(eta_peak))``.
    """
    if not (0.0 < peak_efficiency <= 1.0):
        raise ValueError(f"peak_efficiency must lie in (0, 1], got {peak_efficiency!r}")
    return EfficiencyModel(
        interaction_length=float(interaction_length),
        f_tpm=float(_f_center(hw) if f_tpm is None else f_tpm),
        refractive_index=float(refractive_index),
        gamma_peak=math.asin(math.sqrt(peak_efficiency)),
        angular_coupling=float(angular_coupling),
    )


def _f_center(hw: AODLParams) -> float:
    """The (common) channel carrier [Hz]; the web hardware spec drives all four alike."""
    return float(hw.channels[CHANNELS[0]].f_center)


def _band(hw: AODLParams) -> tuple[float, float]:
    """The (common) usable absolute band [Hz]."""
    lo, hi = hw.channels[CHANNELS[0]].band
    return float(lo), float(hi)


def mismatch_scale(model: EfficiencyModel, hw: AODLParams) -> float:
    """``C2 = pi lambda L / (2 n v^2)`` [rad/Hz^2] — the TPM curvature of the module docstring."""
    v = hw.sound_speed
    return (
        math.pi
        * hw.optics.wavelength
        * model.interaction_length
        / (2.0 * model.refractive_index * v * v)
    )


# ------------------------------------------------------------------------------ the laws


def bragg_efficiency(gamma: ArrayLike, delta: ArrayLike) -> Float:
    """Kogelnik efficiency ``Gamma^2 sin^2(sqrt(Gamma^2 + delta^2)) / (Gamma^2 + delta^2)``.

    ``gamma`` is the coupling strength ``Gamma`` [rad] and ``delta`` the mismatch parameter
    ``Delta k L / 2`` [rad]; both broadcast.  Even in ``delta``; equals ``sin^2 Gamma`` at
    ``delta = 0``.
    """
    g = np.asarray(gamma, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    s2 = g * g + d * d
    root = np.sqrt(s2)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(s2 > 0.0, g * g * np.sin(root) ** 2 / np.where(s2 > 0.0, s2, 1.0), 0.0)
    return np.asarray(out, dtype=np.float64)


def tpm_mismatch(
    f: ArrayLike, f_tpm: float, L: float, n: float, v: float, wavelength: float
) -> Float:
    """``delta(f) = pi lambda L (f - f_tpm)^2 / (2 n v^2)`` [rad] — quadratic TPM detuning."""
    df = np.asarray(f, dtype=np.float64) - float(f_tpm)
    return np.asarray(math.pi * wavelength * L / (2.0 * n * v * v) * df * df, dtype=np.float64)


def deflection_angle(f: ArrayLike, f_center: float, v: float, wavelength: float) -> Float:
    """External deflection angle ``lambda (f - f_center) / v`` [rad] of a tone at RF ``f``."""
    return np.asarray(
        wavelength * (np.asarray(f, dtype=np.float64) - float(f_center)) / v, dtype=np.float64
    )


def first_aod_efficiency(f: ArrayLike, model: EfficiencyModel, hw: AODLParams) -> Float:
    """``eta_A(f)`` of a first (A) AOD at absolute RF ``f`` [Hz]: TPM mismatch into Kogelnik."""
    delta = tpm_mismatch(
        f,
        model.f_tpm,
        model.interaction_length,
        model.refractive_index,
        hw.sound_speed,
        hw.optics.wavelength,
    )
    return bragg_efficiency(model.gamma_peak, delta)


def second_aod_mismatch(
    f_B: ArrayLike, f_A: ArrayLike, model: EfficiencyModel, hw: AODLParams
) -> Float:
    """``delta_B = C2 [(f_B - f_tpm)^2 - 2 gamma f_tpm (f_A - f_tpm)]`` [rad] (module docstring).

    ``f_B``, ``f_A`` are absolute RF frequencies [Hz] and broadcast against each other.  The
    first term is B's own TPM detuning; the second is the *additive* incidence mismatch the A
    crystal's deflection imposes — linear in ``f_A``, hence the asymmetric map.
    """
    fb = np.asarray(f_B, dtype=np.float64) - model.f_tpm
    fa = np.asarray(f_A, dtype=np.float64) - model.f_tpm
    c2 = mismatch_scale(model, hw)
    return np.asarray(c2 * (fb * fb - 2.0 * model.angular_coupling * model.f_tpm * fa))


def second_aod_efficiency(
    f_B: ArrayLike, f_A: ArrayLike, model: EfficiencyModel, hw: AODLParams
) -> Float:
    """``eta_B(f_B; f_A)`` of the second AOD of a pair, illuminated by A's deflected beam."""
    return bragg_efficiency(model.gamma_peak, second_aod_mismatch(f_B, f_A, model, hw))


def pair_map(f_A: ArrayLike, f_B: ArrayLike, model: EfficiencyModel, hw: AODLParams) -> Float:
    """``eta_pair(f_A, f_B) = eta_A(f_A) eta_B(f_B; f_A)``, shape ``(len(f_A), len(f_B))``."""
    fa = np.asarray(f_A, dtype=np.float64).ravel()
    fb = np.asarray(f_B, dtype=np.float64).ravel()
    eta_a = first_aod_efficiency(fa, model, hw)
    eta_b = second_aod_efficiency(fb[None, :], fa[:, None], model, hw)
    return np.asarray(eta_a[:, None] * eta_b, dtype=np.float64)


# --------------------------------------------------------------------------- bandwidths


def half_max_mismatch(model: EfficiencyModel) -> float:
    """``delta_3`` [rad] with ``eta(Gamma, delta_3) = eta(Gamma, 0) / 2`` (numeric, exact root)."""
    g = model.gamma_peak
    target = 0.5 * float(bragg_efficiency(g, 0.0))

    def residual(delta: float) -> float:
        return float(bragg_efficiency(g, delta)) - target

    # eta is monotonically decreasing from delta = 0 to its first zero at
    # sqrt(Gamma^2 + delta^2) = pi, so the half-maximum root is bracketed there.
    upper = math.sqrt(math.pi**2 - g * g)
    return float(brentq(residual, 0.0, upper, xtol=1e-12, rtol=1e-12, maxiter=200))


def bandwidth_3db(model: EfficiencyModel, hw: AODLParams) -> float:
    """Full 3 dB width [Hz] of ``eta_A(f)``: ``2 sqrt(delta_3 / C2)`` — scales as ``L^-1/2``."""
    return 2.0 * math.sqrt(half_max_mismatch(model) / mismatch_scale(model, hw))


def ridge(
    f_A: ArrayLike, model: EfficiencyModel, hw: AODLParams, band: tuple[float, float]
) -> tuple[Float, Float]:
    """Per ``f_A``: ``max_{f_B in band} eta_pair`` and the ``f_B`` where it is reached [Hz].

    The ``f_B`` scan is a uniform in-band grid of :data:`_RIDGE_GRID` points.
    """
    fa = np.asarray(f_A, dtype=np.float64).ravel()
    fb = np.linspace(float(band[0]), float(band[1]), _RIDGE_GRID)
    eta = pair_map(fa, fb, model, hw)
    idx = np.argmax(eta, axis=1)
    return eta[np.arange(fa.size), idx], fb[idx]


def _half_max_extent(x: Float, y: Float) -> tuple[float, float, bool]:
    """``(left, right, bounded)`` of the contiguous region about the maximum with ``y >= max/2``.

    Crossings are linearly interpolated; ``bounded`` is False when the region touches an end
    of ``x`` (the caller then widens the grid).
    """
    i_max = int(np.argmax(y))
    half = 0.5 * float(y[i_max])
    left_i = i_max
    while left_i > 0 and y[left_i - 1] >= half:
        left_i -= 1
    right_i = i_max
    while right_i < y.size - 1 and y[right_i + 1] >= half:
        right_i += 1
    bounded = left_i > 0 and right_i < y.size - 1
    left = float(x[left_i])
    right = float(x[right_i])
    if left_i > 0:
        y0, y1 = float(y[left_i - 1]), float(y[left_i])
        left = float(x[left_i - 1] + (half - y0) / (y1 - y0) * (x[left_i] - x[left_i - 1]))
    if right_i < y.size - 1:
        y0, y1 = float(y[right_i]), float(y[right_i + 1])
        right = float(x[right_i] + (y0 - half) / (y0 - y1) * (x[right_i + 1] - x[right_i]))
    return left, right, bounded


def first_aod_usable_width(
    model: EfficiencyModel, hw: AODLParams, band: tuple[float, float]
) -> float:
    """Full width [Hz] in ``f_A`` over which ``max_{f_B in band} eta_pair >= 1/2 its maximum``.

    Computed numerically from the map (:func:`ridge` on a :data:`_USABLE_GRID`-point ``f_A``
    grid that is widened until the half-maximum region is bracketed).  For reference, the
    analytic cut-off-side half width is ``delta_3 / (2 C2 f_tpm |gamma|)`` and the ridge side
    is wider wherever the parabolic ridge stays inside the band.
    """
    span = max(bandwidth_3db(model, hw), float(band[1] - band[0]))
    for _ in range(8):
        fa = np.linspace(model.f_tpm - span, model.f_tpm + span, _USABLE_GRID)
        eta_max, _ = ridge(fa, model, hw, band)
        left, right, bounded = _half_max_extent(fa, eta_max)
        if bounded:
            return right - left
        span *= 2.0
    return right - left


# --------------------------------------------------------------------- along a drive


def _live_table(cw: ChannelWaveform, t_c: Float) -> tuple[Float, Float]:
    """``(f, A^2)`` of every tone at the drive times ``t_c``, shape ``(n_tones, n_t)``."""
    f = np.empty((cw.n_tones, t_c.size), dtype=np.float64)
    w = np.empty((cw.n_tones, t_c.size), dtype=np.float64)
    for i, tone in enumerate(cw.tones):
        f[i] = np.asarray(tone.f(t_c), dtype=np.float64)
        w[i] = np.asarray(tone.env.A(t_c), dtype=np.float64) ** 2
    return f, w


def _weighted_mean(values: Float, weights: Float) -> tuple[Float, Float]:
    """Amplitude^2-weighted mean over tones, and the total weight (zero where no tone is live)."""
    total = weights.sum(axis=0)
    safe = np.where(total > 0.0, total, 1.0)
    mean = np.where(total > 0.0, (values * weights).sum(axis=0) / safe, 0.0)
    return np.asarray(mean, dtype=np.float64), np.asarray(total, dtype=np.float64)


def trajectory_efficiency(
    wfs: WaveformSet, model: EfficiencyModel, t: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-channel efficiency along a drive at observation times ``t`` [s], plus ``"product"``.

    For channel ``mu`` at frame time ``t`` the drive seen at the beam centre is the one at the
    retarded time ``t_c = t - tau/2`` (``conventions.beam_center_time``), and the channel's
    efficiency is the amplitude^2-weighted mean over its *live* tones

        eta_mu(t) = sum_n A_n^2 eta(f_center + f_n(t_c)) / sum_n A_n^2

    with ``eta = eta_A`` on the A channels and ``eta = eta_B(.; f_A)`` on the B channels, where
    ``f_A`` is the A channel's own amplitude^2-weighted live frequency at ``t_c`` (exact for
    Eq. S19, the ladder-weighted closure for a Shepard drive).  A channel with no live tone
    transmits nothing and reports ``0``.  ``"product"`` multiplies the four channels: the
    predicted relative trap power.  Undriven channels are left out of the product.
    """
    from aodl.device.conventions import beam_center_time

    times = np.atleast_1d(np.asarray(t, dtype=np.float64)).ravel()
    out: dict[str, np.ndarray] = {}
    f_a_eff: dict[str, Float] = {}
    for name in CHANNELS:
        cw = wfs.channels.get(name)
        if cw is None or not name.startswith("A"):
            continue
        aod = wfs.params.channels[name]
        t_c = np.asarray([beam_center_time(float(ti), aod) for ti in times])
        f, w = _live_table(cw, t_c)
        eta = first_aod_efficiency(aod.f_center + f, model, wfs.params)
        out[name], _ = _weighted_mean(eta, w)
        f_a_eff[name[-1]], _ = _weighted_mean(aod.f_center + f, w)
    for name in CHANNELS:
        cw = wfs.channels.get(name)
        if cw is None or not name.startswith("B"):
            continue
        aod = wfs.params.channels[name]
        t_c = np.asarray([beam_center_time(float(ti), aod) for ti in times])
        f, w = _live_table(cw, t_c)
        axis = name[-1]
        f_a = f_a_eff.get(axis)
        if f_a is None:  # no A channel on this axis: B is illuminated on-axis at TPM
            f_a = np.full(times.shape, model.f_tpm)
        eta = second_aod_efficiency(aod.f_center + f, f_a[None, :], model, wfs.params)
        out[name], _ = _weighted_mean(eta, w)
    product = np.ones(times.shape, dtype=np.float64)
    for name in CHANNELS:
        if name in out:
            product = product * out[name]
    ordered: dict[str, np.ndarray] = {name: out[name] for name in CHANNELS if name in out}
    ordered["product"] = product
    return ordered


def channel_relative_amplitude(
    wfs: WaveformSet, model: EfficiencyModel, name: str, t_c: float
) -> Any:
    """``a(f_abs) = sqrt(eta(f_abs) / eta_peak)`` for channel ``name`` at drive time ``t_c``.

    Returns a callable on absolute RF frequency [Hz] (the ``rel_amp`` of
    :func:`aodl_advanced.pupil.complex_drive`).  For a B channel the efficiency is closed over
    the A channel's live frequency at ``t_c`` (amplitude^2-weighted over a Shepard A ladder),
    as :func:`trajectory_efficiency` does.
    """
    peak = model.peak_efficiency
    if name.startswith("A"):

        def a_first(f_abs: ArrayLike) -> Float:
            return np.sqrt(first_aod_efficiency(f_abs, model, wfs.params) / peak)

        return a_first

    partner = "A" + name[-1]
    cw_a = wfs.channels.get(partner)
    f_a = model.f_tpm
    if cw_a is not None:
        aod_a = wfs.params.channels[partner]
        f, w = _live_table(cw_a, np.asarray([t_c], dtype=np.float64))
        mean, total = _weighted_mean(aod_a.f_center + f, w)
        if total[0] > 0.0:
            f_a = float(mean[0])

    def a_second(f_abs: ArrayLike) -> Float:
        return np.sqrt(second_aod_efficiency(f_abs, f_a, model, wfs.params) / peak)

    return a_second


def assumptions_text(model: EfficiencyModel, hw: AODLParams) -> str:
    """One line for the UI: what the model is and what it leaves out."""
    return (
        "Kogelnik plane-wave Bragg efficiency with a quadratic TPM mismatch "
        f"delta = C2 (f - f_tpm)^2 (L = {model.interaction_length * 1e3:.3g} mm, n = "
        f"{model.refractive_index:.3g}, eta_peak = {model.peak_efficiency:.3g}, f_tpm = "
        f"{model.f_tpm / MHz:.4g} MHz); the second AOD sees the first one's deflection as an "
        f"additive mismatch with angular coupling gamma = {model.angular_coupling:+.3g}. "
        "No acoustic walk-off/divergence, no optical activity, no mid-band degeneracy dip; a "
        "model with user-set parameters, not a measurement."
    )


__all__ = [
    "DEFAULT_ANGULAR_COUPLING",
    "DEFAULT_INTERACTION_LENGTH",
    "DEFAULT_PEAK_EFFICIENCY",
    "DEFAULT_REFRACTIVE_INDEX",
    "EfficiencyModel",
    "assumptions_text",
    "bandwidth_3db",
    "bragg_efficiency",
    "channel_relative_amplitude",
    "default_model",
    "deflection_angle",
    "first_aod_efficiency",
    "first_aod_usable_width",
    "half_max_mismatch",
    "mismatch_scale",
    "pair_map",
    "ridge",
    "second_aod_efficiency",
    "second_aod_mismatch",
    "tpm_mismatch",
    "trajectory_efficiency",
]
