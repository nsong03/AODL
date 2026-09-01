r"""**The** sign authority for the whole package (``docs/ARCHITECTURE.md`` §0.4).

Every orientation, sound-direction, diffraction-order and defocus sign used anywhere in
``aodl`` is defined here and nowhere else.  Downstream modules read the table and the
helper functions below; they never re-derive a sign.  The narrative derivation of
everything in this module lives in ``docs/conventions.md``.

Summary of the conventions (all of them pinned by ``tests/test_conventions.py``):

**Axes.**  ``axis = 0`` is x, ``axis = 1`` is y.  ``Ax``/``Bx`` deflect along x,
``Ay``/``By`` along y (Eq. S7).  The aperture coordinate ``u`` runs along the channel's own
axis with ``u = 0`` at the beam center.

**Sound direction.**  ``sound_sign = +1`` means the acoustic wave travels toward ``+axis``
(transducer at ``u = -D/2``), ``-1`` toward ``-axis`` (transducer at ``u = +D/2``).

**Retarded phase.**  A channel with sound sign ``s`` imprints the optical phase
``-Phi(t - s u / v) + const`` on the beam (Eq. S4 generalized), where
``Phi(t') = 2 pi f_center t' + phase(t')`` is the drive phase.  Taylor-expanding about the
beam center (Eqs. S5-S6) and dropping the carrier ``f_center`` (a common tilt that *defines*
the optical axis, plus a common constant phase) gives the per-channel pupil contributions

    theta1 += s * 2 pi f(t_c) / v          [rad/m]     -- deflection
    theta2 += -2 pi fdot(t_c) / (2 v^2)    [rad/m^2]   -- cylindrical chirp lens

``theta2`` does **not** depend on ``s``: that is exactly why a counter-propagating pair
cancels deflection while adding lensing (``docs/PLAN.md`` §1.2).  The amplitude envelope
expands to the aperture polynomial (Eq. S5)

    alpha(u) = A - s (A' / v) u + (A'' / (2 v^2)) u^2

whose 0th term is intensity control, 1st is tilt and 2nd is acoustic irising.

**Diffraction order.**  ``+1`` on every channel: the diffracted light is *up*-shifted by
``f_center + f`` (absolute).  The common ``f_center`` shift is global and ignored; the
per-term optical frequency tag is ``df_opt = sum f(t_c)`` over the participating channels.

**Defocus.**  ``Z_LAB_SIGN = -1``; see :data:`Z_LAB_SIGN`.

**Acoustic timing.**  The drive starts at ``t = 0``.  The sample emitted at drive time
``t'`` sits where ``t - (s u + D/2) / v = t'``, so the beam-center retarded time is
``t_c = t - tau/2`` with ``tau = D / v``, and the aperture holds drive content only where
``s u <= v t - D/2`` (fully filled once ``t >= tau``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..params import AODParams

#: Aperture-axis index of the x channels.
AXIS_X = 0
#: Aperture-axis index of the y channels.
AXIS_Y = 1
#: Number of transverse axes (x, y).
N_AXES = 2
#: Human-readable axis names, indexed by axis number.
AXIS_NAMES: tuple[str, str] = ("x", "y")

#: Which side of the fill edge holds acoustic content.  ``"lower"`` means the filled
#: region is ``u >= u_edge`` (use :func:`aodl.field.gaussian.gauss_moments_lower`),
#: ``"upper"`` means ``u <= u_edge`` (:func:`~aodl.field.gaussian.gauss_moments_upper`).
Side = Literal["lower", "upper"]


@dataclass(frozen=True)
class ChannelGeometry:
    """Orientation of one AOD channel (Eq. S7).

    Attributes
    ----------
    axis:
        ``0`` = x, ``1`` = y — the transverse axis this channel deflects along, and the
        axis its aperture coordinate ``u`` runs along.
    sound_sign:
        ``+1`` if the acoustic wave travels toward ``+axis`` (transducer at ``u = -D/2``),
        ``-1`` if toward ``-axis`` (transducer at ``u = +D/2``).
    """

    axis: int
    sound_sign: int

    def __post_init__(self) -> None:
        if self.axis not in (AXIS_X, AXIS_Y):
            raise ValueError(f"axis must be 0 (x) or 1 (y), got {self.axis!r}")
        if self.sound_sign not in (-1, +1):
            raise ValueError(f"sound_sign must be -1 or +1, got {self.sound_sign!r}")

    @property
    def axis_name(self) -> str:
        """``"x"`` or ``"y"``."""
        return AXIS_NAMES[self.axis]

    @property
    def transducer_u(self) -> float:
        """Transducer position in units of the aperture ``D``: ``-sound_sign / 2``."""
        return -0.5 * self.sound_sign


#: The four-AOD stack of ``docs/PLAN.md`` §1.1 (Eq. S7): ``Ax`` sends sound toward -x,
#: ``Bx`` toward +x, ``Ay`` toward -y, ``By`` toward +y.  The A/B pairs counter-propagate,
#: which is what makes ``X = deflection_scale * (f_Bx - f_Ax)`` (Table I) come out right.
CHANNEL_GEOMETRY: dict[str, ChannelGeometry] = {
    "Ax": ChannelGeometry(axis=AXIS_X, sound_sign=-1),
    "Bx": ChannelGeometry(axis=AXIS_X, sound_sign=+1),
    "Ay": ChannelGeometry(axis=AXIS_Y, sound_sign=-1),
    "By": ChannelGeometry(axis=AXIS_Y, sound_sign=+1),
}

#: Diffraction order used on every channel.  ``+1`` up-shifts the optical frequency by
#: ``f_center + f`` and imprints ``-Phi(t_ret)`` (see the module docstring).
DIFFRACTION_ORDER: int = +1

#: Sign relating the lab axial coordinate to the Eq. S11 defocus variable:
#: ``Z_lab = Z_LAB_SIGN * Z_S11``, equivalently ``Z_S11 = Z_LAB_SIGN * Z_lab``.
#:
#: Why it is ``-1``.  A pure Eq. S11 evaluation focuses sharpest at
#: ``Z_S11 = 2 F^2 theta2 / k`` (pinned by WO-01's ``tests/test_focal_geometry.py``), and
#: this module's ``theta2_axis = -pi * sum(fdot) / v^2``, so
#: ``Z_S11 = -lambda F^2 sum(fdot) / v^2 = -lens_scale * sum(fdot)``.  The paper's Table I
#: instead reports ``Zbar = +(1/2) (lambda F^2 / v^2) sum_all fdot`` — an up-chirp puts the
#: focus *above* the static focal plane.  We adopt **Table I's sign as the lab axis**, so
#: ``Z_axis_lab = +lens_scale * sum_axis(fdot)`` and every user-facing / trajectory Z is lab
#: Z, while ``field/`` evaluates Eq. S11 at ``Z_S11 = Z_LAB_SIGN * Z_lab``.
Z_LAB_SIGN: int = -1


class FillEdge(NamedTuple):
    """Leading edge of the acoustic column inside the aperture.

    Attributes
    ----------
    u_edge:
        Aperture coordinate [m] of the leading acoustic wavefront.
    side:
        ``"lower"`` — drive content occupies ``u >= u_edge``; ``"upper"`` — ``u <= u_edge``.
        The filled half-line always contains the transducer (``u = -sound_sign * D / 2``),
        because that is where the sound comes from.
    """

    u_edge: float
    side: Side


def geometry(channel: str) -> ChannelGeometry:
    """Look up a channel's orientation, with a helpful error for unknown names."""
    try:
        return CHANNEL_GEOMETRY[channel]
    except KeyError:
        raise KeyError(
            f"unknown channel {channel!r}; expected one of {tuple(CHANNEL_GEOMETRY)}"
        ) from None


def filled_side(geom: ChannelGeometry) -> Side:
    """Which side of the fill edge carries drive content.

    The filled region is ``s u <= v t - D/2`` (module docstring), i.e. ``u <= u_edge`` for
    ``s = +1`` (an *upper* edge) and ``u >= u_edge`` for ``s = -1`` (a *lower* edge).
    """
    return "upper" if geom.sound_sign > 0 else "lower"


def beam_center_time(t: float, aod: AODParams) -> float:
    """Retarded drive time seen at the beam center: ``t_c = t - tau / 2``.

    The transducer sits half an aperture away from the beam center, so the drive sample
    illuminating ``u = 0`` at frame time ``t`` was emitted ``tau / 2 = D / (2 v)`` earlier.
    """
    return float(t) - 0.5 * aod.transit_time


def retarded_time(
    t: float, u: ArrayLike, geom: ChannelGeometry, aod: AODParams
) -> NDArray[np.float64]:
    """Drive time whose sample sits at aperture coordinate ``u`` at frame time ``t``.

    ``t_ret(u) = t - (s u + D/2) / v = t_c - s u / v``.  Negative values mean the aperture
    is not yet filled at that ``u`` (the drive starts at ``t = 0``).
    """
    u_arr = np.asarray(u, dtype=np.float64)
    return beam_center_time(t, aod) - geom.sound_sign * u_arr / aod.sound_speed


def is_filled(u: ArrayLike, t: float, geom: ChannelGeometry, aod: AODParams) -> NDArray[np.bool_]:
    """Mask of aperture positions already carrying drive content: ``s u <= v t - D/2``."""
    u_arr = np.asarray(u, dtype=np.float64)
    reach = aod.sound_speed * float(t) - 0.5 * aod.aperture
    return np.asarray(geom.sound_sign * u_arr <= reach, dtype=np.bool_)


def theta1_contribution(
    f: ArrayLike, geom: ChannelGeometry, sound_speed: float
) -> NDArray[np.float64]:
    """Per-channel pupil tilt ``s * 2 pi f / v`` [rad/m] (Eq. S6).

    ``f`` is the rotating-frame detuning at the beam-center retarded time ``t_c``; the
    carrier ``f_center`` is dropped because its tilt defines the optical axis.  Summing
    this over a channel's axis and mapping through Eq. S11 (``X = theta1 F / k``) gives
    ``X = deflection_scale * sum(s f)`` — Table I's ``X = (lambda F / v)(f_Bx - f_Ax)``.
    """
    return geom.sound_sign * 2.0 * math.pi * np.asarray(f, dtype=np.float64) / sound_speed


def theta2_contribution(fdot: ArrayLike, sound_speed: float) -> NDArray[np.float64]:
    """Per-channel chirp-lens curvature ``-2 pi fdot / (2 v^2) = -pi fdot / v^2`` [rad/m^2].

    Independent of ``sound_sign`` (Eq. S6) — counter-propagating pairs cancel deflection
    but *add* lensing (``docs/PLAN.md`` §1.2).  Through Eq. S11 the axis focus lands at
    ``Z_S11 = 2 F^2 theta2 / k = -lens_scale * sum(fdot)``, i.e. at lab
    ``Z = +lens_scale * sum(fdot)`` (see :data:`Z_LAB_SIGN`).
    """
    return -math.pi * np.asarray(fdot, dtype=np.float64) / sound_speed**2


def amplitude_poly(
    amp: ArrayLike,
    d_amp: ArrayLike,
    d2_amp: ArrayLike,
    geom: ChannelGeometry,
    sound_speed: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Aperture amplitude polynomial ``(alpha0, alpha1, alpha2)`` of Eq. S5.

    ``alpha(u) = A - s (A' / v) u + (A'' / (2 v^2)) u^2`` — the envelope evaluated at the
    retarded time, Taylor-expanded about the beam center.  ``alpha1`` carries the sound
    sign (the envelope sweeps across the aperture in the sound direction); ``alpha2`` does
    not, and is the acoustic-irising term.

    ``device/aodl.py`` uses the *shape* of this polynomial only (normalized to
    ``alpha0 = 1``), because ``A(t_c)`` is already carried in the line amplitude.
    """
    a0 = np.asarray(amp, dtype=np.float64)
    a1 = -geom.sound_sign * np.asarray(d_amp, dtype=np.float64) / sound_speed
    a2 = np.asarray(d2_amp, dtype=np.float64) / (2.0 * sound_speed**2)
    return a0, a1, a2


def z_s11_from_lab(z_lab: ArrayLike) -> NDArray[np.float64]:
    """Lab axial coordinate -> Eq. S11 defocus variable: ``Z_S11 = Z_LAB_SIGN * Z_lab``."""
    return Z_LAB_SIGN * np.asarray(z_lab, dtype=np.float64)


def z_lab_from_s11(z_s11: ArrayLike) -> NDArray[np.float64]:
    """Eq. S11 defocus variable -> lab axial coordinate: ``Z_lab = Z_LAB_SIGN * Z_S11``."""
    return Z_LAB_SIGN * np.asarray(z_s11, dtype=np.float64)


__all__ = [
    "AXIS_NAMES",
    "AXIS_X",
    "AXIS_Y",
    "CHANNEL_GEOMETRY",
    "DIFFRACTION_ORDER",
    "N_AXES",
    "Z_LAB_SIGN",
    "ChannelGeometry",
    "FillEdge",
    "Side",
    "amplitude_poly",
    "beam_center_time",
    "filled_side",
    "geometry",
    "is_filled",
    "retarded_time",
    "theta1_contribution",
    "theta2_contribution",
    "z_lab_from_s11",
    "z_s11_from_lab",
]
