"""Pins ``device/conventions.py`` — the one place in the package that owns a sign.

Two things are checked here: that the geometry table is Eq. S7 verbatim, and that the
per-channel Taylor coefficients this module hands out reproduce the paper's Table I
control mapping when pushed through the Eq. S11 geometry that WO-01's
``tests/test_focal_geometry.py`` pinned (``X = theta1 F / k``, ``Z_S11 = 2 F^2 theta2 / k``).
Everything downstream reads these signs instead of re-deriving them, so if this file is
green the rest of the package cannot silently flip an axis.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aodl.device.conventions import (
    AXIS_NAMES,
    AXIS_X,
    AXIS_Y,
    CHANNEL_GEOMETRY,
    DIFFRACTION_ORDER,
    N_AXES,
    Z_LAB_SIGN,
    ChannelGeometry,
    amplitude_poly,
    beam_center_time,
    filled_side,
    geometry,
    is_filled,
    retarded_time,
    theta1_contribution,
    theta2_contribution,
    z_lab_from_s11,
    z_s11_from_lab,
)
from aodl.params import CHANNELS
from aodl.units import MHz, mm, ms, us

#: Eq. S7 / ``docs/PLAN.md`` §1.1, written out again so the test does not read the table
#: it is supposed to be checking.
EQ_S7 = {"Ax": (0, -1), "Bx": (0, +1), "Ay": (1, -1), "By": (1, +1)}


def _axis_z_lab(optics, theta2: float) -> float:
    """Lab focus of one axis from its pupil curvature (the composition under test)."""
    z_s11 = 2.0 * optics.focal_length**2 * theta2 / optics.k
    return float(z_lab_from_s11(z_s11))


# --------------------------------------------------------------------- the table


def test_channel_geometry_is_eq_s7() -> None:
    """The four channels, their axes and their sound directions — exactly Eq. S7."""
    assert set(CHANNEL_GEOMETRY) == set(CHANNELS)
    for name, (axis, sound_sign) in EQ_S7.items():
        geom = CHANNEL_GEOMETRY[name]
        assert (geom.axis, geom.sound_sign) == (axis, sound_sign), name
    # x channels on axis 0, y channels on axis 1; A counter-propagates against B.
    assert {n for n, g in CHANNEL_GEOMETRY.items() if g.axis == AXIS_X} == {"Ax", "Bx"}
    assert {n for n, g in CHANNEL_GEOMETRY.items() if g.axis == AXIS_Y} == {"Ay", "By"}
    for a, b in (("Ax", "Bx"), ("Ay", "By")):
        assert CHANNEL_GEOMETRY[a].sound_sign == -CHANNEL_GEOMETRY[b].sound_sign
    assert AXIS_NAMES == ("x", "y")
    assert N_AXES == 2
    assert CHANNEL_GEOMETRY["Ax"].axis_name == "x"
    assert CHANNEL_GEOMETRY["By"].axis_name == "y"


def test_diffraction_order_and_axial_sign_are_the_documented_values() -> None:
    """``+1`` order on every channel; ``Z_LAB_SIGN`` is ``-1`` (see docs/conventions.md)."""
    assert DIFFRACTION_ORDER == +1
    assert Z_LAB_SIGN == -1


def test_geometry_lookup_rejects_unknown_channels() -> None:
    assert geometry("By") is CHANNEL_GEOMETRY["By"]
    with pytest.raises(KeyError, match="unknown channel"):
        geometry("Cz")
    with pytest.raises(ValueError):
        ChannelGeometry(axis=2, sound_sign=+1)
    with pytest.raises(ValueError):
        ChannelGeometry(axis=0, sound_sign=0)


def test_transducer_sits_opposite_the_sound_direction() -> None:
    """Sound travelling toward ``+axis`` is launched from ``u = -D/2`` and vice versa."""
    for geom in CHANNEL_GEOMETRY.values():
        assert geom.transducer_u == pytest.approx(-0.5 * geom.sound_sign)


# ------------------------------------------------------- Taylor coefficients (Eq. S6)


def test_theta1_flips_with_sound_sign_and_theta2_does_not(params1030) -> None:
    """The reason counter-propagating pairs cancel deflection but add lensing."""
    v = params1030.sound_speed
    f, fdot = 4.0 * MHz, 30.0 * MHz / ms
    a = theta1_contribution(f, geometry("Ax"), v)
    b = theta1_contribution(f, geometry("Bx"), v)
    assert float(a) == pytest.approx(-2.0 * math.pi * f / v, rel=1e-14)
    assert float(b) == pytest.approx(+2.0 * math.pi * f / v, rel=1e-14)
    assert float(a + b) == pytest.approx(0.0, abs=1e-9)

    qa = theta2_contribution(fdot, v)
    qb = theta2_contribution(fdot, v)
    assert float(qa) == pytest.approx(-math.pi * fdot / v**2, rel=1e-14)
    assert float(qa) == float(qb)  # independent of sound_sign by construction


def test_table_i_lateral_mapping(params1030) -> None:
    """``X = theta1_x F / k`` reproduces Table I's ``X = (lambda F / v)(f_Bx - f_Ax)``."""
    optics = params1030.optics
    v = params1030.sound_speed
    f_ax, f_bx = -2.0 * MHz, 3.5 * MHz
    theta1 = float(
        theta1_contribution(f_ax, geometry("Ax"), v) + theta1_contribution(f_bx, geometry("Bx"), v)
    )
    x_spot = theta1 * optics.focal_length / optics.k
    assert x_spot == pytest.approx(params1030.deflection_scale * (f_bx - f_ax), rel=1e-12)
    # The handy scale of docs/PLAN.md §1.5: 1 MHz of difference <-> 10.3 um.
    assert x_spot == pytest.approx(5.5 * 10.3e-6, rel=5e-3)


def test_table_i_axial_mapping(params1030) -> None:
    """Per-axis foci from ``theta2`` give Table I's ``Zbar`` and astigmatic interval."""
    optics = params1030.optics
    v = params1030.sound_speed
    rates = {
        "Ax": 10.0 * MHz / ms,
        "Bx": -4.0 * MHz / ms,
        "Ay": 25.0 * MHz / ms,
        "By": 7.0 * MHz / ms,
    }
    theta2 = [0.0, 0.0]
    for name, rate in rates.items():
        theta2[geometry(name).axis] += float(theta2_contribution(rate, v))

    z_x = _axis_z_lab(optics, theta2[0])
    z_y = _axis_z_lab(optics, theta2[1])
    lens = params1030.lens_scale
    # Each axis focuses at + lens_scale * (sum of its channels' chirp rates).
    assert z_x == pytest.approx(lens * (rates["Ax"] + rates["Bx"]), rel=1e-12)
    assert z_y == pytest.approx(lens * (rates["Ay"] + rates["By"]), rel=1e-12)
    # Table I: Zbar = 1/2 lens_scale sum(all), Delta F = lens_scale (x-sum - y-sum).
    assert 0.5 * (z_x + z_y) == pytest.approx(0.5 * lens * sum(rates.values()), rel=1e-12)
    assert z_x - z_y == pytest.approx(
        lens * (rates["Ax"] + rates["Bx"] - rates["Ay"] - rates["By"]), rel=1e-12
    )


def test_single_channel_chirp_reproduces_the_table_i_combination(params1030) -> None:
    """Ay alone: y focuses at ``+lens_scale fdot``, x stays put — i.e. Zbar + DeltaF/2 = 0."""
    optics = params1030.optics
    v = params1030.sound_speed
    lens = params1030.lens_scale
    fdot = 50.0 * MHz / ms

    z_y = _axis_z_lab(optics, float(theta2_contribution(fdot, v)))
    z_x = _axis_z_lab(optics, 0.0)
    assert z_y == pytest.approx(lens * fdot, rel=1e-12)
    assert z_x == 0.0
    # Table I with only fdot_Ay nonzero: Zbar = lens/2 * fdot, DeltaF = -lens * fdot,
    # so Z_x = Zbar + DeltaF/2 = 0 and Z_y = Zbar - DeltaF/2 = lens * fdot.
    z_bar, delta_f = 0.5 * lens * fdot, -lens * fdot
    assert z_bar + 0.5 * delta_f == pytest.approx(z_x, abs=1e-18)
    assert z_bar - 0.5 * delta_f == pytest.approx(z_y, rel=1e-12)


def test_z_lab_and_z_s11_are_inverse_and_opposite() -> None:
    z_lab = np.array([-3.0e-6, 0.0, 7.5e-6])
    assert np.allclose(z_s11_from_lab(z_lab), -z_lab)
    assert np.allclose(z_lab_from_s11(z_s11_from_lab(z_lab)), z_lab)


def test_amplitude_poly_signs(params1030) -> None:
    """Eq. S5: the tilt term carries the sound sign, the irising term does not."""
    v = params1030.sound_speed
    amp, d_amp, d2_amp = 0.7, 1.3e5, -2.5e10
    for name in CHANNELS:
        geom = geometry(name)
        a0, a1, a2 = amplitude_poly(amp, d_amp, d2_amp, geom, v)
        assert float(a0) == pytest.approx(amp)
        assert float(a1) == pytest.approx(-geom.sound_sign * d_amp / v, rel=1e-14)
        assert float(a2) == pytest.approx(d2_amp / (2.0 * v**2), rel=1e-14)


# ------------------------------------------------------------- acoustic timing / fill


def test_beam_center_time_is_half_a_transit_back(params1030) -> None:
    aod = params1030.channels["Ay"]
    tau = aod.transit_time
    assert tau == pytest.approx(11.54 * us, rel=1e-3)
    assert beam_center_time(3.0 * tau, aod) == pytest.approx(2.5 * tau, rel=1e-14)
    # The beam center is where the retarded time equals t_c, for either sound direction.
    for name in ("Ay", "By"):
        geom = geometry(name)
        assert float(retarded_time(3.0 * tau, 0.0, geom, aod)) == pytest.approx(2.5 * tau)
        # One aperture-half of travel is half a transit time, with the sign of the sound.
        u = 0.5 * aod.aperture
        got = float(retarded_time(3.0 * tau, u, geom, aod))
        assert got == pytest.approx(2.5 * tau - geom.sound_sign * 0.5 * tau, rel=1e-14)


def test_fill_starts_at_the_transducer_and_completes_after_one_transit(params1030) -> None:
    """Content exists where ``s u <= v t - D/2``: the transducer side fills first."""
    aod = params1030.channels["Ax"]
    tau, half = aod.transit_time, 0.5 * aod.aperture
    u = np.linspace(-half, half, 401)
    for name in ("Ax", "Bx"):
        geom = geometry(name)
        transducer = -geom.sound_sign * half
        # Nothing before the drive starts; everything after one transit time.
        assert not is_filled(u, -1e-9, geom, aod).any()
        assert is_filled(u, tau * (1.0 + 1e-9), geom, aod).all()
        # Half way through, exactly the transducer half is filled.
        mask = is_filled(u, 0.5 * tau, geom, aod)
        assert mask[np.argmin(np.abs(u - transducer))]
        assert not mask[np.argmin(np.abs(u + transducer))]
        assert mask.mean() == pytest.approx(0.5, abs=0.01)
        # The filled region is a half-line on the side named by filled_side().
        side = filled_side(geom)
        assert side == ("upper" if geom.sound_sign > 0 else "lower")
        boundary = u[mask].max() if side == "upper" else u[mask].min()
        assert boundary == pytest.approx(0.0, abs=1.1 * (u[1] - u[0]))
    # Retarded time and the fill mask are the same statement.
    geom = geometry("Ay")
    aod_y = params1030.channels["Ay"]
    t = 0.37 * tau
    assert np.array_equal(retarded_time(t, u, geom, aod_y) >= 0.0, is_filled(u, t, geom, aod_y))


def test_fill_edge_hand_computation(params1030) -> None:
    """At ``t = 0.6 tau`` the wavefront sits ``0.1 D`` past the aperture center."""
    aod = params1030.channels["Ay"]
    tau, aperture = aod.transit_time, aod.aperture
    reach = aod.sound_speed * 0.6 * tau - 0.5 * aperture
    assert reach == pytest.approx(0.1 * aperture, rel=1e-12)
    assert reach == pytest.approx(0.75 * mm, rel=1e-12)
    # Ay sends sound toward -y, so the wavefront is at u = -0.1 D and everything above it
    # (toward the transducer at u = +D/2) is filled.  (The boundary itself is a float-exact
    # tie; probe a nanometre either side of it.)
    assert bool(is_filled(-0.1 * aperture + 1e-9, 0.6 * tau, geometry("Ay"), aod))
    assert not bool(is_filled(-0.1 * aperture - 1e-9, 0.6 * tau, geometry("Ay"), aod))


# ------------------------------------------------------------------------- the doc


def test_conventions_doc_states_the_axial_sign() -> None:
    """``docs/conventions.md`` carries the derivation, including ``Z_LAB_SIGN = -1``."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "conventions.md"
    text = doc.read_text(encoding="utf-8")
    for needle in ("Z_LAB_SIGN", "-1", "Eq. S7", "t_c = t", "sound_sign"):
        assert needle in text, needle
