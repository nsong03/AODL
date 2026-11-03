"""The zoom transform and the Gaussian-beam geometry it must reproduce (W2 §5)."""

from __future__ import annotations

import numpy as np
import pytest
from _advanced_fixtures import LENS, PARAMS, TAU, W0, Z_R, lift_hold, run_frame

from aodl.units import um
from aodl_advanced.engine import XzSpec
from aodl_advanced.metrics import best_focus, moments_1d, width_scan
from aodl_advanced.propagate import defocus_phase, focal_intensity, xz_slice, zoom_dft
from aodl_advanced.pupil import pupil_grid

OPTICS = PARAMS.optics
K_OVER_F = OPTICS.k / OPTICS.focal_length


def test_zoom_dft_matches_direct_dft() -> None:
    rng = np.random.default_rng(7)
    p = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    dx, dy = 12e-6, 9e-6
    xs = (np.arange(32) - 15.5) * dx
    ys = (np.arange(32) - 15.5) * dy
    x_out = np.linspace(-3e-6, 4e-6, 7)
    y_out = np.linspace(-2e-6, 5e-6, 5)
    direct = (
        np.einsum(
            "mn,jn,im->ij",
            p,
            np.exp(-1j * K_OVER_F * np.outer(x_out, xs)),
            np.exp(-1j * K_OVER_F * np.outer(y_out, ys)),
        )
        * dx
        * dy
    )
    mine = zoom_dft(p, dx, dy, x_out, y_out, K_OVER_F)
    assert mine.shape == (5, 7)
    assert np.max(np.abs(mine - direct)) / np.max(np.abs(direct)) < 1e-10
    # a single output point takes the direct branch
    single = zoom_dft(p, dx, dy, x_out[:1], y_out[:1], K_OVER_F)
    assert single[0, 0] == pytest.approx(direct[0, 0], rel=1e-10)


def test_gaussian_pupil_gives_the_waist_and_rayleigh_law() -> None:
    x, y = pupil_grid(OPTICS.w_in, 256, 2.5)
    pupil = np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / OPTICS.w_in**2).astype(complex)
    out = np.linspace(-6 * um, 6 * um, 121)
    img = focal_intensity(pupil, x, y, out, out, OPTICS.k, OPTICS.focal_length, 0.0)
    _, wx = moments_1d(img.sum(axis=0), out)
    _, wy = moments_1d(img.sum(axis=1), out)
    assert wx == pytest.approx(W0, rel=5e-3)
    assert wy == pytest.approx(W0, rel=5e-3)
    img_r = focal_intensity(pupil, x, y, out, out, OPTICS.k, OPTICS.focal_length, Z_R)
    _, wx_r = moments_1d(img_r.sum(axis=0), out)
    assert wx_r == pytest.approx(np.sqrt(2.0) * W0, rel=0.01)
    # and the Rayleigh law holds on the other side too
    img_m = focal_intensity(pupil, x, y, out, out, OPTICS.k, OPTICS.focal_length, -Z_R)
    assert moments_1d(img_m.sum(axis=0), out)[1] == pytest.approx(wx_r, rel=1e-6)


def test_defocus_sign_through_conventions_only() -> None:
    """A pure co-chirp on all four channels focuses at Table I's ``Z = 2 lens_scale fdot``."""
    plan = lift_hold()
    wfs = plan.wfs
    t = 60e-6 + 40e-6 + 0.5 * TAU  # mid-hold: Z = 10 um sustained
    fdot = float(wfs.channels["Ax"].tones[0].fdot(t - 0.5 * TAU))
    z_table = 2.0 * LENS * fdot
    assert z_table == pytest.approx(10 * um, rel=1e-6)
    # scan the static focal plane +- 12 um: both signs of Z are inside, so a wrong defocus sign
    # would put the minimum at -10 um, not at Table I's +10 um
    frame = run_frame(wfs, t, plane=0.0, order_model="weak", xz=XzSpec(32, 12 * um))
    assert frame.metrics.z_best_x is not None and frame.metrics.z_best_y is not None
    assert abs(frame.metrics.z_best_x - z_table) < 0.1 * Z_R
    assert abs(frame.metrics.z_best_y - z_table) < 0.1 * Z_R
    assert frame.metrics.astig == pytest.approx(0.0, abs=0.02 * Z_R)


def test_xz_slice_focus_follows_defocus_phase() -> None:
    x, y = pupil_grid(OPTICS.w_in, 128, 2.5)
    gauss = np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / OPTICS.w_in**2)
    # pre-curve the pupil so that it focuses at lab Z = +2 um: the propagation's own defocus
    # factor at that plane must cancel the curvature exactly
    z0 = 2 * um
    pupil = gauss * np.conj(defocus_phase(x, y, z0, OPTICS.k, OPTICS.focal_length))
    xs = np.linspace(-5 * um, 5 * um, 81)
    zs = np.linspace(-6 * um, 6 * um, 25)
    img = xz_slice(pupil, x, y, xs, zs, 0.0, OPTICS.k, OPTICS.focal_length)
    assert img.shape == (25, 81)
    z_best = best_focus(zs, width_scan(img, xs))
    assert z_best == pytest.approx(z0, abs=0.05 * Z_R)


def test_best_focus_parabolic_refinement_is_exact_for_a_gaussian_beam() -> None:
    z = np.linspace(-8 * um, 8 * um, 24)
    z0 = 1.37 * um
    w = W0 * np.sqrt(1.0 + ((z - z0) / Z_R) ** 2)
    assert best_focus(z, w) == pytest.approx(z0, abs=1e-9 * um)
