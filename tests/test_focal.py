r"""Focal-field assembly: closed-form frames, frequency grouping, patch accumulation.

The device layer is deliberately **not** imported: ``field/focal.py`` consumes terms
structurally, so the fixtures here build a synthetic term array with the frozen layout of
``device.aodl.TermArray`` (WO-03 §3).  That keeps this suite an independent check of the field
physics — the reference oracle is ``field/reference.py``'s brute-force Eq. S11 quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aodl.field.focal import (
    Z_LAB_SIGN,
    FrameGrid,
    group_terms,
    intensity_frame,
    intensity_slice_xz,
    spot_params,
    term_field,
)
from aodl.field.reference import reference_field_separable
from aodl.units import MHz

#: Aperture samples for the hard-edged reference pupil: ``span * w_in`` is 12 mm on each side
#: and ``n - 1 = 8016 = 24 * 334``, so a fill edge at ``+-0.5 w_in`` lands exactly on a
#: quadrature sample (the trapezoid rule then resolves the step to O(du^2), see
#: ``_hard_edge`` below).
EDGE_REF_N = 8017


@dataclass
class TermArray:
    """Structural stand-in for ``device.aodl.TermArray`` (WO-03 §3)."""

    c: Any
    theta1: Any
    theta2: Any
    alpha: Any
    df_opt: Any
    edge: Any = (None, None)


def make_terms(
    *,
    c: Any,
    theta1: Any,
    theta2: Any,
    alpha: Any = None,
    df_opt: Any = 0.0,
    edge: Any = (None, None),
) -> TermArray:
    """Build a synthetic term array; ``theta*`` are ``(2, N)`` (or broadcastable)."""
    c_arr = np.atleast_1d(np.asarray(c, dtype=np.complex128))
    n = c_arr.size
    th1 = np.broadcast_to(np.asarray(theta1, dtype=np.float64).reshape(2, -1), (2, n)).copy()
    th2 = np.broadcast_to(np.asarray(theta2, dtype=np.float64).reshape(2, -1), (2, n)).copy()
    if alpha is None:
        alpha_arr = np.zeros((2, 3, n), dtype=np.complex128)
        alpha_arr[:, 0, :] = 1.0
    else:
        alpha_arr = np.broadcast_to(
            np.asarray(alpha, dtype=np.complex128).reshape(2, 3, -1), (2, 3, n)
        ).copy()
    df = np.broadcast_to(np.asarray(df_opt, dtype=np.float64).ravel(), (n,)).copy()
    return TermArray(c=c_arr, theta1=th1, theta2=th2, alpha=alpha_arr, df_opt=df, edge=edge)


def take(terms: TermArray, idx: Any) -> TermArray:
    """Subset of a term array (per frequency group)."""
    return TermArray(
        c=terms.c[idx],
        theta1=terms.theta1[:, idx],
        theta2=terms.theta2[:, idx],
        alpha=terms.alpha[:, :, idx],
        df_opt=terms.df_opt[idx],
        edge=terms.edge,
    )


def _hard_edge(u: Any, u_edge: float, side: str) -> Any:
    """Hard aperture window, one-half at the edge sample (the symmetric step convention,
    which the trapezoid rule integrates to O(du^2) when ``u_edge`` is on the grid)."""
    filled = u > u_edge if side == "lower" else u < u_edge
    return np.where(filled, 1.0, np.where(u == u_edge, 0.5, 0.0))


def _pupil(optics: Any, alpha: Any, th1: float, th2: float, edge: Any = None) -> Any:
    """Separable pupil factor matching one axis of a term (input Gaussian included)."""

    def pupil(u: Any) -> Any:
        poly = alpha[0] + alpha[1] * u + alpha[2] * u * u
        val = poly * np.exp(-(u**2) / optics.w_in**2 + 1j * th2 * u * u + 1j * th1 * u)
        if edge is not None:
            val = val * _hard_edge(u, edge[0], edge[1])
        return val

    return pupil


def _gauss_alpha() -> Any:
    """``alpha = (1, 0, 0)`` for one axis."""
    return np.array([1.0, 0.0, 0.0])


def _fit_waist(offset: Any, intensity: Any) -> float:
    """1/e^2 intensity radius from a log fit of a 1D cut (``offset`` centred on the spot)."""
    slope = np.polyfit(offset**2, np.log(intensity / intensity.max()), 1)[0]
    return float(np.sqrt(-2.0 / slope))


def test_unaberrated_term_is_the_textbook_focal_gaussian(params1030):
    """theta1 = theta2 = 0: peak at the origin, both waists = ``optics.waist0``."""
    optics = params1030.optics
    w0 = optics.waist0
    terms = make_terms(c=[1.0], theta1=np.zeros((2, 1)), theta2=np.zeros((2, 1)))

    grid = FrameGrid(-3.0 * w0, 3.0 * w0, 61, -3.0 * w0, 3.0 * w0, 61)
    frame = intensity_frame(terms, optics, grid, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(frame)), frame.shape)
    assert (grid.x[ix], grid.y[iy]) == (0.0, 0.0)

    assert _fit_waist(grid.x, frame[iy, :]) == pytest.approx(w0, rel=1e-4)
    assert _fit_waist(grid.y, frame[:, ix]) == pytest.approx(w0, rel=1e-4)


def test_unaberrated_frame_matches_reference_quadrature(params1030):
    """31x31 patch against brute-force Eq. S11, relative 1e-6 after peak normalization."""
    optics = params1030.optics
    w0 = optics.waist0
    terms = make_terms(c=[1.0], theta1=np.zeros((2, 1)), theta2=np.zeros((2, 1)))

    grid = FrameGrid(-1.5 * w0, 1.5 * w0, 31, -1.5 * w0, 1.5 * w0, 31)
    frame = intensity_frame(terms, optics, grid, 0.0)

    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    pupil = _pupil(optics, _gauss_alpha(), 0.0, 0.0)
    reference = np.abs(reference_field_separable(pupil, pupil, optics, xx, yy, 0.0)) ** 2

    deviation = np.abs(frame / frame.max() - reference / reference.max())
    assert deviation.max() < 1e-6


def test_astigmatic_term_focuses_per_axis_at_the_predicted_plane(params1030):
    """theta1x != 0, theta2y != 0: deflected spot, and each axis focuses at its own lab Z.

    The work order asks for the *on-axis intensity* to peak at ``z_lab = Z_LAB_SIGN 2F^2
    theta2y/k``; for a single-axis defocus that is not true (on-axis intensity goes like
    ``1/(|a_x| |a_y|)``, which peaks at the circle of least confusion, half way to the y
    focus, and is exactly equal at Z = 0 and Z = Z_y).  What *is* true, and is the physics the
    bullet is after, is that the **y waist** is minimal there, so that is what is scanned; the
    on-axis-intensity form is exercised by the isotropic term in
    :func:`test_slice_xz_brightest_z_tracks_the_predicted_focus`.
    """
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    w0, z_r = optics.waist0, optics.rayleigh

    x_spot = 2.0 * w0
    theta1x = k * x_spot / focal
    z_y_s11 = 1.0 * z_r
    theta2y = k * z_y_s11 / (2.0 * focal**2)
    z_y_lab = Z_LAB_SIGN * z_y_s11
    terms = make_terms(c=[1.0], theta1=[[theta1x], [0.0]], theta2=[[0.0], [theta2y]], df_opt=0.0)

    grid = FrameGrid(x_spot - 4.0 * w0, x_spot + 4.0 * w0, 81, -4.0 * w0, 4.0 * w0, 81)
    frame = intensity_frame(terms, optics, grid, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(frame)), frame.shape)
    assert grid.x[ix] == pytest.approx(x_spot, rel=1e-12)
    assert grid.y[iy] == 0.0

    # In the nominal plane the x axis is at its focus and the y axis is one z_R out.
    wx0 = _fit_waist(grid.x - x_spot, frame[iy, :])
    wy0 = _fit_waist(grid.y, frame[:, ix])
    assert wx0 == pytest.approx(w0, rel=1e-4)
    assert wy0 == pytest.approx(w0 * np.sqrt(2.0), rel=1e-4)
    assert wy0 > wx0  # astigmatism is visible

    # Scanning lab Z, the y waist is minimal (and equal to waist0) at the predicted plane.
    scan = z_y_lab + np.linspace(-1.5, 1.5, 13) * z_r
    waists = [_fit_waist(grid.y, intensity_frame(terms, optics, grid, z)[:, ix]) for z in scan]
    assert int(np.argmin(waists)) == 6
    assert scan[6] == pytest.approx(z_y_lab, rel=1e-12)
    assert waists[6] == pytest.approx(w0, rel=1e-4)

    # ... and the x waist is minimal in the plane z_lab = 0 (theta2x = 0), not at z_y_lab.
    wx_at_y_focus = _fit_waist(
        grid.x - x_spot, intensity_frame(terms, optics, grid, z_y_lab)[iy, :]
    )
    assert wx_at_y_focus == pytest.approx(w0 * np.sqrt(2.0), rel=1e-4)

    # spot_params agrees with the fitted frames (it is the shared closed form).
    xc, yc, wx, wy = spot_params(terms, optics, 0.0)
    assert xc[0] == pytest.approx(x_spot, rel=1e-12)
    assert yc[0] == 0.0
    assert (wx[0], wy[0]) == (pytest.approx(wx0, rel=1e-4), pytest.approx(wy0, rel=1e-4))


def _twin_terms(optics: Any, phases: Any, df_opt: Any) -> TermArray:
    """Two co-located terms with the given phases and optical frequencies."""
    theta1x = optics.k * (1.5 * optics.waist0) / optics.focal_length
    return make_terms(
        c=np.exp(1j * np.asarray(phases, dtype=np.float64)),
        theta1=[[theta1x, theta1x], [0.0, 0.0]],
        theta2=np.zeros((2, 2)),
        df_opt=df_opt,
    )


def _twin_grid(optics: Any) -> FrameGrid:
    w0 = optics.waist0
    return FrameGrid(-2.0 * w0, 5.0 * w0, 57, -3.0 * w0, 3.0 * w0, 49)


def test_degenerate_terms_interfere_coherently(params1030):
    """Same ``df_opt``, same place, phases 0 and pi: the group cancels."""
    optics = params1030.optics
    grid = _twin_grid(optics)
    pair = _twin_terms(optics, [0.0, np.pi], [0.0, 0.0])
    assert len(group_terms(pair)) == 1

    single = take(pair, np.array([0]))
    reference = intensity_frame(single, optics, grid, 0.0)
    frame = intensity_frame(pair, optics, grid, 0.0)
    assert frame.max() < 1e-10 * reference.max()


def test_distinct_frequencies_add_in_intensity(params1030):
    """The same two terms 1 MHz apart no longer interfere: intensities simply add."""
    optics = params1030.optics
    grid = _twin_grid(optics)
    pair = _twin_terms(optics, [0.0, np.pi], [0.0, 1.0 * MHz])
    assert len(group_terms(pair)) == 2

    frame = intensity_frame(pair, optics, grid, 0.0)
    parts = [intensity_frame(take(pair, np.array([i])), optics, grid, 0.0) for i in (0, 1)]
    total = parts[0] + parts[1]
    assert np.abs(frame - total).max() < 1e-10 * total.max()


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_fill_edge_matches_hard_edged_reference(params1030, side):
    """A partially filled aperture: closed-form edge moments vs the hard-edged quadrature."""
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    w0, z_r = optics.waist0, optics.rayleigh

    u_edge = -0.5 * optics.w_in if side == "lower" else 0.5 * optics.w_in
    x_spot = 2.0 * w0
    theta1x = k * x_spot / focal
    theta2x = k * (0.5 * z_r) / (2.0 * focal**2)
    z_lab = 0.3 * z_r
    alpha_x = np.array([1.0, 0.4 / optics.w_in, -0.6 / optics.w_in**2])
    alpha = np.zeros((2, 3, 1), dtype=np.complex128)
    alpha[0, :, 0] = alpha_x
    alpha[1, 0, 0] = 1.0

    terms = make_terms(
        c=[1.0],
        theta1=[[theta1x], [0.0]],
        theta2=[[theta2x], [0.0]],
        alpha=alpha,
        edge=((u_edge, side), None),
    )

    x_axis = x_spot + np.linspace(-4.0, 4.0, 41) * w0
    field = term_field(terms, optics, x_axis, 0.0, z_lab)[0]
    reference = reference_field_separable(
        _pupil(optics, alpha_x, theta1x, theta2x, (u_edge, side)),
        _pupil(optics, _gauss_alpha(), 0.0, 0.0),
        optics,
        x_axis,
        0.0,
        Z_LAB_SIGN * z_lab,
        n=EDGE_REF_N,
    )
    got = np.abs(field) ** 2
    want = np.abs(reference) ** 2
    assert np.abs(got / got.max() - want / want.max()).max() < 1e-3

    # The window really matters: ignoring it would be a gross error, not a 1e-3 one.
    unwindowed = term_field(
        make_terms(c=[1.0], theta1=[[theta1x], [0.0]], theta2=[[theta2x], [0.0]], alpha=alpha),
        optics,
        x_axis,
        0.0,
        z_lab,
    )[0]
    naive = np.abs(unwindowed) ** 2
    assert np.abs(naive / naive.max() - want / want.max()).max() > 0.05


def test_slice_xz_brightest_z_tracks_the_predicted_focus(params1030):
    """Chirp-like term (equal theta2 on both axes): the XZ panel focuses at the predicted Z."""
    optics = params1030.optics
    k, focal = optics.k, optics.focal_length
    w0, z_r = optics.waist0, optics.rayleigh

    x_spot = 1.5 * w0
    theta1x = k * x_spot / focal
    z_focus_s11 = 0.8 * z_r
    theta2 = k * z_focus_s11 / (2.0 * focal**2)
    z_pred = Z_LAB_SIGN * z_focus_s11
    terms = make_terms(c=[1.0], theta1=[[theta1x], [0.0]], theta2=[[theta2], [theta2]])

    x_axis = x_spot + np.linspace(-2.0, 2.0, 41) * w0
    z_axis = z_pred + np.linspace(-2.0, 2.0, 201) * z_r
    panel = intensity_slice_xz(terms, optics, x_axis, z_axis, 0.0)
    assert panel.shape == (z_axis.size, x_axis.size)

    on_axis = panel[:, 20]
    assert x_axis[20] == pytest.approx(x_spot, rel=1e-12)
    imax = int(np.argmax(on_axis))
    assert 0 < imax < on_axis.size - 1
    y0, y1, y2 = on_axis[imax - 1 : imax + 2]
    dz = z_axis[1] - z_axis[0]
    z_peak = z_axis[imax] + 0.5 * dz * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    assert abs(z_peak - z_pred) < 0.02 * z_r

    # The brightest column is the spot column, and the panel is symmetric about the focus.
    assert int(np.argmax(panel[imax, :])) == 20
    assert panel[imax - 50, 20] == pytest.approx(panel[imax + 50, 20], rel=1e-3)


def _random_scene(rng: Any, optics: Any, n: int = 5) -> TermArray:
    """A 3-group, 5-term scene: random places, foci, amplitudes and irising polynomials."""
    k, focal = optics.k, optics.focal_length
    centres = rng.uniform(-5.0, 5.0, (2, n)) * optics.waist0
    foci = rng.uniform(-1.0, 1.0, (2, n)) * optics.rayleigh
    alpha = np.zeros((2, 3, n), dtype=np.complex128)
    alpha[:, 0, :] = 1.0
    alpha[:, 1, :] = rng.uniform(-1.0, 1.0, (2, n)) / optics.w_in
    alpha[:, 2, :] = rng.uniform(-1.0, 1.0, (2, n)) / optics.w_in**2
    return make_terms(
        c=rng.uniform(0.5, 1.5, n) * np.exp(2j * np.pi * rng.uniform(0.0, 1.0, n)),
        theta1=centres * k / focal,
        theta2=foci * k / (2.0 * focal**2),
        alpha=alpha,
        df_opt=np.array([0.0, 0.0, 1.0, 2.0, 2.0]) * MHz,
    )


def test_patch_accumulation_matches_full_grid_evaluation(params1030, rng):
    """Patched rendering == brute full-grid rendering, group by group."""
    optics = params1030.optics
    w0 = optics.waist0
    terms = _random_scene(rng, optics)
    grid = FrameGrid(-12.0 * w0, 12.0 * w0, 121, -12.0 * w0, 12.0 * w0, 121)
    z_lab = 0.25 * optics.rayleigh

    groups = group_terms(terms)
    assert [len(g) for g in groups] == [2, 1, 2]

    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    brute_total = np.zeros_like(xx)
    patched_sum = np.zeros_like(xx)
    for idx in groups:
        sub = take(terms, idx)
        brute = np.abs(term_field(sub, optics, xx, yy, z_lab).sum(axis=0)) ** 2
        patched = intensity_frame(sub, optics, grid, z_lab)
        covered = patched > 0.0
        assert covered.any() and not covered.all()  # the patch is a real restriction
        assert np.abs(patched[covered] - brute[covered]).max() < 1e-12 * brute.max()
        assert brute[~covered].sum() < 1e-6 * brute.sum()
        brute_total += brute
        patched_sum += patched

    frame = intensity_frame(terms, optics, grid, z_lab)
    assert np.abs(frame - patched_sum).max() < 1e-12 * frame.max()
    assert abs(frame.sum() - brute_total.sum()) < 1e-6 * brute_total.sum()


def test_group_terms_clusters_by_optical_frequency(params1030):
    """Grouping tolerance (default 1 kHz): 0.5 kHz apart is one tweezer, 1 MHz apart is two."""
    optics = params1030.optics
    theta = np.zeros((2, 3))
    close = make_terms(c=np.ones(3), theta1=theta, theta2=theta, df_opt=[0.0, 5e2, 1.0 * MHz])
    groups = group_terms(close)
    assert [g.tolist() for g in groups] == [[0, 1], [2]]
    assert [g.tolist() for g in group_terms(close, tol=2.0 * MHz)] == [[0, 1, 2]]
    assert [g.tolist() for g in group_terms(close, tol=1e2)] == [[0], [1], [2]]
    assert len(spot_params(close, optics, 0.0)[0]) == 3
