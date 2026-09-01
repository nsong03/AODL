r"""M6 §5: the measurement layer — profile fits, best focus, blob audit, accumulation.

These are the numbers a checker verdict is made of, so every one of them is pinned against a
closed form: an exact Gaussian for the fits, an exact ``w^2(Z)`` parabola for the focus, a
planted scene for the blob finder, and a brute-force per-snapshot 2D average for the outer
product.  The contaminated cases matter as much as the clean ones — a fit that is exact on a
lone Gaussian but drifts when a neighbouring trap and a 1 % ghost are present would report
position errors that are really contamination.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aodl.check.metrics import (
    Blob,
    TrapFit,
    accumulate_intensity,
    accumulate_marginals,
    best_focus,
    find_blobs,
    fit_gaussian_1d,
    profile_moments,
)
from aodl.units import um


def _gaussian(x, center, radius, peak):
    """``I = peak exp(-2 (x - c)^2 / w^2)`` — the 1/e^2 *intensity* convention."""
    return peak * np.exp(-2.0 * (np.asarray(x) - center) ** 2 / radius**2)


# =============================================================== 1. the profile fit


@pytest.mark.parametrize("radius", [1.07 * um, 2.4 * um])
@pytest.mark.parametrize("center", [0.0, 3.3 * um])
def test_fit_is_exact_on_a_gaussian(center, radius) -> None:
    """A weighted log-parabola fit reproduces a Gaussian to round-off, off-centre included."""
    x = np.linspace(-8.0 * um, 12.0 * um, 321)
    peak = 3.7
    got_center, got_radius, got_peak = fit_gaussian_1d(x, _gaussian(x, center, radius, peak))
    assert got_center == pytest.approx(center, abs=1e-12 * um)
    assert got_radius == pytest.approx(radius, rel=1e-12)
    assert got_peak == pytest.approx(peak, rel=1e-12)

    # The model-free companion: for a Gaussian the RMS width is exactly w/2.
    moment_center, rms = profile_moments(x, _gaussian(x, center, radius, peak))
    assert moment_center == pytest.approx(center, abs=1e-3 * um)
    assert rms == pytest.approx(0.5 * radius, rel=1e-3)


def test_fit_survives_a_neighbour_and_a_ghost() -> None:
    """One pitch away and 1 % of ghost: the fit still lands within a thousandth of a waist.

    The ``I^2`` weights plus the contiguous ``e^{-2}`` window are what do it — the neighbour
    contributes ``e^{-2 (pitch/w)^2}`` inside that window, which for a 10.3 um pitch and a
    1.07 um waist is zero to any precision, and the ghost's 1 % sits outside it.
    """
    waist, pitch = 1.07 * um, 10.3 * um
    x = np.linspace(-6.0 * um, 16.0 * um, 881)
    clean = _gaussian(x, 0.0, waist, 1.0)
    scene = clean + _gaussian(x, pitch, waist, 1.0) + 0.01 * _gaussian(x, 0.45 * pitch, waist, 1.0)

    center, radius, peak = fit_gaussian_1d(x, scene)
    assert abs(center) < 1e-3 * waist
    assert radius == pytest.approx(waist, rel=1e-3)
    assert peak == pytest.approx(1.0, rel=1e-3)

    # The raw moments, by contrast, are dragged right across by the neighbour — which is
    # exactly why both are reported and only the fit is used for a verdict.
    moment_center, _ = profile_moments(x, scene)
    assert moment_center > 2.0 * waist


def test_fit_refuses_what_is_not_a_peak() -> None:
    """A trough, an empty profile or a two-sample profile are errors, not silent numbers."""
    x = np.linspace(-5.0 * um, 5.0 * um, 51)
    with pytest.raises(ValueError, match="opens upward"):
        fit_gaussian_1d(x, np.ones_like(x))  # flat: no peak at all
    with pytest.raises(ValueError, match="opens upward"):
        fit_gaussian_1d(x, np.exp((x / um) ** 2))  # a valley, sampled at its walls
    with pytest.raises(ValueError, match="no positive samples"):
        fit_gaussian_1d(x, np.zeros_like(x))
    with pytest.raises(ValueError, match="at least 3 samples"):
        fit_gaussian_1d(x[:2], np.ones(2))
    with pytest.raises(ValueError, match="needs 3; sample the profile more finely"):
        fit_gaussian_1d(np.arange(5.0), np.array([0.0, 0.0, 1.0, 2.0, 0.0]))
    with pytest.raises(ValueError, match="matching 1-D arrays"):
        fit_gaussian_1d(x, np.ones(7))
    with pytest.raises(ValueError, match="no power"):
        profile_moments(x, np.zeros_like(x))

    # Coarse sampling still fits: three points is the minimum and the window widens to find
    # them (a waist0/3 canvas gives about six).
    coarse = np.linspace(-2.0 * um, 2.0 * um, 5)
    center, radius, _ = fit_gaussian_1d(coarse, _gaussian(coarse, 0.2 * um, 1.07 * um, 1.0))
    assert center == pytest.approx(0.2 * um, rel=1e-9)
    assert radius == pytest.approx(1.07 * um, rel=1e-9)


# =============================================================== 2. best focus


def test_best_focus_is_exact_on_the_through_focus_parabola() -> None:
    """``w^2(Z) = w0^2 (1 + ((Z - Zf)/z_R)^2)``: the vertex comes back exactly."""
    w0, z_r, focus = 1.07 * um, 3.46 * um, 5.15 * um
    planes = focus + np.linspace(-1.0, 1.0, 7) * z_r
    w2 = w0**2 * (1.0 + ((planes - focus) / z_r) ** 2)
    assert best_focus(planes, w2) == pytest.approx(focus, rel=1e-12)
    # Three planes are enough — the fit is exact, not a regression.
    assert best_focus(planes[::3], w2[::3]) == pytest.approx(focus, rel=1e-12)
    # ... and an off-centre stack still finds it.
    shifted = focus + np.linspace(-0.2, 1.8, 9) * z_r
    assert best_focus(shifted, w0**2 * (1.0 + ((shifted - focus) / z_r) ** 2)) == pytest.approx(
        focus, rel=1e-12
    )


def test_best_focus_refuses_a_curve_with_no_waist_on_it() -> None:
    """No upward curvature anywhere means no vertex: raise rather than invent one."""
    planes = np.linspace(0.0, 10.0 * um, 5)
    with pytest.raises(ValueError, match="does not curve upward"):
        best_focus(planes, np.linspace(1.0, 4.0, 5))  # straight: curvature exactly zero
    with pytest.raises(ValueError, match="does not curve upward"):
        best_focus(planes, np.array([1.0, 4.0, 6.0, 7.0, 7.5]))  # concave
    with pytest.raises(ValueError, match="at least 3 planes"):
        best_focus(planes[:2], np.ones(2))
    with pytest.raises(ValueError, match="matching 1-D arrays"):
        best_focus(planes, np.ones(3))

    # A stack that misses the waist still reports it — the parabola is the exact law, so a
    # modest extrapolation is a measurement, and a large one is itself the finding.
    w0, z_r, focus = 1.07 * um, 3.46 * um, 5.15 * um
    off = focus + np.linspace(1.0, 3.0, 5) * z_r
    assert best_focus(off, w0**2 * (1.0 + ((off - focus) / z_r) ** 2)) == pytest.approx(
        focus, rel=1e-12
    )


def test_the_on_axis_peak_cannot_replace_the_per_axis_parabola() -> None:
    """Why ``best_focus`` fits ``w^2`` and not the on-axis intensity (module docstring).

    An astigmatic beam with foci at ``Z_x`` and ``Z_y`` has its brightest on-axis point at the
    circle of least confusion: midway between them while they are within ``2 z_R``, and split
    into a pair of maxima at ``(Z_x + Z_y)/2 -+ sqrt(((Z_x - Z_y)/2)^2 - z_R^2)`` beyond that.
    Neither case is a per-axis focus and neither reveals ``Delta F``.
    """
    w0, z_r = 1.07 * um, 3.46 * um
    planes = np.linspace(-8.0, 9.0, 1701) * z_r

    def on_axis_peak(zx, zy):
        """Brightest plane of the separable astigmatic Gaussian ``1 / sqrt(wx^2 wy^2)``."""
        wx2 = w0**2 * (1.0 + ((planes - zx) / z_r) ** 2)
        wy2 = w0**2 * (1.0 + ((planes - zy) / z_r) ** 2)
        assert best_focus(planes, wx2) == pytest.approx(zx, rel=1e-12)
        assert best_focus(planes, wy2) == pytest.approx(zy, rel=1e-12)
        assert best_focus(planes, wx2) - best_focus(planes, wy2) == pytest.approx(
            zx - zy, rel=1e-12
        )
        return float(planes[int(np.argmax(1.0 / np.sqrt(wx2 * wy2)))])

    # Weak astigmatism: one maximum, at the midpoint — not at either focus.
    weak = on_axis_peak(-0.3 * z_r, +0.3 * z_r)
    assert weak == pytest.approx(0.0, abs=0.02 * z_r)

    # Strong astigmatism: the midpoint has become a local *minimum* between two line foci.
    zx, zy = -2.0 * z_r, +3.0 * z_r
    strong = on_axis_peak(zx, zy)
    half = 0.5 * (zy - zx)
    roots = 0.5 * (zx + zy) + np.array([-1.0, 1.0]) * math.sqrt(half**2 - z_r**2)
    assert float(np.min(np.abs(strong - roots))) < 0.02 * z_r
    assert abs(strong - zx) > 0.2 * z_r and abs(strong - zy) > 0.2 * z_r


# =============================================================== 3. the blob audit


def test_find_blobs_finds_planted_ghosts() -> None:
    """Local maxima above the floor, sub-pixel, merged within a waist, brightest first."""
    waist, pitch = 1.07 * um, 10.3 * um
    xs = np.linspace(-16.0 * um, 16.0 * um, 321)
    ys = np.linspace(-16.0 * um, 16.0 * um, 321)
    planted = [
        (0.0, 0.0, 1.0),
        (pitch, 0.0, 0.9),
        (0.0, pitch, 0.9),
        (0.5 * pitch, -0.5 * pitch, 0.03),  # the ghost
    ]
    canvas = np.zeros((xs.size, ys.size))
    for cx, cy, amp in planted:
        canvas += amp * np.outer(_gaussian(xs, cx, waist, 1.0), _gaussian(ys, cy, waist, 1.0))

    found = find_blobs(canvas, xs, ys, floor=0.01, merge_radius=waist)
    assert len(found) == len(planted)
    assert [blob[2] for blob in found] == sorted((blob[2] for blob in found), reverse=True)
    assert found[0][2] == pytest.approx(1.0, rel=0.02)  # brightest first
    assert found[-1][2] == pytest.approx(0.03, rel=0.02)  # the ghost, last
    for cx, cy, amp in planted:  # the two equal-brightness nodes may come back either way
        match = min(found, key=lambda blob: math.hypot(blob[0] - cx, blob[1] - cy))
        assert match[0] == pytest.approx(cx, abs=0.05 * waist)
        assert match[1] == pytest.approx(cy, abs=0.05 * waist)
        assert match[2] == pytest.approx(amp, rel=0.02)

    # The floor hides the ghost; a reference rescales what is reported.
    assert len(find_blobs(canvas, xs, ys, floor=0.5, merge_radius=waist)) == 3
    doubled = find_blobs(canvas, xs, ys, floor=0.01, merge_radius=waist, reference=0.5)
    assert doubled[0][2] == pytest.approx(2.0, rel=0.02)
    # A merge radius wider than the pitch collapses the array into one blob.
    assert len(find_blobs(canvas, xs, ys, floor=0.01, merge_radius=2.0 * pitch)) == 1

    assert find_blobs(np.zeros((4, 4)), np.arange(4.0), np.arange(4.0), 0.0, merge_radius=1.0) == []
    with pytest.raises(ValueError, match="merge_radius must be positive"):
        find_blobs(canvas, xs, ys, 0.01, merge_radius=0.0)
    with pytest.raises(ValueError, match=r"must be shaped"):
        find_blobs(canvas[:, :10], xs, ys, 0.01, merge_radius=waist)


# =============================================================== 4. accumulation


def _axis_field(coords, tones, t):
    """Two-tone axis factor: a beating pair of displaced Gaussians."""
    out = np.zeros(coords.shape, dtype=np.complex128)
    for center, amplitude, frequency in tones:
        out += amplitude * np.exp(
            -((coords - center) ** 2) / (1.07 * um) ** 2 + 2j * np.pi * frequency * t
        )
    return out


@pytest.mark.parametrize("y_beat, gap", [(2.0e6, 0.23), (3.0e6, 0.0)])
def test_outer_product_accumulation_matches_a_brute_snapshot_average(y_beat, gap) -> None:
    r"""Average the product, never the factors — and this is exactly when it matters.

    Two overlapping tones per axis, so ``|U_x|^2`` beats at ``df_x`` and ``|U_y|^2`` at
    ``df_y``.  When the two beat frequencies are **equal** the product of the averages misses
    the ``<cos cos> = 1/2`` cross term and is wrong by 23 % of the peak; when they differ (and
    the window holds whole periods of both) the two agree exactly.  An atom array's tone
    ladders share one spacing per axis, so the equal case is the normal one, not the corner.
    """
    xs = np.linspace(-4.0 * um, 4.0 * um, 121)
    ys = np.linspace(-4.0 * um, 4.0 * um, 97)
    x_tones = ((-0.7 * um, 1.0, 0.0), (0.7 * um, 0.8, 2.0e6))
    y_tones = ((-0.7 * um, 1.0, 0.0), (0.7 * um, 0.9, y_beat))
    times = np.linspace(0.0, 1.0e-6, 32, endpoint=False)

    ux = np.stack([_axis_field(xs, x_tones, t) for t in times])
    uy = np.stack([_axis_field(ys, y_tones, t) for t in times])

    brute = np.zeros((xs.size, ys.size))
    for i in range(times.size):
        brute += np.abs(np.outer(ux[i], uy[i])) ** 2
    brute /= times.size

    got = accumulate_intensity(ux, uy)
    np.testing.assert_allclose(got, brute, rtol=1e-13, atol=1e-15 * brute.max())

    naive = np.outer((np.abs(ux) ** 2).mean(axis=0), (np.abs(uy) ** 2).mean(axis=0))
    assert np.abs(naive - brute).max() / brute.max() == pytest.approx(gap, abs=0.01)

    # The marginals are the canvas's, computed without building it.
    mx, my = accumulate_marginals(ux, uy)
    np.testing.assert_allclose(mx, brute.sum(axis=1), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(my, brute.sum(axis=0), rtol=1e-12, atol=0.0)


def test_accumulation_carries_batch_axes_and_checks_shapes() -> None:
    """Middle axes (Z planes, traps) pass through; mismatched ones raise."""
    rng = np.random.default_rng(3)
    ux = rng.normal(size=(5, 3, 7)) + 1j * rng.normal(size=(5, 3, 7))
    uy = rng.normal(size=(5, 3, 4)) + 1j * rng.normal(size=(5, 3, 4))
    canvas = accumulate_intensity(ux, uy)
    assert canvas.shape == (3, 7, 4)
    for plane in range(3):
        np.testing.assert_allclose(
            canvas[plane], accumulate_intensity(ux[:, plane], uy[:, plane]), rtol=1e-13
        )
    mx, my = accumulate_marginals(ux, uy)
    np.testing.assert_allclose(mx, canvas.sum(axis=-1), rtol=1e-12)
    np.testing.assert_allclose(my, canvas.sum(axis=-2), rtol=1e-12)

    with pytest.raises(ValueError, match="every axis but the last"):
        accumulate_intensity(ux, uy[:, :2])
    with pytest.raises(ValueError, match="no sub-times"):
        accumulate_intensity(np.zeros((0, 7)), np.zeros((0, 4)))


# =============================================================== 5. the record types


def test_the_result_records_are_plain_frozen_values() -> None:
    """``TrapFit`` and ``Blob`` are the shapes WO-22 fills in; keep them boring and frozen."""
    fit = TrapFit(
        x=1.0 * um,
        y=-2.0 * um,
        z_lab=0.5 * um,
        delta_f=0.1 * um,
        sigma_astig=0.03,
        wx=1.1 * um,
        wy=1.2 * um,
        peak=4.0,
        power=9.0,
        beat_std=0.02,
    )
    assert fit.sigma_astig == pytest.approx(0.03)
    with pytest.raises(Exception):  # frozen dataclass
        fit.x = 0.0  # type: ignore[misc]

    blob = Blob(time=1.0e-5, x=0.0, y=0.0, rel_intensity=0.02, on_lattice=True)
    assert blob.on_lattice is True
    assert math.isfinite(blob.rel_intensity)
