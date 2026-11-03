"""The TPM Bragg efficiency model (W2 §2/§5, DESIGN §1.6)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from _advanced_fixtures import PARAMS, TAU, pick_and_place, static_array

from aodl.units import MHz, mm, us
from aodl_advanced import efficiency as eff

MODEL = eff.default_model(PARAMS)
BAND = PARAMS.channels["Ax"].band
F_TPM = MODEL.f_tpm


def test_peak_at_tpm_and_even() -> None:
    f = F_TPM + np.linspace(-12, 12, 25) * MHz
    eta = eff.first_aod_efficiency(f, MODEL, PARAMS)
    assert eta[12] == pytest.approx(MODEL.peak_efficiency, rel=1e-12)
    assert eta[12] == pytest.approx(0.85, rel=1e-12)
    np.testing.assert_allclose(eta, eta[::-1], rtol=1e-12)
    assert np.all(eta <= eta[12] + 1e-15)


def test_bandwidth_scales_as_inverse_sqrt_L_and_default_value() -> None:
    bw = eff.bandwidth_3db(MODEL, PARAMS)
    assert bw / MHz == pytest.approx(27.9, abs=0.5)
    quad = eff.default_model(PARAMS, interaction_length=4 * 4 * mm)
    assert bw / eff.bandwidth_3db(quad, PARAMS) == pytest.approx(2.0, rel=0.02)
    # the half-maximum root really is the half maximum
    f_half = F_TPM + 0.5 * bw
    assert eff.first_aod_efficiency(f_half, MODEL, PARAMS) == pytest.approx(
        0.5 * MODEL.peak_efficiency, rel=1e-9
    )


def test_pair_map_separable_without_coupling() -> None:
    model = eff.default_model(PARAMS, angular_coupling=0.0)
    f_a = F_TPM + np.linspace(-15, 15, 31) * MHz
    f_b = F_TPM + np.linspace(-20, 20, 41) * MHz
    pair = eff.pair_map(f_a, f_b, model, PARAMS)
    outer = np.outer(
        eff.first_aod_efficiency(f_a, model, PARAMS), eff.first_aod_efficiency(f_b, model, PARAMS)
    )
    np.testing.assert_allclose(pair, outer, rtol=1e-12)
    assert pair.shape == (31, 41)


def test_pair_map_ridge_and_cutoff_sides() -> None:
    model = eff.default_model(PARAMS, angular_coupling=1.0)
    f_b = F_TPM + np.linspace(-20, 20, 8001) * MHz
    plus, minus = F_TPM + 1 * MHz, F_TPM - 1 * MHz
    pair = eff.pair_map(np.array([plus, minus]), f_b, model, PARAMS)
    eta_a = eff.first_aod_efficiency(np.array([plus, minus]), model, PARAMS)
    # ridge side: B re-matches (eta_B = eta_peak) where (f_B - f_tpm)^2 = 2 gamma f_tpm * 1 MHz
    ridge_offset = math.sqrt(2.0 * 1.0 * F_TPM * 1 * MHz)
    i_max = int(np.argmax(pair[0]))
    assert abs(abs(f_b[i_max] - F_TPM) - ridge_offset) < 20e3
    assert pair[0].max() == pytest.approx(eta_a[0] * model.peak_efficiency, rel=0.01)
    # cut-off side: nothing in f_B compensates
    assert pair[1].max() / eta_a[1] <= 0.6


def test_first_aod_usable_width_matches_brute_force() -> None:
    model = eff.default_model(PARAMS, angular_coupling=0.2)
    width = eff.first_aod_usable_width(model, PARAMS, BAND)
    # brute force on a fine map, f_B restricted to the band
    f_a = F_TPM + np.linspace(-30, 30, 12001) * MHz
    f_b = np.linspace(BAND[0], BAND[1], 2001)
    ridge = eff.pair_map(f_a, f_b, model, PARAMS).max(axis=1)
    above = ridge >= 0.5 * ridge.max()
    brute = float(f_a[above].max() - f_a[above].min())
    assert width == pytest.approx(brute, rel=0.01)
    # the cut-off side alone is the analytic delta_3 / (2 C2 f_tpm |gamma|), within eta_A's own
    # roll-off over those few MHz
    c2 = eff.mismatch_scale(model, PARAMS)
    one_sided = eff.half_max_mismatch(model) / (2.0 * c2 * F_TPM * 0.2)
    i_max = int(np.argmax(ridge))
    left = f_a[above & (f_a <= f_a[i_max])].min()
    assert (F_TPM - left) == pytest.approx(one_sided, rel=0.05)
    assert width > one_sided  # the ridge side is wider


def test_usable_width_without_coupling_is_the_tpm_bandwidth() -> None:
    model = eff.default_model(PARAMS, angular_coupling=0.0)
    assert eff.first_aod_usable_width(model, PARAMS, BAND) == pytest.approx(
        eff.bandwidth_3db(model, PARAMS), rel=1e-3
    )


def test_deflection_angle_and_second_aod_at_center() -> None:
    f = np.array([F_TPM - 5 * MHz, F_TPM, F_TPM + 5 * MHz])
    angle = eff.deflection_angle(f, F_TPM, PARAMS.sound_speed, PARAMS.optics.wavelength)
    expected = PARAMS.optics.wavelength * 5 * MHz / PARAMS.sound_speed
    np.testing.assert_allclose(angle, [-expected, 0.0, expected], rtol=1e-12)
    # with f_A at TPM the second AOD is the first one again
    np.testing.assert_allclose(
        eff.second_aod_efficiency(f, F_TPM, MODEL, PARAMS),
        eff.first_aod_efficiency(f, MODEL, PARAMS),
        rtol=1e-12,
    )


def test_trajectory_efficiency_static_tweezer_is_eta_peak_fourth() -> None:
    plan = static_array(1, 1)
    t = np.array([2 * TAU, 3 * TAU, 50 * us])
    table = eff.trajectory_efficiency(plan.wfs, MODEL, t)
    for name in ("Ax", "Bx", "Ay", "By"):
        np.testing.assert_allclose(table[name], MODEL.peak_efficiency, rtol=1e-12)
    np.testing.assert_allclose(table["product"], MODEL.peak_efficiency**4, rtol=1e-12)


def test_trajectory_efficiency_dips_during_a_move() -> None:
    plan = pick_and_place()
    t = np.linspace(TAU, 140 * us + TAU, 60)
    table = eff.trajectory_efficiency(plan.wfs, MODEL, t)
    product = table["product"]
    assert product[0] == pytest.approx(MODEL.peak_efficiency**4, rel=1e-6)
    assert product.min() < product[0]
    assert np.all((product > 0.0) & (product <= MODEL.peak_efficiency**4 + 1e-12))


def test_model_validation() -> None:
    with pytest.raises(ValueError):
        eff.default_model(PARAMS, interaction_length=0.0)
    with pytest.raises(ValueError):
        eff.default_model(PARAMS, refractive_index=1.0)
    with pytest.raises(ValueError):
        eff.default_model(PARAMS, peak_efficiency=1.5)
    with pytest.raises(ValueError):
        eff.default_model(PARAMS, angular_coupling=1.5)
    assert eff.default_model(PARAMS).f_tpm == PARAMS.channels["Ax"].f_center
    assert eff.default_model(PARAMS).gamma_peak == pytest.approx(math.asin(math.sqrt(0.85)))
