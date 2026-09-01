"""AWG sample rendering: the carrier comes back, and the samples are the analytic signal.

Recovering the instantaneous frequency from the samples with
``np.unwrap(np.angle(hilbert(...)))`` is noisy at the band edges, so instead we assert the
two things that actually matter: every rendered sample equals the closed-form
``A(t) cos(2 pi f_center t + phase(t))`` it is supposed to be, and the zero-crossing count
reproduces the mean frequency of the chirp.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aodl.params import default_1030
from aodl.trajectory import ramps
from aodl.units import MHz, us
from aodl.waveform.export import (
    DEFAULT_SAMPLE_RATE,
    SAMPLES_SUFFIX,
    render_samples,
    sample_times,
    save_samples,
)
from aodl.waveform.tones import ChannelWaveform, ConstantEnvelope, ToneTrack, WaveformSet

TWO_PI = 2.0 * np.pi
CHIRP_T = 100.0 * us
CHIRP_DF = 2.0 * MHz
RATE = DEFAULT_SAMPLE_RATE  # 625 MS/s


def _chirp_tone(amp: float = 1.0, phase0: float = 0.3) -> ToneTrack:
    """f: 0 -> 2 MHz detuning over 100 us (constant chirp, Table I: a fixed axial offset)."""
    return ToneTrack(ramps.linear(0.0, CHIRP_T, 0.0, CHIRP_DF), ConstantEnvelope(amp), phase0)


def _chirp_set(**kwargs) -> WaveformSet:
    return WaveformSet(
        {"Ay": ChannelWaveform((_chirp_tone(**kwargs),))},
        params=default_1030(),
        description="single linear chirp",
    )


def test_rendered_samples_equal_the_closed_form_signal(rng):
    wfs = _chirp_set()
    tone = wfs.channels["Ay"].tones[0]
    f_center = wfs.params.channels["Ay"].f_center
    assert f_center == 100.0 * MHz

    samples, scale = render_samples(wfs, sample_rate=RATE, return_scale=True)
    arr = samples["Ay"]
    assert arr.dtype == np.float32
    assert arr.size == int(round(CHIRP_T * RATE)) + 1 == 62501
    assert scale == pytest.approx(1.0, rel=1e-3)  # unit envelope, peak of a cosine

    idx = rng.integers(0, arr.size, size=64)
    t = idx / RATE
    expected = tone.env.A(t) * np.cos(TWO_PI * f_center * t + tone.phase(t))
    np.testing.assert_allclose(arr[idx] * scale, expected, rtol=0, atol=1e-6)

    # float64 rendering is exact to round-off
    exact = render_samples(wfs, sample_rate=RATE, dtype=np.float64)["Ay"]
    np.testing.assert_allclose(exact[idx] * scale, expected, rtol=0, atol=1e-12)


def test_zero_crossing_count_recovers_the_mean_frequency():
    wfs = _chirp_set()
    arr = render_samples(wfs, sample_rate=RATE)["Ay"]
    duration = (arr.size - 1) / RATE
    crossings = int(np.count_nonzero(np.diff(np.signbit(arr))))
    f_mean_est = crossings / (2.0 * duration)
    # carrier + mean detuning of a linear ramp = 100 MHz + 1 MHz
    assert f_mean_est == pytest.approx(100.0 * MHz + 0.5 * CHIRP_DF, rel=1e-3)
    assert crossings == pytest.approx(2 * 10100, abs=2)  # 10100 cycles in 100 us


def test_chunking_does_not_change_the_result():
    wfs = _chirp_set()
    whole = render_samples(wfs, sample_rate=RATE)["Ay"]
    chunked = render_samples(wfs, sample_rate=RATE, chunk=997)["Ay"]
    np.testing.assert_array_equal(whole, chunked)


def test_normalization_is_one_common_factor_across_channels():
    params = default_1030()
    wfs = WaveformSet(
        {
            "Ay": ChannelWaveform((_chirp_tone(amp=1.0),)),
            "By": ChannelWaveform((_chirp_tone(amp=0.25),)),
        },
        params=params,
    )
    samples, scale = render_samples(wfs, sample_rate=RATE, return_scale=True)
    peak_ay = float(np.max(np.abs(samples["Ay"])))
    peak_by = float(np.max(np.abs(samples["By"])))
    assert peak_ay == pytest.approx(1.0, rel=1e-6)  # the global peak is full scale
    assert peak_by / peak_ay == pytest.approx(0.25, rel=1e-6)  # relative amplitude preserved
    assert scale == pytest.approx(1.0, rel=1e-3)
    assert all(np.max(np.abs(v)) <= 1.0 + 1e-6 for v in samples.values())


def test_render_over_an_extended_span_after_with_hold_until():
    """Beyond a tone's programmed domain the phase clamps; with_hold_until keeps it going."""
    wfs = _chirp_set().with_hold_until(150.0 * us)
    tone = wfs.channels["Ay"].tones[0]
    f_center = wfs.params.channels["Ay"].f_center
    samples, scale = render_samples(wfs, sample_rate=RATE, return_scale=True)
    n_expected, _ = sample_times((0.0, 150.0 * us), RATE)
    assert samples["Ay"].size == n_expected

    idx = np.array([70_000, 80_000, 90_000, n_expected - 1])  # all past the 100 us mark
    t = idx / RATE
    np.testing.assert_allclose(tone.f(t), CHIRP_DF, rtol=1e-12)  # frequency held
    expected = tone.env.A(t) * np.cos(TWO_PI * f_center * t + tone.phase(t))
    np.testing.assert_allclose(samples["Ay"][idx] * scale, expected, rtol=0, atol=1e-6)


def test_explicit_t_span_and_input_validation():
    wfs = _chirp_set()
    samples = render_samples(wfs, sample_rate=RATE, t_span=(10.0 * us, 20.0 * us))
    assert samples["Ay"].size == int(round(10.0 * us * RATE)) + 1
    with pytest.raises(ValueError, match="t_span must be increasing"):
        render_samples(wfs, t_span=(20.0 * us, 10.0 * us))
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        render_samples(wfs, sample_rate=0.0)
    with pytest.raises(ValueError, match="floating"):
        render_samples(wfs, dtype=np.int16)
    with pytest.raises(ValueError, match="chunk"):
        render_samples(wfs, chunk=0)


def test_save_samples_writes_metadata_and_enforces_the_suffix(tmp_path):
    wfs = _chirp_set()
    with pytest.raises(ValueError, match=SAMPLES_SUFFIX):
        save_samples(wfs, tmp_path / "chirp.npz")

    path = save_samples(wfs, tmp_path / f"chirp{SAMPLES_SUFFIX}", sample_rate=RATE)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"meta", "Ay"}
        arr = data["Ay"]
        meta = json.loads(str(data["meta"].item()))
    assert meta["kind"] == "samples"
    assert meta["sample_rate"] == RATE
    assert meta["t_span"] == [0.0, CHIRP_T]
    assert meta["n_samples"] == arr.size == 62501
    assert meta["channels"] == ["Ay"]
    assert meta["f_center"] == {"Ay": 100.0 * MHz}
    assert meta["dtype"] == "float32"

    # samples * normalization reproduces the physical signal
    tone = wfs.channels["Ay"].tones[0]
    t = np.array([0.0, 3.7 * us, 55.0 * us, CHIRP_T])
    idx = np.round(t * RATE).astype(int)
    expected = tone.env.A(idx / RATE) * np.cos(
        TWO_PI * 100.0 * MHz * (idx / RATE) + tone.phase(idx / RATE)
    )
    np.testing.assert_allclose(arr[idx] * meta["normalization"], expected, rtol=0, atol=1e-6)
