"""Parametric NPZ round-trip: bit-exact parameters in, bit-exact parameters out.

The product decision (``docs/ARCHITECTURE.md`` §5.5) is that a waveform file holds the
*function*, never its samples — so these tests assert both halves of that: the round trip
is float-identical, and nothing in the file is remotely sample-sized.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aodl.params import default_1030, paper_808
from aodl.poly import PiecewisePoly
from aodl.trajectory import ramps
from aodl.units import MHz, us
from aodl.waveform import serialize
from aodl.waveform.tones import (
    ChannelWaveform,
    ConstantEnvelope,
    SmoothOnOff,
    ToneTrack,
    WaveformSet,
)

T_END = 120.0 * us


def _mixed_set() -> WaveformSet:
    """Three tones on Bx (min-jerk+const-accel chain, SCJ, hold) plus one on Ax."""
    chain = PiecewisePoly.concat(
        [
            ramps.min_jerk(0.0, 60.0 * us, 0.0, 2.0 * MHz),
            ramps.constant_accel(60.0 * us, 40.0 * us, 2.0 * MHz, -1.0 * MHz),
        ]
    )
    tone0 = ToneTrack(chain, ConstantEnvelope(0.8), phase0=0.37).with_hold_until(T_END)
    tone1 = ToneTrack(
        ramps.switching_constant_jerk(0.0, 100.0 * us, -3.0 * MHz, 4.0 * MHz),
        SmoothOnOff(t_on=5.0 * us, t_off=115.0 * us, ramp=20.0 * us),
        phase0=-1.25,
    ).with_hold_until(T_END)
    tone2 = ToneTrack(ramps.linear(0.0, T_END, 1.5 * MHz, -0.25 * MHz), phase0=2.0)
    tone3 = ToneTrack(ramps.hold(0.0, T_END, -0.5 * MHz), ConstantEnvelope(0.5))
    return WaveformSet(
        channels={
            "Bx": ChannelWaveform((tone0, tone1, tone2)),
            "Ax": ChannelWaveform((tone3,)),
        },
        params=default_1030(),
        description="mixed-ramp, mixed-envelope test set",
    )


def _assert_same_waveform(a: WaveformSet, b: WaveformSet) -> None:
    assert list(a.channels) == list(b.channels)
    assert a.description == b.description
    assert a.params == b.params
    for name in a.channels:
        tones_a, tones_b = a.channels[name].tones, b.channels[name].tones
        assert len(tones_a) == len(tones_b)
        for ta, tb in zip(tones_a, tones_b, strict=True):
            np.testing.assert_array_equal(ta.freq.breaks, tb.freq.breaks)
            np.testing.assert_array_equal(ta.freq.coeffs, tb.freq.coeffs)
            assert ta.phase0 == tb.phase0
            assert type(ta.env) is type(tb.env)
            assert vars(ta.env) == vars(tb.env)


def test_round_trip_is_float_identical(tmp_path):
    wfs = _mixed_set()
    path = serialize.save(wfs, tmp_path / "mixed.npz")
    assert path.exists()
    loaded = serialize.load(path)
    _assert_same_waveform(wfs, loaded)

    # and the reconstructed waveform evaluates identically, bit for bit
    t = np.linspace(*wfs.t_span, 1001)
    for name in wfs.channels:
        before = wfs.channels[name].eval_table(t)
        after = loaded.channels[name].eval_table(t)
        for key, values in before.items():
            np.testing.assert_array_equal(values, after[key], err_msg=f"{name}.{key}")


def test_waveform_set_save_load_methods(tmp_path):
    wfs = _mixed_set()
    path = wfs.save(tmp_path / "via_method.npz")
    loaded = WaveformSet.load(path)
    _assert_same_waveform(wfs, loaded)


def test_params_snapshot_round_trips(tmp_path):
    for preset in (default_1030(), paper_808()):
        wfs = WaveformSet(
            {"Ay": ChannelWaveform((ToneTrack(ramps.linear(0.0, T_END, 0.0, 1.0 * MHz)),))},
            params=preset,
        )
        loaded = serialize.load(serialize.save(wfs, tmp_path / "p.npz"))
        assert loaded.params == preset
        assert loaded.params.optics.wavelength == preset.optics.wavelength
        assert loaded.params.deflection_scale == preset.deflection_scale
        assert loaded.params.lens_scale == preset.lens_scale
        for name, aod in preset.channels.items():
            assert loaded.params.channels[name].band == aod.band
            assert loaded.params.channels[name].drive_strength == aod.drive_strength


def test_file_contains_parameters_not_samples(tmp_path):
    """A 120 us waveform at 625 MS/s would be 75000 samples per channel; assert we store
    nothing remotely that size."""
    path = serialize.save(_mixed_set(), tmp_path / "params_only.npz")
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"meta", "Bx_segments", "Bx_tones", "Ax_segments", "Ax_tones"}
        for key in data.files:
            assert data[key].size <= 10_000, f"{key} looks like samples ({data[key].size})"
        assert data["Bx_segments"].shape[1] == serialize.SEGMENT_COLUMNS
        assert data["Bx_tones"].shape == (3, serialize.TONE_COLUMNS)
        meta = json.loads(str(data["meta"].item()))
    assert meta["schema_version"] == serialize.SCHEMA_VERSION
    assert meta["channels"] == ["Bx", "Ax"]
    assert path.stat().st_size < 20_000


def test_env_kinds_are_encoded_as_documented(tmp_path):
    path = serialize.save(_mixed_set(), tmp_path / "envs.npz")
    with np.load(path, allow_pickle=False) as data:
        bx = data["Bx_tones"]
    # tone 0 constant(0.8), tone 1 smooth_on_off(5us, 115us, 20us), tone 2 constant(1.0)
    assert bx[0, 2] == serialize.ENV_CONSTANT
    assert bx[0, 3] == 0.8
    assert bx[1, 2] == serialize.ENV_SMOOTH_ON_OFF
    np.testing.assert_array_equal(bx[1, 3:6], [5.0 * us, 115.0 * us, 20.0 * us])
    assert bx[2, 2] == serialize.ENV_CONSTANT
    assert bx[2, 3] == 1.0
    np.testing.assert_array_equal(bx[:, 1], [0.37, -1.25, 2.0])  # phase0 column


def test_unknown_schema_version_is_a_clear_error(tmp_path):
    path = tmp_path / "future.npz"
    serialize.save(_mixed_set(), path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "meta"}
        meta = json.loads(str(data["meta"].item()))
    meta["schema_version"] = 99
    np.savez(path, meta=json.dumps(meta), **arrays)
    with pytest.raises(ValueError, match="unsupported schema_version 99"):
        serialize.load(path)


def test_unknown_env_kind_is_a_clear_error(tmp_path):
    path = tmp_path / "weird_env.npz"
    serialize.save(_mixed_set(), path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "meta"}
        meta_json = str(data["meta"].item())
    arrays["Bx_tones"][0, 2] = 7.0
    np.savez(path, meta=meta_json, **arrays)
    with pytest.raises(ValueError, match="unknown env_kind 7"):
        serialize.load(path)


def test_unserializable_envelope_is_rejected(tmp_path):
    class Weird:
        def A(self, t):
            return np.zeros_like(np.asarray(t, dtype=float))

        dA = d2A = A

    wfs = WaveformSet(
        {"Ay": ChannelWaveform((ToneTrack(ramps.linear(0.0, T_END, 0.0, 1.0 * MHz), Weird()),))},
        params=default_1030(),
    )
    with pytest.raises(TypeError, match="cannot serialize envelope"):
        serialize.save(wfs, tmp_path / "nope.npz")


def test_non_waveform_file_is_rejected(tmp_path):
    path = tmp_path / "random.npz"
    np.savez(path, junk=np.arange(3.0))
    with pytest.raises(ValueError, match="no 'meta' entry"):
        serialize.load(path)
