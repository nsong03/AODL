"""``WaveformSet`` <-> NPZ, **parameters only** — never samples.

The stored object is the parametric function representation: piecewise-polynomial
frequency laws (segment breakpoints + normalized-time coefficients), envelope parameters,
tone phases and a full hardware snapshot.  A file is therefore a few kilobytes regardless
of how long the move is, round-trips bit-exactly, and lets the simulator and the AWG
exporter derive *exactly the same* waveform (``docs/ARCHITECTURE.md`` §0.1, decision §5.5).

The schema is documented, with a worked example, in ``docs/waveform_format.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..params import CHANNELS, AODLParams, AODParams, OpticsParams
from ..poly import MAX_DEGREE, PiecewisePoly
from .shepard import FadeZoneEnvelope
from .tones import ChannelWaveform, ConstantEnvelope, Envelope, SmoothOnOff, ToneTrack, WaveformSet

#: Baseline on-disk schema version, written into the ``meta`` JSON for any file the v1
#: layout can express.
SCHEMA_VERSION = 1

#: Schema version written when at least one tone carries a fading-Shepard envelope
#: (:class:`~aodl.waveform.shepard.FadeZoneEnvelope`).  v2 is purely **additive**: the v1
#: arrays are unchanged and one ``<ch>_env_polys`` table per affected channel is added, so a
#: file only claims v2 when it actually needs the new table.  See ``docs/waveform_format.md``.
SCHEMA_VERSION_FADE = 2

#: Schema versions this build can read.
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (SCHEMA_VERSION, SCHEMA_VERSION_FADE)

#: ``mixing_order`` assumed for a v1 file whose params block predates that key (the package
#: default at the time such a file could have been written).  See :func:`params_from_dict`.
LEGACY_MIXING_ORDER = 1

#: ``<ch>_segments`` row layout: ``tone_idx, t0, T, degree, c0..c9``.  Shared verbatim by the
#: v2 ``<ch>_env_polys`` table, whose rows are the fade coordinate ``g(t)`` of one tone.
SEGMENT_COLUMNS = 4 + MAX_DEGREE + 1

#: ``<ch>_tones`` row layout: ``tone_idx, phase0, env_kind, env_p0..env_p3``.
TONE_COLUMNS = 3 + 4

#: Envelope discriminators stored in the ``env_kind`` column.
ENV_CONSTANT = 0
ENV_SMOOTH_ON_OFF = 1
ENV_FADE_ZONE = 2

_ENV_PARAMS = 4


# ------------------------------------------------------------------ envelope codec


def _encode_env(env: Envelope) -> tuple[int, list[float], PiecewisePoly | None]:
    """``Envelope -> (env_kind, [p0, p1, p2, p3], g-poly or None)``.

    Only :class:`~aodl.waveform.shepard.FadeZoneEnvelope` returns a polynomial: its fade
    coordinate ``g(t) = f_Z(t) + (n + xi) delta_f`` is a piecewise polynomial like any
    frequency law and goes into the channel's ``<ch>_env_polys`` table (schema v2), which is
    also where the rung index and ``xi`` live — inside ``g``'s constant term.
    """
    if isinstance(env, ConstantEnvelope):
        return ENV_CONSTANT, [env.amp, 0.0, 0.0, 0.0], None
    if isinstance(env, SmoothOnOff):
        return ENV_SMOOTH_ON_OFF, [env.t_on, env.t_off, env.ramp, 0.0], None
    if isinstance(env, FadeZoneEnvelope):
        if env.amp != 1.0:
            raise ValueError(
                f"cannot serialize a FadeZoneEnvelope with amp={env.amp!r}: schema "
                f"v{SCHEMA_VERSION_FADE}'s four env parameter slots are "
                f"(delta_f, eta, p, M) and carry no room for a peak amplitude.  Synthesize "
                f"with amp=1.0 (the drive's absolute scale belongs to the AWG export anyway, "
                f"see waveform/export.py) if the waveform has to round-trip through a file."
            )
        return ENV_FADE_ZONE, [env.delta_f, env.eta, env.p, float(env.m)], env.g
    raise TypeError(
        f"cannot serialize envelope of type {type(env).__name__!r}: schema "
        f"v{SCHEMA_VERSION_FADE} knows ConstantEnvelope (env_kind={ENV_CONSTANT}), "
        f"SmoothOnOff (env_kind={ENV_SMOOTH_ON_OFF}) and "
        f"FadeZoneEnvelope (env_kind={ENV_FADE_ZONE})"
    )


def _decode_env(kind: int, params: NDArray[np.float64], g: PiecewisePoly | None) -> Envelope:
    """``(env_kind, [p0..p3], g-poly) -> Envelope``."""
    if kind == ENV_CONSTANT:
        return ConstantEnvelope(amp=float(params[0]))
    if kind == ENV_SMOOTH_ON_OFF:
        return SmoothOnOff(t_on=float(params[0]), t_off=float(params[1]), ramp=float(params[2]))
    if kind == ENV_FADE_ZONE:
        if g is None:
            raise ValueError(
                f"env_kind {ENV_FADE_ZONE} (fade_zone) needs the tone's fade coordinate, but no "
                f"matching '<ch>_env_polys' rows were found; the file is incomplete"
            )
        return FadeZoneEnvelope(
            g=g,
            delta_f=float(params[0]),
            eta=float(params[1]),
            p=float(params[2]),
            m=int(round(float(params[3]))),
        )
    raise ValueError(
        f"unknown env_kind {kind!r}: schema v{SCHEMA_VERSION_FADE} defines "
        f"{ENV_CONSTANT} (constant), {ENV_SMOOTH_ON_OFF} (smooth_on_off) and "
        f"{ENV_FADE_ZONE} (fade_zone)"
    )


# -------------------------------------------------------------------- params codec


def params_to_dict(params: AODLParams) -> dict[str, Any]:
    """JSON-ready snapshot of every :class:`~aodl.params.AODLParams` field."""
    return {
        "optics": {
            "wavelength": params.optics.wavelength,
            "focal_length": params.optics.focal_length,
            "w_in": params.optics.w_in,
        },
        "channels": {
            name: {
                "sound_speed": aod.sound_speed,
                "aperture": aod.aperture,
                "f_center": aod.f_center,
                "band": [aod.band[0], aod.band[1]],
                "drive_strength": aod.drive_strength,
                "mixing_order": aod.mixing_order,
            }
            for name, aod in params.channels.items()
        },
    }


def params_from_dict(data: dict[str, Any]) -> AODLParams:
    """Inverse of :func:`params_to_dict` (exact: JSON floats round-trip).

    ``mixing_order`` was added to the params block after the first files were written, so a
    channel that lacks the key is read back as :data:`LEGACY_MIXING_ORDER` — the value that
    was in force when such a file could have been produced.  The schema version is unchanged
    (v1): the key is purely additive, and older readers simply ignore it.
    """
    try:
        optics = OpticsParams(**data["optics"])
        channels = {
            name: AODParams(
                sound_speed=ch["sound_speed"],
                aperture=ch["aperture"],
                f_center=ch["f_center"],
                band=(ch["band"][0], ch["band"][1]),
                drive_strength=ch["drive_strength"],
                mixing_order=int(ch.get("mixing_order", LEGACY_MIXING_ORDER)),
            )
            for name, ch in data["channels"].items()
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed params block in waveform metadata: {exc}") from exc
    return AODLParams(optics=optics, channels=channels)


# --------------------------------------------------------------------------- save


def _poly_rows(poly: PiecewisePoly, tone_idx: int) -> list[NDArray[np.float64]]:
    """One ``tone_idx, t0, T, degree, c0..c9`` row per segment of ``poly``, in time order."""
    widths = np.diff(poly.breaks)
    width = poly.coeffs.shape[1]
    rows: list[NDArray[np.float64]] = []
    for k in range(poly.n_segments):
        row = np.zeros(SEGMENT_COLUMNS, dtype=np.float64)
        row[0] = tone_idx
        row[1] = poly.breaks[k]
        row[2] = widths[k]
        row[3] = poly.degree
        row[4 : 4 + width] = poly.coeffs[k]
        rows.append(row)
    return rows


def _channel_arrays(
    cw: ChannelWaveform,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(<ch>_segments, <ch>_tones, <ch>_env_polys)``; the last is empty without fades."""
    segments: list[NDArray[np.float64]] = []
    tone_rows: list[NDArray[np.float64]] = []
    env_segments: list[NDArray[np.float64]] = []
    for tone_idx, tone in enumerate(cw.tones):
        segments.extend(_poly_rows(tone.freq, tone_idx))
        kind, env_params, env_poly = _encode_env(tone.env)
        if env_poly is not None:
            env_segments.extend(_poly_rows(env_poly, tone_idx))
        tone_rows.append(
            np.array([tone_idx, tone.phase0, float(kind), *env_params], dtype=np.float64)
        )
    seg = np.array(segments, dtype=np.float64).reshape(-1, SEGMENT_COLUMNS)
    ton = np.array(tone_rows, dtype=np.float64).reshape(-1, TONE_COLUMNS)
    env = np.array(env_segments, dtype=np.float64).reshape(-1, SEGMENT_COLUMNS)
    return seg, ton, env


def save(wfs: WaveformSet, path: str | Path) -> Path:
    """Write ``wfs`` to a parametric NPZ and return the path actually written.

    The file holds one ``meta`` JSON string plus, per driven channel, a ``<ch>_segments``
    and a ``<ch>_tones`` float64 table — and, for a channel carrying fading-Shepard
    envelopes, a ``<ch>_env_polys`` table with their fade coordinates (schema
    v:data:`SCHEMA_VERSION_FADE`).  No samples: rendering is
    :func:`aodl.waveform.export.render_samples`.
    """
    path = Path(path)
    arrays: dict[str, Any] = {}
    version = SCHEMA_VERSION
    for name, cw in wfs.channels.items():
        seg, ton, env = _channel_arrays(cw)
        arrays[f"{name}_segments"] = seg
        arrays[f"{name}_tones"] = ton
        if env.shape[0]:
            arrays[f"{name}_env_polys"] = env
            version = SCHEMA_VERSION_FADE
    meta = {
        "schema_version": version,
        "description": wfs.description,
        "params": params_to_dict(wfs.params),
        "channels": list(wfs.channels),
    }
    np.savez(path, meta=json.dumps(meta), **arrays)
    # numpy appends ".npz" when the name lacks it; report the file that really exists.
    return path if path.name.endswith(".npz") else path.with_name(path.name + ".npz")


# --------------------------------------------------------------------------- load


def _poly_from_rows(rows: NDArray[np.float64], where: str) -> PiecewisePoly:
    """Rebuild one tone's frequency law from its ``<ch>_segments`` rows (in file order)."""
    if rows.shape[0] == 0:
        raise ValueError(f"{where}: no segment rows found")
    degrees = np.unique(rows[:, 3].astype(int))
    if degrees.size != 1:
        raise ValueError(f"{where}: segments disagree on the polynomial degree {degrees.tolist()}")
    degree = int(degrees[0])
    if not 0 <= degree <= MAX_DEGREE:
        raise ValueError(f"{where}: degree {degree} outside [0, {MAX_DEGREE}]")
    starts = rows[:, 1]
    breaks = np.empty(rows.shape[0] + 1, dtype=np.float64)
    breaks[:-1] = starts
    breaks[-1] = starts[-1] + rows[-1, 2]
    coeffs = np.ascontiguousarray(rows[:, 4 : 4 + degree + 1])
    return PiecewisePoly(breaks, coeffs)


def _channel_from_arrays(
    name: str,
    seg: NDArray[np.float64],
    ton: NDArray[np.float64],
    env_seg: NDArray[np.float64] | None = None,
) -> ChannelWaveform:
    if seg.ndim != 2 or seg.shape[1] != SEGMENT_COLUMNS:
        raise ValueError(
            f"{name}_segments must have {SEGMENT_COLUMNS} columns, got shape {seg.shape}"
        )
    if ton.ndim != 2 or ton.shape[1] != TONE_COLUMNS:
        raise ValueError(f"{name}_tones must have {TONE_COLUMNS} columns, got shape {ton.shape}")
    if env_seg is not None and (env_seg.ndim != 2 or env_seg.shape[1] != SEGMENT_COLUMNS):
        raise ValueError(
            f"{name}_env_polys must have {SEGMENT_COLUMNS} columns, got shape {env_seg.shape}"
        )
    tones: list[ToneTrack] = []
    for row in ton:
        tone_idx = int(row[0])
        rows = seg[seg[:, 0].astype(int) == tone_idx]
        poly = _poly_from_rows(rows, f"{name} tone {tone_idx}")
        env_poly = None
        if env_seg is not None:
            env_rows = env_seg[env_seg[:, 0].astype(int) == tone_idx]
            if env_rows.shape[0]:
                env_poly = _poly_from_rows(env_rows, f"{name} tone {tone_idx} envelope")
        env = _decode_env(int(row[2]), row[3 : 3 + _ENV_PARAMS], env_poly)
        tones.append(ToneTrack(freq=poly, env=env, phase0=float(row[1])))
    return ChannelWaveform(tuple(tones))


def load(path: str | Path) -> WaveformSet:
    """Read a parametric NPZ written by :func:`save`.

    Raises ``ValueError`` for an unknown ``schema_version`` or ``env_kind``, and for
    structurally malformed files.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data:
            raise ValueError(f"{path}: not an AODL waveform file (no 'meta' entry)")
        meta = json.loads(str(data["meta"].item()))
        version = meta.get("schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"{path}: unsupported schema_version {version!r}; "
                f"this build reads version(s) {list(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        names = list(meta.get("channels", []))
        unknown = [n for n in names if n not in CHANNELS]
        if unknown:
            raise ValueError(f"{path}: unknown channel name(s) {unknown} in metadata")
        channels: dict[str, ChannelWaveform] = {}
        for name in names:
            try:
                seg = np.asarray(data[f"{name}_segments"], dtype=np.float64)
                ton = np.asarray(data[f"{name}_tones"], dtype=np.float64)
            except KeyError as exc:
                raise ValueError(f"{path}: missing array {exc} for channel {name!r}") from exc
            key = f"{name}_env_polys"
            env_seg = np.asarray(data[key], dtype=np.float64) if key in data else None
            channels[name] = _channel_from_arrays(name, seg, ton, env_seg)
    return WaveformSet(
        channels=channels,
        params=params_from_dict(meta["params"]),
        description=str(meta.get("description", "")),
    )


__all__ = [
    "ENV_CONSTANT",
    "ENV_FADE_ZONE",
    "ENV_SMOOTH_ON_OFF",
    "LEGACY_MIXING_ORDER",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_FADE",
    "SEGMENT_COLUMNS",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TONE_COLUMNS",
    "load",
    "params_from_dict",
    "params_to_dict",
    "save",
]
