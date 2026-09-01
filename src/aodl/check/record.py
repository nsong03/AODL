r"""The checker's input boundary: literal AWG sample buffers (``docs/PLAN.md`` §1.1).

M6 verifies the simulator by rebuilding the tweezers from the **rendered RF samples** — the
buffers a Spectrum card would actually clock out — instead of from the parametric waveform
IR.  A :class:`SampleRecord` is therefore the only thing this package's checker ever learns
about a drive besides :class:`~aodl.params.AODLParams` (and, in WO-22, the requested
trajectory): four real arrays, a sample rate, a start time and one scale factor.

**Normalization is load-bearing.**  :func:`aodl.waveform.export.render_samples` divides
*all* channels by a single global peak so the full-scale sample is ``1.0`` on exactly one
channel, and reports that peak as the ``normalization`` factor.  The phase modulation a
channel imprints on the light is (Eq. S1)

.. math::

    C\,V_\mu(t) = \text{drive\_strength} \times \text{normalization} \times \text{sample},

so a checker that forgets the factor mis-scales the *nonlinear* pupil model by exactly the
drive's crest factor — 2.70 for a 3x3 fading-Shepard drive at the product defaults, which
evaluates the ``2 J_1(C)/C`` fundamental compression at ``C = 0.30`` instead of ``0.81`` and
so reports a 1.1 % loss where the crystal has 8.0 %.  The linear (``weak``) model is immune,
which is precisely why forgetting it is a silent error.  :attr:`SampleRecord.drive` is the
one place the factor is applied; nothing downstream re-applies it.

**Timing.**  ``t = 0`` is the instant the drive starts (``docs/conventions.md`` §7), which
is what makes ``retarded_time(t, u, ...) < 0`` mean "this part of the aperture holds no
sound yet".  ``t_start`` is the time of sample 0 and is normally ``0.0``; a record that
starts later simply has no content for the earlier aperture positions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..params import CHANNELS, AODLParams
from ..waveform.export import DEFAULT_SAMPLE_RATE, SAMPLES_SCHEMA_VERSION, SAMPLES_SUFFIX

Float = NDArray[np.float64]

#: Fewest samples a record may carry: the cubic-Hermite gather of
#: :func:`aodl.check.demod.sample_baseband` reads four neighbours.
MIN_SAMPLES = 4

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "MIN_SAMPLES",
    "SampleRecord",
    "from_arrays",
    "load_samples",
]


@dataclass(frozen=True)
class SampleRecord:
    """Rendered RF sample buffers for one drive, plus everything needed to read them.

    Attributes
    ----------
    channels:
        ``{channel name: samples}``; equal-length 1D ``float64`` arrays of the *normalized*
        samples, exactly as :func:`aodl.waveform.export.render_samples` produced them.  Names
        are a subset of :data:`aodl.params.CHANNELS`; an absent channel is undriven.
    sample_rate:
        Samples per second [S/s].  Sample ``k`` sits at ``t_start + k / sample_rate``.
    t_start:
        Time of sample 0 [s].  ``t = 0`` is the drive start (``docs/conventions.md`` §7).
    params:
        Hardware the drive was rendered for — carriers, band, aperture, drive strength.
    normalization:
        Multiply the samples by this to recover the Eq. S1-unit drive ``V`` (see the module
        docstring).  ``1.0`` means the samples already carry physical units.
    """

    channels: dict[str, Float]
    sample_rate: float
    t_start: float
    params: AODLParams
    normalization: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.params, AODLParams):
            raise TypeError(f"SampleRecord.params must be an AODLParams, got {type(self.params)!r}")
        if not isinstance(self.channels, Mapping):
            raise TypeError("SampleRecord.channels must be a mapping {name: samples}")
        if not self.channels:
            raise ValueError("SampleRecord needs at least one channel")
        unknown = [name for name in self.channels if name not in CHANNELS]
        if unknown:
            raise ValueError(f"unknown channel name(s) {unknown}; valid names are {list(CHANNELS)}")
        arrays: dict[str, Float] = {}
        length: int | None = None
        for name, samples in self.channels.items():
            arr = np.array(samples, dtype=np.float64, copy=True)
            if arr.ndim != 1:
                raise ValueError(f"channel {name!r} samples must be 1-D, got shape {arr.shape}")
            if arr.size < MIN_SAMPLES:
                raise ValueError(
                    f"channel {name!r} carries {arr.size} samples; at least {MIN_SAMPLES} are "
                    "needed for the cubic-Hermite baseband gather"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"channel {name!r} samples must all be finite")
            if length is None:
                length = arr.size
            elif arr.size != length:
                raise ValueError(
                    "all channels must carry the same number of samples (they share one clock); "
                    f"got { ({k: np.size(v) for k, v in self.channels.items()}) }"
                )
            arr.flags.writeable = False
            arrays[name] = arr
        rate = float(self.sample_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"sample_rate must be positive and finite, got {self.sample_rate!r}")
        start = float(self.t_start)
        if not math.isfinite(start):
            raise ValueError(f"t_start must be finite, got {self.t_start!r}")
        scale = float(self.normalization)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"normalization must be positive and finite, got {self.normalization!r} — it is "
                "the global peak render_samples divided out, so the physical drive is "
                "samples * normalization"
            )
        object.__setattr__(self, "channels", arrays)
        object.__setattr__(self, "sample_rate", rate)
        object.__setattr__(self, "t_start", start)
        object.__setattr__(self, "normalization", scale)

    @property
    def n_samples(self) -> int:
        """Samples per channel."""
        return int(next(iter(self.channels.values())).size)

    @property
    def t_span(self) -> tuple[float, float]:
        """``(t_start, t_last)`` [s] — the closed span the record covers."""
        return self.t_start, self.t_start + (self.n_samples - 1) / self.sample_rate

    def times(self) -> Float:
        """Sample times [s] (``t_start + k / sample_rate``)."""
        return self.t_start + np.arange(self.n_samples, dtype=np.float64) / self.sample_rate

    def drive(self, channel: str) -> Float:
        """The Eq. S1-unit drive ``V_mu`` [same units as ``drive_strength`` expects].

        ``samples * normalization`` — see the module docstring.  This is the only place the
        normalization factor is applied.
        """
        try:
            samples = self.channels[channel]
        except KeyError:
            raise KeyError(
                f"channel {channel!r} is not in this record; it carries {tuple(self.channels)}"
            ) from None
        return samples * self.normalization


def from_arrays(
    samples: Mapping[str, ArrayLike],
    sample_rate: float,
    params: AODLParams,
    *,
    t_start: float = 0.0,
    normalization: float = 1.0,
) -> SampleRecord:
    """Build a :class:`SampleRecord` from in-memory arrays.

    The companion of ``render_samples(..., return_scale=True)``::

        arrays, scale = render_samples(wfs, rate, dtype=np.float64, return_scale=True)
        rec = from_arrays(arrays, rate, wfs.params, normalization=scale)

    Passing ``normalization=1.0`` when the arrays *were* normalized is the one mistake this
    boundary cannot detect for you (see the module docstring).
    """
    return SampleRecord(
        channels={name: np.asarray(arr, dtype=np.float64) for name, arr in samples.items()},
        sample_rate=float(sample_rate),
        t_start=float(t_start),
        params=params,
        normalization=float(normalization),
    )


def load_samples(path: str | Path, params: AODLParams) -> SampleRecord:
    """Read a ``*_samples.npz`` file written by :func:`aodl.waveform.export.save_samples`.

    Schema 1 (``SAMPLES_SCHEMA_VERSION``): one array per channel plus a ``meta`` JSON blob
    carrying ``sample_rate``, ``t_start``, ``normalization`` and the per-channel carriers.

    ``params`` supplies the hardware the checker will model with.  The file's own
    ``f_center`` values are compared against it and a mismatch **raises**: a carrier offset
    by even a kilohertz silently rotates every rebuilt pupil, so this is the one place it can
    be caught.
    """
    path = Path(path)
    if not path.name.endswith(SAMPLES_SUFFIX):
        raise ValueError(
            f"{path.name!r} is not a rendered-sample file (expected '*{SAMPLES_SUFFIX}'); the "
            "parametric waveform NPZ holds segment parameters, not samples, and the checker "
            "deliberately never reads it"
        )
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data.files:
            raise ValueError(f"{path.name!r} has no 'meta' block — not a save_samples() file")
        meta: dict[str, Any] = json.loads(str(data["meta"].item()))
        version = int(meta.get("schema_version", -1))
        if version != SAMPLES_SCHEMA_VERSION:
            raise ValueError(
                f"{path.name!r} carries samples schema version {version}, this checker reads "
                f"version {SAMPLES_SCHEMA_VERSION}"
            )
        if meta.get("kind") != "samples":
            raise ValueError(f"{path.name!r} has kind {meta.get('kind')!r}, expected 'samples'")
        names = list(meta.get("channels", []))
        missing = [name for name in names if name not in data.files]
        if missing:
            raise ValueError(f"{path.name!r} names channels {missing} but carries no such arrays")
        arrays = {name: np.asarray(data[name], dtype=np.float64) for name in names}

    carriers: dict[str, float] = dict(meta.get("f_center", {}))
    for name in names:
        if name not in CHANNELS:
            raise ValueError(
                f"{path.name!r} names channel {name!r}; valid names are {list(CHANNELS)}"
            )
        if name not in carriers:
            raise ValueError(f"{path.name!r} records no f_center for channel {name!r}")
        want = params.channels[name].f_center
        got = float(carriers[name])
        if got != want:
            raise ValueError(
                f"carrier mismatch on channel {name!r}: the file was rendered at "
                f"f_center = {got!r} Hz but the params passed to load_samples say {want!r} Hz.  "
                "Demodulating at the wrong carrier rotates the rebuilt pupil by "
                "2 pi (f_file - f_params) t_ret, so this is refused rather than approximated."
            )
    return SampleRecord(
        channels=arrays,
        sample_rate=float(meta["sample_rate"]),
        t_start=float(meta.get("t_start", 0.0)),
        params=params,
        normalization=float(meta.get("normalization", 1.0)),
    )
