"""Pydantic v2 models of ``webapp/docs/API.md`` — these classes **are** the JSON contract.

Units are in the field names (``_um``, ``_us``, ``_MHz``, ``_mm``, ``_nm``, ``_m_s``); SI
conversion happens at the boundary in :mod:`aodl_webapp.hardware` and
:mod:`aodl_webapp.trajectory`, never here.  Request models forbid unknown fields so a typo on
the client side is a 422, not a silently ignored option.  ``NaN`` floats (the instantaneous
frequency where an aperture is empty) serialise as JSON ``null`` — pydantic's default — and
the corresponding fields are typed ``float | None`` to say so.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import MAX_FRAMES, MAX_IMAGE_SIDE

Channel = Literal["Ax", "Bx", "Ay", "By"]
ProfileName = Literal[
    "min_jerk", "constant_jerk", "constant_accel", "switching_constant_jerk", "linear"
]

#: Ramp profiles offered by the UI, in :data:`aodl.trajectory.ramps.RAMPS` order.
PROFILES: tuple[str, ...] = (
    "min_jerk",
    "constant_jerk",
    "constant_accel",
    "switching_constant_jerk",
    "linear",
)


class _Request(BaseModel):
    """Base of every request model: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class _Response(BaseModel):
    """Base of every response model."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ §3 shared request models


class HardwareSpec(_Request):
    """API.md §3.1 — one :class:`aodl.AODLParams` with identical AODs on the four channels."""

    wavelength_nm: float = Field(default=1030.0, gt=0.0)
    focal_length_mm: float = Field(default=6.5, gt=0.0)
    w_in_mm: float = Field(default=2.0, gt=0.0)
    sound_speed_m_s: float = Field(default=650.0, gt=0.0)
    aperture_mm: float = Field(default=7.5, gt=0.0)
    f_center_MHz: float = Field(default=100.0, gt=0.0)
    band_MHz: tuple[float, float] = (90.0, 110.0)
    drive_strength: float = Field(default=0.3, gt=0.0)
    mixing_order: Literal[1, 3] = 3

    @model_validator(mode="after")
    def _band_is_increasing(self) -> HardwareSpec:
        lo, hi = self.band_MHz
        if not (lo > 0.0 and lo < hi):
            raise ValueError(f"band_MHz must be (lo, hi) with 0 < lo < hi, got {self.band_MHz!r}")
        return self


class ArraySpecModel(_Request):
    """API.md §3.2 — ``mx x my`` tweezers at the given tone spacings."""

    mx: int = Field(default=1, ge=1)
    my: int = Field(default=1, ge=1)
    delta_f_x_MHz: float = 1.0
    delta_f_y_MHz: float = 1.3


class LiftMove(_Request):
    kind: Literal["lift"]
    dz_um: float
    duration_us: float = Field(gt=0.0)
    profile: ProfileName = "min_jerk"


class TranslateMove(_Request):
    kind: Literal["translate"]
    dx_um: float
    dy_um: float
    duration_us: float = Field(gt=0.0)
    profile: ProfileName = "min_jerk"


class HoldMove(_Request):
    kind: Literal["hold"]
    duration_us: float = Field(gt=0.0)


class PathMove(_Request):
    """API.md §3.3 — waypoints relative to where the path starts (DESIGN §1.3)."""

    kind: Literal["path"]
    points_um: list[tuple[float, float, float]] = Field(min_length=1)
    duration_us: float = Field(gt=0.0)
    closed: bool = False


MoveSpec = Annotated[LiftMove | TranslateMove | HoldMove | PathMove, Field(discriminator="kind")]


class PlanOptions(_Request):
    """API.md §3.4 — the ``plan_motion`` options exposed by the UI."""

    shepard: Literal["auto", "never", "always"] = "auto"
    shepard_delta_f_MHz: tuple[float, float] | None = None
    eta: float = Field(default=0.5, gt=0.0, le=0.5)
    phases: Literal["schroeder", "zero", "random"] = "schroeder"
    phase_seed: int = Field(default=0, ge=0)
    amp: float = Field(default=1.0, gt=0.0, le=1.0)
    retard_compensate: bool = False
    f_z_bias: Literal["none", "auto"] = "none"
    t_pad_us: float | None = Field(default=None, ge=0.0)
    check_band: bool = True

    @model_validator(mode="after")
    def _always_needs_spacings(self) -> PlanOptions:
        if self.shepard == "always":
            if self.shepard_delta_f_MHz is None:
                raise ValueError("shepard='always' requires shepard_delta_f_MHz = [dfx, dfy]")
            if any(not v > 0.0 for v in self.shepard_delta_f_MHz):
                raise ValueError("shepard_delta_f_MHz entries must be positive")
        return self


class PlanRequest(_Request):
    """API.md §3.5 — the whole input; every route takes it (the API is stateless)."""

    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    array: ArraySpecModel = Field(default_factory=ArraySpecModel)
    moves: list[MoveSpec] = Field(min_length=1)
    options: PlanOptions = Field(default_factory=PlanOptions)


# ------------------------------------------------------------------------ §4 plan response


class BandUsage(_Response):
    f_min_MHz: float
    f_max_MHz: float
    margin_MHz: float


class ZBudget(_Response):
    """``requested`` and the Eq. 1 ``ceiling`` [m·s]; ``ceiling`` is ``null`` when unbounded
    (fading-Shepard).  ``fraction`` is always finite: requested over the *plain Eq. 1* ceiling
    (doubled under ``f_z_bias``), so under Shepard it reads how far past the budget the request
    went (e.g. ``2.9`` = 290 %)."""

    requested_m_s: float
    ceiling_m_s: float | None
    fraction: float


class FadeEventModel(_Response):
    time_us: float
    axis: Literal["x", "y"]
    shadow_um: float


class PlanConstants(_Response):
    """Derived hardware numbers the frontend draws with (API.md §4 ``constants``)."""

    tau_us: float
    half_tau_us: float
    waist0_um: float
    rayleigh_um: float
    um_per_MHz: float
    lens_scale_m_s: float
    chirp_per_um_MHz_per_ms: float
    beam_transit_us: float
    eq1_hold_10um_us: float


class TrajectorySamples(_Response):
    """The *requested* ``X, Y, Z`` on a time grid, plus the move seams."""

    t_us: list[float]
    x_um: list[float]
    y_um: list[float]
    z_um: list[float]
    seams_us: list[float]


class Poly(_Response):
    """API.md §4.1 — normalized-local-time piecewise polynomial, clamp-held outside."""

    breaks_us: list[float]
    coeffs: list[list[float]]


class ConstantEnv(_Response):
    kind: Literal["constant"] = "constant"
    amp: float


class SmoothOnOffEnv(_Response):
    kind: Literal["smooth_on_off"] = "smooth_on_off"
    t_on_us: float
    t_off_us: float
    ramp_us: float


class FadeZoneEnv(_Response):
    kind: Literal["fade_zone"] = "fade_zone"
    g: Poly
    delta_f_MHz: float
    eta: float
    p: float
    m: int
    amp: float


EnvelopeModel = Annotated[ConstantEnv | SmoothOnOffEnv | FadeZoneEnv, Field(discriminator="kind")]


class ToneIR(_Response):
    freq: Poly
    phase: Poly
    phase0: float
    env: EnvelopeModel


class ChannelIR(_Response):
    f_center_MHz: float
    tau_us: float
    sound_sign: Literal[-1, 1]
    axis: Literal["x", "y"]
    aperture_mm: float
    sound_speed_m_s: float
    band_MHz: tuple[float, float]
    tones: list[ToneIR]


class WaveformIR(_Response):
    channels: dict[str, ChannelIR]
    t_span_us: tuple[float, float]


class PlanResponse(_Response):
    plan_id: str
    mode: Literal["s19", "shepard"]
    description: str
    summary_text: str
    band_usage: dict[str, BandUsage]
    tone_counts: dict[str, int]
    z_budget: ZBudget
    fade_events: list[FadeEventModel]
    notes: list[str]
    constants: PlanConstants
    t_span_us: tuple[float, float]
    duration_us: float
    seams_us: list[float]
    trajectory: TrajectorySamples
    waveform_ir: WaveformIR
    code_python: str


# ---------------------------------------------------------------------- §5 basic-mode routes


class Spot(_Response):
    """One spatial cluster of one frequency group (API.md §5.1), scalars."""

    x_um: float
    y_um: float
    z_um: float
    delta_f_um: float
    sigma_astig: float
    wx_um: float
    wy_um: float
    power: float
    power_coherent: float
    df_opt_MHz: float
    group: int
    n_terms: int


class SpotFrame(_Response):
    """The same twelve fields as equal-length lists — struct-of-arrays."""

    x_um: list[float]
    y_um: list[float]
    z_um: list[float]
    delta_f_um: list[float]
    sigma_astig: list[float]
    wx_um: list[float]
    wy_um: list[float]
    power: list[float]
    power_coherent: list[float]
    df_opt_MHz: list[float]
    group: list[int]
    n_terms: list[int]


class SimulateRequest(_Request):
    plan: PlanRequest
    n_frames: int = Field(default=60, ge=2, le=MAX_FRAMES)
    t_start_us: float | None = None
    t_end_us: float | None = None
    movie_mixing_order: Literal[1, 3] | None = None


class Bounds(_Response):
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float


class SimulateResponse(_Response):
    plan_id: str
    times_us: list[float]
    tracked_z_um: list[float]
    frames: list[SpotFrame]
    constants: PlanConstants
    bounds_um: Bounds
    power_ref: float
    notes: list[str]


class FrameGridModel(_Request):
    x0_um: float
    x1_um: float
    nx: int = Field(ge=2, le=MAX_IMAGE_SIDE)
    y0_um: float
    y1_um: float
    ny: int = Field(ge=2, le=MAX_IMAGE_SIDE)

    @model_validator(mode="after")
    def _increasing(self) -> FrameGridModel:
        if not (self.x1_um > self.x0_um and self.y1_um > self.y0_um):
            raise ValueError("grid bounds must satisfy x0 < x1 and y0 < y1")
        return self


class XzModel(_Request):
    nx: int = Field(default=160, ge=2, le=MAX_IMAGE_SIDE)
    nz: int = Field(default=96, ge=2, le=MAX_IMAGE_SIDE)
    z_half_um: float = Field(default=8.0, gt=0.0)
    row_y_um: float | None = None


class PlaneZ(_Request):
    z_um: float


class PackedImage(_Response):
    """API.md §0.3 — little-endian uint16, row-major, ``value = peak * code / 65535``."""

    shape: tuple[int, int]
    peak: float
    data: str


class ImageXY(_Response):
    x_um: list[float]
    y_um: list[float]
    image: PackedImage


class ImageXZ(_Response):
    x_um: list[float]
    z_um: list[float]
    row_y_um: float
    image: PackedImage


class CutPoint(_Response):
    x_um: float
    y_um: float


class Cuts(_Response):
    """API.md §5.2 — closed-form line cuts through the brightest spot (SI power per m²).

    ``i_x`` runs over ``x_um`` (the XY row nearest ``through.y_um``), ``i_y`` over the frame's
    ``xy.y_um`` (the XY column nearest ``through.x_um``), ``i_z`` over ``z_um``.
    """

    x_um: list[float]
    i_x: list[float]
    i_y: list[float]
    z_um: list[float]
    i_z: list[float]
    through: CutPoint


class FrameRequest(_Request):
    plan: PlanRequest
    time_us: float
    plane: Literal["tracked", "fixed"] | PlaneZ = "tracked"
    grid: FrameGridModel
    xz: XzModel | None = None
    movie_mixing_order: Literal[1, 3] | None = None


class FrameResponse(_Response):
    plan_id: str
    time_us: float
    plane_z_um: float
    xy: ImageXY
    xz: ImageXZ | None
    cuts: Cuts
    spots: list[Spot]
    #: ``sum 2 P / (pi wx wy)`` over the spots within one waist of the brightest (SI W/m²
    #: scale of ``xy.image.peak``); ``0`` for a dark frame.
    gauss_peak: float
    #: ``max`` of the XY image (``== xy.image.peak``), the exact closed-form peak.
    exact_peak: float


class ApertureRequest(_Request):
    plan: PlanRequest
    time_us: float
    n: int = Field(default=512, ge=16, le=8192)
    include_voltage: bool = False
    voltage_window_mm: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _window_is_increasing(self) -> ApertureRequest:
        if self.voltage_window_mm is not None:
            u0, u1 = self.voltage_window_mm
            if not u1 > u0:
                raise ValueError("voltage_window_mm must be (u0, u1) with u0 < u1")
        return self


class ApertureVoltage(_Response):
    u_mm: list[float]
    v: list[float]


class ApertureChannel(_Response):
    u_mm: list[float]
    envelope: list[float]
    f_inst_MHz: list[float | None]
    filled: list[bool]
    t_ret_us: list[float]
    voltage: ApertureVoltage | None


class ApertureResponse(_Response):
    time_us: float
    channels: dict[str, ApertureChannel]


# ------------------------------------------------------------------------ §1, §2, §5.5, §0.1


class HealthResponse(_Response):
    status: str
    aodl_version: str
    version: str


class ExampleSpec(_Response):
    id: str
    title: str
    blurb: str
    tags: list[str]
    request: PlanRequest


class PresetsResponse(_Response):
    hardware_presets: dict[str, HardwareSpec]
    profiles: list[str]
    examples: list[ExampleSpec]


class CodeResponse(_Response):
    code_python: str


class ErrorDetail(_Response):
    code: str
    message: str
    detail: Any = None


class ErrorBody(_Response):
    error: ErrorDetail


__all__ = [
    "PROFILES",
    "ApertureChannel",
    "ApertureRequest",
    "ApertureResponse",
    "ApertureVoltage",
    "ArraySpecModel",
    "BandUsage",
    "Bounds",
    "Channel",
    "ChannelIR",
    "CodeResponse",
    "ConstantEnv",
    "CutPoint",
    "Cuts",
    "EnvelopeModel",
    "ErrorBody",
    "ErrorDetail",
    "ExampleSpec",
    "FadeEventModel",
    "FadeZoneEnv",
    "FrameGridModel",
    "FrameRequest",
    "FrameResponse",
    "HardwareSpec",
    "HealthResponse",
    "HoldMove",
    "ImageXY",
    "ImageXZ",
    "LiftMove",
    "MoveSpec",
    "PackedImage",
    "PathMove",
    "PlanConstants",
    "PlanOptions",
    "PlanRequest",
    "PlanResponse",
    "PlaneZ",
    "Poly",
    "PresetsResponse",
    "ProfileName",
    "SimulateRequest",
    "SimulateResponse",
    "SmoothOnOffEnv",
    "Spot",
    "SpotFrame",
    "ToneIR",
    "TrajectorySamples",
    "TranslateMove",
    "WaveformIR",
    "XzModel",
    "ZBudget",
]
