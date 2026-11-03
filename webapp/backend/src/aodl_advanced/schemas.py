"""Pydantic v2 models of ``webapp/docs/API.md`` §6 — the ``/api/advanced`` JSON contract.

Units are in the field names (``_um``, ``_us``, ``_MHz``, ``_mm``, ``_ns``, ``_mrad``); the
route converts to SI at the boundary.  The shared request models (``PlanRequest``,
``HardwareSpec``) and the packed image come from ``aodl_webapp.schemas`` (W1's frozen list),
which is why this module is not imported by ``aodl_advanced/__init__.py``: the engine itself
does not depend on the web package.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aodl_webapp.schemas import HardwareSpec, PackedImage, PlanRequest

#: Largest number of frame times one ``/advanced/simulate`` call renders.
MAX_TIMES = 24
#: Largest focal patch side [px].
MAX_GRID_N = 192
#: Allowed pupil grid sides.
PUPIL_SIZES = (128, 256, 512)
#: Largest axial scan length.
MAX_NZ = 32
#: Largest comb DFT length ``K``.
MAX_COMB_POINTS = 64
#: Largest number of rendered frequency groups.
MAX_GROUPS = 256

Channel = Literal["Ax", "Bx", "Ay", "By"]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------ §6.1 model


class EfficiencyModelModel(_Request):
    """API.md §6.1 — the TPM Bragg model's parameters."""

    interaction_length_mm: float = Field(default=4.0, gt=0.0)
    f_tpm_MHz: float | None = Field(default=None, gt=0.0)
    refractive_index: float = Field(default=2.26, gt=1.0)
    peak_efficiency: float = Field(default=0.85, gt=0.0, le=1.0)
    angular_coupling: float = Field(default=0.2, ge=-1.0, le=1.0)


# ------------------------------------------------------------------------- §6.2 efficiency


class EfficiencyRequest(_Request):
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    model: EfficiencyModelModel = Field(default_factory=EfficiencyModelModel)
    f_range_MHz: tuple[float, float] | None = None
    n: int = Field(default=241, ge=8, le=4001)

    @model_validator(mode="after")
    def _range_is_increasing(self) -> EfficiencyRequest:
        if self.f_range_MHz is not None and not self.f_range_MHz[0] < self.f_range_MHz[1]:
            raise ValueError("f_range_MHz must be (lo, hi) with lo < hi")
        return self


class PairMap(_Response):
    f_A_MHz: list[float]
    f_B_MHz: list[float]
    eta: list[list[float]]


class Ridge(_Response):
    f_A_MHz: list[float]
    eta_max: list[float]
    f_B_at_max_MHz: list[float]


class EfficiencyResponse(_Response):
    f_MHz: list[float]
    deflection_mrad: list[float]
    eta_first: list[float]
    eta_second_at_center: list[float]
    bandwidth_3dB_MHz: float
    first_aod_usable_MHz: float
    band_MHz: tuple[float, float]
    f_tpm_MHz: float
    pair: PairMap
    ridge: Ridge
    assumptions: str


# ------------------------------------------------------------- §6.3 trajectory efficiency


class TrajectoryEfficiencyRequest(_Request):
    plan: PlanRequest
    model: EfficiencyModelModel = Field(default_factory=EfficiencyModelModel)
    n: int = Field(default=300, ge=2, le=2000)


class TrajectoryEfficiencyResponse(_Response):
    t_us: list[float]
    per_channel: dict[str, list[float]]
    product: list[float]
    min_product: float
    assumptions: str


# ----------------------------------------------------------------------- §6.4 misalignment


class AODMisalignmentModel(_Request):
    """API.md §6.4 — one AOD's seven knobs; all default to the aligned device."""

    offset_par_um: float = 0.0
    offset_perp_um: float = 0.0
    rotation_mrad: float = 0.0
    delay_ns: float = 0.0
    velocity_scale: float = Field(default=1.0, gt=0.0)
    tilt_x_mrad: float = 0.0
    tilt_y_mrad: float = 0.0


class MisalignmentModel(_Request):
    """The four channels' misalignment; any channel may be omitted (aligned)."""

    Ax: AODMisalignmentModel = Field(default_factory=AODMisalignmentModel)
    Bx: AODMisalignmentModel = Field(default_factory=AODMisalignmentModel)
    Ay: AODMisalignmentModel = Field(default_factory=AODMisalignmentModel)
    By: AODMisalignmentModel = Field(default_factory=AODMisalignmentModel)


# --------------------------------------------------------------------------- §6.5 simulate


class GridCenter(_Request):
    x_um: float
    y_um: float


class AdvancedGrid(_Request):
    half_x_um: float = Field(default=6.0, gt=0.0)
    half_y_um: float = Field(default=6.0, gt=0.0)
    n: int = Field(default=96, ge=8, le=MAX_GRID_N)
    center: Literal["tracked"] | GridCenter = "tracked"


class AdvancedXz(_Request):
    nz: int = Field(default=24, ge=2, le=MAX_NZ)
    z_half_um: float = Field(default=8.0, gt=0.0)


class AdvancedModel(_Request):
    pupil_n: Literal[128, 256, 512] = 256
    span_w_in: float = Field(default=2.5, ge=1.5, le=6.0)
    order_model: Literal["j1", "weak"] = "j1"
    comb_dft_points: int = Field(default=16, ge=1, le=MAX_COMB_POINTS)
    max_groups: int = Field(default=128, ge=1, le=MAX_GROUPS)
    efficiency: EfficiencyModelModel | None = None


class AdvancedSimulateRequest(_Request):
    plan: PlanRequest
    misalignment: MisalignmentModel = Field(default_factory=MisalignmentModel)
    times_us: list[float] = Field(min_length=1, max_length=MAX_TIMES)
    grid: AdvancedGrid = Field(default_factory=AdvancedGrid)
    xz: AdvancedXz | None = None
    model: AdvancedModel = Field(default_factory=AdvancedModel)
    reference: bool = True


class AdvImageXY(_Response):
    x_um: list[float]
    y_um: list[float]
    image: PackedImage


class AdvImageXZ(_Response):
    x_um: list[float]
    z_um: list[float]
    row_y_um: float
    image: PackedImage


class AdvMetricsModel(_Response):
    """API.md §6.5 ``AdvMetrics``; ``*_rel``/``strehl`` are relative to the aligned reference at
    the same time (``null`` without one), ``z_best_*``/``astig_um`` need an ``xz`` scan with
    ``nz >= 16``."""

    x_um: float
    y_um: float
    wx_um: float
    wy_um: float
    z_best_x_um: float | None
    z_best_y_um: float | None
    astig_um: float | None
    peak_rel: float | None
    power_rel: float | None
    strehl: float | None


class AdvFrame(_Response):
    time_us: float
    plane_z_um: float
    n_groups: int
    groups_dropped: int
    xy: AdvImageXY
    xz: AdvImageXZ | None
    metrics: AdvMetricsModel
    reference_metrics: AdvMetricsModel | None


class StaticOffset(_Response):
    x: float
    y: float


class AdvancedSimulateResponse(_Response):
    plan_id: str
    frames: list[AdvFrame]
    static_offset_um: dict[str, StaticOffset]
    notes: list[str]


# --------------------------------------------------------------------------- §6.6 spectrum


class SpectrumRequest(_Request):
    plan: PlanRequest
    channel: Channel = "Ax"
    t_center_us: float
    window_us: float = Field(default=20.0, gt=0.0, le=2000.0)
    sample_rate_MSps: float = Field(default=625.0, gt=0.0, le=10000.0)


class SpectrumResponse(_Response):
    f_MHz: list[float]
    psd_dB: list[float]
    crest_factor: float
    papr_dB: float
    peak_depth_rad: float
    band_MHz: tuple[float, float]
    notes: list[str]


__all__ = [
    "MAX_COMB_POINTS",
    "MAX_GRID_N",
    "MAX_GROUPS",
    "MAX_NZ",
    "MAX_TIMES",
    "PUPIL_SIZES",
    "AODMisalignmentModel",
    "AdvFrame",
    "AdvImageXY",
    "AdvImageXZ",
    "AdvMetricsModel",
    "AdvancedGrid",
    "AdvancedModel",
    "AdvancedSimulateRequest",
    "AdvancedSimulateResponse",
    "AdvancedXz",
    "EfficiencyModelModel",
    "EfficiencyRequest",
    "EfficiencyResponse",
    "GridCenter",
    "MisalignmentModel",
    "PairMap",
    "Ridge",
    "SpectrumRequest",
    "SpectrumResponse",
    "StaticOffset",
    "TrajectoryEfficiencyRequest",
    "TrajectoryEfficiencyResponse",
]
