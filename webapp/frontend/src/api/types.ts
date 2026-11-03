/**
 * TypeScript mirror of `webapp/docs/API.md` (normative). Field names and units are the
 * API's display units: `_um`, `_us`, `_ns`, `_MHz`, `_mm`, `_nm`, `_mrad`, `_m_s`.
 * Keep this file in lock-step with `aodl_webapp/schemas.py` / `aodl_advanced/schemas.py`.
 */

// ---------------------------------------------------------------- §0 conventions

export type ChannelName = 'Ax' | 'Bx' | 'Ay' | 'By'
/** API.md §0.4 — always this order. */
export const CHANNELS: readonly ChannelName[] = ['Ax', 'Bx', 'Ay', 'By'] as const

export type ErrorCode =
  | 'band_violation'
  | 'invalid_spec'
  | 'too_many_terms'
  | 'plan_not_cached'
  | 'coverage'
  | 'unsupported'

/** API.md §0.1 — the HTTP 400 body. */
export interface ErrorBody {
  error: { code: ErrorCode | string; message: string; detail: unknown }
}

/** API.md §0.3 — packed uint16 image. */
export interface PackedImage {
  shape: [number, number] // [ny, nx]
  peak: number
  data: string // base64 little-endian uint16, row-major (row 0 = y0)
}

// ---------------------------------------------------------------- §1–§2

export interface HealthResponse {
  status: string
  aodl_version: string
  version: string
}

export interface ExampleSpec {
  id: string
  title: string
  blurb: string
  tags: string[]
  request: PlanRequest
}

export type ProfileName =
  | 'min_jerk'
  | 'constant_jerk'
  | 'constant_accel'
  | 'switching_constant_jerk'
  | 'linear'

export interface PresetsResponse {
  hardware_presets: Record<string, HardwareSpec>
  profiles: ProfileName[]
  examples: ExampleSpec[]
}

// ---------------------------------------------------------------- §3 shared request models

export interface HardwareSpec {
  wavelength_nm: number
  focal_length_mm: number
  w_in_mm: number
  sound_speed_m_s: number
  aperture_mm: number
  f_center_MHz: number
  band_MHz: [number, number]
  drive_strength: number
  mixing_order: 1 | 3
}

export interface ArraySpec {
  mx: number
  my: number
  delta_f_x_MHz: number
  delta_f_y_MHz: number
}

export interface LiftMove {
  kind: 'lift'
  dz_um: number
  duration_us: number
  profile: ProfileName
}
export interface TranslateMove {
  kind: 'translate'
  dx_um: number
  dy_um: number
  duration_us: number
  profile: ProfileName
}
export interface HoldMove {
  kind: 'hold'
  duration_us: number
}
export interface PathMove {
  kind: 'path'
  points_um: [number, number, number][]
  duration_us: number
  closed: boolean
}
export type MoveSpec = LiftMove | TranslateMove | HoldMove | PathMove
export type MoveKind = MoveSpec['kind']

export interface PlanOptions {
  shepard: 'auto' | 'never' | 'always'
  shepard_delta_f_MHz: [number, number] | null
  eta: number
  phases: 'schroeder' | 'zero' | 'random'
  phase_seed: number
  amp: number
  retard_compensate: boolean
  f_z_bias: 'none' | 'auto'
  t_pad_us: number | null
  check_band: boolean
}

export interface PlanRequest {
  hardware: HardwareSpec
  array: ArraySpec
  moves: MoveSpec[]
  options: PlanOptions
}

// ---------------------------------------------------------------- §4 plan response

export interface BandUsage {
  f_min_MHz: number
  f_max_MHz: number
  margin_MHz: number
}

export interface ZBudget {
  requested_m_s: number
  ceiling_m_s: number | null
  fraction: number
}

export interface FadeEvent {
  time_us: number
  axis: 'x' | 'y'
  shadow_um: number
}

export interface PlanConstants {
  tau_us: number
  half_tau_us: number
  waist0_um: number
  rayleigh_um: number
  um_per_MHz: number
  lens_scale_m_s: number
  chirp_per_um_MHz_per_ms: number
  beam_transit_us: number
  eq1_hold_10um_us: number
}

export interface TrajectorySamples {
  t_us: number[]
  x_um: number[]
  y_um: number[]
  z_um: number[]
  seams_us?: number[] // repeated from PlanResponse.seams_us
}

/** API.md §4.1 — normalized-local-time piecewise polynomial, clamp-held outside its domain. */
export interface Poly {
  breaks_us: number[] // K+1
  coeffs: number[][] // K rows × (D+1)
}

export interface ConstantEnv {
  kind: 'constant'
  amp: number
}
export interface SmoothOnOffEnv {
  kind: 'smooth_on_off'
  t_on_us: number
  t_off_us: number
  ramp_us: number
}
export interface FadeZoneEnv {
  kind: 'fade_zone'
  g: Poly // fade coordinate [MHz]
  delta_f_MHz: number
  eta: number
  p: number
  m: number
  amp: number
}
export type Envelope = ConstantEnv | SmoothOnOffEnv | FadeZoneEnv

export interface ToneIR {
  freq: Poly // detuning from f_center [MHz]
  phase: Poly // [rad], already 2π ∫ f dt
  phase0: number // [rad]
  env: Envelope
}

export interface ChannelIR {
  f_center_MHz: number
  tau_us: number
  sound_sign: 1 | -1
  axis: 'x' | 'y'
  aperture_mm: number
  sound_speed_m_s: number
  band_MHz: [number, number]
  tones: ToneIR[]
}

export interface WaveformIR {
  channels: Record<ChannelName, ChannelIR>
  t_span_us: [number, number]
}

export type PlanMode = 's19' | 'shepard'

export interface PlanResponse {
  plan_id: string
  mode: PlanMode
  description: string
  summary_text: string
  band_usage: Record<ChannelName, BandUsage>
  tone_counts: Record<ChannelName, number>
  z_budget: ZBudget
  fade_events: FadeEvent[]
  notes: string[]
  constants: PlanConstants
  t_span_us: [number, number]
  duration_us: number
  seams_us: number[]
  trajectory: TrajectorySamples
  waveform_ir: WaveformIR
  code_python: string
}

// ---------------------------------------------------------------- §5 basic-mode routes

export interface SimulateRequest {
  plan: PlanRequest
  n_frames: number
  t_start_us: number | null
  t_end_us: number | null
  movie_mixing_order: 1 | 3 | null
}

/** API.md §5.1 — struct-of-arrays; one entry per spatial cluster. */
export interface SpotFrame {
  x_um: number[]
  y_um: number[]
  z_um: number[]
  delta_f_um: number[]
  sigma_astig: number[]
  wx_um: number[]
  wy_um: number[]
  power: number[]
  power_coherent: number[]
  df_opt_MHz: number[]
  group: number[]
  n_terms: number[]
}

/** The same twelve fields as scalars (used by /api/frame and /api/advanced). */
export interface Spot {
  x_um: number
  y_um: number
  z_um: number
  delta_f_um: number
  sigma_astig: number
  wx_um: number
  wy_um: number
  power: number
  power_coherent: number
  df_opt_MHz: number
  group: number
  n_terms: number
}

export interface Bounds {
  x0: number
  x1: number
  y0: number
  y1: number
  z0: number
  z1: number
}

export interface SimulateResponse {
  plan_id: string
  times_us: number[]
  tracked_z_um: number[]
  frames: SpotFrame[]
  constants: PlanConstants // the full §4 block (superset of waist0_um, rayleigh_um, tau_us)
  bounds_um: Bounds
  power_ref: number
  notes: string[]
}

export type PlaneSpec = 'tracked' | 'fixed' | { z_um: number }

export interface FrameGridSpec {
  x0_um: number
  x1_um: number
  nx: number
  y0_um: number
  y1_um: number
  ny: number
}

export interface XzSpec {
  nx: number
  nz: number
  z_half_um: number
  row_y_um: number | null
}

export interface FrameRequest {
  plan: PlanRequest
  time_us: number
  plane: PlaneSpec
  grid: FrameGridSpec
  xz: XzSpec | null
  movie_mixing_order: 1 | 3 | null
}

export interface ImageXY {
  x_um: number[]
  y_um: number[]
  image: PackedImage
}
export interface ImageXZ {
  x_um: number[]
  z_um: number[]
  row_y_um: number
  image: PackedImage
}
export interface Cuts {
  x_um: number[]
  i_x: number[]
  z_um: number[]
  i_z: number[]
  through: { x_um: number; y_um: number }
}

export interface FrameResponse {
  plan_id: string
  time_us: number
  plane_z_um: number
  xy: ImageXY
  xz: ImageXZ | null
  cuts: Cuts
  spots: Spot[]
}

export interface ApertureRequest {
  plan: PlanRequest
  time_us: number
  n: number
  include_voltage: boolean
  voltage_window_mm: [number, number] | null
}

export interface ApertureVoltage {
  u_mm: number[]
  v: number[]
}

export interface ApertureChannel {
  u_mm: number[]
  envelope: number[]
  f_inst_MHz: (number | null)[] // JSON has no NaN: null where the envelope is 0 (the frontend maps it back to NaN)
  filled: boolean[]
  t_ret_us: number[]
  voltage: ApertureVoltage | null
}

export interface ApertureResponse {
  time_us: number
  channels: Record<ChannelName, ApertureChannel>
}

export interface CodeResponse {
  code_python: string
}

// ---------------------------------------------------------------- §6 advanced-mode routes

export interface EfficiencyModel {
  interaction_length_mm: number
  f_tpm_MHz: number | null
  refractive_index: number
  peak_efficiency: number
  angular_coupling: number
}

export interface EfficiencyRequest {
  hardware: HardwareSpec
  model: EfficiencyModel
  f_range_MHz: [number, number] | null
  n: number
}

export interface EfficiencyResponse {
  f_MHz: number[]
  deflection_mrad: number[]
  eta_first: number[]
  eta_second_at_center: number[]
  bandwidth_3dB_MHz: number
  first_aod_usable_MHz: number
  band_MHz: [number, number]
  f_tpm_MHz: number
  pair: { f_A_MHz: number[]; f_B_MHz: number[]; eta: number[][] } // row = f_A
  ridge: { f_A_MHz: number[]; eta_max: number[]; f_B_at_max_MHz: number[] }
  assumptions: string
}

export interface TrajectoryEfficiencyRequest {
  plan: PlanRequest
  model: EfficiencyModel
  n: number
}

export interface TrajectoryEfficiencyResponse {
  t_us: number[]
  per_channel: Record<ChannelName, number[]>
  product: number[]
  min_product: number
  assumptions: string
}

export interface ChannelMisalignment {
  offset_par_um: number
  offset_perp_um: number
  rotation_mrad: number
  delay_ns: number
  velocity_scale: number
  tilt_x_mrad: number
  tilt_y_mrad: number
}

export type Misalignment = Partial<Record<ChannelName, ChannelMisalignment>>

export interface AdvancedModel {
  pupil_n: 128 | 256 | 512
  span_w_in: number
  order_model: 'j1' | 'weak'
  comb_dft_points: number
  max_groups: number
  efficiency: EfficiencyModel | null
}

export interface AdvancedSimulateRequest {
  plan: PlanRequest
  misalignment: Misalignment
  times_us: number[]
  grid: {
    half_x_um: number
    half_y_um: number
    n: number
    center: 'tracked' | { x_um: number; y_um: number }
  }
  xz: { nz: number; z_half_um: number } | null
  model: AdvancedModel
  reference: boolean
}

export interface AdvMetrics {
  x_um: number
  y_um: number
  wx_um: number
  wy_um: number
  z_best_x_um: number | null
  z_best_y_um: number | null
  astig_um: number | null
  peak_rel: number
  power_rel: number
  strehl: number
}

export interface AdvancedFrame {
  time_us: number
  plane_z_um: number
  n_groups: number
  groups_dropped: number
  xy: ImageXY
  xz: ImageXZ | null
  metrics: AdvMetrics
  reference_metrics: AdvMetrics | null
}

export interface AdvancedSimulateResponse {
  plan_id: string
  frames: AdvancedFrame[]
  static_offset_um: Record<ChannelName, { x: number; y: number }>
  notes: string[]
}

export interface SpectrumRequest {
  plan: PlanRequest
  channel: ChannelName
  t_center_us: number
  window_us: number
  sample_rate_MSps: number
}

export interface SpectrumResponse {
  f_MHz: number[]
  psd_dB: number[]
  crest_factor: number
  papr_dB: number
  peak_depth_rad: number
  band_MHz: [number, number]
  notes: string[]
}
