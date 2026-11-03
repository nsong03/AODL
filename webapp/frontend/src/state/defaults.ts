import type { ArraySpec, HardwareSpec, MoveKind, MoveSpec, PlanOptions, PlanRequest } from '../api/types'

/** `default_1030` (API.md §3.1). */
export const DEFAULT_HARDWARE: HardwareSpec = {
  wavelength_nm: 1030.0,
  focal_length_mm: 6.5,
  w_in_mm: 2.0,
  sound_speed_m_s: 650.0,
  aperture_mm: 7.5,
  f_center_MHz: 100.0,
  band_MHz: [90.0, 110.0],
  drive_strength: 0.3,
  mixing_order: 3,
}

/** Fallback presets shown before `/api/presets` answers. */
export const FALLBACK_HARDWARE_PRESETS: Record<string, HardwareSpec> = {
  default_1030: DEFAULT_HARDWARE,
  paper_808: { ...DEFAULT_HARDWARE, wavelength_nm: 808.0 },
}

export const DEFAULT_ARRAY: ArraySpec = { mx: 1, my: 1, delta_f_x_MHz: 1.0, delta_f_y_MHz: 1.3 }

export const DEFAULT_OPTIONS: PlanOptions = {
  shepard: 'auto',
  shepard_delta_f_MHz: null,
  eta: 0.5,
  phases: 'schroeder',
  phase_seed: 0,
  amp: 1.0,
  retard_compensate: false,
  f_z_bias: 'none',
  t_pad_us: null,
  check_band: true,
}

/** The pick-and-place starter (W1 §2 `examples.py`), used until the presets arrive. */
export const DEFAULT_MOVES: MoveSpec[] = [
  { kind: 'lift', dz_um: 5.0, duration_us: 40.0, profile: 'min_jerk' },
  { kind: 'translate', dx_um: 20.0, dy_um: 0.0, duration_us: 60.0, profile: 'min_jerk' },
  { kind: 'lift', dz_um: -5.0, duration_us: 40.0, profile: 'min_jerk' },
]

export const DEFAULT_REQUEST: PlanRequest = {
  hardware: DEFAULT_HARDWARE,
  array: DEFAULT_ARRAY,
  moves: DEFAULT_MOVES,
  options: DEFAULT_OPTIONS,
}

export function newMove(kind: MoveKind): MoveSpec {
  switch (kind) {
    case 'lift':
      return { kind: 'lift', dz_um: 5.0, duration_us: 40.0, profile: 'min_jerk' }
    case 'translate':
      return { kind: 'translate', dx_um: 10.0, dy_um: 0.0, duration_us: 60.0, profile: 'min_jerk' }
    case 'hold':
      return { kind: 'hold', duration_us: 100.0 }
    case 'path':
      return { kind: 'path', points_um: [[5, 0, 0], [5, 5, 0], [0, 5, 0]], duration_us: 200.0, closed: true }
  }
}

export const MOVE_KINDS: readonly MoveKind[] = ['lift', 'translate', 'hold', 'path']
