/**
 * Unit constants, mirroring `aodl.units`: each constant is "how many SI units one of these
 * is", so multiplying converts *into* SI and dividing converts *out of* SI.
 *
 * The JSON API speaks display units (`_um`, `_us`, `_MHz`, `_mm`, `_nm`, `_mrad`, `_ns`);
 * the helpers below are the only place a conversion lives.
 */
export const kHz = 1e3
export const MHz = 1e6
export const GHz = 1e9

export const nm = 1e-9
export const um = 1e-6
export const mm = 1e-3

export const us = 1e-6
export const ms = 1e-3

export const mrad = 1e-3
export const ns = 1e-9

/** µs → s */
export const usToS = (t_us: number): number => t_us * us
/** s → µs */
export const sToUs = (t_s: number): number => t_s / us
/** MHz → Hz */
export const MHzToHz = (f: number): number => f * MHz
/** Hz → MHz */
export const HzToMHz = (f: number): number => f / MHz
/** µm → m */
export const umToM = (x: number): number => x * um
/** m → µm */
export const mToUm = (x: number): number => x / um
/** mm → m */
export const mmToM = (x: number): number => x * mm
/** m → mm */
export const mToMm = (x: number): number => x / mm
/** nm → m */
export const nmToM = (x: number): number => x * nm

/** µm per MHz of frequency difference: `λF/v` (Table I's deflection scale). */
export function umPerMHz(wavelength_nm: number, focal_length_mm: number, sound_speed_m_s: number): number {
  return (nmToM(wavelength_nm) * mmToM(focal_length_mm) * MHz) / sound_speed_m_s / um
}

/** Acoustic transit time `τ = D / v` in µs. */
export function tauUs(aperture_mm: number, sound_speed_m_s: number): number {
  return sToUs(mmToM(aperture_mm) / sound_speed_m_s)
}

/** Deflection angle `λ (f − f_center) / v` in mrad for a detuning in MHz. */
export function deflectionMrad(df_MHz: number, wavelength_nm: number, sound_speed_m_s: number): number {
  return (nmToM(wavelength_nm) * MHzToHz(df_MHz)) / sound_speed_m_s / mrad
}

/**
 * Intensity per m² → per µm². `/api/frame` and `/api/advanced` images carry `peak` on the SI
 * `intensity_frame` scale (power per m²) while the spot metrics the movie uses are in µm, so
 * the Gaussian peak `2P/(π w_x w_y)` with `w` in µm is `1e12` times the SI number.
 */
export const PER_M2_TO_PER_UM2 = um * um
