/**
 * Line cuts of the astigmatic Gaussian approximation (DESIGN §1 item 2) along the same lines as
 * the closed-form cuts of `/api/frame` (API.md §5.2), so the two can be overlaid:
 *
 *     I(X, Y; Z) = Σ_k P_k · 2/(π w_x w_y) · exp(−2 (X−x_k)²/w_x² − 2 (Y−y_k)²/w_y²)
 *
 * with `w_x(Z), w_y(Z)` the per-axis Gaussian-beam radii about `Z_x = z + ΔF/2`,
 * `Z_y = z − ΔF/2` (`spotRadii`). Units: µm, intensities per µm² on the `SpotMetrics.power`
 * scale (the closed-form cuts are per m²: multiply them by `PER_M2_TO_PER_UM2`).
 */
import type { Spot } from '../api/types'
import { spotRadii, type SpotLike } from './gaussianBeam'
import { PAIR_RADIUS_W0 } from './traps'

/** Gaussian-approximation intensity at `(x, y)` on plane `z` [per µm²]. */
export function gaussIntensity(spots: readonly SpotLike[], x: number, y: number, z: number, w0: number, zR: number): number {
  let sum = 0
  for (const s of spots) {
    const [wx, wy] = spotRadii(s, z, w0, zR)
    const u = (x - s.x_um) / wx
    const v = (y - s.y_um) / wy
    sum += ((2 * s.power) / (Math.PI * wx * wy)) * Math.exp(-2 * (u * u + v * v))
  }
  return sum
}

/** `I(x)` along the row `y` on plane `z`. */
export function gaussCutX(spots: readonly SpotLike[], xs: readonly number[], y: number, z: number, w0: number, zR: number): number[] {
  return xs.map((x) => gaussIntensity(spots, x, y, z, w0, zR))
}

/** `I(y)` along the column `x` on plane `z`. */
export function gaussCutY(spots: readonly SpotLike[], ys: readonly number[], x: number, z: number, w0: number, zR: number): number[] {
  return ys.map((y) => gaussIntensity(spots, x, y, z, w0, zR))
}

/** `I(z)` along the axial line through `(x, y)`. */
export function gaussCutZ(spots: readonly SpotLike[], zs: readonly number[], x: number, y: number, w0: number, zR: number): number[] {
  return zs.map((z) => gaussIntensity(spots, x, y, z, w0, zR))
}

/**
 * Astigmatic-Gaussian trap depth `Σ 2P/(π w_x w_y)` of the spots within `0.5·w0` of the
 * brightest spot, with `w_x, w_y` the radii at each spot's own best focus (`Spot.wx_um/wy_um`)
 * [per µm²]. The client-side twin of `FrameResponse.gauss_peak` (used when an older backend
 * omits it).
 */
export function gaussPeakOfBrightest(spots: readonly Spot[], w0: number): number {
  let top: Spot | null = null
  for (const s of spots) if (!top || s.power > top.power) top = s
  if (!top) return 0
  const r = PAIR_RADIUS_W0 * w0
  let sum = 0
  for (const s of spots) {
    if (Math.hypot(s.x_um - top.x_um, s.y_um - top.y_um) > r) continue
    if (s.wx_um > 0 && s.wy_um > 0) sum += (2 * s.power) / (Math.PI * s.wx_um * s.wy_um)
  }
  return sum
}

/** Index of the entry of `v` nearest `x`. */
export function nearestIndex(v: readonly number[], x: number): number {
  let best = 0
  for (let i = 1; i < v.length; i++) if (Math.abs(v[i] - x) < Math.abs(v[best] - x)) best = i
  return best
}
