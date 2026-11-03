/**
 * The astigmatic Gaussian beam approximation of the movie (DESIGN §1 item 2).
 *
 * For a spot with centre `(x, y)`, best focus `z`, astigmatic interval `ΔF = Z_x − Z_y`, power
 * `P` and the hardware constants `w0`, `z_R`:
 *
 *     Z_x = z + ΔF/2,   Z_y = z − ΔF/2
 *     w_x(Z) = w0 √(1 + ((Z − Z_x)/z_R)²),   w_y(Z) likewise with Z_y
 *     I(X, Y; Z) = P · 2/(π w_x w_y) · exp(−2 (X−x)²/w_x² − 2 (Y−y)²/w_y²)
 *
 * (`aodl.field.focal.spot_params`; `SpotMetrics.wx/wy` are these radii at `Z = z`). `P` is on
 * the `intensity_frame` scale, so the approximation and an exact frame share one colour scale.
 * Spots add in intensity; each is tinted by its own `z` (`zColor`) and composited additively
 * on black through a gamma, with one global peak over the whole movie (honest dimming), as
 * `aodl.viz.style.composite` does. Units: µm throughout.
 */
import type { SpotFrame } from '../api/types'
import { GAMMA, zColor } from './zColor'

/** One spot of a frame, as scalars (µm, power on the `SpotMetrics.power` scale). */
export interface SpotLike {
  x_um: number
  y_um: number
  z_um: number
  delta_f_um: number
  power: number
}

export interface Grid2D {
  x0: number
  x1: number
  nx: number
  y0: number
  y1: number
  ny: number
}

/** 1/e² intensity radii `(w_x, w_y)` of a spot at plane `z` [µm]. */
export function spotRadii(spot: SpotLike, z: number, w0: number, zR: number): [number, number] {
  const zx = spot.z_um + 0.5 * spot.delta_f_um
  const zy = spot.z_um - 0.5 * spot.delta_f_um
  const ax = (z - zx) / zR
  const ay = (z - zy) / zR
  return [w0 * Math.sqrt(1 + ax * ax), w0 * Math.sqrt(1 + ay * ay)]
}

/** Peak intensity `P · 2/(π w_x w_y)` of a spot at plane `z`. */
export function spotPeak(spot: SpotLike, z: number, w0: number, zR: number): number {
  const [wx, wy] = spotRadii(spot, z, w0, zR)
  return (spot.power * 2) / (Math.PI * wx * wy)
}

/** Unpack a struct-of-arrays frame into spot records. */
export function frameSpots(frame: SpotFrame): SpotLike[] {
  const n = frame.x_um.length
  const out: SpotLike[] = new Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = {
      x_um: frame.x_um[i],
      y_um: frame.y_um[i],
      z_um: frame.z_um[i],
      delta_f_um: frame.delta_f_um[i],
      power: frame.power[i],
    }
  }
  return out
}

/**
 * Global normalisation: the brightest pixel the movie ever shows. For each frame, spots whose
 * centres coincide (within half a 1/e² radius — the dying and rising rung pairs of a fading
 * hand-over, API.md §5.1) have their peaks `P·2/(π w_x w_y)` at the frame's plane summed; the
 * maximum over spots and frames is returned. This is what `aodl.viz.movie._peaks` measures
 * (the max of the summed group frames): summing *every* spot's peak would dim a 16-trap array
 * sixteen-fold, which is not honest dimming. Every frame of the movie is scaled by this one
 * number so a dimming spot looks dimmer.
 */
export function peakNormalization(frames: readonly SpotFrame[], planes: ArrayLike<number>, w0: number, zR: number): number {
  let peak = 0
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]
    const z = planes[i]
    const n = f.x_um.length
    const pk = new Float64Array(n)
    const wmax = new Float64Array(n)
    for (let k = 0; k < n; k++) {
      const zx = f.z_um[k] + 0.5 * f.delta_f_um[k]
      const zy = f.z_um[k] - 0.5 * f.delta_f_um[k]
      const ax = (z - zx) / zR
      const ay = (z - zy) / zR
      const wx = w0 * Math.sqrt(1 + ax * ax)
      const wy = w0 * Math.sqrt(1 + ay * ay)
      pk[k] = (f.power[k] * 2) / (Math.PI * wx * wy)
      wmax[k] = Math.max(wx, wy)
    }
    for (let k = 0; k < n; k++) {
      let sum = pk[k]
      const r = 0.5 * wmax[k]
      for (let j = 0; j < n; j++) {
        if (j === k) continue
        if (Math.abs(f.x_um[j] - f.x_um[k]) < r && Math.abs(f.y_um[j] - f.y_um[k]) < r) sum += pk[j]
      }
      if (sum > peak) peak = sum
    }
  }
  return peak
}

/** Patch half-width in 1/e² radii around each spot (`aodl.field.focal.PATCH_WAISTS`). */
export const PATCH_WAISTS = 4

/**
 * Render an XY frame at `plane_z` → RGB `Float32Array` of length `ny·nx·3` (row 0 = y0, imshow
 * convention), additive, hue by each spot's `z`, patch-based with separable 1-D profiles.
 */
export function renderXY(
  spots: readonly SpotLike[],
  grid: Grid2D,
  plane_z: number,
  w0: number,
  zR: number,
  peakNorm: number,
  zMax: number,
  gamma = GAMMA,
  out?: Float32Array,
): Float32Array {
  const { nx, ny } = grid
  const rgb = out ?? new Float32Array(nx * ny * 3)
  rgb.fill(0)
  if (!(peakNorm > 0)) return rgb
  const dx = (grid.x1 - grid.x0) / (nx - 1)
  const dy = (grid.y1 - grid.y0) / (ny - 1)
  const fx = new Float64Array(nx)
  const fy = new Float64Array(ny)
  for (const s of spots) {
    const [wx, wy] = spotRadii(s, plane_z, w0, zR)
    const amp = (s.power * 2) / (Math.PI * wx * wy) / peakNorm
    if (!(amp > 0)) continue
    const mx = PATCH_WAISTS * wx
    const my = PATCH_WAISTS * wy
    let i0 = Math.floor((s.x_um - mx - grid.x0) / dx)
    let i1 = Math.ceil((s.x_um + mx - grid.x0) / dx)
    let j0 = Math.floor((s.y_um - my - grid.y0) / dy)
    let j1 = Math.ceil((s.y_um + my - grid.y0) / dy)
    if (i1 < 0 || j1 < 0 || i0 >= nx || j0 >= ny) continue
    i0 = Math.max(i0, 0)
    j0 = Math.max(j0, 0)
    i1 = Math.min(i1, nx - 1)
    j1 = Math.min(j1, ny - 1)
    const [cr, cg, cb] = zColor(s.z_um, zMax)
    for (let i = i0; i <= i1; i++) {
      const u = (grid.x0 + i * dx - s.x_um) / wx
      fx[i] = Math.exp(-2 * u * u)
    }
    for (let j = j0; j <= j1; j++) {
      const v = (grid.y0 + j * dy - s.y_um) / wy
      fy[j] = Math.exp(-2 * v * v)
    }
    for (let j = j0; j <= j1; j++) {
      const row = j * nx
      const ay = amp * fy[j]
      for (let i = i0; i <= i1; i++) {
        let val = ay * fx[i]
        if (val > 1) val = 1
        const scaled = Math.pow(val, gamma)
        const p = (row + i) * 3
        rgb[p] += scaled * cr
        rgb[p + 1] += scaled * cg
        rgb[p + 2] += scaled * cb
      }
    }
  }
  for (let p = 0; p < rgb.length; p++) if (rgb[p] > 1) rgb[p] = 1
  return rgb
}

/**
 * Render an XZ slice at `Y = rowY` → RGB `Float32Array` of length `nz·nx·3` (row = z, tinted
 * row by row by its own Z as `aodl.viz.style.tint_by_row` does). No patching along z — a
 * spot sweeps through focus along this axis.
 */
export function renderXZ(
  spots: readonly SpotLike[],
  xgrid: { x0: number; x1: number; nx: number },
  zgrid: { z0: number; z1: number; nz: number },
  rowY: number,
  w0: number,
  zR: number,
  peakNorm: number,
  zMax: number,
  gamma = GAMMA,
  out?: Float32Array,
): Float32Array {
  const { nx } = xgrid
  const { nz } = zgrid
  const rgb = out ?? new Float32Array(nx * nz * 3)
  rgb.fill(0)
  if (!(peakNorm > 0)) return rgb
  const dx = (xgrid.x1 - xgrid.x0) / (nx - 1)
  const dz = (zgrid.z1 - zgrid.z0) / (nz - 1)
  const intensity = new Float64Array(nx * nz)
  for (const s of spots) {
    for (let j = 0; j < nz; j++) {
      const z = zgrid.z0 + j * dz
      const [wx, wy] = spotRadii(s, z, w0, zR)
      const v = (rowY - s.y_um) / wy
      const amp = ((s.power * 2) / (Math.PI * wx * wy) / peakNorm) * Math.exp(-2 * v * v)
      if (!(amp > 1e-9)) continue
      const m = PATCH_WAISTS * wx
      let i0 = Math.max(0, Math.floor((s.x_um - m - xgrid.x0) / dx))
      const i1 = Math.min(nx - 1, Math.ceil((s.x_um + m - xgrid.x0) / dx))
      const row = j * nx
      for (; i0 <= i1; i0++) {
        const u = (xgrid.x0 + i0 * dx - s.x_um) / wx
        intensity[row + i0] += amp * Math.exp(-2 * u * u)
      }
    }
  }
  for (let j = 0; j < nz; j++) {
    const [cr, cg, cb] = zColor(zgrid.z0 + j * dz, zMax)
    const row = j * nx
    for (let i = 0; i < nx; i++) {
      let val = intensity[row + i]
      if (val <= 0) continue
      if (val > 1) val = 1
      const scaled = Math.pow(val, gamma)
      const p = (row + i) * 3
      rgb[p] = scaled * cr
      rgb[p + 1] = scaled * cg
      rgb[p + 2] = scaled * cb
    }
  }
  return rgb
}

/** Integrated power of an RGB-free intensity render — test helper: `Σ I dx dy` on a grid. */
export function integrateXY(spots: readonly SpotLike[], grid: Grid2D, plane_z: number, w0: number, zR: number): number {
  const dx = (grid.x1 - grid.x0) / (grid.nx - 1)
  const dy = (grid.y1 - grid.y0) / (grid.ny - 1)
  let total = 0
  for (const s of spots) {
    const [wx, wy] = spotRadii(s, plane_z, w0, zR)
    const amp = (s.power * 2) / (Math.PI * wx * wy)
    for (let j = 0; j < grid.ny; j++) {
      const v = (grid.y0 + j * dy - s.y_um) / wy
      const fy = Math.exp(-2 * v * v)
      for (let i = 0; i < grid.nx; i++) {
        const u = (grid.x0 + i * dx - s.x_um) / wx
        total += amp * fy * Math.exp(-2 * u * u)
      }
    }
  }
  return total * dx * dy
}

/** Write an RGB float buffer into an `ImageData`-compatible RGBA byte array. */
export function rgbToRGBA(rgb: Float32Array, out: Uint8ClampedArray): void {
  const n = rgb.length / 3
  for (let i = 0; i < n; i++) {
    out[i * 4] = rgb[i * 3] * 255
    out[i * 4 + 1] = rgb[i * 3 + 1] * 255
    out[i * 4 + 2] = rgb[i * 3 + 2] * 255
    out[i * 4 + 3] = 255
  }
}
