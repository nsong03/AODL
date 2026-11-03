/** Shared per-simulation render constants of the tweezer view. */
import type { Bounds, SimulateResponse } from '../../api/types'
import { fmt, ticks } from '../../physics/format'
import { peakNormalization } from '../../physics/gaussianBeam'
import { zExtent } from '../../physics/spots'
import { zMaxFrom } from '../../physics/zColor'

export interface ViewScale {
  w0: number
  zR: number
  bounds: Bounds
  /** global peak over the movie at the *tracked* planes (honest dimming) */
  peakTracked: number
  /** global peak with the plane fixed at Z = 0 */
  peakFixed: number
  zMax: number
  /** XZ panel half range: the spots' axial reach (incl. ΔF/2) plus a Rayleigh range, at least the bounds */
  zHalf: number
}

export function viewScale(sim: SimulateResponse): ViewScale {
  const w0 = sim.constants.waist0_um
  const zR = sim.constants.rayleigh_um
  const peakTracked = peakNormalization(sim.frames, sim.tracked_z_um, w0, zR)
  const peakFixed = peakNormalization(sim.frames, new Float64Array(sim.frames.length), w0, zR)
  const zMax = zMaxFrom([zExtent(sim.frames)])
  let reach = 0
  for (const f of sim.frames) for (let k = 0; k < f.z_um.length; k++) reach = Math.max(reach, Math.abs(f.z_um[k]) + 0.5 * Math.abs(f.delta_f_um[k]))
  const zHalf = Math.max(reach + zR, Math.abs(sim.bounds_um.z0), Math.abs(sim.bounds_um.z1), zMax)
  return { w0, zR, bounds: sim.bounds_um, peakTracked, peakFixed, zMax, zHalf }
}

/** Canvas pixel size for a bounds box with the long side at `longSide` px. */
export function fitGrid(b: Bounds, longSide: number): { nx: number; ny: number } {
  const sx = b.x1 - b.x0
  const sy = b.y1 - b.y0
  if (sx >= sy) return { nx: longSide, ny: Math.max(24, Math.round((longSide * sy) / sx)) }
  return { nx: Math.max(24, Math.round((longSide * sx) / sy)), ny: longSide }
}

export function drawFrame(ctx: CanvasRenderingContext2D, width: number, height: number, b: Bounds, requested: { x: number; y: number } | null, label: string, planeZ: number) {
  const sx = width / (b.x1 - b.x0)
  const sy = height / (b.y1 - b.y0)
  const X = (x: number) => (x - b.x0) * sx
  const Y = (y: number) => height - (y - b.y0) * sy
  ctx.font = '10px sans-serif'
  ctx.fillStyle = 'rgba(154,165,179,0.9)'
  ctx.strokeStyle = 'rgba(154,165,179,0.5)'
  ctx.lineWidth = 1
  // ticks
  ctx.textAlign = 'center'
  for (const v of ticks(b.x0, b.x1, 6)) {
    ctx.beginPath()
    ctx.moveTo(X(v), height)
    ctx.lineTo(X(v), height - 4)
    ctx.stroke()
    ctx.fillText(fmt(v, 0), X(v), height - 6)
  }
  ctx.textAlign = 'left'
  for (const v of ticks(b.y0, b.y1, 4)) {
    if (Y(v) < 18) continue // keep clear of the title
    ctx.beginPath()
    ctx.moveTo(0, Y(v))
    ctx.lineTo(4, Y(v))
    ctx.stroke()
    ctx.fillText(fmt(v, 0), 6, Y(v) + 3)
  }
  ctx.fillStyle = 'rgba(242,245,248,0.9)'
  ctx.textAlign = 'left'
  ctx.fillText(label, 6, 12)
  ctx.textAlign = 'right'
  ctx.fillText(`plane Z = ${fmt(planeZ, 2)} µm`, width - 6, 12)
  ctx.fillText('X [µm] →', width - 6, height - 6)
  if (requested) {
    ctx.strokeStyle = 'rgba(242,245,248,0.85)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.arc(X(requested.x), Y(requested.y), 5, 0, 2 * Math.PI)
    ctx.stroke()
  }
}
