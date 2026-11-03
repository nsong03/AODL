import { describe, expect, it } from 'vitest'
import type { SpotFrame } from '../src/api/types'
import { frameSpots, integrateXY, peakNormalization, renderXY, renderXZ, spotPeak, spotRadii, type SpotLike } from '../src/physics/gaussianBeam'

const w0 = 1.0655
const zR = 3.463

describe('spotRadii', () => {
  it('reduces to w0 at the per-axis foci and grows as the textbook law', () => {
    const s: SpotLike = { x_um: 0, y_um: 0, z_um: 2, delta_f_um: 1, power: 1 }
    const [wx, wy] = spotRadii(s, 2.5, w0, zR) // Z_x = 2.5, Z_y = 1.5
    expect(wx).toBeCloseTo(w0, 12)
    expect(wy).toBeCloseTo(w0 * Math.sqrt(1 + 1 / zR ** 2), 12)
    const [wx2] = spotRadii(s, 2.5 + zR, w0, zR)
    expect(wx2).toBeCloseTo(w0 * Math.SQRT2, 12)
  })
})

describe('renderXY', () => {
  const spots: SpotLike[] = [
    { x_um: 3.2, y_um: -1.1, z_um: 0, delta_f_um: 0, power: 2.0 },
    { x_um: -4, y_um: 2, z_um: 3, delta_f_um: 0, power: 1.0 },
  ]
  const grid = { x0: -10, x1: 10, nx: 401, y0: -8, y1: 8, ny: 321 }
  it('peaks at the brightest spot and uses the global normalisation', () => {
    const peak = spots.reduce((m, s) => Math.max(m, spotPeak(s, 0, w0, zR)), 0)
    const rgb = renderXY(spots, grid, 0, w0, zR, peak, 3)
    let best = -1
    let bestV = -1
    for (let p = 0; p < rgb.length / 3; p++) {
      const v = rgb[p * 3] + rgb[p * 3 + 1] + rgb[p * 3 + 2]
      if (v > bestV) {
        bestV = v
        best = p
      }
    }
    const i = best % grid.nx
    const j = Math.floor(best / grid.nx)
    expect(grid.x0 + (i * 20) / 400).toBeCloseTo(3.2, 1)
    expect(grid.y0 + (j * 16) / 320).toBeCloseTo(-1.1, 1)
    expect(rgb[best * 3]).toBeCloseTo(1, 6) // red channel of a Z = 0 (white) spot at full scale
  })
  it('integrates to the spot power within 2 % on a fine grid', () => {
    const one = [spots[0]]
    const total = integrateXY(one, grid, 0, w0, zR)
    expect(Math.abs(total - 2.0) / 2.0).toBeLessThan(0.02)
    const defocused = integrateXY(one, grid, 2.5, w0, zR) // power is conserved through focus
    expect(Math.abs(defocused - 2.0) / 2.0).toBeLessThan(0.02)
  })
  it('tints a spot above the focal plane red and below blue', () => {
    const peak = spotPeak(spots[1], 3, w0, zR)
    const rgb = renderXY([spots[1]], grid, 3, w0, zR, peak, 3)
    const i = Math.round(((-4 - grid.x0) / 20) * 400)
    const j = Math.round(((2 - grid.y0) / 16) * 320)
    const p = (j * grid.nx + i) * 3
    expect(rgb[p]).toBeCloseTo(1, 3)
    expect(rgb[p + 2]).toBeLessThan(0.4)
  })
})

describe('peakNormalization / renderXZ', () => {
  const frame: SpotFrame = {
    x_um: [0, 5], y_um: [0, 0], z_um: [0, 1], delta_f_um: [0, 0], sigma_astig: [0, 0], wx_um: [w0, w0], wy_um: [w0, w0],
    power: [1, 0.5], power_coherent: [1, 0.5], df_opt_MHz: [0, 1], group: [0, 1], n_terms: [1, 1],
  }
  it('is the brightest pixel over frames: separated spots do not add, coincident ones do', () => {
    const n = peakNormalization([frame, frame], [0, 1], w0, zR)
    const s = frameSpots(frame)
    const expected = Math.max(spotPeak(s[0], 0, w0, zR), spotPeak(s[1], 0, w0, zR), spotPeak(s[0], 1, w0, zR), spotPeak(s[1], 1, w0, zR))
    expect(n).toBeCloseTo(expected, 12)
    // a fading hand-over: the dying and rising rung pairs sit on the same trap and sum
    const handover: SpotFrame = { ...frame, x_um: [0, 0.05], y_um: [0, 0], z_um: [0, 0], power: [0.5, 0.5], power_coherent: [0.5, 0.5] }
    expect(peakNormalization([handover], [0], w0, zR)).toBeCloseTo(spotPeak({ x_um: 0, y_um: 0, z_um: 0, delta_f_um: 0, power: 1 }, 0, w0, zR), 9)
  })
  it('renderXZ peaks at the spot focus row', () => {
    const spots = frameSpots(frame).slice(0, 1)
    const peak = spotPeak(spots[0], 0, w0, zR)
    const nx = 101
    const nz = 81
    const rgb = renderXZ(spots, { x0: -5, x1: 5, nx }, { z0: -4, z1: 4, nz }, 0, w0, zR, peak, 1)
    let best = 0
    let bestV = -1
    for (let p = 0; p < nx * nz; p++) {
      const v = rgb[p * 3]
      if (v > bestV) {
        bestV = v
        best = p
      }
    }
    expect(best % nx).toBe(50)
    expect(Math.floor(best / nx)).toBe(40)
    expect(bestV).toBeCloseTo(1, 6)
  })
})
