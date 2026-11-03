import { describe, expect, it } from 'vitest'
import type { SpotFrame } from '../src/api/types'
import { frameSpots, integrateXY, mergeCoincident, peakNormalization, renderXY, renderXZ, spotPeak, spotRadii, type SpotLike } from '../src/physics/gaussianBeam'

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
  it('adds coincident spots in intensity before the gamma (a hand-over looks like one trap)', () => {
    const one: SpotLike = { x_um: 1.3, y_um: 0.4, z_um: 1.5, delta_f_um: 0.2, power: 1 }
    const halves: SpotLike[] = [
      { ...one, power: 0.5 },
      { ...one, power: 0.5 },
    ]
    const peak = 0.8 * spotPeak(one, 0.5, w0, zR) // saturating centre on purpose
    const a = renderXY([one], grid, 0.5, w0, zR, peak, 3)
    const b = renderXY(halves, grid, 0.5, w0, zR, peak, 3)
    let worst = 0
    for (let p = 0; p < a.length; p++) worst = Math.max(worst, Math.abs(a[p] - b[p]))
    expect(worst).toBeLessThan(1e-6)
    // the frame peak of the two halves equals that of the full trap
    const fr = (ps: number[]): SpotFrame => ({
      x_um: ps.map(() => 1.3), y_um: ps.map(() => 0.4), z_um: ps.map(() => 1.5), delta_f_um: ps.map(() => 0.2), sigma_astig: ps.map(() => 0),
      wx_um: ps.map(() => w0), wy_um: ps.map(() => w0), power: ps, power_coherent: ps, df_opt_MHz: ps.map(() => 0), group: ps.map((_, i) => i), n_terms: ps.map(() => 1),
    })
    expect(peakNormalization([fr([0.5, 0.5])], [0.5], w0, zR)).toBeCloseTo(peakNormalization([fr([1])], [0.5], w0, zR), 12)
  })
  it('leaves distinct spots unchanged (each renders as it would alone)', () => {
    const peak = Math.max(...spots.map((s) => spotPeak(s, 0, w0, zR)))
    const both = renderXY(spots, grid, 0, w0, zR, peak, 3)
    for (const s of spots) {
      const alone = renderXY([s], grid, 0, w0, zR, peak, 3)
      const i = Math.round(((s.x_um - grid.x0) / 20) * 400)
      const j = Math.round(((s.y_um - grid.y0) / 16) * 320)
      // the 4-waist patches of the two spots do not overlap: identical within each patch
      for (let dj = -8; dj <= 8; dj++) {
        for (let di = -8; di <= 8; di++) {
          const p = ((j + dj) * grid.nx + i + di) * 3
          for (let c = 0; c < 3; c++) expect(Math.abs(both[p + c] - alone[p + c])).toBeLessThan(1e-6)
        }
      }
    }
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
    const handover: SpotFrame = { ...frame, x_um: [0, 0], y_um: [0, 0], z_um: [0, 0], power: [0.5, 0.5], power_coherent: [0.5, 0.5] }
    expect(peakNormalization([handover], [0], w0, zR)).toBeCloseTo(spotPeak({ x_um: 0, y_um: 0, z_um: 0, delta_f_um: 0, power: 1 }, 0, w0, zR), 12)
  })
  it('is the maximum of the merged intensity for partly overlapping spots', () => {
    const near: SpotFrame = { ...frame, x_um: [0, 0.8], y_um: [0, 0.3], z_um: [0, 0.5], delta_f_um: [0, 0.4], power: [1, 0.7], power_coherent: [1, 0.7] }
    const s = frameSpots(near)
    let brute = 0
    for (let i = -400; i <= 400; i++) {
      for (let j = -300; j <= 300; j++) {
        const x = i * 0.0025
        const y = j * 0.0025
        let I = 0
        for (const q of s) {
          const [wx, wy] = spotRadii(q, 0.2, w0, zR)
          I += spotPeak(q, 0.2, w0, zR) * Math.exp(-2 * ((x - q.x_um) ** 2 / wx ** 2 + (y - q.y_um) ** 2 / wy ** 2))
        }
        brute = Math.max(brute, I)
      }
    }
    expect(Math.abs(peakNormalization([near], [0.2], w0, zR) / brute - 1)).toBeLessThan(1e-5)
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

describe('mergeCoincident', () => {
  it('sums exact duplicates and leaves distinct spots and the input alone', () => {
    const a = { x_um: 1, y_um: 2, z_um: 3, delta_f_um: 0, power: 0.25 }
    const b = { ...a, power: 0.5 }
    const c = { ...a, x_um: 1.01 }
    const d = { ...a, delta_f_um: 0.2 }
    const input = [a, b, c, d]
    const out = mergeCoincident(input)
    expect(out).toHaveLength(3)
    expect(out[0].power).toBeCloseTo(0.75, 15)
    expect(out[1]).toEqual(c)
    expect(out[2]).toEqual(d)
    expect(a.power).toBe(0.25) // not mutated
  })
})
