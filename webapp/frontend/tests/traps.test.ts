import { describe, expect, it } from 'vitest'
import type { SpotFrame } from '../src/api/types'
import { frameShift, frameSites, median, pairFrames, trapDepths } from '../src/physics/traps'

const w0 = 1.0655

function frame(xs: number[], ys: number[], ps: number[], w = w0): SpotFrame {
  return {
    x_um: xs, y_um: ys, z_um: xs.map(() => 0), delta_f_um: xs.map(() => 0), sigma_astig: xs.map(() => 0),
    wx_um: xs.map(() => w), wy_um: xs.map(() => w), power: ps, power_coherent: ps, df_opt_MHz: xs.map((_, i) => i),
    group: xs.map((_, i) => i), n_terms: xs.map(() => 1),
  }
}
const peak1 = 2 / (Math.PI * w0 * w0)

describe('frameSites', () => {
  it('merges coincident spots into one site whose depth is Σ 2P/(π w_x w_y)', () => {
    const sites = frameSites(frame([0, 0.1, 10.3, 20.6], [0, 0, 0, 0], [0.5, 0.5, 1, 0.5]), w0)
    expect(sites.length).toBe(3)
    const first = sites.find((s) => Math.abs(s.x_um) < 1)!
    expect(first.depth).toBeCloseTo(peak1, 12)
    expect(first.x_um).toBeCloseTo(0.05, 12)
    expect(first.members.sort()).toEqual([0, 1])
  })
})

describe('pairFrames / frameShift', () => {
  it('finds no shift for a reordered static frame and the carry for a moved one', () => {
    const a = [
      { x_um: 0, y_um: 0, power: 1 },
      { x_um: 10, y_um: 0, power: 1 },
    ]
    const b = [a[1], a[0]]
    expect(frameShift(a, b, 0.5 * w0, 2 * w0)).toEqual([0, 0])
    expect(Array.from(pairFrames(a, b, w0).pair)).toEqual([1, 0])
    const moved = a.map((s) => ({ ...s, x_um: s.x_um + 1.5 }))
    const p = pairFrames(a, moved, w0)
    expect(p.shift[0]).toBeCloseTo(1.5, 12)
    expect(Array.from(p.pair)).toEqual([0, 1])
    const far = a.map((s) => ({ ...s, x_um: s.x_um + 3 * w0 }))
    expect(Array.from(pairFrames(a, far, w0).pair)).toEqual([-1, -1])
  })
})

describe('trapDepths', () => {
  it('reads 1 on the plateau and through a hand-over, tracks by position, drops faint tracks', () => {
    const frames = [
      frame([], [], []), // dark before τ/2
      frame([0, 10], [0, 0], [1, 1]),
      frame([10, 0.05, 0, 20, -10, 30], [0, 0, 0, 0, 0, 0], [1, 0.5, 0.5, 0.5, 0.02, 0.5]), // hand-over: two halves + a shadow + faint ghost + extra column
      frame([10, 0], [0, 0], [1, 1]),
    ]
    const d = trapDepths(frames, w0)
    expect(d.plateau).toBeCloseTo(peak1, 12)
    expect(median([3, 1, 2])).toBe(2)
    // the two traps, the shadow at 20 and the column at 30 (the 2 % ghost is below 5 %)
    expect(d.depth.length).toBe(4)
    const at = (x: number) => d.depth[d.where.findIndex((w) => Math.abs(w.x_um - x) < 0.5)]
    expect(Array.from(at(0))).toEqual([0, 1, expect.closeTo(1, 12), 1])
    expect(Array.from(at(10))).toEqual([0, 1, 1, 1])
    expect(Array.from(at(20))).toEqual([0, 0, 0.5, 0])
    expect(Array.from(d.brightest)).toEqual([0, 1, 1, 1])
  })
})
