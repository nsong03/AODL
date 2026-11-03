import { describe, expect, it } from 'vitest'
import type { SimulateResponse, SpotFrame } from '../src/api/types'
import { frameIndexAt, nearestFrame, offCanvas, spotRow, spotsAt, trackSpots } from '../src/physics/spots'

function frame(xs: number[], ys: number[], ps: number[]): SpotFrame {
  return {
    x_um: xs, y_um: ys, z_um: xs.map(() => 0), delta_f_um: xs.map(() => 0), sigma_astig: xs.map(() => 0),
    wx_um: xs.map(() => 1), wy_um: xs.map(() => 1), power: ps, power_coherent: ps, df_opt_MHz: xs.map((_, i) => i),
    group: xs.map((_, i) => i), n_terms: xs.map(() => 1),
  }
}

describe('spotRow', () => {
  it('snaps the power-weighted mean onto a real row, ignoring faint spots', () => {
    const spots = [
      { x_um: 0, y_um: -6.5, z_um: 0, delta_f_um: 0, power: 1 },
      { x_um: 0, y_um: 6.5, z_um: 0, delta_f_um: 0, power: 1 },
      { x_um: 0, y_um: -0.1, z_um: 0, delta_f_um: 0, power: 0.01 }, // IM3 ghost near the centre line
    ]
    expect(spotRow(spots)).toBe(-6.5)
    expect(spotRow([])).toBe(0)
  })
})

describe('frame lookup', () => {
  const times = [0, 10, 20, 30]
  it('frameIndexAt / nearestFrame', () => {
    expect(frameIndexAt(times, -1)).toBe(-1)
    expect(frameIndexAt(times, 0)).toBe(0)
    expect(frameIndexAt(times, 15)).toBe(1)
    expect(frameIndexAt(times, 30)).toBe(3)
    expect(nearestFrame(times, 14)).toBe(1)
    expect(nearestFrame(times, 16)).toBe(2)
    expect(nearestFrame(times, 99)).toBe(3)
  })
  const constants = { waist0_um: 1, rayleigh_um: 3, tau_us: 11, half_tau_us: 5.5, um_per_MHz: 10.3, lens_scale_m_s: 1e-16, chirp_per_um_MHz_per_ms: 4.85, beam_transit_us: 6.15, eq1_hold_10um_us: 206 }
  const mkSim = (frames: SpotFrame[], t = times): SimulateResponse => ({
    plan_id: 'x', times_us: t, tracked_z_um: t.map((_, i) => i), frames, constants,
    bounds_um: { x0: -1, x1: 15, y0: -1, y1: 1, z0: 0, z1: 1 }, power_ref: 1, notes: [],
  })
  it('spotsAt interpolates paired spots and cross-fades the rest', () => {
    const sim = mkSim([frame([], [], []), frame([0], [0], [1]), frame([0.4], [0], [1]), frame([0.4, 20], [0, 0], [1, 1])])
    const early = spotsAt(sim, 5) // dark → lit: the spot fades in where it is
    expect(early.spots.length).toBe(1)
    expect(early.spots[0].power).toBeCloseTo(0.5, 12)
    expect(early.tracked_z).toBe(1)
    const mid = spotsAt(sim, 15)
    expect(mid.spots.length).toBe(1)
    expect(mid.spots[0].x_um).toBeCloseTo(0.2, 12)
    expect(mid.tracked_z).toBeCloseTo(1.5, 12)
    const late = spotsAt(sim, 24)
    expect(late.spots.length).toBe(2)
    expect(late.spots.find((s) => s.x_um === 20)?.power).toBeCloseTo(0.4, 12)
    expect(offCanvas(spotsAt(sim, 30).spots, sim.bounds_um).map((s) => s.x_um)).toEqual([20])
  })
  it('the same spots in a different order produce no motion', () => {
    const a = frame([0, 10, 20], [0, 0, 5], [1, 0.5, 0.25])
    const b = frame([20, 0, 10], [5, 0, 0], [0.25, 1, 0.5])
    const sim = mkSim([a, b], [0, 10])
    for (const t of [0, 2.5, 5, 7.5, 10]) {
      const got = spotsAt(sim, t).spots.map((s) => [s.x_um, s.y_um, s.power]).sort((p, q) => p[0] - q[0])
      expect(got).toEqual([
        [0, 0, 1],
        [10, 0, 0.5],
        [20, 5, 0.25],
      ])
    }
  })
  it('a spot present only in frame b ramps from 0 to P', () => {
    const sim = mkSim([frame([0], [0], [1]), frame([0, 10], [0, 0], [1, 0.8])], [0, 10])
    for (const w of [0.25, 0.5, 0.75]) {
      const s = spotsAt(sim, 10 * w).spots.find((q) => q.x_um === 10)
      expect(s?.power).toBeCloseTo(0.8 * w, 12)
    }
    expect(spotsAt(sim, 10).spots.find((q) => q.x_um === 10)?.power).toBeCloseTo(0.8, 12)
    expect(spotsAt(sim, 0).spots.length).toBe(1)
  })
  it('a spot that really moved by 3 w0 is not paired and cross-fades', () => {
    const sim = mkSim([frame([0], [0], [1]), frame([3], [0], [1])], [0, 10])
    const s = spotsAt(sim, 4).spots.sort((p, q) => p.x_um - q.x_um)
    expect(s.map((q) => q.x_um)).toEqual([0, 3])
    expect(s[0].power).toBeCloseTo(0.6, 12)
    expect(s[1].power).toBeCloseTo(0.4, 12)
  })
  it('a rigidly carried array interpolates even when a step exceeds 0.5 w0', () => {
    // a 3-trap row carried by 0.9 w0 between frames (fast carry); a sliding fade ladder order on top
    const sim = mkSim([frame([0, 10, 20], [0, 0, 0], [1, 1, 1]), frame([20.9, 0.9, 10.9], [0, 0, 0], [1, 1, 1])], [0, 10])
    const s = spotsAt(sim, 5).spots.map((q) => q.x_um).sort((p, q) => p - q)
    expect(s.length).toBe(3)
    expect(s[0]).toBeCloseTo(0.45, 12)
    expect(s[1]).toBeCloseTo(10.45, 12)
    expect(s[2]).toBeCloseTo(20.45, 12)
  })
})

describe('trackSpots', () => {
  it('follows spots by nearest position across frames', () => {
    const frames = [frame([0, 10], [0, 0], [1, 1]), frame([1, 11], [0, 0], [1, 1]), frame([2], [0], [1]), frame([3, 13], [0, 0], [1, 1])]
    const tracks = trackSpots(frames, 3)
    expect(tracks.length).toBe(2)
    expect(tracks[0]).toEqual([0, 0, 0, 0])
    expect(tracks[1]).toEqual([1, 1, -1, 1])
  })
})
