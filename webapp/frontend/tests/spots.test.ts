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
  it('spotsAt interpolates linearly when counts match and snaps otherwise', () => {
    const sim: SimulateResponse = {
      plan_id: 'x', times_us: times, tracked_z_um: [0, 1, 2, 3],
      frames: [frame([], [], []), frame([0], [0], [1]), frame([10], [0], [1]), frame([10, 20], [0, 0], [1, 1])],
      constants: { waist0_um: 1, rayleigh_um: 3, tau_us: 11, half_tau_us: 5.5, um_per_MHz: 10.3, lens_scale_m_s: 1e-16, chirp_per_um_MHz_per_ms: 4.85, beam_transit_us: 6.15, eq1_hold_10um_us: 206 }, bounds_um: { x0: -1, x1: 15, y0: -1, y1: 1, z0: 0, z1: 1 }, power_ref: 1, notes: [],
    }
    expect(spotsAt(sim, 5).spots.length).toBe(1) // nearest of (empty, one) at w = 0.5 → frame 1
    const mid = spotsAt(sim, 15)
    expect(mid.spots[0].x_um).toBeCloseTo(5, 12)
    expect(mid.tracked_z).toBeCloseTo(1.5, 12)
    expect(spotsAt(sim, 24).spots.length).toBe(1)
    expect(spotsAt(sim, 26).spots.length).toBe(2)
    expect(offCanvas(spotsAt(sim, 30).spots, sim.bounds_um).map((s) => s.x_um)).toEqual([20])
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
