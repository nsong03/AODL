/**
 * Frame-level helpers on the simulate response: frame lookup / interpolation at the cursor,
 * the XZ row choice (`aodl.engine.SimResult.spot_row`), the shadow-tweezer badge and the
 * frame-to-frame spot matching used by the per-spot metric plots.
 */
import type { Bounds, FadeEvent, SimulateResponse, SpotFrame } from '../api/types'
import { frameSpots, type SpotLike } from './gaussianBeam'

/** Index of the last frame time `≤ t` (−1 before the first frame). */
export function frameIndexAt(times: readonly number[], t: number): number {
  let lo = 0
  let hi = times.length
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (times[mid] <= t) lo = mid + 1
    else hi = mid
  }
  return lo - 1
}

/** Nearest frame index to `t`. */
export function nearestFrame(times: readonly number[], t: number): number {
  if (times.length === 0) return -1
  const i = frameIndexAt(times, t)
  if (i < 0) return 0
  if (i >= times.length - 1) return times.length - 1
  return t - times[i] <= times[i + 1] - t ? i : i + 1
}

export interface InterpolatedFrame {
  spots: SpotLike[]
  /** tracked plane at `t` (linear between frames) */
  tracked_z: number
  /** the two frame indices used (equal when not interpolating) */
  i0: number
  i1: number
}

/**
 * Spots at cursor time `t`: linear interpolation by index between the bracketing frames when
 * their spot counts match (the backend emits spots in frequency-group order), otherwise the
 * nearest frame. Returns no spots before the first frame or when the frame is empty.
 */
export function spotsAt(sim: SimulateResponse, t: number): InterpolatedFrame {
  const times = sim.times_us
  const n = times.length
  if (n === 0) return { spots: [], tracked_z: 0, i0: -1, i1: -1 }
  const i = frameIndexAt(times, t)
  if (i < 0) return { spots: frameSpots(sim.frames[0]), tracked_z: sim.tracked_z_um[0], i0: 0, i1: 0 }
  if (i >= n - 1) return { spots: frameSpots(sim.frames[n - 1]), tracked_z: sim.tracked_z_um[n - 1], i0: n - 1, i1: n - 1 }
  const a = sim.frames[i]
  const b = sim.frames[i + 1]
  const w = (t - times[i]) / (times[i + 1] - times[i])
  const zt = sim.tracked_z_um[i] * (1 - w) + sim.tracked_z_um[i + 1] * w
  if (a.x_um.length !== b.x_um.length || a.x_um.length === 0) {
    const j = w < 0.5 ? i : i + 1
    return { spots: frameSpots(sim.frames[j]), tracked_z: a.x_um.length === 0 || b.x_um.length === 0 ? sim.tracked_z_um[j] : zt, i0: j, i1: j }
  }
  const spots: SpotLike[] = new Array(a.x_um.length)
  for (let k = 0; k < a.x_um.length; k++) {
    spots[k] = {
      x_um: a.x_um[k] * (1 - w) + b.x_um[k] * w,
      y_um: a.y_um[k] * (1 - w) + b.y_um[k] * w,
      z_um: a.z_um[k] * (1 - w) + b.z_um[k] * w,
      delta_f_um: a.delta_f_um[k] * (1 - w) + b.delta_f_um[k] * w,
      power: a.power[k] * (1 - w) + b.power[k] * w,
    }
  }
  return { spots, tracked_z: zt, i0: i, i1: i + 1 }
}

/**
 * Lab Y the XZ slice cuts through (`SimResult.spot_row`): the power-weighted mean y snapped
 * onto the spot nearest to it among spots carrying ≥ half the frame's maximum power.
 */
export function spotRow(spots: readonly SpotLike[]): number {
  if (spots.length === 0) return 0
  let wsum = 0
  let ysum = 0
  let pmax = 0
  for (const s of spots) {
    wsum += s.power
    ysum += s.power * s.y_um
    if (s.power > pmax) pmax = s.power
  }
  const centre = wsum > 0 ? ysum / wsum : spots.reduce((acc, s) => acc + s.y_um, 0) / spots.length
  const cut = 0.5 * pmax
  let bright = spots.filter((s) => s.power >= cut)
  if (bright.length === 0) bright = [...spots]
  let best = bright[0]
  for (const s of bright) if (Math.abs(s.y_um - centre) < Math.abs(best.y_um - centre)) best = s
  return best.y_um
}

/** Brightest spot of a list (or `null`). */
export function brightest(spots: readonly SpotLike[]): SpotLike | null {
  let best: SpotLike | null = null
  for (const s of spots) if (!best || s.power > best.power) best = s
  return best
}

/** `max |z|` over every spot of every frame [µm] (before the 1 µm floor). */
export function zExtent(frames: readonly SpotFrame[]): number {
  let peak = 0
  for (const f of frames) for (const z of f.z_um) if (Math.abs(z) > peak) peak = Math.abs(z)
  return peak
}

/** Spots outside the canvas bounds (laterally) — shadow tweezers during a fade. */
export function offCanvas(spots: readonly SpotLike[], b: Bounds): SpotLike[] {
  return spots.filter((s) => s.x_um < b.x0 || s.x_um > b.x1 || s.y_um < b.y0 || s.y_um > b.y1)
}

/** Fade events deduplicated within `tol_us` (aodl's `fade_schedule` can list one instant twice at the 1e-16 level). */
export function dedupeFades(events: readonly FadeEvent[], tol_us = 1e-3): FadeEvent[] {
  const out: FadeEvent[] = []
  for (const e of events) {
    if (!out.some((o) => o.axis === e.axis && Math.abs(o.time_us - e.time_us) <= tol_us)) out.push(e)
  }
  return out
}

/** Fade events whose fade is in progress at drive time `t_c` (within ± half a transit of the hand-over). */
export function activeFades(events: readonly FadeEvent[], t_c: number, halfWidth_us: number): FadeEvent[] {
  return events.filter((e) => Math.abs(e.time_us - t_c) <= halfWidth_us)
}

/**
 * Match spots frame to frame by nearest position, for per-spot time series. Returns, per track,
 * the spot index in each frame (−1 where lost). Tracks are seeded from the frame with the most
 * spots; the matching radius is `radius_um`.
 */
export function trackSpots(frames: readonly SpotFrame[], radius_um: number): number[][] {
  if (frames.length === 0) return []
  let seed = 0
  for (let i = 1; i < frames.length; i++) if (frames[i].x_um.length > frames[seed].x_um.length) seed = i
  const nTracks = frames[seed].x_um.length
  const tracks: number[][] = Array.from({ length: nTracks }, () => new Array(frames.length).fill(-1))
  const r2 = radius_um * radius_um
  const run = (order: number[]) => {
    // last known position of every track along this sweep
    const last: { x: number; y: number; z: number }[] = []
    for (let k = 0; k < nTracks; k++) {
      tracks[k][seed] = k
      last[k] = { x: frames[seed].x_um[k], y: frames[seed].y_um[k], z: frames[seed].z_um[k] }
    }
    for (const to of order) {
      const f = frames[to]
      const used = new Uint8Array(f.x_um.length)
      for (let k = 0; k < nTracks; k++) {
        let best = -1
        let bestD = r2
        for (let j = 0; j < f.x_um.length; j++) {
          if (used[j]) continue
          const dx = f.x_um[j] - last[k].x
          const dy = f.y_um[j] - last[k].y
          const dz = 0.25 * (f.z_um[j] - last[k].z)
          const d = dx * dx + dy * dy + dz * dz
          if (d < bestD) {
            bestD = d
            best = j
          }
        }
        if (best >= 0) {
          used[best] = 1
          tracks[k][to] = best
          last[k] = { x: f.x_um[best], y: f.y_um[best], z: f.z_um[best] }
        }
      }
    }
  }
  const forward: number[] = []
  for (let i = seed + 1; i < frames.length; i++) forward.push(i)
  const backward: number[] = []
  for (let i = seed - 1; i >= 0; i--) backward.push(i)
  run(forward)
  run(backward)
  return tracks
}

/** Requested trajectory point at drive time `t` (clamped to [0, T]). */
export function requestedAt(traj: { t_us: number[]; x_um: number[]; y_um: number[]; z_um: number[] }, t: number): { x: number; y: number; z: number } {
  const ts = traj.t_us
  const n = ts.length
  if (n === 0) return { x: 0, y: 0, z: 0 }
  if (t <= ts[0]) return { x: traj.x_um[0], y: traj.y_um[0], z: traj.z_um[0] }
  if (t >= ts[n - 1]) return { x: traj.x_um[n - 1], y: traj.y_um[n - 1], z: traj.z_um[n - 1] }
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (ts[mid] <= t) lo = mid
    else hi = mid
  }
  const w = (t - ts[lo]) / (ts[hi] - ts[lo])
  return {
    x: traj.x_um[lo] * (1 - w) + traj.x_um[hi] * w,
    y: traj.y_um[lo] * (1 - w) + traj.y_um[hi] * w,
    z: traj.z_um[lo] * (1 - w) + traj.z_um[hi] * w,
  }
}

