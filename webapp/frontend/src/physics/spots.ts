/**
 * Frame-level helpers on the simulate response: frame lookup / interpolation at the cursor,
 * the XZ row choice (`aodl.engine.SimResult.spot_row`), the shadow-tweezer badge and the
 * frame-to-frame spot matching used by the per-spot metric plots.
 */
import type { Bounds, FadeEvent, SimulateResponse, SpotFrame } from '../api/types'
import { frameSpots, type SpotLike } from './gaussianBeam'
import { pairFrames, type FramePairing } from './traps'

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

const pairingCache = new WeakMap<SpotFrame, { b: SpotFrame; w0: number; spotsA: SpotLike[]; spotsB: SpotLike[]; pairing: FramePairing }>()

/** Position pairing of two consecutive frames, cached per frame object (it does not depend on `t`). */
function cachedPairing(a: SpotFrame, b: SpotFrame, w0: number) {
  const hit = pairingCache.get(a)
  if (hit && hit.b === b && hit.w0 === w0) return hit
  const spotsA = frameSpots(a)
  const spotsB = frameSpots(b)
  const entry = { b, w0, spotsA, spotsB, pairing: pairFrames(spotsA, spotsB, w0) }
  pairingCache.set(a, entry)
  return entry
}

/**
 * Spots at cursor time `t`, blended between the bracketing frames `a`, `b` (weight `w` of `b`)
 * **by position**, never by array index (the backend orders spots by frequency group, and on a
 * fading drive the ladder slides through the groups): spot `i` of `a` pairs with the nearest
 * unpaired spot of `b` within `0.5·w0`, brightest first, in the frame co-moving with the array
 * (`physics/traps.ts`). Paired spots interpolate position, `z`, `ΔF` and power; an unpaired spot
 * of `a` stays where it is with power `(1 − w)·P`, an unpaired spot of `b` with `w·P` — a column
 * that switches on fades in over one frame interval instead of jumping, and nothing moves that
 * the physics did not move. `w0 = sim.constants.waist0_um`. Before the first / after the last
 * frame the end frames are returned unchanged.
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
  const empty = a.x_um.length === 0 || b.x_um.length === 0
  // an empty (dark, t < τ/2) frame has no meaningful tracked plane: take the lit neighbour's
  const zt = empty ? sim.tracked_z_um[a.x_um.length === 0 ? i + 1 : i] : sim.tracked_z_um[i] * (1 - w) + sim.tracked_z_um[i + 1] * w
  const { spotsA, spotsB, pairing } = cachedPairing(a, b, sim.constants.waist0_um)
  const spots: SpotLike[] = []
  for (let k = 0; k < spotsA.length; k++) {
    const sa = spotsA[k]
    const j = pairing.pair[k]
    if (j < 0) {
      const p = (1 - w) * sa.power
      if (p > 0) spots.push({ ...sa, power: p })
      continue
    }
    const sb = spotsB[j]
    spots.push({
      x_um: sa.x_um * (1 - w) + sb.x_um * w,
      y_um: sa.y_um * (1 - w) + sb.y_um * w,
      z_um: sa.z_um * (1 - w) + sb.z_um * w,
      delta_f_um: sa.delta_f_um * (1 - w) + sb.delta_f_um * w,
      power: sa.power * (1 - w) + sb.power * w,
    })
  }
  for (let j = 0; j < spotsB.length; j++) {
    if (pairing.paired[j]) continue
    const p = w * spotsB[j].power
    if (p > 0) spots.push({ ...spotsB[j], power: p })
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

