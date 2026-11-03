/**
 * Position-based identity of tweezers across frames, and the per-site trap depth.
 *
 * The backend emits one spot per spatial cluster of a frequency group (API.md §5.1), ordered by
 * frequency group. On a fading drive the ladders slide through the groups, so array index `k`
 * names a different trap in consecutive frames: identity must come from *position*. Two
 * ingredients:
 *
 * - **Pairing** (`pairFrames`): spot `i` of frame `a` pairs with the nearest unpaired spot of
 *   frame `b` within `0.5·w0`, greedily, brightest first. Positions are compared in the frame
 *   co-moving with the array (`frameShift`): the whole spot pattern of an AODL drive translates
 *   rigidly (Table I: every lateral position is `(λF/v)·(f_B − f_A)` and the A tones are common
 *   to all traps), and a fast carry can move it by more than `0.5·w0` between two movie frames.
 *   The shift is chosen among the displacements of the brightest spots (≤ `2·w0`, zero
 *   preferred), so a spot that really jumped by several waists is never paired.
 * - **Sites** (`frameSites`): spots within `0.5·w0` of each other are one trap; its depth is the
 *   astigmatic-Gaussian peak `Σ 2P/(π w_x w_y)` (DESIGN §1 item 2, `aodl.field.focal.spot_params`;
 *   exact for an unwindowed single-term trap), with `w_x, w_y` the radii at the spot's own
 *   best focus (`SpotMetrics.wx/wy`).
 *
 * Units: µm; power on the `SpotMetrics.power` scale, so depths are per µm².
 */
import type { SpotFrame } from '../api/types'

/** Anything with a lateral position and a power. */
export interface Positioned {
  x_um: number
  y_um: number
  power: number
}

/** Pairing radius as a fraction of `w0` (the work order's `0.5·w0`). */
export const PAIR_RADIUS_W0 = 0.5
/** Largest rigid array displacement between two frames the pairing compensates [w0]. */
export const MAX_SHIFT_W0 = 2

/** Indices sorted by descending power. */
function byPower(s: readonly Positioned[]): number[] {
  return s.map((_, i) => i).sort((i, j) => s[j].power - s[i].power)
}

/**
 * Greedy brightest-first pairing of `a` onto `b` after shifting `a` by `(sx, sy)`. Returns
 * `pair[i]` = index into `b` or −1, plus the power overlap `Σ min(P_a, P_b)` of the pairs.
 */
function greedyPair(a: readonly Positioned[], b: readonly Positioned[], r: number, sx: number, sy: number): { pair: Int32Array; score: number } {
  const pair = new Int32Array(a.length).fill(-1)
  const used = new Uint8Array(b.length)
  const r2 = r * r
  let score = 0
  for (const i of byPower(a)) {
    const x = a[i].x_um + sx
    const y = a[i].y_um + sy
    let best = -1
    let bestD = r2
    for (let j = 0; j < b.length; j++) {
      if (used[j]) continue
      const dx = b[j].x_um - x
      const dy = b[j].y_um - y
      const d = dx * dx + dy * dy
      if (d < bestD) {
        bestD = d
        best = j
      }
    }
    if (best >= 0) {
      used[best] = 1
      pair[i] = best
      score += Math.min(a[i].power, b[best].power)
    }
  }
  return { pair, score }
}

/**
 * Rigid lateral displacement `(sx, sy)` of the spot pattern from `a` to `b` [µm]: the candidate
 * (zero, or the displacement from one of the three brightest spots of `a` to a spot of `b` within
 * `maxShift`) whose greedy pairing carries the most power; ties go to the smaller shift.
 */
export function frameShift(a: readonly Positioned[], b: readonly Positioned[], r: number, maxShift: number): [number, number] {
  if (a.length === 0 || b.length === 0) return [0, 0]
  const cands: [number, number][] = [[0, 0]]
  const m2 = maxShift * maxShift
  for (const i of byPower(a).slice(0, 3)) {
    for (const s of b) {
      const dx = s.x_um - a[i].x_um
      const dy = s.y_um - a[i].y_um
      if (dx * dx + dy * dy <= m2) cands.push([dx, dy])
    }
  }
  let best: [number, number] = [0, 0]
  let bestScore = -1
  let bestNorm = Infinity
  let total = 0
  for (const s of a) total += s.power
  const tol = 1e-9 * total
  for (const c of cands) {
    const { score } = greedyPair(a, b, r, c[0], c[1])
    const norm = Math.hypot(c[0], c[1])
    if (score > bestScore + tol || (Math.abs(score - bestScore) <= tol && norm < bestNorm)) {
      best = c
      bestScore = score
      bestNorm = norm
    }
  }
  return best
}

/** Pairing of two frames: `pair[i]` = index into `b` paired with spot `i` of `a` (−1 if none). */
export interface FramePairing {
  pair: Int32Array
  /** `paired[j]` = 1 when spot `j` of `b` is paired */
  paired: Uint8Array
  shift: [number, number]
}

/** Pair the spots of two frames by position (module doc). `w0` in µm. */
export function pairFrames(a: readonly Positioned[], b: readonly Positioned[], w0: number): FramePairing {
  const r = PAIR_RADIUS_W0 * w0
  const shift = frameShift(a, b, r, MAX_SHIFT_W0 * w0)
  const { pair } = greedyPair(a, b, r, shift[0], shift[1])
  const paired = new Uint8Array(b.length)
  for (const j of pair) if (j >= 0) paired[j] = 1
  return { pair, paired, shift }
}

/** One trap of a frame: spots within `0.5·w0` merged. */
export interface Site {
  x_um: number
  y_um: number
  z_um: number
  power: number
  /** astigmatic-Gaussian peak `Σ 2P/(π w_x w_y)` [per µm², `SpotMetrics.power` scale] */
  depth: number
  /** spot indices of the frame merged into this site */
  members: number[]
}

/** Peak `2P/(π w_x w_y)` of spot `k` at its own best focus. */
export function gaussPeakAtFocus(f: SpotFrame, k: number): number {
  const wx = f.wx_um[k]
  const wy = f.wy_um[k]
  if (!(wx > 0 && wy > 0)) return 0
  return (2 * f.power[k]) / (Math.PI * wx * wy)
}

/**
 * Merge the spots of a frame into sites, brightest first: a spot within `0.5·w0` of an existing
 * site's seed joins it (the dying and rising rung pairs of a hand-over, API.md §5.1), else it
 * seeds a new site. Positions are power-weighted.
 */
export function frameSites(f: SpotFrame, w0: number): Site[] {
  const r2 = (PAIR_RADIUS_W0 * w0) ** 2
  const n = f.x_um.length
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => f.power[j] - f.power[i])
  const sites: (Site & { sx: number; sy: number })[] = []
  for (const k of order) {
    let host: (typeof sites)[number] | null = null
    let bestD = r2
    for (const s of sites) {
      const dx = f.x_um[k] - s.sx
      const dy = f.y_um[k] - s.sy
      const d = dx * dx + dy * dy
      if (d < bestD) {
        bestD = d
        host = s
      }
    }
    const p = f.power[k]
    const pk = gaussPeakAtFocus(f, k)
    if (!host) {
      sites.push({ sx: f.x_um[k], sy: f.y_um[k], x_um: f.x_um[k], y_um: f.y_um[k], z_um: f.z_um[k], power: p, depth: pk, members: [k] })
      continue
    }
    const tot = host.power + p
    if (tot > 0) {
      host.x_um = (host.x_um * host.power + f.x_um[k] * p) / tot
      host.y_um = (host.y_um * host.power + f.y_um[k] * p) / tot
      host.z_um = (host.z_um * host.power + f.z_um[k] * p) / tot
    }
    host.power = tot
    host.depth += pk
    host.members.push(k)
  }
  return sites.map(({ x_um, y_um, z_um, power, depth, members }) => ({ x_um, y_um, z_um, power, depth, members }))
}

/** Trap-depth time series of the whole movie. */
export interface TrapDepthSeries {
  /** per track, depth / plateau per frame (0 where the site is absent) */
  depth: Float64Array[]
  /** per track, representative position (where first seen) [µm] */
  where: { x_um: number; y_um: number }[]
  /** the plateau: median over lit frames of the brightest site's depth [per µm²] */
  plateau: number
  /** per frame, the brightest site's depth / plateau */
  brightest: Float64Array
}

/** Median of a list (NaN when empty). */
export function median(v: readonly number[]): number {
  if (v.length === 0) return NaN
  const s = [...v].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m])
}

/**
 * Track the sites of every frame through the movie by position (within `0.5·w0` of a track's
 * last position carried along with the array's rigid shift, brightest first; an unmatched site
 * opens a new track) and normalise every depth to the plateau. Tracks whose peak never reaches
 * `minRel` of the plateau are dropped.
 */
export function trapDepths(frames: readonly SpotFrame[], w0: number, minRel = 0.05): TrapDepthSeries {
  const n = frames.length
  const sites = frames.map((f) => frameSites(f, w0))
  const brightestAbs = sites.map((ss) => ss.reduce((m, s) => Math.max(m, s.depth), 0))
  const plateau = median(brightestAbs.filter((d) => d > 0)) || 0
  const r2 = (PAIR_RADIUS_W0 * w0) ** 2
  const tracks: { x: number; y: number; d: Float64Array; where: { x_um: number; y_um: number } }[] = []
  let prev: Site[] = []
  for (let i = 0; i < n; i++) {
    const cur = sites[i]
    const [sx, sy] = frameShift(prev, cur, PAIR_RADIUS_W0 * w0, MAX_SHIFT_W0 * w0)
    for (const tk of tracks) {
      tk.x += sx
      tk.y += sy
    }
    const used: boolean[] = tracks.map(() => false)
    const order = cur.map((_, k) => k).sort((a, b) => cur[b].depth - cur[a].depth)
    for (const k of order) {
      const s = cur[k]
      let best = -1
      let bestD = r2
      for (let q = 0; q < tracks.length; q++) {
        if (used[q]) continue
        const dx = tracks[q].x - s.x_um
        const dy = tracks[q].y - s.y_um
        const d = dx * dx + dy * dy
        if (d < bestD) {
          bestD = d
          best = q
        }
      }
      if (best < 0) {
        tracks.push({ x: s.x_um, y: s.y_um, d: new Float64Array(n), where: { x_um: s.x_um, y_um: s.y_um } })
        best = tracks.length - 1
        used.push(false)
      }
      used[best] = true
      tracks[best].x = s.x_um
      tracks[best].y = s.y_um
      tracks[best].d[i] = plateau > 0 ? s.depth / plateau : 0
    }
    prev = cur
  }
  const kept = tracks.filter((tk) => tk.d.some((v) => v >= minRel))
  return {
    depth: kept.map((tk) => tk.d),
    where: kept.map((tk) => tk.where),
    plateau,
    brightest: Float64Array.from(brightestAbs, (d) => (plateau > 0 ? d / plateau : 0)),
  }
}
