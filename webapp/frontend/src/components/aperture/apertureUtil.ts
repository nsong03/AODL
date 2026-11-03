import type { ToneIR } from '../../api/types'

/** Frequency colour scale across the band: blue (low) → cyan → white (centre) → orange → red (high). */
const F_ANCHORS: [number, number, number][] = [
  [60, 100, 230],
  [76, 201, 240],
  [245, 245, 245],
  [255, 183, 3],
  [230, 70, 60],
]
export function freqColor(x: number): string {
  const xx = x < 0 ? 0 : x > 1 ? 1 : x
  const pos = xx * (F_ANCHORS.length - 1)
  const i = Math.min(Math.floor(pos), F_ANCHORS.length - 2)
  const f = pos - i
  const a = F_ANCHORS[i]
  const b = F_ANCHORS[i + 1]
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)}, ${Math.round(a[1] + (b[1] - a[1]) * f)}, ${Math.round(a[2] + (b[2] - a[2]) * f)})`
}

export interface LiveTone {
  tone: ToneIR
  a: number
  b: number
}

/** Tones whose live interval can intersect `[t − τ, t]` (the aperture holds that drive segment). */
export function tonesInWindow(live: LiveTone[], t: number, tau: number): ToneIR[] {
  const lo = t - tau - 1e-9
  const hi = t + 1e-9
  return live.filter((l) => l.b >= lo && l.a <= hi).map((l) => l.tone)
}

