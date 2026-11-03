/**
 * Tone / channel evaluation from the waveform IR (API.md §4.1).
 *
 * `V(t) = Σ_n A_n(t) cos(2π f_center t + phase_n(t) + phase0_n)` — carrier included, **no**
 * renormalisation (the golden fixtures use the same convention). `f_center` is in MHz and `t`
 * in µs, so `2π f_center[Hz] t[s] = 2π f_center_MHz t_us` exactly.
 */
import type { ChannelIR, ToneIR } from '../api/types'
import { envA } from './envelope'
import { evalPoly, polyDomain } from './piecewisePoly'

const TWO_PI = 2 * Math.PI

/** Detuning from `f_center` [MHz] at `t` [µs] (clamp-held). */
export function toneFreqMHz(tone: ToneIR, t: number): number {
  return evalPoly(tone.freq, t)
}

/** Absolute RF frequency `f_center + f(t)` [MHz]. */
export function toneAbsFreqMHz(channel: ChannelIR, tone: ToneIR, t: number): number {
  return channel.f_center_MHz + evalPoly(tone.freq, t)
}

/** Rotating-frame phase `phase(t) + phase0` [rad]. */
export function tonePhaseRad(tone: ToneIR, t: number): number {
  return evalPoly(tone.phase, t) + tone.phase0
}

/** Envelope `A_n(t)`. */
export function toneAmp(tone: ToneIR, t: number): number {
  return envA(tone.env, t)
}

/** `V(t)` of one channel on the given times [µs], carrier included. */
export function voltage(channel: ChannelIR, t_us: ArrayLike<number>, out?: Float64Array): Float64Array {
  const n = t_us.length
  const v = out ?? new Float64Array(n)
  v.fill(0)
  const fc = channel.f_center_MHz
  for (const tone of channel.tones) {
    for (let i = 0; i < n; i++) {
      const t = t_us[i]
      const a = envA(tone.env, t)
      if (a === 0) continue
      v[i] += a * Math.cos(TWO_PI * fc * t + evalPoly(tone.phase, t) + tone.phase0)
    }
  }
  return v
}

/** `V(t')` on 2001 points over `[t − half, t + half]` (the RF panel's voltage zoom). */
export function voltageZoom(channel: ChannelIR, t_us: number, half_us = 0.2, n = 2001): { t: Float64Array; v: Float64Array } {
  const t = new Float64Array(n)
  for (let i = 0; i < n; i++) t[i] = t_us - half_us + (2 * half_us * i) / (n - 1)
  return { t, v: voltage(channel, t) }
}

/** One tone's plot-ready track: absolute frequency [MHz] with `NaN` where `A == 0`, and `alpha = A`. */
export interface ToneTrack {
  t: Float64Array
  f_MHz: Float64Array // NaN where the envelope is zero so the line breaks
  alpha: Float64Array // envelope value in [0, 1]
  live: boolean // any sample with A > 0
  f_live_min: number // over the live samples (Infinity if never live)
  f_live_max: number
  t_live_start: number // first/last sample time with A > 0 (NaN if never live)
  t_live_end: number
}

/**
 * Tracks of every tone of a channel on `n` samples over the channel's programmed span
 * (`liveTracks`, computed once per plan). The frequency is absolute (`f_center + f`).
 */
export function liveTracks(channel: ChannelIR, n = 1500, span?: [number, number]): ToneTrack[] {
  const [t0, t1] = span ?? channelSpan(channel)
  const t = new Float64Array(n)
  for (let i = 0; i < n; i++) t[i] = t0 + ((t1 - t0) * i) / (n - 1)
  const fc = channel.f_center_MHz
  return channel.tones.map((tone) => {
    const f = new Float64Array(n)
    const alpha = new Float64Array(n)
    let live = false
    let fmin = Infinity
    let fmax = -Infinity
    let tStart = NaN
    let tEnd = NaN
    for (let i = 0; i < n; i++) {
      const a = envA(tone.env, t[i])
      alpha[i] = a
      if (a > 0) {
        const fi = fc + evalPoly(tone.freq, t[i])
        f[i] = fi
        live = true
        if (fi < fmin) fmin = fi
        if (fi > fmax) fmax = fi
        if (Number.isNaN(tStart)) tStart = t[i]
        tEnd = t[i]
      } else {
        f[i] = NaN
      }
    }
    return { t, f_MHz: f, alpha, live, f_live_min: fmin, f_live_max: fmax, t_live_start: tStart, t_live_end: tEnd }
  })
}

/** Union of the tone frequency domains of a channel [µs]. */
export function channelSpan(channel: ChannelIR): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const tone of channel.tones) {
    const [a, b] = polyDomain(tone.freq)
    if (a < lo) lo = a
    if (b > hi) hi = b
  }
  if (!Number.isFinite(lo)) return [0, 0]
  return [lo, hi]
}

/**
 * Conservative live interval `[t_a, t_b]` of a tone (where `A` can be non-zero), used to skip
 * dead rungs in per-frame evaluations. Exact for `constant` and `smooth_on_off`; for
 * `fade_zone` it brackets the sampled live span by `margin` samples on each side.
 */
export function toneLiveInterval(tone: ToneIR, span: [number, number], n = 2000, margin = 2): [number, number] {
  const env = tone.env
  if (env.kind === 'constant') return env.amp > 0 ? [-Infinity, Infinity] : [Infinity, -Infinity]
  if (env.kind === 'smooth_on_off') return [env.t_on_us, env.t_off_us]
  const [t0, t1] = span
  const dt = (t1 - t0) / (n - 1)
  let first = -1
  let last = -1
  for (let i = 0; i < n; i++) {
    if (envA(env, t0 + dt * i) > 0) {
      if (first < 0) first = i
      last = i
    }
  }
  if (first < 0) return [Infinity, -Infinity]
  // Beyond the sampled span the polynomial clamp-holds: if live at an end, extend to infinity.
  const a = first === 0 ? -Infinity : t0 + dt * (first - margin)
  const b = last === n - 1 ? Infinity : t0 + dt * (last + margin)
  return [a, b]
}
