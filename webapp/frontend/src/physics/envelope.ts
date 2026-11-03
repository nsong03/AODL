/**
 * Parametric amplitude envelopes of the waveform IR, evaluated exactly (DESIGN §1 item 4):
 * `ConstantEnvelope`, `SmoothOnOff` (aodl.waveform.tones) and `FadeZoneEnvelope`
 * (aodl.waveform.shepard, Eqs. S26/S27). Times in µs, as in the IR.
 */
import type { Envelope, FadeZoneEnv, SmoothOnOffEnv } from '../api/types'
import { evalPoly } from './piecewisePoly'

const PI_2 = 0.5 * Math.PI
const PI_4 = 0.25 * Math.PI

/** `aodl.waveform.tones.SmoothOnOff.A`: off, sin² rise, flat, sin² fall, off (half-open). */
export function smoothOnOff(env: SmoothOnOffEnv, t: number): number {
  const { t_on_us: tOn, t_off_us: tOff, ramp_us: ramp } = env
  const c = Math.PI / (2 * ramp)
  if (t < tOn) return 0
  if (t < tOn + ramp) {
    const s = Math.sin(c * (t - tOn))
    return s * s
  }
  if (t < tOff - ramp) return 1
  if (t < tOff) {
    const s = Math.sin(c * (tOff - t))
    return s * s
  }
  return 0
}

/**
 * The Eq. S26/S27 fade window as a function of the fade coordinate `g` [MHz]:
 * `A = amp` for `|g| ≤ (m − η)Δf/2`, `0` for `|g| ≥ (m + η)Δf/2`, `amp·cos^p θ` between with
 * `θ = (π/2η)(|g|/Δf − m/2) + π/4`  (`aodl.waveform.shepard.fade_window`).
 */
export function fadeWindow(g: number, deltaF: number, eta: number, p: number, m: number, amp: number): number {
  const u = Math.abs(g)
  const inner = 0.5 * (m - eta) * deltaF
  const outer = 0.5 * (m + eta) * deltaF
  if (u <= inner) return amp
  if (u >= outer) return 0
  const k = Math.PI / (2 * eta * deltaF)
  let theta = k * (u - 0.5 * m * deltaF) + PI_4
  if (theta < 0) theta = 0
  if (theta > PI_2) theta = PI_2
  return amp * Math.pow(Math.cos(theta), p)
}

/** `FadeZoneEnvelope.A(t) = fade_window(g(t))` with `g` its own piecewise polynomial [MHz]. */
export function fadeZone(env: FadeZoneEnv, t: number): number {
  const g = evalPoly(env.g, t)
  return fadeWindow(g, env.delta_f_MHz, env.eta, env.p, env.m, env.amp)
}

/** Envelope value `A(t)` for any IR envelope record. */
export function envA(env: Envelope, t: number): number {
  switch (env.kind) {
    case 'constant':
      return env.amp
    case 'smooth_on_off':
      return smoothOnOff(env, t)
    case 'fade_zone':
      return fadeZone(env, t)
  }
}

/** Vectorised `envA`. */
export function envAArray(env: Envelope, t: ArrayLike<number>, out?: Float64Array): Float64Array {
  const res = out ?? new Float64Array(t.length)
  for (let i = 0; i < t.length; i++) res[i] = envA(env, t[i])
  return res
}

/** Peak amplitude of an envelope (`amp` for the parametric ones, 1 for a gate). */
export function envPeak(env: Envelope): number {
  return env.kind === 'smooth_on_off' ? 1 : env.amp
}
