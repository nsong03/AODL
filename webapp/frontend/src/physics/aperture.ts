/**
 * Acoustic content inside one AOD crystal at observation time `t` (API.md §5.3,
 * `docs/conventions.md` §7). Aperture coordinate `u` runs along the channel's axis with the
 * transducer at `u = −s·D/2`; the sample at `u` was emitted at the retarded time
 * `t_ret = t − τ/2 − s·u/v` and the aperture holds content only where `s·u ≤ v·t − D/2`
 * (`aodl.device.conventions.is_filled`). Signs (`sound_sign`) come from the IR, never from here.
 */
import type { ChannelIR, ToneIR } from '../api/types'
import { envA } from './envelope'
import { evalPoly } from './piecewisePoly'
import { MHz, mm, us } from './units'

const TWO_PI = 2 * Math.PI

export interface ApertureContent {
  u_mm: Float64Array
  envelope: Float64Array // Σ A_n(t_ret(u)); 0 where unfilled
  f_inst_MHz: Float64Array // f_center + Σ A_n f_n / Σ A_n; NaN where the envelope is 0
  filled: Uint8Array // 1 where s·u ≤ v·t − D/2
  t_ret_us: Float64Array
}

/** `t_ret(u) = t − τ/2 − s·u/v` in µs for `u` in mm. */
export function retardedTimeUs(channel: ChannelIR, t_us: number, u_mm: number): number {
  return t_us - 0.5 * channel.tau_us - (channel.sound_sign * (u_mm * mm)) / channel.sound_speed_m_s / us
}

/** Fill mask `s·u ≤ v·t − D/2` evaluated in SI, as `aodl.device.conventions.is_filled` does. */
export function isFilled(channel: ChannelIR, t_us: number, u_mm: number): boolean {
  const reach = channel.sound_speed_m_s * (t_us * us) - 0.5 * channel.aperture_mm * mm
  return channel.sound_sign * (u_mm * mm) <= reach
}

/**
 * Envelope and instantaneous frequency across the crystal on `n` points over `[−D/2, D/2]`.
 * `tones` may restrict the evaluation to the tones that can be live (a display optimisation;
 * dead rungs contribute exactly zero).
 */
export function apertureContent(channel: ChannelIR, t_us: number, n = 512, tones?: readonly ToneIR[]): ApertureContent {
  const half = 0.5 * channel.aperture_mm
  const u = new Float64Array(n)
  const tRet = new Float64Array(n)
  const filled = new Uint8Array(n)
  const env = new Float64Array(n)
  const fsum = new Float64Array(n)
  const fInst = new Float64Array(n)
  const reach = channel.sound_speed_m_s * (t_us * us) - 0.5 * channel.aperture_mm * mm
  for (let i = 0; i < n; i++) {
    u[i] = -half + (2 * half * i) / (n - 1)
    tRet[i] = retardedTimeUs(channel, t_us, u[i])
    filled[i] = channel.sound_sign * (u[i] * mm) <= reach ? 1 : 0
  }
  const list = tones ?? channel.tones
  for (const tone of list) {
    for (let i = 0; i < n; i++) {
      if (!filled[i]) continue
      const a = envA(tone.env, tRet[i])
      if (a === 0) continue
      env[i] += a
      fsum[i] += a * evalPoly(tone.freq, tRet[i])
    }
  }
  for (let i = 0; i < n; i++) {
    fInst[i] = env[i] > 0 ? channel.f_center_MHz + fsum[i] / env[i] : NaN
  }
  return { u_mm: u, envelope: env, f_inst_MHz: fInst, filled, t_ret_us: tRet }
}

/**
 * The literal carrier-included waveform on the crystal, `V(u) = Σ A_n(t_ret) cos(2π f_center
 * t_ret + phase_n(t_ret) + phase0_n)`, zero where unfilled (`aodl.device.aod.aperture_window`),
 * over `[u0, u1]` mm on `m ≥ 4096` points — the only honest way to draw the 100 MHz carrier.
 */
export function apertureVoltage(channel: ChannelIR, t_us: number, u0_mm: number, u1_mm: number, m = 4096, tones?: readonly ToneIR[]): { u_mm: Float64Array; v: Float64Array } {
  const n = Math.max(m, 4096)
  const u = new Float64Array(n)
  const v = new Float64Array(n)
  const fc = channel.f_center_MHz
  const list = tones ?? channel.tones
  for (let i = 0; i < n; i++) {
    u[i] = u0_mm + ((u1_mm - u0_mm) * i) / (n - 1)
    if (!isFilled(channel, t_us, u[i])) continue
    const tr = retardedTimeUs(channel, t_us, u[i])
    let s = 0
    for (const tone of list) {
      const a = envA(tone.env, tr)
      if (a === 0) continue
      s += a * Math.cos(TWO_PI * fc * tr + evalPoly(tone.phase, tr) + tone.phase0)
    }
    v[i] = s
  }
  return { u_mm: u, v }
}

/** Acoustic wavelength `v / f_center` in mm (for the zoom-window caption). */
export function acousticWavelengthMm(channel: ChannelIR): number {
  return channel.sound_speed_m_s / (channel.f_center_MHz * MHz) / mm
}
