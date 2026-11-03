import { describe, expect, it } from 'vitest'
import type { FadeZoneEnv, SmoothOnOffEnv } from '../src/api/types'
import { envA, fadeWindow, fadeZone, smoothOnOff } from '../src/physics/envelope'

describe('smoothOnOff', () => {
  const env: SmoothOnOffEnv = { kind: 'smooth_on_off', t_on_us: 10, t_off_us: 50, ramp_us: 8 }
  it('is 0 before t_on, 1 on the plateau, 0 at and after t_off (half-open intervals)', () => {
    expect(smoothOnOff(env, 9.999)).toBe(0)
    expect(smoothOnOff(env, 10)).toBe(0) // sin²(0)
    expect(smoothOnOff(env, 18)).toBe(1)
    expect(smoothOnOff(env, 30)).toBe(1)
    expect(smoothOnOff(env, 42)).toBe(1) // t_off − ramp belongs to the fall branch: sin²(c·8) = 1
    expect(smoothOnOff(env, 50)).toBe(0)
    expect(smoothOnOff(env, 60)).toBe(0)
  })
  it('rises as sin² with c = π/(2 ramp)', () => {
    const c = Math.PI / 16
    expect(smoothOnOff(env, 12)).toBeCloseTo(Math.sin(c * 2) ** 2, 14)
    expect(smoothOnOff(env, 14)).toBeCloseTo(0.5, 14)
    expect(smoothOnOff(env, 46)).toBeCloseTo(0.5, 14)
  })
})

describe('fadeWindow (Eq. S26/S27)', () => {
  it('plateau, shoulder, zero with the documented boundaries', () => {
    const df = 8
    const eta = 0.5
    expect(fadeWindow(0, df, eta, 0.5, 1, 1)).toBe(1)
    expect(fadeWindow(0.25 * df, df, eta, 0.5, 1, 1)).toBe(1) // |g| = (m−η)Δf/2 = 2
    expect(fadeWindow(0.75 * df, df, eta, 0.5, 1, 1)).toBe(0) // (m+η)Δf/2 = 6
    expect(fadeWindow(-7, df, eta, 0.5, 1, 1)).toBe(0)
    // mid-shoulder: θ = (π/2η)(|g|/Δf − m/2) + π/4 with |g|/Δf = 0.5 → θ = π/4
    expect(fadeWindow(4, df, eta, 0.5, 1, 1)).toBeCloseTo(Math.cos(Math.PI / 4) ** 0.5, 14)
    expect(fadeWindow(4, df, eta, 1, 1, 0.5)).toBeCloseTo(0.5 * Math.cos(Math.PI / 4), 14)
  })
  it('p = 0 is a rectangle of width (m+η)Δf', () => {
    expect(fadeWindow(5.9, 8, 0.5, 0, 1, 1)).toBe(1)
    expect(fadeWindow(6.0, 8, 0.5, 0, 1, 1)).toBe(0)
  })
  it('m = M widens the window (Eq. S27)', () => {
    expect(fadeWindow(10, 8, 0.5, 0.5, 4, 1)).toBe(1) // inner = (4−0.5)·4 = 14
    expect(fadeWindow(17.9, 8, 0.5, 0.5, 4, 1)).toBeGreaterThan(0)
    expect(fadeWindow(18, 8, 0.5, 0.5, 4, 1)).toBe(0)
  })
  it('fadeZone composes the window with g(t)', () => {
    const env: FadeZoneEnv = { kind: 'fade_zone', g: { breaks_us: [0, 100], coeffs: [[-10, 20]] }, delta_f_MHz: 8, eta: 0.5, p: 0.5, m: 1, amp: 1 }
    expect(fadeZone(env, 50)).toBe(1) // g = 0
    expect(fadeZone(env, 0)).toBe(0) // g = −10
    expect(envA(env, 70)).toBeCloseTo(fadeWindow(4, 8, 0.5, 0.5, 1, 1), 14) // g = 4
    expect(envA({ kind: 'constant', amp: 0.3 }, 123)).toBe(0.3)
  })
})
