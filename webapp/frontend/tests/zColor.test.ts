import { describe, expect, it } from 'vitest'
import { GAMMA, RDBU_R_ANCHORS, rdbuR, zColor, zColorCss, zMaxFrom, zNorm } from '../src/physics/zColor'

describe('zColor', () => {
  it('is white-ish (1, 1, 1) at Z = 0 after luminance normalisation', () => {
    const c = zColor(0, 5)
    expect(c[0]).toBeCloseTo(1, 12)
    expect(c[1]).toBeCloseTo(246 / 247, 12)
    expect(c[2]).toBeCloseTo(246 / 247, 12)
    expect(zColorCss(0, 5)).toBe('rgb(255, 254, 254)')
  })
  it('has its brightest channel at 1 everywhere, red above and blue below', () => {
    for (const z of [-10, -3, -1, 0.2, 2, 10]) {
      const c = zColor(z, 4)
      expect(Math.max(...c)).toBeCloseTo(1, 12)
    }
    const above = zColor(4, 4)
    const below = zColor(-4, 4)
    expect(above[0]).toBe(1)
    expect(above[2]).toBeLessThan(0.5)
    expect(below[2]).toBe(1)
    expect(below[0]).toBeLessThan(0.1)
  })
  it('interpolates the ColorBrewer anchors linearly', () => {
    expect(rdbuR(0)).toEqual(RDBU_R_ANCHORS[0].map((v) => v / 255))
    expect(rdbuR(1)).toEqual(RDBU_R_ANCHORS[10].map((v) => v / 255))
    const mid = rdbuR(0.05)
    expect(mid[0]).toBeCloseTo((5 + 32) / 2 / 255, 12)
  })
  it('z_norm is symmetric about zero, saturates and floors z_max at 1 µm', () => {
    expect(zNorm(0, 3)).toBe(0.5)
    expect(zNorm(3, 3)).toBe(1)
    expect(zNorm(-30, 3)).toBe(0)
    expect(zNorm(0.5, 0.1)).toBe(0.75)
    expect(zMaxFrom([0.1, -0.2])).toBe(1)
    expect(zMaxFrom([2, -7])).toBe(7)
    expect(GAMMA).toBe(0.7)
  })
})
