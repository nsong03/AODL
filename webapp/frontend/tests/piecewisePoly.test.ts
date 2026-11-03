import { describe, expect, it } from 'vitest'
import type { Poly } from '../src/api/types'
import { derivative, evalPoly, evalPolyArray, polyRangeSampled, segmentIndex } from '../src/physics/piecewisePoly'

// Two segments: [0, 10) -> 1 + 2τ, [10, 30] -> 3 − τ² (values in "MHz", breaks in µs).
const poly: Poly = { breaks_us: [0, 10, 30], coeffs: [[1, 2], [3, 0, -1]] }

describe('evalPoly', () => {
  it('evaluates in normalized local time', () => {
    expect(evalPoly(poly, 0)).toBeCloseTo(1, 12)
    expect(evalPoly(poly, 5)).toBeCloseTo(2, 12)
    expect(evalPoly(poly, 20)).toBeCloseTo(3 - 0.25, 12)
  })
  it('assigns an interior break to the right-hand segment (searchsorted side="right")', () => {
    expect(evalPoly(poly, 10)).toBeCloseTo(3, 12) // not 1 + 2 = 3 by coincidence? check derivative-free: segment 2 at τ=0 is 3
    const p2: Poly = { breaks_us: [0, 10, 30], coeffs: [[1, 2], [7, 0, -1]] }
    expect(evalPoly(p2, 10)).toBeCloseTo(7, 12)
  })
  it('gives the last break to the last segment', () => {
    expect(segmentIndex(poly.breaks_us, 30)).toBe(1)
    expect(evalPoly(poly, 30)).toBeCloseTo(2, 12)
  })
  it('clamp-holds outside the domain', () => {
    expect(evalPoly(poly, -100)).toBeCloseTo(1, 12)
    expect(evalPoly(poly, 1e6)).toBeCloseTo(2, 12)
  })
  it('vectorised form matches the scalar one', () => {
    const t = [-1, 0, 3, 10, 15, 30, 31]
    const v = evalPolyArray(poly, t)
    t.forEach((ti, i) => expect(v[i]).toBe(evalPoly(poly, ti)))
  })
})

describe('derivative', () => {
  it('carries the 1/T Jacobian', () => {
    const d = derivative(poly)
    expect(evalPoly(d, 5)).toBeCloseTo(2 / 10, 12)
    // d/dt (3 − τ²) = −2τ/T, τ = 0.5 at t = 20, T = 20
    expect(evalPoly(d, 20)).toBeCloseTo(-2 * 0.5 / 20, 12)
  })
  it('of a constant is zero', () => {
    const c: Poly = { breaks_us: [0, 1], coeffs: [[4]] }
    expect(evalPoly(derivative(c), 0.5)).toBe(0)
  })
})

describe('polyRangeSampled', () => {
  it('brackets the sampled values', () => {
    const [lo, hi] = polyRangeSampled(poly, 0, 30)
    expect(lo).toBeCloseTo(1, 6)
    expect(hi).toBeCloseTo(3, 6)
  })
})
