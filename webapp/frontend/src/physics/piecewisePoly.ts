/**
 * Port of `aodl.poly.PiecewisePoly.__call__` / `derivative` for the API's `Poly` record.
 *
 * Segment `k` covers `[breaks[k], breaks[k+1])` and evaluates as
 * `Σ_j coeffs[k][j] · τ^j`, `τ = (t − breaks[k]) / (breaks[k+1] − breaks[k])` — the
 * normalized local time, so the value does not depend on the time unit the breaks are in
 * (the API ships µs). The last break belongs to the last segment (`searchsorted(...,
 * side="right") − 1`, clipped), and the polynomial is **clamp-held outside its domain**.
 */
import type { Poly } from '../api/types'

/** `numpy.searchsorted(breaks, t, side="right") - 1`, clipped to `[0, K-1]`. */
export function segmentIndex(breaks: readonly number[], t: number): number {
  const nSeg = breaks.length - 1
  // binary search for the last index i with breaks[i] <= t
  let lo = 0
  let hi = breaks.length // exclusive
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (breaks[mid] <= t) lo = mid + 1
    else hi = mid
  }
  let idx = lo - 1
  if (idx < 0) idx = 0
  if (idx > nSeg - 1) idx = nSeg - 1
  return idx
}

/** Evaluate `poly` at one time (same unit as `breaks_us`), clamp-held outside the domain. */
export function evalPoly(poly: Poly, t: number): number {
  const breaks = poly.breaks_us
  const nSeg = breaks.length - 1
  if (nSeg < 1) return NaN
  const t0 = breaks[0]
  const t1 = breaks[nSeg]
  const tc = t < t0 ? t0 : t > t1 ? t1 : t
  const k = segmentIndex(breaks, tc)
  const width = breaks[k + 1] - breaks[k]
  const tau = (tc - breaks[k]) / width
  const c = poly.coeffs[k]
  let out = c[c.length - 1]
  for (let j = c.length - 2; j >= 0; j--) out = out * tau + c[j]
  return out
}

/** Vectorised `evalPoly` into a `Float64Array` (or into `out` when given). */
export function evalPolyArray(poly: Poly, t: ArrayLike<number>, out?: Float64Array): Float64Array {
  const n = t.length
  const res = out ?? new Float64Array(n)
  const breaks = poly.breaks_us
  const nSeg = breaks.length - 1
  if (nSeg < 1) {
    res.fill(NaN)
    return res
  }
  const t0 = breaks[0]
  const t1 = breaks[nSeg]
  for (let i = 0; i < n; i++) {
    const ti = t[i]
    const tc = ti < t0 ? t0 : ti > t1 ? t1 : ti
    const k = segmentIndex(breaks, tc)
    const tau = (tc - breaks[k]) / (breaks[k + 1] - breaks[k])
    const c = poly.coeffs[k]
    let v = c[c.length - 1]
    for (let j = c.length - 2; j >= 0; j--) v = v * tau + c[j]
    res[i] = v
  }
  return res
}

/**
 * Exact derivative `dp/dt` in units of value per break unit (µs for the API), carrying the
 * `1/T` Jacobian of normalized time — `aodl.poly.PiecewisePoly.derivative`.
 */
export function derivative(poly: Poly): Poly {
  const breaks = poly.breaks_us
  const nSeg = breaks.length - 1
  const degree = poly.coeffs.reduce((d, row) => Math.max(d, row.length - 1), 0)
  if (degree <= 0) {
    return { breaks_us: breaks.slice(), coeffs: Array.from({ length: nSeg }, () => [0]) }
  }
  const coeffs = poly.coeffs.map((row, k) => {
    const width = breaks[k + 1] - breaks[k]
    const out: number[] = new Array(degree).fill(0)
    for (let j = 1; j < row.length; j++) out[j - 1] = (row[j] * j) / width
    return out
  })
  return { breaks_us: breaks.slice(), coeffs }
}

/** `(t_start, t_end)` of the polynomial's domain. */
export function polyDomain(poly: Poly): [number, number] {
  return [poly.breaks_us[0], poly.breaks_us[poly.breaks_us.length - 1]]
}

/**
 * `(min, max)` of the polynomial over `[t0, t1]` by dense sampling plus the breakpoints
 * (display helper; `aodl.waveform.shepard.poly_range` is exact but the UI only needs
 * axis limits and the recipe table's frequency range).
 */
export function polyRangeSampled(poly: Poly, t0: number, t1: number, n = 512): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  const consider = (v: number) => {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  for (let i = 0; i < n; i++) consider(evalPoly(poly, t0 + ((t1 - t0) * i) / (n - 1)))
  for (const b of poly.breaks_us) if (b >= t0 && b <= t1) consider(evalPoly(poly, b))
  return [lo, hi]
}
