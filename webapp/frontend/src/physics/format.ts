/** Number formatting by magnitude, with units. */

export function fmt(x: number | null | undefined, digits?: number): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞'
  if (digits !== undefined) return x.toFixed(digits)
  const a = Math.abs(x)
  if (a === 0) return '0'
  if (a >= 1000) return x.toFixed(0)
  if (a >= 100) return x.toFixed(1)
  if (a >= 10) return x.toFixed(2)
  if (a >= 1) return x.toFixed(3)
  if (a >= 0.01) return x.toFixed(4)
  return x.toExponential(2)
}

export function fmtUnit(x: number | null | undefined, unit: string, digits?: number): string {
  const s = fmt(x, digits)
  return s === '—' ? s : `${s} ${unit}`
}

export const fmtUs = (x: number | null | undefined, digits = 2): string => fmtUnit(x, 'µs', digits)
export const fmtUm = (x: number | null | undefined, digits = 2): string => fmtUnit(x, 'µm', digits)
export const fmtMHz = (x: number | null | undefined, digits = 3): string => fmtUnit(x, 'MHz', digits)

export function fmtSigned(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return fmt(x)
  const s = x.toFixed(digits)
  return x >= 0 ? `+${s}` : s.replace('-', '−')
}

export function fmtPct(x: number, digits = 1): string {
  return `${(100 * x).toFixed(digits)} %`
}

/** Nice tick step for a span (1/2/5 × 10^k). */
export function niceStep(span: number, targetTicks = 5): number {
  if (!(span > 0)) return 1
  const raw = span / targetTicks
  const p = Math.pow(10, Math.floor(Math.log10(raw)))
  const r = raw / p
  const m = r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10
  return m * p
}

/** Ticks covering `[lo, hi]` at `niceStep`. */
export function ticks(lo: number, hi: number, targetTicks = 5): number[] {
  const step = niceStep(hi - lo, targetTicks)
  const out: number[] = []
  const start = Math.ceil(lo / step) * step
  for (let v = start; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-12 * step ? 0 : v)
  return out
}
