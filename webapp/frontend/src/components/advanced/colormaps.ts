/** Sequential colour map for heat maps (inferno-like anchors). */
const INFERNO: [number, number, number][] = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
]

export function inferno(x: number): [number, number, number] {
  const xx = x < 0 ? 0 : x > 1 ? 1 : x
  const pos = xx * (INFERNO.length - 1)
  const i = Math.min(Math.floor(pos), INFERNO.length - 2)
  const f = pos - i
  const a = INFERNO[i]
  const b = INFERNO[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

export function infernoCss(x: number): string {
  const [r, g, b] = inferno(x)
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`
}
