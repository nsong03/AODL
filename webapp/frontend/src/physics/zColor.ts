/**
 * Port of `aodl.viz.style.z_color`: hue carries Z, luminance carries intensity.
 *
 * `RdBu_r` from the 11 ColorBrewer anchors (linear interpolation; matches matplotlib's
 * 256-entry LUT to < 0.01), `z_norm = clip(0.5 + z/(2 z_max), 0, 1)`, then each RGB is divided
 * by its brightest channel so a tweezer far from the focal plane is not drawn dim merely
 * because of its depth. `z_max = max(|z|, 1 µm)` (`Z_FLOOR`).
 */

/** ColorBrewer RdBu reversed (blue → white → red), anchors at x = 0, 0.1, …, 1.0. */
export const RDBU_R_ANCHORS: readonly [number, number, number][] = [
  [5, 48, 97],
  [32, 101, 171],
  [67, 147, 195],
  [144, 196, 221],
  [209, 229, 240],
  [247, 246, 246],
  [253, 219, 199],
  [243, 164, 129],
  [214, 96, 77],
  [177, 24, 43],
  [103, 0, 31],
]

/** Smallest half-range of the Z colour scale [µm] (`aodl.viz.style.Z_FLOOR`). */
export const Z_FLOOR_UM = 1.0

/** Display gamma for intensity (`aodl.viz.style.GAMMA`). */
export const GAMMA = 0.7

/** `max |z|` floored at 1 µm. */
export function zMaxFrom(z: ArrayLike<number>, floor = Z_FLOOR_UM): number {
  let peak = 0
  for (let i = 0; i < z.length; i++) {
    const a = Math.abs(z[i])
    if (a > peak) peak = a
  }
  return Math.max(peak, floor)
}

/** `clip(0.5 + z / (2 z_max), 0, 1)`, symmetric about Z = 0. */
export function zNorm(z: number, zMax: number): number {
  const scale = Math.max(zMax, Z_FLOOR_UM)
  const v = 0.5 + z / (2 * scale)
  return v < 0 ? 0 : v > 1 ? 1 : v
}

/** `RdBu_r(x)` as RGB in [0, 1] by linear interpolation of the anchors. */
export function rdbuR(x: number): [number, number, number] {
  const xx = x < 0 ? 0 : x > 1 ? 1 : x
  const pos = xx * (RDBU_R_ANCHORS.length - 1)
  const i = Math.min(Math.floor(pos), RDBU_R_ANCHORS.length - 2)
  const f = pos - i
  const a = RDBU_R_ANCHORS[i]
  const b = RDBU_R_ANCHORS[i + 1]
  return [
    (a[0] + (b[0] - a[0]) * f) / 255,
    (a[1] + (b[1] - a[1]) * f) / 255,
    (a[2] + (b[2] - a[2]) * f) / 255,
  ]
}

/** Luminance-normalised tint for lab Z [µm]: brightest channel is 1. */
export function zColor(z: number, zMax: number): [number, number, number] {
  const rgb = rdbuR(zNorm(z, zMax))
  const peak = Math.max(rgb[0], rgb[1], rgb[2])
  if (peak <= 0) return [0, 0, 0]
  return [rgb[0] / peak, rgb[1] / peak, rgb[2] / peak]
}

/** CSS `rgb()` string of `zColor`, optionally scaled by a brightness in [0, 1]. */
export function zColorCss(z: number, zMax: number, brightness = 1): string {
  const [r, g, b] = zColor(z, zMax)
  const s = 255 * Math.max(0, Math.min(1, brightness))
  return `rgb(${Math.round(r * s)}, ${Math.round(g * s)}, ${Math.round(b * s)})`
}
