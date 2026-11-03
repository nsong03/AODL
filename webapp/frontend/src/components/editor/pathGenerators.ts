/** Waypoint generators for `path` moves: points relative to the start (DESIGN §1.3). */

export type Plane = 'xy' | 'xz' | 'yz'
export type Vec3 = [number, number, number]

function inPlane(plane: Plane, a: number, b: number): Vec3 {
  switch (plane) {
    case 'xy':
      return [a, b, 0]
    case 'xz':
      return [a, 0, b]
    case 'yz':
      return [0, a, b]
  }
}

function round(v: Vec3): Vec3 {
  return v.map((x) => (Math.abs(x) < 1e-9 ? 0 : Number(x.toFixed(4)))) as Vec3
}

/** Square of side `side` starting at a corner, `perSide` waypoints per side, open (set `closed`). */
export function squarePath(side: number, plane: Plane, perSide: number): Vec3[] {
  const n = Math.max(1, Math.round(perSide))
  const corners: [number, number][] = [
    [0, 0],
    [side, 0],
    [side, side],
    [0, side],
  ]
  const pts: Vec3[] = []
  for (let e = 0; e < 4; e++) {
    const [ax, ay] = corners[e]
    const [bx, by] = corners[(e + 1) % 4]
    for (let i = 1; i <= n; i++) {
      const f = i / n
      if (e === 3 && i === n) break // the return to the start is `closed`
      pts.push(round(inPlane(plane, ax + (bx - ax) * f, ay + (by - ay) * f)))
    }
  }
  return pts
}

/** Circle through the start, `turns` revolutions, `points` waypoints (the last one dropped when it returns to the start). */
export function circlePath(radius: number, plane: Plane, turns: number, points: number): { points: Vec3[]; closed: boolean } {
  const n = Math.max(3, Math.round(points))
  const pts: Vec3[] = []
  const integral = Math.abs(turns - Math.round(turns)) < 1e-9
  for (let i = 1; i <= n; i++) {
    const th = (2 * Math.PI * turns * i) / n
    if (integral && i === n) break
    pts.push(round(inPlane(plane, radius * Math.cos(th) - radius, radius * Math.sin(th))))
  }
  return { points: pts, closed: integral }
}

/** Spiral on a sphere of radius `r` centred at (0, 0, r): start = south pole, θ 0→π, φ = 3θ (the `sphere_loop` example). */
export function sphereSpiralPath(radius: number, points: number, windings = 3): Vec3[] {
  const n = Math.max(4, Math.round(points))
  const pts: Vec3[] = []
  for (let i = 1; i <= n; i++) {
    const th = (Math.PI * i) / n
    const ph = windings * th
    pts.push(round([radius * Math.sin(th) * Math.cos(ph), radius * Math.sin(th) * Math.sin(ph), radius - radius * Math.cos(th)]))
  }
  return pts
}

/** Helix about z through the start: radius, pitch per turn, turns. */
export function helixPath(radius: number, pitch: number, turns: number, points: number): Vec3[] {
  const n = Math.max(4, Math.round(points))
  const pts: Vec3[] = []
  for (let i = 1; i <= n; i++) {
    const th = (2 * Math.PI * turns * i) / n
    pts.push(round([radius * Math.cos(th) - radius, radius * Math.sin(th), (pitch * th) / (2 * Math.PI)]))
  }
  return pts
}

/** Parse "dx, dy, dz" lines. Returns null on any malformed line. */
export function parsePoints(text: string): Vec3[] | null {
  const out: Vec3[] = []
  for (const raw of text.split(/\n|;/)) {
    const line = raw.trim()
    if (!line) continue
    const parts = line.split(/[,\s]+/).filter(Boolean).map(Number)
    if (parts.length !== 3 || parts.some((v) => !Number.isFinite(v))) return null
    out.push([parts[0], parts[1], parts[2]])
  }
  return out
}

export function formatPoints(points: readonly Vec3[]): string {
  return points.map((p) => p.map((v) => fmtCoord(v)).join(', ')).join('\n')
}

function fmtCoord(v: number): string {
  return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(4)))
}

/** Chord length of a path from the origin, for a speed estimate. */
export function pathLength(points: readonly Vec3[], closed: boolean): number {
  let prev: Vec3 = [0, 0, 0]
  let total = 0
  const all = closed ? [...points, [0, 0, 0] as Vec3] : points
  for (const p of all) {
    total += Math.hypot(p[0] - prev[0], p[1] - prev[1], p[2] - prev[2])
    prev = p
  }
  return total
}
