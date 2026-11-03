import { describe, expect, it } from 'vitest'
import { circlePath, helixPath, parsePoints, pathLength, sphereSpiralPath, squarePath } from '../src/components/editor/pathGenerators'

describe('path generators', () => {
  it('square: 4n − 1 waypoints (the return is `closed`), corners hit', () => {
    const pts = squarePath(15, 'xy', 3)
    expect(pts.length).toBe(11)
    expect(pts[2]).toEqual([15, 0, 0])
    expect(pts[5]).toEqual([15, 15, 0])
    expect(pts[8]).toEqual([0, 15, 0])
    expect(pts.every((p) => p[2] === 0)).toBe(true)
    expect(squarePath(4, 'xz', 1).map((p) => p[1])).toEqual([0, 0, 0])
  })
  it('circle through the start: integral turns close, fractional stay open', () => {
    const c = circlePath(8, 'xy', 1, 24)
    expect(c.closed).toBe(true)
    expect(c.points.length).toBe(23)
    for (const p of c.points) expect(Math.hypot(p[0] + 8, p[1])).toBeCloseTo(8, 3)
    const half = circlePath(8, 'yz', 0.5, 12)
    expect(half.closed).toBe(false)
    expect(half.points.length).toBe(12)
    expect(half.points[11][1]).toBeCloseTo(-16, 3)
  })
  it('sphere spiral starts at the south pole and ends at the north pole', () => {
    const pts = sphereSpiralPath(6, 24)
    expect(pts.length).toBe(24)
    expect(pts[23][2]).toBeCloseTo(12, 3)
    for (const p of pts) expect(Math.hypot(p[0], p[1], p[2] - 6)).toBeCloseTo(6, 3)
  })
  it('helix climbs pitch per turn', () => {
    const pts = helixPath(5, 4, 2, 16)
    expect(pts[15][2]).toBeCloseTo(8, 3)
    expect(Math.hypot(pts[15][0] + 5, pts[15][1])).toBeCloseTo(5, 3)
  })
  it('parsePoints / pathLength', () => {
    expect(parsePoints('1, 2, 3\n4 5 6')).toEqual([[1, 2, 3], [4, 5, 6]])
    expect(parsePoints('1, 2')).toBeNull()
    expect(pathLength([[3, 4, 0]], true)).toBeCloseTo(10, 12)
  })
})
