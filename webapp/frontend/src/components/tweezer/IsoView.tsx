import { useEffect, useRef } from 'react'
import type { TrajectorySamples } from '../../api/types'
import { fmt } from '../../physics/format'
import { spotRadii, type SpotLike } from '../../physics/gaussianBeam'
import { zColor } from '../../physics/zColor'
import type { ViewScale } from './view'
import { useElementWidth } from '../../hooks/useCanvas'

const HEIGHT = 230

/**
 * Isometric view of the bounds box: each spot's 1/e² caustic as 9 ellipses over `z ± 2 z_R`
 * tinted by z, the requested trajectory as a trail and the current requested point as a hollow
 * marker — so the τ/2 lag between request and tweezer is visible.
 */
export function IsoView({
  spots,
  scale,
  trajectory,
  requested,
  powerRef,
}: {
  spots: readonly SpotLike[]
  scale: ViewScale
  trajectory: TrajectorySamples | null
  requested: { x: number; y: number; z: number } | null
  powerRef: number
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 320
    const height = HEIGHT
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, width, height)
    const b = scale.bounds
    const zLo = Math.min(b.z0, -scale.zHalf)
    const zHi = Math.max(b.z1, scale.zHalf)
    const c30 = Math.cos(Math.PI / 6)
    const s30 = Math.sin(Math.PI / 6)
    // iso projection: x right-down, y right-up (the far corner (+x, +y) sits higher on screen), z up
    const proj = (x: number, y: number, z: number) => ({ px: (x - y) * c30, py: (x + y) * s30 + z })
    const corners = [
      [b.x0, b.y0, zLo],
      [b.x1, b.y0, zLo],
      [b.x1, b.y1, zLo],
      [b.x0, b.y1, zLo],
      [b.x0, b.y0, zHi],
      [b.x1, b.y0, zHi],
      [b.x1, b.y1, zHi],
      [b.x0, b.y1, zHi],
    ].map(([x, y, z]) => proj(x, y, z))
    const minX = Math.min(...corners.map((c) => c.px))
    const maxX = Math.max(...corners.map((c) => c.px))
    const minY = Math.min(...corners.map((c) => c.py))
    const maxY = Math.max(...corners.map((c) => c.py))
    const k = 0.86 * Math.min(width / (maxX - minX || 1), height / (maxY - minY || 1))
    const P = (x: number, y: number, z: number) => {
      const { px, py } = proj(x, y, z)
      return { X: width / 2 + (px - (minX + maxX) / 2) * k, Y: height / 2 - (py - (minY + maxY) / 2) * k }
    }
    const line = (a: number[], c: number[], style: string, dash?: number[]) => {
      const p = P(a[0], a[1], a[2])
      const q = P(c[0], c[1], c[2])
      ctx.strokeStyle = style
      ctx.setLineDash(dash ?? [])
      ctx.beginPath()
      ctx.moveTo(p.X, p.Y)
      ctx.lineTo(q.X, q.Y)
      ctx.stroke()
      ctx.setLineDash([])
    }
    ctx.lineWidth = 1
    const edge = 'rgba(90,100,114,0.8)'
    // bottom face
    line([b.x0, b.y0, zLo], [b.x1, b.y0, zLo], edge)
    line([b.x1, b.y0, zLo], [b.x1, b.y1, zLo], edge)
    line([b.x1, b.y1, zLo], [b.x0, b.y1, zLo], edge)
    line([b.x0, b.y1, zLo], [b.x0, b.y0, zLo], edge)
    // verticals
    for (const [x, y] of [
      [b.x0, b.y0],
      [b.x1, b.y0],
      [b.x1, b.y1],
      [b.x0, b.y1],
    ])
      line([x, y, zLo], [x, y, zHi], edge, [2, 3])
    // Z = 0 plane outline
    const z0style = 'rgba(154,165,179,0.35)'
    line([b.x0, b.y0, 0], [b.x1, b.y0, 0], z0style, [3, 3])
    line([b.x1, b.y0, 0], [b.x1, b.y1, 0], z0style, [3, 3])
    line([b.x1, b.y1, 0], [b.x0, b.y1, 0], z0style, [3, 3])
    line([b.x0, b.y1, 0], [b.x0, b.y0, 0], z0style, [3, 3])
    // requested trail
    if (trajectory && trajectory.t_us.length > 1) {
      ctx.strokeStyle = 'rgba(242,245,248,0.45)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let i = 0; i < trajectory.t_us.length; i++) {
        const p = P(trajectory.x_um[i], trajectory.y_um[i], trajectory.z_um[i])
        if (i === 0) ctx.moveTo(p.X, p.Y)
        else ctx.lineTo(p.X, p.Y)
      }
      ctx.stroke()
    }
    // spots: 9 ellipses over z ± 2 z_R
    const sorted = [...spots].sort((a, c) => a.x_um + a.y_um - (c.x_um + c.y_um))
    for (const s of sorted) {
      const bright = Math.min(1, Math.sqrt(Math.max(0, s.power / Math.max(powerRef, 1e-300))))
      const zc = s.z_um
      for (let m = -4; m <= 4; m++) {
        const z = zc + (m / 2) * scale.zR
        const [wx, wy] = spotRadii(s, z, scale.w0, scale.zR)
        const [r, g, bl] = zColor(z, scale.zMax)
        const alpha = (0.18 + 0.5 * (1 - Math.abs(m) / 5)) * (0.35 + 0.65 * bright)
        ctx.strokeStyle = `rgba(${r * 255}, ${g * 255}, ${bl * 255}, ${alpha})`
        ctx.lineWidth = m === 0 ? 1.4 : 0.9
        ctx.beginPath()
        const n = 24
        for (let i = 0; i <= n; i++) {
          const th = (2 * Math.PI * i) / n
          const p = P(s.x_um + wx * Math.cos(th), s.y_um + wy * Math.sin(th), z)
          if (i === 0) ctx.moveTo(p.X, p.Y)
          else ctx.lineTo(p.X, p.Y)
        }
        ctx.stroke()
      }
      // centre dot
      const c = P(s.x_um, s.y_um, zc)
      const [r, g, bl] = zColor(zc, scale.zMax)
      ctx.fillStyle = `rgba(${r * 255}, ${g * 255}, ${bl * 255}, ${0.5 + 0.5 * bright})`
      ctx.beginPath()
      ctx.arc(c.X, c.Y, 1.8, 0, 2 * Math.PI)
      ctx.fill()
    }
    // requested point (hollow)
    if (requested) {
      const p = P(requested.x, requested.y, requested.z)
      ctx.strokeStyle = 'rgba(242,245,248,0.95)'
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.arc(p.X, p.Y, 5, 0, 2 * Math.PI)
      ctx.stroke()
      // drop line to Z = 0
      line([requested.x, requested.y, requested.z], [requested.x, requested.y, 0], 'rgba(242,245,248,0.35)', [2, 2])
    }
    // axis labels at the box corner
    ctx.font = '10px sans-serif'
    ctx.fillStyle = 'rgba(154,165,179,0.9)'
    const ox = P(b.x1, b.y0, zLo)
    ctx.textAlign = 'left'
    ctx.fillText(`X → ${fmt(b.x1, 0)}`, ox.X + 4, ox.Y + 4)
    const oy = P(b.x0, b.y1, zLo)
    ctx.textAlign = 'right'
    ctx.fillText(`${fmt(b.y1, 0)} ← Y`, oy.X - 4, oy.Y + 4)
    const oz = P(b.x0, b.y0, zHi)
    ctx.textAlign = 'right'
    ctx.fillText(`Z ${fmt(zHi, 0)} µm`, oz.X - 4, oz.Y + 3)
    ctx.textAlign = 'left'
    ctx.fillStyle = 'rgba(242,245,248,0.9)'
    ctx.fillText('isometric · 1/e² caustics over z ± 2 z_R · ○ requested point (atoms lag by τ/2)', 6, 12)
  }, [spots, scale, trajectory, requested, powerRef, cssWidth])
  return <canvas ref={ref} className="plot" style={{ width: '100%', height: HEIGHT }} aria-label="isometric view" />
}
