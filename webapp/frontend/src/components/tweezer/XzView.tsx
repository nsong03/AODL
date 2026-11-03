import { useEffect, useRef } from 'react'
import { fmt, ticks } from '../../physics/format'
import { renderXZ, type SpotLike } from '../../physics/gaussianBeam'
import { GAMMA } from '../../physics/zColor'
import type { ViewScale } from './view'
import { useElementWidth } from '../../hooks/useCanvas'

const NX = 192
const NZ = 96
const HEIGHT = 170

/** XZ slice at `Y = rowY`, tinted row by row by its own Z; the plane `planeZ` is marked. */
export function XzView({ spots, rowY, planeZ, scale, peak, label }: { spots: readonly SpotLike[]; rowY: number; planeZ: number; scale: ViewScale; peak: number; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const buf = useRef<{ canvas: HTMLCanvasElement; rgb: Float32Array; img: ImageData } | null>(null)
  const b = scale.bounds
  const zHalf = scale.zHalf
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 320
    const height = HEIGHT
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const pw = Math.round(width * dpr)
    const ph = Math.round(height * dpr)
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!buf.current) {
      const off = document.createElement('canvas')
      off.width = NX
      off.height = NZ
      buf.current = { canvas: off, rgb: new Float32Array(NX * NZ * 3), img: new ImageData(NX, NZ) }
    }
    const { canvas: off, rgb, img } = buf.current
    renderXZ(spots, { x0: b.x0, x1: b.x1, nx: NX }, { z0: -zHalf, z1: zHalf, nz: NZ }, rowY, scale.w0, scale.zR, peak, scale.zMax, GAMMA, rgb)
    const data = img.data
    for (let j = 0; j < NZ; j++) {
      const src = (NZ - 1 - j) * NX
      const dst = j * NX
      for (let i = 0; i < NX; i++) {
        const p = (src + i) * 3
        const q = (dst + i) * 4
        data[q] = rgb[p] * 255
        data[q + 1] = rgb[p + 1] * 255
        data[q + 2] = rgb[p + 2] * 255
        data[q + 3] = 255
      }
    }
    off.getContext('2d')!.putImageData(img, 0, 0)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.drawImage(off, 0, 0, pw, ph)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const Y = (z: number) => height - ((z + zHalf) / (2 * zHalf)) * height
    const X = (x: number) => ((x - b.x0) / (b.x1 - b.x0)) * width
    // tracked plane
    ctx.setLineDash([4, 3])
    ctx.strokeStyle = 'rgba(242,245,248,0.7)'
    ctx.beginPath()
    ctx.moveTo(0, Y(planeZ))
    ctx.lineTo(width, Y(planeZ))
    ctx.stroke()
    ctx.setLineDash([])
    // Z = 0
    ctx.strokeStyle = 'rgba(154,165,179,0.35)'
    ctx.beginPath()
    ctx.moveTo(0, Y(0))
    ctx.lineTo(width, Y(0))
    ctx.stroke()
    ctx.font = '10px sans-serif'
    ctx.fillStyle = 'rgba(154,165,179,0.9)'
    ctx.textAlign = 'left'
    for (const v of ticks(-zHalf, zHalf, 4)) ctx.fillText(fmt(v, 0), 4, Y(v) + 3)
    ctx.textAlign = 'center'
    for (const v of ticks(b.x0, b.x1, 6)) ctx.fillText(fmt(v, 0), X(v), height - 5)
    ctx.fillStyle = 'rgba(242,245,248,0.9)'
    ctx.textAlign = 'left'
    ctx.fillText(label, 6, 12)
    ctx.textAlign = 'right'
    ctx.fillText(`y = ${fmt(rowY, 1)} µm · Z lab [µm] ↑ · X, Z to different scales`, width - 6, 12)
  }, [spots, rowY, planeZ, scale, peak, b, zHalf, label, cssWidth])
  return <canvas ref={ref} className="plot" style={{ width: '100%', height: HEIGHT }} aria-label={label} />
}
