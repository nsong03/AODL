import { useEffect, useRef } from 'react'
import { renderXY, type SpotLike } from '../../physics/gaussianBeam'
import { GAMMA } from '../../physics/zColor'
import { drawFrame, fitGrid, type ViewScale } from './view'
import { useElementWidth } from '../../hooks/useCanvas'

/**
 * XY plane at `planeZ` under the astigmatic Gaussian approximation: the render grid has its long
 * side at `longSide` px (≤ 320), displayed with the data aspect ratio, row 0 = y0 at the bottom.
 */
export function XyView({
  spots,
  planeZ,
  scale,
  peak,
  requested,
  longSide = 320,
  label,
}: {
  spots: readonly SpotLike[]
  planeZ: number
  scale: ViewScale
  peak: number
  requested?: { x: number; y: number } | null
  longSide?: number
  label: string
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const buf = useRef<{ canvas: HTMLCanvasElement; rgb: Float32Array; img: ImageData; nx: number; ny: number } | null>(null)
  const b = scale.bounds
  const { nx, ny } = fitGrid(b, longSide)
  const aspect = (b.y1 - b.y0) / (b.x1 - b.x0)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 320
    const height = Math.round(width * aspect)
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.style.height = `${height}px`
    const pw = Math.round(width * dpr)
    const ph = Math.round(height * dpr)
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (!buf.current || buf.current.nx !== nx || buf.current.ny !== ny) {
      const off = document.createElement('canvas')
      off.width = nx
      off.height = ny
      buf.current = { canvas: off, rgb: new Float32Array(nx * ny * 3), img: new ImageData(nx, ny), nx, ny }
    }
    const { canvas: off, rgb, img } = buf.current
    renderXY(spots, { x0: b.x0, x1: b.x1, nx, y0: b.y0, y1: b.y1, ny }, planeZ, scale.w0, scale.zR, peak, scale.zMax, GAMMA, rgb)
    // flip rows so y0 is at the bottom (imshow origin="lower")
    const data = img.data
    for (let j = 0; j < ny; j++) {
      const src = (ny - 1 - j) * nx
      const dst = j * nx
      for (let i = 0; i < nx; i++) {
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
    drawFrame(ctx, width, height, b, requested ?? null, label, planeZ)
  }, [spots, planeZ, scale, peak, nx, ny, aspect, b, requested, label, cssWidth])

  return <canvas ref={ref} className="plot" style={{ width: '100%' }} aria-label={label} />
}

