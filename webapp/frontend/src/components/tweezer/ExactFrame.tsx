import { useEffect, useMemo, useRef } from 'react'
import { decodePacked } from '../../api/client'
import type { FrameResponse, ImageXY, ImageXZ } from '../../api/types'
import { fmt, ticks } from '../../physics/format'
import { useElementWidth } from '../../hooks/useCanvas'
import { PER_M2_TO_PER_UM2 } from '../../physics/units'
import { GAMMA, zColor } from '../../physics/zColor'
import type { ViewScale } from './view'

/**
 * The closed-form Eq. S11 frame from `/api/frame`, drawn on the **same** colour scale as the
 * approximation: `value / peakNorm` through the same gamma, tinted by the plane's Z (a single
 * intensity image carries no per-group Z, so the tint is uniform — stated in the label). The
 * packed values are per m² (API.md §0.3, SI `intensity_frame`); the movie's peaks are per µm²,
 * hence `PER_M2_TO_PER_UM2`.
 */
export function ExactXy({ frame, scale, peak, aspect }: { frame: FrameResponse; scale: ViewScale; peak: number; aspect: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const img = useMemo(() => decodePacked(frame.xy.image), [frame])
  useEffect(() => {
    drawPacked(ref.current, cssWidth, frame.xy, img, scale, peak, frame.plane_z_um, aspect, `closed form (Eq. S11) · t = ${fmt(frame.time_us, 1)} µs`, true)
  }, [frame, img, scale, peak, aspect, cssWidth])
  return <canvas ref={ref} className="plot" style={{ width: '100%' }} aria-label="exact XY frame" />
}

export function ExactXz({ frame, scale, peak }: { frame: FrameResponse; scale: ViewScale; peak: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const xz = frame.xz
  const img = useMemo(() => (xz ? decodePacked(xz.image) : null), [xz])
  useEffect(() => {
    if (!xz || !img) return
    drawPacked(ref.current, cssWidth, xz, img, scale, peak, frame.plane_z_um, 170 / (cssWidth || ref.current?.clientWidth || 320), `closed form XZ · y = ${fmt(xz.row_y_um, 1)} µm`, false)
  }, [xz, img, scale, peak, frame.plane_z_um, cssWidth])
  if (!xz) return null
  return <canvas ref={ref} className="plot" style={{ width: '100%', height: 170 }} aria-label="exact XZ frame" />
}

function drawPacked(
  canvas: HTMLCanvasElement | null,
  cssWidth: number,
  im: ImageXY | ImageXZ,
  values: Float32Array,
  scale: ViewScale,
  peak: number,
  planeZ: number,
  aspect: number,
  label: string,
  tintByPlane: boolean,
) {
  if (!canvas) return
  const [ny, nx] = im.image.shape
  const width = cssWidth || canvas.clientWidth || 320
  const height = Math.round(width * aspect)
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  canvas.style.height = `${height}px`
  canvas.width = Math.round(width * dpr)
  canvas.height = Math.round(height * dpr)
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const off = document.createElement('canvas')
  off.width = nx
  off.height = ny
  const data = new ImageData(nx, ny)
  const rows = 'z_um' in im ? im.z_um : null
  const [cr, cg, cb] = zColor(planeZ, scale.zMax)
  const norm = peak > 0 ? PER_M2_TO_PER_UM2 / peak : 0
  for (let j = 0; j < ny; j++) {
    const src = (ny - 1 - j) * nx // row 0 = y0 → bottom
    let r = cr
    let g = cg
    let b = cb
    if (!tintByPlane && rows) [r, g, b] = zColor(rows[ny - 1 - j], scale.zMax)
    for (let i = 0; i < nx; i++) {
      let v = values[src + i] * norm
      if (v > 1) v = 1
      const s = Math.pow(v, GAMMA) * 255
      const q = (j * nx + i) * 4
      data.data[q] = s * r
      data.data[q + 1] = s * g
      data.data[q + 2] = s * b
      data.data[q + 3] = 255
    }
  }
  off.getContext('2d')!.putImageData(data, 0, 0)
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.font = '10px sans-serif'
  ctx.fillStyle = 'rgba(242,245,248,0.9)'
  ctx.textAlign = 'left'
  ctx.fillText(label, 6, 12)
  ctx.textAlign = 'right'
  ctx.fillText(`peak/scale ${fmt((im.image.peak * PER_M2_TO_PER_UM2) / peak, 2)} · plane Z = ${fmt(planeZ, 2)} µm`, width - 6, 12)
  const xs = im.x_um
  const x0 = xs[0]
  const x1 = xs[xs.length - 1]
  ctx.fillStyle = 'rgba(154,165,179,0.9)'
  ctx.textAlign = 'center'
  for (const v of ticks(x0, x1, 6)) ctx.fillText(fmt(v, 0), ((v - x0) / (x1 - x0)) * width, height - 6)
  if (rows) {
    // the exact XZ z axis is centred on the evaluation plane (API.md §5.2), unlike the approximation's ±z_half about 0
    const z0 = rows[0]
    const z1 = rows[rows.length - 1]
    ctx.textAlign = 'left'
    for (const v of ticks(z0, z1, 4)) {
      const y = height - ((v - z0) / (z1 - z0)) * height
      if (y < 18 || y > height - 4) continue
      ctx.fillText(fmt(v, 0), 4, y + 3)
    }
    ctx.fillText('Z lab [µm] (about the plane)', 4, height - 6)
  }
}
