import { useEffect, useRef } from 'react'
import { fmt } from '../../physics/format'
import { zColor } from '../../physics/zColor'

/** The luminance-normalised Z tint scale (blue below, white at 0, red above), labelled in µm. */
export function ColorBar({ zMax, height = 180 }: { zMax: number; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const w = 14
    c.width = w
    c.height = height
    const ctx = c.getContext('2d')
    if (!ctx) return
    for (let j = 0; j < height; j++) {
      const z = zMax * (1 - (2 * j) / (height - 1))
      const [r, g, b] = zColor(z, zMax)
      ctx.fillStyle = `rgb(${r * 255}, ${g * 255}, ${b * 255})`
      ctx.fillRect(0, j, w, 1)
    }
  }, [zMax, height])
  return (
    <div className="colorbar" title="hue = spot Z (lab, Table I sign); brightness = intensity">
      <span>+{fmt(zMax, 1)}</span>
      <canvas ref={ref} style={{ width: 14, height }} />
      <span>−{fmt(zMax, 1)}</span>
      <span style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', fontSize: 10.5 }}>spot Z lab [µm]</span>
    </div>
  )
}
