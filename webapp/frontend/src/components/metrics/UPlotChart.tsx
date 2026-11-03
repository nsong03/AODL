import { useEffect, useRef } from 'react'
import uPlot, { type AlignedData, type Options } from 'uplot'

/**
 * Minimal uPlot wrapper: builds the chart once per `opts` identity, pushes `data` with
 * `setData`, and draws the shared time cursor on an overlay canvas (no uPlot redraw during
 * playback). Click sets the cursor through `onPick`.
 */
export function UPlotChart({
  opts,
  data,
  cursorX,
  onPick,
  height,
}: {
  opts: Omit<Options, 'width' | 'height'>
  data: AlignedData
  cursorX?: number | null
  onPick?: (x: number) => void
  height: number
}) {
  const host = useRef<HTMLDivElement>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const chart = useRef<uPlot | null>(null)
  const onPickRef = useRef(onPick)
  onPickRef.current = onPick

  useEffect(() => {
    const el = host.current
    if (!el) return
    const width = el.clientWidth || 400
    const u = new uPlot({ ...opts, width, height }, data, el)
    chart.current = u
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => u.setSize({ width: el.clientWidth || 400, height })) : null
    ro?.observe(el)
    const onClick = (e: MouseEvent) => {
      const rect = u.over.getBoundingClientRect()
      const x = e.clientX - rect.left
      if (x < 0 || x > rect.width) return
      onPickRef.current?.(u.posToVal(x, 'x'))
    }
    u.over.addEventListener('click', onClick)
    return () => {
      ro?.disconnect()
      u.over.removeEventListener('click', onClick)
      u.destroy()
      chart.current = null
    }
    // the chart is rebuilt when opts change; data flows through setData below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opts, height])

  useEffect(() => {
    chart.current?.setData(data)
  }, [data])

  useEffect(() => {
    const u = chart.current
    const c = overlay.current
    if (!u || !c) return
    const draw = () => {
      const rect = u.root.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      c.style.width = `${rect.width}px`
      c.style.height = `${rect.height}px`
      c.width = Math.round(rect.width * dpr)
      c.height = Math.round(rect.height * dpr)
      const ctx = c.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, rect.width, rect.height)
      if (cursorX === null || cursorX === undefined) return
      const over = u.over.getBoundingClientRect()
      const x = u.valToPos(cursorX, 'x') + (over.left - rect.left)
      if (x < over.left - rect.left || x > over.right - rect.left) return
      ctx.strokeStyle = 'rgba(242,245,248,0.85)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, over.top - rect.top)
      ctx.lineTo(x, over.bottom - rect.top)
      ctx.stroke()
    }
    draw()
  }, [cursorX, data, opts])

  return (
    <div className="uplot-wrap">
      <div ref={host} />
      <canvas ref={overlay} className="cursor-overlay" />
    </div>
  )
}

