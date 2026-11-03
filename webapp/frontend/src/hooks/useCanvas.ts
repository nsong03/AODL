import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react'

export interface CanvasSize {
  width: number // CSS px
  height: number
  dpr: number
}

/**
 * A canvas that follows its CSS width (ResizeObserver) at device-pixel resolution and calls
 * `draw(ctx, size)` whenever `deps` change or the element resizes. `draw` receives a context
 * already scaled by `dpr`, so it works in CSS pixels. `height` is the CSS height in px, or a
 * function of the width (aspect ratio).
 */
export function useCanvas(
  draw: (ctx: CanvasRenderingContext2D, size: CanvasSize) => void,
  deps: readonly unknown[],
  height: number | ((width: number) => number),
): RefObject<HTMLCanvasElement> {
  const ref = useRef<HTMLCanvasElement>(null)
  const drawRef = useRef(draw)
  drawRef.current = draw
  const sizeRef = useRef<CanvasSize>({ width: 0, height: 0, dpr: 1 })

  const render = () => {
    const canvas = ref.current
    if (!canvas) return
    const width = canvas.clientWidth || canvas.parentElement?.clientWidth || 300
    const h = typeof height === 'function' ? height(width) : height
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const pw = Math.max(1, Math.round(width * dpr))
    const ph = Math.max(1, Math.round(h * dpr))
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    canvas.style.height = `${h}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    sizeRef.current = { width, height: h, dpr }
    drawRef.current(ctx, sizeRef.current)
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(render, deps)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => render())
    ro.observe(canvas)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return ref
}

/** Map data → CSS pixel helpers for a plot area. */
export interface Scale {
  x: (v: number) => number
  y: (v: number) => number
  invX: (px: number) => number
  invY: (px: number) => number
  left: number
  right: number
  top: number
  bottom: number
}

export function makeScale(
  xlim: [number, number],
  ylim: [number, number],
  box: { left: number; right: number; top: number; bottom: number },
): Scale {
  const sx = (box.right - box.left) / (xlim[1] - xlim[0] || 1)
  const sy = (box.bottom - box.top) / (ylim[1] - ylim[0] || 1)
  return {
    x: (v) => box.left + (v - xlim[0]) * sx,
    y: (v) => box.bottom - (v - ylim[0]) * sy,
    invX: (px) => xlim[0] + (px - box.left) / sx,
    invY: (px) => ylim[0] + (box.bottom - px) / sy,
    ...box,
  }
}

/**
 * CSS width of an element, tracked with a ResizeObserver, so a canvas effect can list it as a
 * dependency and redraw at the right resolution when the layout changes (a canvas drawn for a
 * wider column would otherwise be CSS-scaled).
 */
export function useElementWidth(ref: RefObject<HTMLElement>): number {
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setW(Math.round(e.contentRect.width))
    })
    ro.observe(el)
    setW(el.clientWidth)
    return () => ro.disconnect()
  }, [ref])
  return w
}
