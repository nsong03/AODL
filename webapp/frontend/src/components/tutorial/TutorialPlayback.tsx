/**
 * The tutorial's single time bar (shared by both columns): move seams (grey), the Shepard
 * drive's hand-overs and the plain drive's first band exit as the atoms see them (+τ/2), the
 * aperture window `[t − τ, t]`, the simulated coverage and the cursor; click / drag to scrub,
 * play loops over the simulated range. Mirrors `Playback`, on the page-local cursor.
 */
import { useEffect, useRef } from 'react'
import type { FadeEvent, PlanResponse, SimulateResponse } from '../../api/types'
import { makeScale, useElementWidth } from '../../hooks/useCanvas'
import { fmt, ticks } from '../../physics/format'
import { useTutorialCursor } from './cursor'
import type { BandExit } from './tutorialMath'

export function TutorialPlayback({ plan, sim, fades, exit }: { plan: PlanResponse; sim: SimulateResponse | null; fades: FadeEvent[]; exit: BandExit | null }) {
  const t = useTutorialCursor((s) => s.t_us)
  const playing = useTutorialCursor((s) => s.playing)
  const speed = useTutorialCursor((s) => s.speed_us_per_s)
  const { setTime, togglePlay, pause, setSpeed } = useTutorialCursor.getState()
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const dragging = useRef(false)
  const [t0, t1] = plan.t_span_us
  const half = plan.constants.half_tau_us
  const tau = plan.constants.tau_us
  const loopStart = sim?.times_us[0] ?? t0
  const loopEnd = sim?.times_us[sim.times_us.length - 1] ?? t1

  useEffect(() => {
    if (!playing) return
    let last = performance.now()
    let raf = 0
    const tick = (now: number) => {
      const s = useTutorialCursor.getState()
      let next = s.t_us + ((now - last) / 1000) * s.speed_us_per_s
      last = now
      if (next > loopEnd) next = loopStart
      s.setTime(next)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, loopStart, loopEnd])

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 400
    const height = 46
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#0f1219'
    ctx.fillRect(0, 0, width, height)
    const sc = makeScale([t0, t1], [0, 1], { left: 8, right: width - 8, top: 6, bottom: height - 14 })
    ctx.fillStyle = 'rgba(76,201,240,0.10)'
    ctx.fillRect(sc.x(loopStart), sc.top, sc.x(loopEnd) - sc.x(loopStart), sc.bottom - sc.top)
    ctx.fillStyle = 'rgba(255,255,255,0.16)'
    const a = Math.max(t0, t - tau)
    ctx.fillRect(sc.x(a), sc.top, Math.max(1, sc.x(t) - sc.x(a)), sc.bottom - sc.top)
    ctx.strokeStyle = 'rgba(154,165,179,0.7)'
    ctx.lineWidth = 1
    for (const s of plan.seams_us) {
      ctx.beginPath()
      ctx.moveTo(sc.x(s), sc.top)
      ctx.lineTo(sc.x(s), sc.bottom)
      ctx.stroke()
    }
    ctx.strokeStyle = 'rgba(255,183,3,0.95)'
    for (const e of fades) {
      const x = sc.x(e.time_us + half)
      ctx.beginPath()
      ctx.moveTo(x, sc.bottom - 9)
      ctx.lineTo(x, sc.bottom)
      ctx.stroke()
    }
    if (exit) {
      ctx.strokeStyle = '#ef6f6c'
      ctx.lineWidth = 2
      const x = sc.x(exit.t_atoms_us)
      ctx.beginPath()
      ctx.moveTo(x, sc.top)
      ctx.lineTo(x, sc.top + 9)
      ctx.stroke()
    }
    ctx.font = '10px sans-serif'
    ctx.fillStyle = '#9aa5b3'
    ctx.textAlign = 'center'
    for (const v of ticks(t0, t1, 8)) ctx.fillText(fmt(v, 0), sc.x(v), height - 3)
    ctx.strokeStyle = 'rgba(242,245,248,0.6)'
    ctx.setLineDash([2, 2])
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(sc.x(t - half), sc.top)
    ctx.lineTo(sc.x(t - half), sc.bottom)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.strokeStyle = '#f2f5f8'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(sc.x(t), sc.top - 2)
    ctx.lineTo(sc.x(t), sc.bottom + 2)
    ctx.stroke()
  }, [plan, t, t0, t1, tau, half, loopStart, loopEnd, fades, exit, cssWidth])

  const scrub = (clientX: number) => {
    const canvas = ref.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const f = (clientX - rect.left - 8) / (rect.width - 16)
    setTime(t0 + Math.min(1, Math.max(0, f)) * (t1 - t0))
  }
  const base = plan.duration_us / 8
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div
        className="playbar"
        onPointerDown={(e) => {
          dragging.current = true
          pause()
          ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
          scrub(e.clientX)
        }}
        onPointerMove={(e) => dragging.current && scrub(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => (dragging.current = false)}
        title="drag to scrub · white: cursor t · dotted: t − τ/2 · shaded: [t − τ, t] · grey: move seams · amber: Shepard hand-overs (+τ/2) · red: plain drive leaves the band (+τ/2)"
      >
        <canvas ref={ref} />
      </div>
      <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
        <button className="primary small" onClick={togglePlay}>
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <span className="mono bright" style={{ minWidth: 110 }}>
          t = {fmt(t, 2)} µs
        </span>
        <span className="small dim">atoms see the drive at t − τ/2 = {fmt(t - half, 2)} µs</span>
        <span className="grow" />
        <span className="row" style={{ gap: 2 }}>
          {[0.5, 1, 2, 4].map((m) => (
            <button key={m} className={`ghost small ${Math.abs(speed - base * m) < 1e-6 ? 'active' : ''}`} onClick={() => setSpeed(base * m)} title={`${m}× (trajectory in ${(8 / m).toFixed(0)} s)`}>
              {m}×
            </button>
          ))}
        </span>
      </div>
    </div>
  )
}
