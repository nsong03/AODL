import { useEffect, useRef } from 'react'
import { fmt, ticks } from '../../physics/format'
import { useStudio } from '../../state/store'
import { makeScale, useElementWidth } from '../../hooks/useCanvas'

/**
 * Time bar over `t_span_us` with seam ticks (move boundaries), fade-event ticks, the shaded
 * `[t − τ, t]` aperture window, the simulated coverage and the cursor; click / drag to scrub.
 * The rAF loop advances `t` at `speed_us_per_s` and loops over the simulated range.
 */
export function Playback() {
  const plan = useStudio((s) => s.plan)
  const sim = useStudio((s) => s.sim)
  const t = Math.max(0, useStudio((s) => s.t_us))
  const playing = useStudio((s) => s.playing)
  const speed = useStudio((s) => s.speed_us_per_s)
  const setTime = useStudio((s) => s.setTime)
  const setSpeed = useStudio((s) => s.setSpeed)
  const togglePlay = useStudio((s) => s.togglePlay)
  const stepFrame = useStudio((s) => s.stepFrame)
  const pause = useStudio((s) => s.pause)
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const dragging = useRef(false)

  const t0 = plan?.t_span_us[0] ?? 0
  const t1 = plan?.t_span_us[1] ?? 1
  const loopEnd = sim && sim.times_us.length ? sim.times_us[sim.times_us.length - 1] : t1
  const loopStart = sim && sim.times_us.length ? sim.times_us[0] : t0

  // rAF loop
  useEffect(() => {
    if (!playing || !plan) return
    let last = performance.now()
    let raf = 0
    const tick = (now: number) => {
      const dt = (now - last) / 1000
      last = now
      const s = useStudio.getState()
      let next = s.t_us + dt * s.speed_us_per_s
      if (next > loopEnd) next = loopStart
      s.setTime(next)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, plan, loopEnd, loopStart])

  // bar
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
    if (!plan) return
    const sc = makeScale([t0, t1], [0, 1], { left: 8, right: width - 8, top: 6, bottom: height - 14 })
    // simulated coverage
    if (sim && sim.times_us.length) {
      ctx.fillStyle = 'rgba(76,201,240,0.10)'
      ctx.fillRect(sc.x(loopStart), sc.top, sc.x(loopEnd) - sc.x(loopStart), sc.bottom - sc.top)
    }
    // trajectory span
    ctx.fillStyle = 'rgba(255,255,255,0.05)'
    ctx.fillRect(sc.x(0), sc.top, sc.x(plan.duration_us) - sc.x(0), sc.bottom - sc.top)
    // aperture window [t − τ, t]
    const tau = plan.constants.tau_us
    const a = Math.max(t0, t - tau)
    ctx.fillStyle = 'rgba(255,255,255,0.16)'
    ctx.fillRect(sc.x(a), sc.top, Math.max(1, sc.x(t) - sc.x(a)), sc.bottom - sc.top)
    // seams
    ctx.strokeStyle = 'rgba(154,165,179,0.7)'
    ctx.lineWidth = 1
    for (const s of plan.seams_us) {
      ctx.beginPath()
      ctx.moveTo(sc.x(s), sc.top)
      ctx.lineTo(sc.x(s), sc.bottom)
      ctx.stroke()
    }
    // fade events (as seen at the atoms: + τ/2)
    ctx.strokeStyle = 'rgba(255,183,3,0.9)'
    for (const e of plan.fade_events) {
      const x = sc.x(e.time_us + plan.constants.half_tau_us)
      ctx.beginPath()
      ctx.moveTo(x, sc.bottom - 8)
      ctx.lineTo(x, sc.bottom)
      ctx.stroke()
    }
    // time ticks
    ctx.font = '10px sans-serif'
    ctx.fillStyle = '#9aa5b3'
    ctx.textAlign = 'center'
    for (const v of ticks(t0, t1, 8)) ctx.fillText(fmt(v, 0), sc.x(v), height - 3)
    // beam-centre marker & cursor
    ctx.strokeStyle = 'rgba(242,245,248,0.6)'
    ctx.setLineDash([2, 2])
    ctx.beginPath()
    ctx.moveTo(sc.x(t - plan.constants.half_tau_us), sc.top)
    ctx.lineTo(sc.x(t - plan.constants.half_tau_us), sc.bottom)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.strokeStyle = '#f2f5f8'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(sc.x(t), sc.top - 2)
    ctx.lineTo(sc.x(t), sc.bottom + 2)
    ctx.stroke()
  }, [plan, sim, t, t0, t1, loopStart, loopEnd, cssWidth])

  const scrub = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = ref.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const f = (e.clientX - rect.left - 8) / (rect.width - 16)
    setTime(t0 + Math.min(1, Math.max(0, f)) * (t1 - t0))
  }

  const base = plan ? plan.duration_us / 8 : speed
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div
        className="playbar"
        onPointerDown={(e) => {
          dragging.current = true
          pause()
          ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
          scrub(e as unknown as React.PointerEvent<HTMLCanvasElement>)
        }}
        onPointerMove={(e) => dragging.current && scrub(e as unknown as React.PointerEvent<HTMLCanvasElement>)}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => (dragging.current = false)}
        title="drag to scrub · white: cursor t · dotted: t − τ/2 · shaded: [t − τ, t] inside the crystals · grey ticks: move seams · amber: fade hand-overs (+τ/2)"
      >
        <canvas ref={ref} />
      </div>
      <div className="row" style={{ gap: 8 }}>
        <button className="primary small" onClick={togglePlay} disabled={!plan} title="space">
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <button className="small" onClick={() => stepFrame(-1)} disabled={!plan} title="←">
          ◀ frame
        </button>
        <button className="small" onClick={() => stepFrame(1)} disabled={!plan} title="→">
          frame ▶
        </button>
        <span className="mono bright" style={{ minWidth: 110 }}>
          t = {fmt(t, 2)} µs
        </span>
        <span className="small dim">atoms see t − τ/2 = {plan ? fmt(t - plan.constants.half_tau_us, 2) : '—'} µs</span>
        <span className="grow" />
        <label className="row small dim" style={{ gap: 6 }}>
          speed
          <input type="number" style={{ width: 70 }} value={Number(speed.toFixed(2))} step={1} min={0.1} onChange={(e) => setSpeed(parseFloat(e.target.value) || 1)} />
          µs/s
        </label>
        <span className="row" style={{ gap: 2 }}>
          {[0.25, 0.5, 1, 2, 4].map((m) => (
            <button key={m} className={`ghost small ${Math.abs(speed - base * m) < 1e-6 ? 'active' : ''}`} onClick={() => setSpeed(base * m)} title={`${m}× (trajectory in ${(8 / m).toFixed(0)} s)`}>
              {m}×
            </button>
          ))}
        </span>
      </div>
    </div>
  )
}
