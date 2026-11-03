/**
 * Compact RF drives of one tutorial column: per channel, the tone tracks `f_center + f_n(t)`
 * [MHz] vs drive time (opacity = envelope `A_n`, broken where `A = 0`; tracks from the shared
 * `useRfTracks`), the usable band (dotted edges, out-of-band shaded), the first band exit of a
 * live tone (from the IR), the hand-over ticks of the fade timetable, the beam-centre tick
 * `t − τ/2` with the live tones on it, and the shared cursor. A thin, cursor-local wrapper
 * around the drawing of `RfPanel` (which is bound to the workbench store).
 */
import { useEffect, useRef } from 'react'
import { CHANNELS, type ChannelIR, type ChannelName, type FadeEvent, type PlanResponse } from '../../api/types'
import { CHANNEL_COLORS, CHANNEL_INFO, withAlpha } from '../../channels'
import { makeScale, useElementWidth } from '../../hooks/useCanvas'
import { fmt, ticks } from '../../physics/format'
import { toneAmp, toneFreqMHz } from '../../physics/waveform'
import { useRfTracks, type ChannelCache } from '../rf/useRfTracks'
import { useTutorialCursor } from './cursor'
import type { BandExit } from './tutorialMath'

const H = 78
const PAD = { left: 40, right: 8, top: 6, bottom: 16 }
const EXIT_COLOR = '#ef6f6c'
const FADE_COLOR = 'rgba(255,183,3,0.95)'

export function TutorialRf({ plan, ylim, exit, fades }: { plan: PlanResponse; ylim: [number, number]; exit: BandExit | null; fades: FadeEvent[] }) {
  const cache = useRfTracks(plan)
  if (!cache) return null
  return (
    <div className="stack" style={{ gap: 4 }}>
      {CHANNELS.map((ch, i) => (
        <RfChannel key={ch} ch={ch} plan={plan} cache={cache[ch]} ylim={ylim} exit={exit} fades={fades.filter((e) => e.axis === plan.waveform_ir.channels[ch].axis)} label={i === 0} />
      ))}
    </div>
  )
}

function RfChannel({ ch, plan, cache, ylim, exit, fades, label }: { ch: ChannelName; plan: PlanResponse; cache: ChannelCache; ylim: [number, number]; exit: BandExit | null; fades: FadeEvent[]; label: boolean }) {
  const t = useTutorialCursor((s) => s.t_us)
  const chan = plan.waveform_ir.channels[ch]
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const off = useRef<{ key: string; canvas: HTMLCanvasElement } | null>(null)
  const color = CHANNEL_COLORS[ch]
  const [t0, t1] = plan.t_span_us

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 360
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const pw = Math.round(width * dpr)
    const ph = Math.round(H * dpr)
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const sc = makeScale([t0, t1], ylim, { left: PAD.left, right: width - PAD.right, top: PAD.top, bottom: H - PAD.bottom })
    const key = `${plan.plan_id}|${ch}|${pw}|${ph}|${ylim.join(',')}|${exit?.t_drive_us ?? ''}|${fades.length}`
    if (!off.current || off.current.key !== key) {
      const c = document.createElement('canvas')
      c.width = pw
      c.height = ph
      const octx = c.getContext('2d')!
      octx.setTransform(dpr, 0, 0, dpr, 0, 0)
      drawStatic(octx, width, sc, chan, cache, color, ylim, exit, fades, label)
      off.current = { key, canvas: c }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, pw, ph)
    ctx.drawImage(off.current.canvas, 0, 0)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    drawCursor(ctx, sc, t, chan, color)
  }, [plan.plan_id, ch, chan, cache, color, t, t0, t1, ylim, exit, fades, label, cssWidth])

  const info = CHANNEL_INFO[ch]
  return (
    <div>
      <div className="row between small" style={{ marginBottom: 1 }}>
        <span>
          <span className="chan-dot" style={{ background: color }} />
          <strong style={{ color }}>{ch}</strong> <span className="dim">· sound {info.soundDir}</span>
        </span>
        <span className="faint">
          {chan.tones.length} tone{chan.tones.length === 1 ? '' : 's'} · live {fmt(cache.liveLo, 1)}–{fmt(cache.liveHi, 1)} MHz · band {chan.band_MHz[0]}–{chan.band_MHz[1]}
        </span>
      </div>
      <canvas ref={ref} className="plot" style={{ height: H, width: '100%' }} aria-label={`${ch} RF tracks`} />
    </div>
  )
}

function drawStatic(
  ctx: CanvasRenderingContext2D,
  width: number,
  sc: ReturnType<typeof makeScale>,
  chan: ChannelIR,
  cache: ChannelCache,
  color: string,
  ylim: [number, number],
  exit: BandExit | null,
  fades: FadeEvent[],
  label: boolean,
) {
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, width, H)
  const [lo, hi] = chan.band_MHz
  // out-of-band shading
  ctx.fillStyle = 'rgba(239,111,108,0.10)'
  if (ylim[1] > hi) ctx.fillRect(sc.left, sc.top, sc.right - sc.left, sc.y(hi) - sc.top)
  if (ylim[0] < lo) ctx.fillRect(sc.left, sc.y(lo), sc.right - sc.left, sc.bottom - sc.y(lo))
  ctx.font = '10px sans-serif'
  ctx.fillStyle = '#9aa5b3'
  ctx.textAlign = 'right'
  ctx.strokeStyle = '#1b2130'
  for (const v of ticks(ylim[0], ylim[1], 3)) {
    const y = sc.y(v)
    ctx.beginPath()
    ctx.moveTo(sc.left, y)
    ctx.lineTo(sc.right, y)
    ctx.stroke()
    ctx.fillText(fmt(v, 0), sc.left - 4, y + 3)
  }
  ctx.textAlign = 'center'
  for (const v of ticks(sc.invX(sc.left), sc.invX(sc.right), 6)) ctx.fillText(fmt(v, 0), sc.x(v), H - 4)
  ctx.textAlign = 'right'
  ctx.fillStyle = '#6b7584'
  ctx.fillText('µs', sc.left - 4, H - 4)
  // band edges
  ctx.setLineDash([2, 3])
  ctx.strokeStyle = '#8b95a3'
  for (const edge of chan.band_MHz) {
    if (edge < ylim[0] || edge > ylim[1]) continue
    ctx.beginPath()
    ctx.moveTo(sc.left, sc.y(edge))
    ctx.lineTo(sc.right, sc.y(edge))
    ctx.stroke()
  }
  ctx.setLineDash([])
  // tracks, batched by quantised alpha (as RfPanel)
  ctx.save()
  ctx.beginPath()
  ctx.rect(sc.left, sc.top, sc.right - sc.left, sc.bottom - sc.top)
  ctx.clip()
  ctx.lineWidth = 1.4
  const BINS = 8
  for (let b = 0; b < BINS; b++) {
    ctx.strokeStyle = withAlpha(color, 0.25 + 0.75 * ((b + 0.5) / BINS))
    ctx.beginPath()
    let any = false
    for (const tr of cache.tracks) {
      if (!tr.live) continue
      let pen = false
      for (let i = 0; i < tr.t.length - 1; i++) {
        const f0 = tr.f_MHz[i]
        const f1 = tr.f_MHz[i + 1]
        if (Number.isNaN(f0) || Number.isNaN(f1)) {
          pen = false
          continue
        }
        const bin = Math.min(BINS - 1, Math.floor(0.5 * (tr.alpha[i] + tr.alpha[i + 1]) * BINS))
        if (bin !== b) {
          pen = false
          continue
        }
        if (!pen) {
          ctx.moveTo(sc.x(tr.t[i]), sc.y(f0))
          pen = true
        }
        ctx.lineTo(sc.x(tr.t[i + 1]), sc.y(f1))
        any = true
      }
    }
    if (any) ctx.stroke()
  }
  ctx.restore()
  // hand-over ticks (drive time)
  ctx.strokeStyle = FADE_COLOR
  ctx.lineWidth = 1.5
  for (const e of fades) {
    const x = sc.x(e.time_us)
    ctx.beginPath()
    ctx.moveTo(x, sc.bottom - 7)
    ctx.lineTo(x, sc.bottom)
    ctx.stroke()
  }
  // first band exit
  if (exit) {
    const x = sc.x(exit.t_drive_us)
    ctx.strokeStyle = EXIT_COLOR
    ctx.lineWidth = 1.2
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(x, sc.top)
    ctx.lineTo(x, sc.bottom)
    ctx.stroke()
    ctx.setLineDash([])
    if (label) {
      ctx.fillStyle = EXIT_COLOR
      ctx.textAlign = x > sc.right - 150 ? 'right' : 'left'
      ctx.fillText(`leaves the band (${exit.edge_MHz} MHz) at ${fmt(exit.t_drive_us, 1)} µs`, x + (ctx.textAlign === 'left' ? 4 : -4), sc.top + 10)
    }
  }
}

function drawCursor(ctx: CanvasRenderingContext2D, sc: ReturnType<typeof makeScale>, t: number, chan: ChannelIR, color: string) {
  ctx.save()
  ctx.beginPath()
  ctx.rect(sc.left, sc.top, sc.right - sc.left, sc.bottom - sc.top)
  ctx.clip()
  const tc = t - 0.5 * chan.tau_us
  if (tc >= sc.invX(sc.left)) {
    ctx.setLineDash([3, 3])
    ctx.strokeStyle = 'rgba(242,245,248,0.6)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(sc.x(tc), sc.top)
    ctx.lineTo(sc.x(tc), sc.bottom)
    ctx.stroke()
    ctx.setLineDash([])
    for (const tone of chan.tones) {
      const a = toneAmp(tone, tc)
      if (a <= 0) continue
      ctx.fillStyle = withAlpha('#f2f5f8', 0.4 + 0.6 * a)
      ctx.strokeStyle = color
      ctx.beginPath()
      ctx.arc(sc.x(tc), sc.y(chan.f_center_MHz + toneFreqMHz(tone, tc)), 2.4, 0, 2 * Math.PI)
      ctx.fill()
      ctx.stroke()
    }
  }
  ctx.strokeStyle = '#f2f5f8'
  ctx.lineWidth = 1.2
  ctx.beginPath()
  ctx.moveTo(sc.x(t), sc.top)
  ctx.lineTo(sc.x(t), sc.bottom)
  ctx.stroke()
  ctx.restore()
}
