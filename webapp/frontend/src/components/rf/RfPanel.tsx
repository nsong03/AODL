/**
 * RF panel: one canvas per channel with the tone frequency tracks `f_center + f_n(t)` [MHz]
 * (opacity = envelope, broken where `A = 0`), dotted band edges, the shaded aperture window
 * `[t − τ, t]` ("inside the crystal", clipped at 0), a tick at `t − τ/2` ("beam centre") and the
 * cursor; optionally the voltage zoom `V(t')` on `[t − 0.2 µs, t + 0.2 µs]` (2001 points).
 * Tracks are computed once per plan (`liveTracks`) and cached on an offscreen canvas.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { CHANNELS, type ChannelIR, type ChannelName, type PlanResponse } from '../../api/types'
import { CHANNEL_COLORS, CHANNEL_INFO, withAlpha } from '../../channels'
import { fmt, ticks } from '../../physics/format'
import { voltageZoom } from '../../physics/waveform'
import { useRfTracks, type ChannelCache } from './useRfTracks'
import { toneFreqMHz, toneAmp } from '../../physics/waveform'
import { useStudio } from '../../state/store'
import { makeScale, useElementWidth } from '../../hooks/useCanvas'
import { Panel, Skeleton } from '../common/Panel'

const TRACK_H = 112
const ZOOM_H = 54
const PAD = { left: 46, right: 10, top: 8, bottom: 18 }

export function RfPanel() {
  const plan = useStudio((s) => s.plan)
  const status = useStudio((s) => s.planStatus)
  const [zoom, setZoom] = useState(false)
  const [fullBand, setFullBand] = useState(false)
  const cache = useRfTracks(plan)
  return (
    <Panel
      title="RF drives"
      sub="f_center + f_n(t) [MHz] vs drive time · opacity = envelope A_n · shaded = inside the crystal"
      actions={
        <>
          <button className={`small ${fullBand ? 'active' : ''}`} onClick={() => setFullBand(!fullBand)} title="y-axis: live excursion or the full usable band">
            full band
          </button>
          <button className={`small ${zoom ? 'active' : ''}`} onClick={() => setZoom(!zoom)} title="draw V(t') on ±0.2 µs around the cursor, carrier included (2001 points)">
            voltage zoom
          </button>
        </>
      }
    >
      {!plan || !cache ? (
        status === 'loading' ? (
          <Skeleton height={4 * TRACK_H} label="planning…" />
        ) : (
          <div className="caption">No plan yet.</div>
        )
      ) : (
        <div className="stack" style={{ gap: 6 }}>
          {CHANNELS.map((ch) => (
            <ChannelCanvas key={ch} ch={ch} plan={plan} cache={cache[ch]} zoom={zoom} fullBand={fullBand} />
          ))}
          <div className="legend">
            <span>
              <span className="sw" style={{ background: '#f2f5f8' }} /> cursor t
            </span>
            <span>
              <span className="sw" style={{ background: '#f2f5f8', opacity: 0.6, height: 1 }} /> t − τ/2 (beam centre: what the atoms see now)
            </span>
            <span>
              <span className="sw" style={{ background: 'rgba(255,255,255,0.18)', height: 10 }} /> [t − τ, t] inside the crystal
            </span>
            <span>
              <span className="sw" style={{ background: 'var(--text-faint)', height: 1 }} /> band edges (dotted)
            </span>
          </div>
        </div>
      )}
    </Panel>
  )
}

function ChannelCanvas({ ch, plan, cache, zoom, fullBand }: { ch: ChannelName; plan: PlanResponse; cache: ChannelCache; zoom: boolean; fullBand: boolean }) {
  const t = useStudio((s) => s.t_us)
  const chan: ChannelIR = plan.waveform_ir.channels[ch]
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const zoomRef = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(canvasRef)
  const offscreen = useRef<HTMLCanvasElement | null>(null)
  const offKey = useRef('')
  const color = CHANNEL_COLORS[ch]
  const [t0, t1] = plan.t_span_us
  const tau = chan.tau_us

  // y range: live excursion (± 8 %) or the band
  const ylim = useMemo<[number, number]>(() => {
    const band = chan.band_MHz
    if (fullBand || !Number.isFinite(cache.liveLo)) {
      const m = 0.04 * (band[1] - band[0])
      return [band[0] - m, band[1] + m]
    }
    let lo = cache.liveLo
    let hi = cache.liveHi
    if (hi - lo < 1e-6) {
      lo -= 0.5
      hi += 0.5
    }
    const m = 0.08 * (hi - lo)
    return [lo - m, hi + m]
  }, [cache, chan.band_MHz, fullBand])

  // main canvas: static tracks from the offscreen cache + dynamic overlay
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 400
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const pw = Math.round(width * dpr)
    const ph = Math.round(TRACK_H * dpr)
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const scale = makeScale([t0, t1], ylim, { left: PAD.left, right: width - PAD.right, top: PAD.top, bottom: TRACK_H - PAD.bottom })

    const key = `${plan.plan_id}|${ch}|${pw}|${ph}|${ylim[0]}|${ylim[1]}`
    if (offKey.current !== key || !offscreen.current) {
      const off = document.createElement('canvas')
      off.width = pw
      off.height = ph
      const octx = off.getContext('2d')!
      octx.setTransform(dpr, 0, 0, dpr, 0, 0)
      drawStatic(octx, width, TRACK_H, scale, chan, cache, color, ylim)
      offscreen.current = off
      offKey.current = key
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, pw, ph)
    ctx.drawImage(offscreen.current, 0, 0)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    drawDynamic(ctx, scale, t, tau, chan, color)
  }, [plan.plan_id, ch, chan, cache, color, t, t0, t1, tau, ylim, cssWidth])

  // zoom canvas: V(t') on ±0.2 µs
  useEffect(() => {
    const canvas = zoomRef.current
    if (!canvas || !zoom) return
    const width = cssWidth || canvas.clientWidth || 400
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(ZOOM_H * dpr)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, width, ZOOM_H)
    const { t: tt, v } = voltageZoom(chan, t, 0.2, 2001)
    let vmax = 0
    for (let i = 0; i < v.length; i++) vmax = Math.max(vmax, Math.abs(v[i]))
    const amp = Math.max(vmax, 1e-9)
    const sc = makeScale([tt[0], tt[tt.length - 1]], [-amp, amp], { left: PAD.left, right: width - PAD.right, top: 4, bottom: ZOOM_H - 4 })
    ctx.strokeStyle = '#2a303a'
    ctx.beginPath()
    ctx.moveTo(sc.left, sc.y(0))
    ctx.lineTo(sc.right, sc.y(0))
    ctx.stroke()
    ctx.strokeStyle = color
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let i = 0; i < tt.length; i++) {
      const x = sc.x(tt[i])
      const y = sc.y(v[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.strokeStyle = 'rgba(242,245,248,0.7)'
    ctx.beginPath()
    ctx.moveTo(sc.x(t), 0)
    ctx.lineTo(sc.x(t), ZOOM_H)
    ctx.stroke()
    ctx.fillStyle = '#9aa5b3'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(`V(t′) · t′ ∈ [t − 0.2, t + 0.2] µs · 2001 pts · |V|max ${fmt(vmax, 2)} (Σ A_n cos, carrier ${chan.f_center_MHz} MHz)`, PAD.left + 4, 11)
    ctx.textAlign = 'right'
    ctx.fillText('−0.2', PAD.left - 4, ZOOM_H - 4)
    ctx.textAlign = 'left'
    ctx.fillText('+0.2 µs', sc.right - 40, ZOOM_H - 4)
  }, [zoom, chan, t, color, cssWidth])

  const info = CHANNEL_INFO[ch]
  const nLive = cache.tracks.filter((tr) => tr.live).length
  return (
    <div>
      <div className="row between small" style={{ marginBottom: 2 }}>
        <span>
          <span className="chan-dot" style={{ background: color }} />
          <strong style={{ color }}>{ch}</strong> <span className="dim">· sound {info.soundDir} · {info.role}</span>
        </span>
        <span className="faint">
          {chan.tones.length} tone{chan.tones.length === 1 ? '' : 's'} {chan.tones.length !== nLive && `(${nLive} ever live)`} · τ {fmt(tau, 2)} µs
        </span>
      </div>
      <canvas ref={canvasRef} className="plot" style={{ height: TRACK_H }} />
      {zoom && <canvas ref={zoomRef} className="plot" style={{ height: ZOOM_H, marginTop: 2 }} />}
    </div>
  )
}

function drawStatic(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  sc: ReturnType<typeof makeScale>,
  chan: ChannelIR,
  cache: ChannelCache,
  color: string,
  ylim: [number, number],
) {
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, width, height)
  // y ticks
  ctx.font = '10px sans-serif'
  ctx.fillStyle = '#9aa5b3'
  ctx.textAlign = 'right'
  ctx.strokeStyle = '#1b2130'
  for (const v of ticks(ylim[0], ylim[1], 4)) {
    const y = sc.y(v)
    ctx.beginPath()
    ctx.moveTo(sc.left, y)
    ctx.lineTo(sc.right, y)
    ctx.stroke()
    ctx.fillText(fmt(v, Math.abs(ylim[1] - ylim[0]) < 2 ? 2 : 1), sc.left - 4, y + 3)
  }
  // x ticks
  ctx.textAlign = 'center'
  for (const v of ticks(sc.invX(sc.left), sc.invX(sc.right), 6)) {
    ctx.fillText(fmt(v, 0), sc.x(v), height - 5)
  }
  ctx.fillStyle = '#6b7584'
  ctx.textAlign = 'left'
  ctx.fillText('µs', sc.right - 14, height - 5)
  ctx.save()
  ctx.translate(10, (sc.top + sc.bottom) / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.fillText('MHz', 0, 0)
  ctx.restore()
  // band edges (dotted)
  ctx.setLineDash([2, 3])
  ctx.strokeStyle = '#6b7584'
  for (const edge of chan.band_MHz) {
    if (edge < ylim[0] || edge > ylim[1]) continue
    const y = sc.y(edge)
    ctx.beginPath()
    ctx.moveTo(sc.left, y)
    ctx.lineTo(sc.right, y)
    ctx.stroke()
    ctx.fillStyle = '#6b7584'
    ctx.textAlign = 'right'
    ctx.fillText(`band ${edge}`, sc.right - 2, edge === chan.band_MHz[1] ? y + 10 : y - 3)
  }
  ctx.setLineDash([])
  // tracks, batched by quantised alpha
  ctx.save()
  ctx.beginPath()
  ctx.rect(sc.left, sc.top, sc.right - sc.left, sc.bottom - sc.top)
  ctx.clip()
  ctx.lineWidth = 1.4
  const BINS = 8
  for (let b = 0; b < BINS; b++) {
    const alpha = 0.25 + 0.75 * ((b + 0.5) / BINS)
    ctx.strokeStyle = withAlpha(color, alpha)
    ctx.beginPath()
    let any = false
    for (const tr of cache.tracks) {
      if (!tr.live) continue
      const n = tr.t.length
      let pen = false
      for (let i = 0; i < n - 1; i++) {
        const f0 = tr.f_MHz[i]
        const f1 = tr.f_MHz[i + 1]
        if (Number.isNaN(f0) || Number.isNaN(f1)) {
          pen = false
          continue
        }
        const mean = 0.5 * (tr.alpha[i] + tr.alpha[i + 1])
        const bin = Math.min(BINS - 1, Math.floor(mean * BINS))
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
}

function drawDynamic(ctx: CanvasRenderingContext2D, sc: ReturnType<typeof makeScale>, t: number, tau: number, chan: ChannelIR, color: string) {
  const tA = Math.max(sc.invX(sc.left), t - tau) // clipped at the start of the drive
  const tB = Math.min(t, sc.invX(sc.right))
  ctx.save()
  ctx.beginPath()
  ctx.rect(sc.left, sc.top, sc.right - sc.left, sc.bottom - sc.top)
  ctx.clip()
  if (tB > tA) {
    ctx.fillStyle = 'rgba(255,255,255,0.14)'
    ctx.fillRect(sc.x(tA), sc.top, sc.x(tB) - sc.x(tA), sc.bottom - sc.top)
  }
  // beam-centre tick t − τ/2
  const tc = t - 0.5 * tau
  if (tc >= sc.invX(sc.left)) {
    ctx.setLineDash([3, 3])
    ctx.strokeStyle = 'rgba(242,245,248,0.75)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(sc.x(tc), sc.top)
    ctx.lineTo(sc.x(tc), sc.bottom)
    ctx.stroke()
    ctx.setLineDash([])
    // live tone markers at the beam centre
    for (const tone of chan.tones) {
      const a = toneAmp(tone, tc)
      if (a <= 0) continue
      const f = chan.f_center_MHz + toneFreqMHz(tone, tc)
      ctx.fillStyle = withAlpha('#f2f5f8', 0.4 + 0.6 * a)
      ctx.beginPath()
      ctx.arc(sc.x(tc), sc.y(f), 2.4, 0, 2 * Math.PI)
      ctx.fill()
      ctx.strokeStyle = color
      ctx.stroke()
    }
  }
  // cursor
  ctx.strokeStyle = '#f2f5f8'
  ctx.lineWidth = 1.2
  ctx.beginPath()
  ctx.moveTo(sc.x(t), sc.top)
  ctx.lineTo(sc.x(t), sc.bottom)
  ctx.stroke()
  ctx.restore()
  ctx.fillStyle = '#d7dde5'
  ctx.font = '10px sans-serif'
  ctx.textAlign = sc.x(t) > sc.right - 120 ? 'right' : 'left'
  ctx.fillText(`t = ${fmt(t, 1)} µs`, sc.x(t) + (ctx.textAlign === 'left' ? 4 : -4), sc.top + 10)
  if (tc >= sc.invX(sc.left)) {
    ctx.fillStyle = 'rgba(242,245,248,0.7)'
    ctx.textAlign = sc.x(tc) > sc.right - 120 ? 'right' : 'left'
    ctx.fillText('beam centre t − τ/2', sc.x(tc) + (ctx.textAlign === 'left' ? 4 : -4), sc.bottom - 4)
  }
}
