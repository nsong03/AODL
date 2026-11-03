/**
 * Acoustic content on each crystal at the cursor time (API.md §5.3, DESIGN §4): the envelope
 * `Σ A_n(t_ret(u))` as a filled curve coloured along `u` by the instantaneous frequency
 * `f_center + Σ A_n f_n / Σ A_n` on the channel's band scale; the unfilled region greyed and
 * labelled while `t < τ`; the transducer side (`u = −s·D/2`) and the sound direction marked;
 * `±w_in` shaded. The 100 MHz carrier (≈ 1150 cycles across 7.5 mm) is drawn only in the zoom
 * (`V(u)` over the central 0.1 mm at 4096 points) — at a few hundred samples it would alias.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { CHANNELS, type ChannelIR, type ChannelName, type PlanResponse } from '../../api/types'
import { CHANNEL_COLORS, CHANNEL_INFO } from '../../channels'
import { acousticWavelengthMm, apertureContent, apertureVoltage } from '../../physics/aperture'
import { fmt } from '../../physics/format'
import { toneLiveInterval } from '../../physics/waveform'
import { useStudio } from '../../state/store'
import { makeScale, useElementWidth } from '../../hooks/useCanvas'
import { freqColor, tonesInWindow, type LiveTone } from './apertureUtil'
import { Panel, Skeleton } from '../common/Panel'

const H = 96
const ZOOM_H = 56
const PAD = { left: 10, right: 10, top: 14, bottom: 16 }
const N_SAMPLES = 320

export function AperturePanel() {
  const plan = useStudio((s) => s.plan)
  const status = useStudio((s) => s.planStatus)
  const [zoom, setZoom] = useState(false)
  const live = useMemo(() => {
    if (!plan) return null
    const out = {} as Record<ChannelName, LiveTone[]>
    for (const ch of CHANNELS) {
      const chan = plan.waveform_ir.channels[ch]
      out[ch] = chan.tones.map((tone) => {
        const [a, b] = toneLiveInterval(tone, plan.t_span_us)
        return { tone, a, b }
      })
    }
    return out
  }, [plan])
  return (
    <Panel
      title="Acoustic window on each crystal"
      sub="envelope Σ A_n(t_ret(u)) at the cursor, coloured by f_inst · t_ret(u) = t − τ/2 − s·u/v"
      actions={
        <button className={`small ${zoom ? 'active' : ''}`} onClick={() => setZoom(!zoom)} title="draw the literal carrier V(u) over the central 0.1 mm (4096 samples)">
          carrier zoom
        </button>
      }
    >
      {!plan || !live ? (
        status === 'loading' ? (
          <Skeleton height={4 * H} label="planning…" />
        ) : (
          <div className="caption">No plan yet.</div>
        )
      ) : (
        <div className="stack" style={{ gap: 6 }}>
          {CHANNELS.map((ch) => (
            <ApertureCanvas key={ch} ch={ch} plan={plan} live={live[ch]} zoom={zoom} />
          ))}
          <div className="caption">
            Content exists where s·u ≤ v·t − D/2: the half-aperture that contains the transducer, bounded by the leading wavefront, full for t ≥ τ. A
            counter-propagating pair's tweezer is dark before τ/2 because both wavefronts must reach the beam.
          </div>
        </div>
      )}
    </Panel>
  )
}

function ApertureCanvas({ ch, plan, live, zoom }: { ch: ChannelName; plan: PlanResponse; live: LiveTone[]; zoom: boolean }) {
  const t = useStudio((s) => s.t_us)
  const wIn = useStudio((s) => s.hardware.w_in_mm)
  const chan: ChannelIR = plan.waveform_ir.channels[ch]
  const ref = useRef<HTMLCanvasElement>(null)
  const zoomRef = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const color = CHANNEL_COLORS[ch]
  const info = CHANNEL_INFO[ch]
  const half = 0.5 * chan.aperture_mm
  const s = chan.sound_sign

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 400
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const pw = Math.round(width * dpr)
    const ph = Math.round(H * dpr)
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw
      canvas.height = ph
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, width, H)
    const tones = tonesInWindow(live, t, chan.tau_us)
    const content = apertureContent(chan, t, N_SAMPLES, tones)
    let envMax = 0
    for (let i = 0; i < N_SAMPLES; i++) envMax = Math.max(envMax, content.envelope[i])
    const ymax = Math.max(1, envMax) * 1.1
    const sc = makeScale([-half, half], [0, ymax], { left: PAD.left, right: width - PAD.right, top: PAD.top, bottom: H - PAD.bottom })
    const band = chan.band_MHz
    // beam ±w_in
    ctx.fillStyle = 'rgba(255,107,107,0.10)'
    ctx.fillRect(sc.x(-wIn), sc.top, sc.x(wIn) - sc.x(-wIn), sc.bottom - sc.top)
    // envelope bars coloured by f_inst
    const du = (sc.right - sc.left) / (N_SAMPLES - 1)
    let unfilledFrom = Infinity
    let unfilledTo = -Infinity
    for (let i = 0; i < N_SAMPLES; i++) {
      const x = sc.x(content.u_mm[i])
      if (!content.filled[i]) {
        unfilledFrom = Math.min(unfilledFrom, x)
        unfilledTo = Math.max(unfilledTo, x)
        continue
      }
      const e = content.envelope[i]
      if (e <= 0) continue
      const f = content.f_inst_MHz[i]
      ctx.fillStyle = freqColor((f - band[0]) / (band[1] - band[0]))
      ctx.fillRect(x - du / 2, sc.y(e), du + 0.6, sc.bottom - sc.y(e))
    }
    // unfilled region (hatched grey)
    if (Number.isFinite(unfilledFrom)) {
      ctx.fillStyle = 'rgba(154,165,179,0.10)'
      ctx.fillRect(unfilledFrom - du / 2, sc.top, unfilledTo - unfilledFrom + du, sc.bottom - sc.top)
      ctx.strokeStyle = 'rgba(154,165,179,0.35)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let x = unfilledFrom - 20; x < unfilledTo + 20; x += 9) {
        ctx.moveTo(x, sc.bottom)
        ctx.lineTo(x + (sc.bottom - sc.top), sc.top)
      }
      ctx.save()
      ctx.beginPath()
      ctx.rect(unfilledFrom - du / 2, sc.top, unfilledTo - unfilledFrom + du, sc.bottom - sc.top)
      ctx.clip()
      ctx.stroke()
      ctx.restore()
      ctx.fillStyle = '#9aa5b3'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('not yet filled', (unfilledFrom + unfilledTo) / 2, (sc.top + sc.bottom) / 2 + 3)
      // leading wavefront
      const edge = s * (chan.sound_speed_m_s * t * 1e-6 - half * 1e-3) / 1e-3
      ctx.strokeStyle = '#f2f5f8'
      ctx.setLineDash([2, 2])
      ctx.beginPath()
      ctx.moveTo(sc.x(edge), sc.top)
      ctx.lineTo(sc.x(edge), sc.bottom)
      ctx.stroke()
      ctx.setLineDash([])
    }
    // frame & baseline
    ctx.strokeStyle = '#2a303a'
    ctx.strokeRect(sc.left, sc.top, sc.right - sc.left, sc.bottom - sc.top)
    // transducer side: u = −s·D/2
    const tx = sc.x(-s * half)
    ctx.fillStyle = '#f2f5f8'
    ctx.fillRect(tx - (s > 0 ? 0 : 4), sc.top - 4, 4, sc.bottom - sc.top + 8)
    ctx.font = '10px sans-serif'
    ctx.fillStyle = '#d7dde5'
    ctx.textAlign = s > 0 ? 'left' : 'right'
    ctx.fillText('transducer', tx + (s > 0 ? 7 : -7), sc.top + 9)
    // sound arrow
    const ax0 = s > 0 ? tx + 60 : tx - 60
    const ax1 = s > 0 ? tx + 92 : tx - 92
    ctx.strokeStyle = '#d7dde5'
    ctx.lineWidth = 1.2
    ctx.beginPath()
    ctx.moveTo(ax0, sc.top + 6)
    ctx.lineTo(ax1, sc.top + 6)
    ctx.lineTo(ax1 - s * 4, sc.top + 3)
    ctx.moveTo(ax1, sc.top + 6)
    ctx.lineTo(ax1 - s * 4, sc.top + 9)
    ctx.stroke()
    ctx.textAlign = s > 0 ? 'left' : 'right'
    ctx.fillText(`sound ${info.soundDir}`, s > 0 ? ax1 + 5 : ax1 - 5, sc.top + 9)
    // u ticks
    ctx.fillStyle = '#9aa5b3'
    ctx.textAlign = 'center'
    for (const u of [-half, -half / 2, 0, half / 2, half]) ctx.fillText(fmt(u, 2), sc.x(u), H - 4)
    ctx.textAlign = 'right'
    ctx.fillText(`u [mm] · ±w_in shaded`, sc.right - 2, H - 4)
    // beam-centre retarded time
    ctx.textAlign = 'center'
    ctx.fillStyle = '#9aa5b3'
    ctx.fillText(`t_ret(0) = ${fmt(t - 0.5 * chan.tau_us, 1)} µs · ${tones.length} tone${tones.length === 1 ? '' : 's'} in the window`, (sc.left + sc.right) / 2, sc.bottom - 4)
  }, [chan, live, t, wIn, half, s, info.soundDir, cssWidth])

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
    const tones = tonesInWindow(live, t, chan.tau_us)
    const { u_mm, v } = apertureVoltage(chan, t, -0.05, 0.05, 4096, tones)
    let vmax = 0
    for (let i = 0; i < v.length; i++) vmax = Math.max(vmax, Math.abs(v[i]))
    const amp = Math.max(vmax, 1e-9)
    const sc = makeScale([-0.05, 0.05], [-amp, amp], { left: PAD.left, right: width - PAD.right, top: 4, bottom: ZOOM_H - 4 })
    ctx.strokeStyle = color
    ctx.lineWidth = 0.8
    ctx.beginPath()
    for (let i = 0; i < u_mm.length; i++) {
      const x = sc.x(u_mm[i])
      const y = sc.y(v[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.fillStyle = '#9aa5b3'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(`V(u) on the central 0.1 mm · 4096 pts · Λ = v/f_center = ${fmt(acousticWavelengthMm(chan) * 1e3, 1)} µm · ${Math.round(0.1 / acousticWavelengthMm(chan))} cycles`, PAD.left + 4, 11)
  }, [zoom, chan, live, t, color, cssWidth])

  return (
    <div>
      <div className="row between small" style={{ marginBottom: 2 }}>
        <span>
          <span className="chan-dot" style={{ background: color }} />
          <strong style={{ color }}>{ch}</strong> <span className="dim">· aperture D = {chan.aperture_mm} mm along {info.axis}</span>
        </span>
        <FreqBar band={chan.band_MHz} />
      </div>
      <canvas ref={ref} className="plot" style={{ height: H }} />
      {zoom && <canvas ref={zoomRef} className="plot" style={{ height: ZOOM_H, marginTop: 2 }} />}
    </div>
  )
}

function FreqBar({ band }: { band: [number, number] }) {
  const stops = Array.from({ length: 9 }, (_, i) => `${freqColor(i / 8)} ${(100 * i) / 8}%`).join(', ')
  return (
    <span className="row faint" style={{ gap: 4, fontSize: 10.5 }}>
      {band[0]}
      <span style={{ display: 'inline-block', width: 70, height: 8, borderRadius: 2, background: `linear-gradient(90deg, ${stops})` }} />
      {band[1]} MHz
    </span>
  )
}
