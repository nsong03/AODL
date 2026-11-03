/**
 * Per-AOD misalignment (DESIGN §1.7): aperture offset (∥, ⊥), sound-axis rotation, acoustic
 * delay (Eq. S29), 4f magnification mismatch (velocity scale), pointing tilt. "Run" calls
 * `/api/advanced/simulate` at four times spread over the trajectory with `xz.nz = 24`, once
 * misaligned and once aligned, and shows both on a shared colour scale with the metrics table.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { api, decodePacked, errorMessage, isAbort } from '../../api/client'
import { CHANNELS, type AdvancedFrame, type AdvancedSimulateResponse, type ChannelMisalignment, type ChannelName, type ImageXY, type ImageXZ, type Misalignment } from '../../api/types'
import { CHANNEL_COLORS, CHANNEL_INFO } from '../../channels'
import { fmt, fmtSigned } from '../../physics/format'
import { useElementWidth } from '../../hooks/useCanvas'
import { GAMMA } from '../../physics/zColor'
import { currentRequest, useStudio } from '../../state/store'
import { Badge, ErrorBanner, Panel, Spinner } from '../common/Panel'
import { infernoCss } from './colormaps'

const ALIGNED: ChannelMisalignment = { offset_par_um: 0, offset_perp_um: 0, rotation_mrad: 0, delay_ns: 0, velocity_scale: 1, tilt_x_mrad: 0, tilt_y_mrad: 0 }

const KNOBS: { key: keyof ChannelMisalignment; label: string; unit: string; min: number; max: number; step: number; title: string }[] = [
  { key: 'offset_par_um', label: 'offset ∥ sound', unit: 'µm', min: -500, max: 500, step: 5, title: 'aperture centre shift along the channel axis (δ∥)' },
  { key: 'offset_perp_um', label: 'offset ⊥', unit: 'µm', min: -500, max: 500, step: 5, title: 'aperture centre shift across the channel axis (δ⊥)' },
  { key: 'rotation_mrad', label: 'sound-axis rotation', unit: 'mrad', min: -50, max: 50, step: 0.5, title: 'rotation θ of the acoustic axis about the optical axis' },
  { key: 'delay_ns', label: 'acoustic delay', unit: 'ns', min: -500, max: 500, step: 5, title: 'timing error δt of this channel (Eq. S29)' },
  { key: 'velocity_scale', label: '4f magnification m', unit: '', min: 0.9, max: 1.1, step: 0.001, title: 'effective sound speed scale v → m·v (relay magnification mismatch)' },
  { key: 'tilt_x_mrad', label: 'pointing tilt x', unit: 'mrad', min: -2, max: 2, step: 0.02, title: 'adds exp(i k α_x x)' },
  { key: 'tilt_y_mrad', label: 'pointing tilt y', unit: 'mrad', min: -2, max: 2, step: 0.02, title: 'adds exp(i k α_y y)' },
]

function isAligned(m: Misalignment): boolean {
  return CHANNELS.every((ch) => {
    const c = m[ch]
    if (!c) return true
    return KNOBS.every((k) => Math.abs(c[k.key] - ALIGNED[k.key]) < 1e-12)
  })
}

export function MisalignmentPanel() {
  const plan = useStudio((s) => s.plan)
  const sim = useStudio((s) => s.sim)
  const adv = useStudio((s) => s.advanced)
  const setMis = useStudio((s) => s.setMisalignment)
  const setAdvanced = useStudio((s) => s.setAdvanced)
  const [ch, setCh] = useState<ChannelName>('Ax')
  const [aligned, setAlignedRes] = useState<AdvancedSimulateResponse | null>(null)
  const [frameIdx, setFrameIdx] = useState(0)
  const abortRef = useRef<AbortController | null>(null)
  const m = adv.misalignment
  const cur = m[ch] ?? ALIGNED

  const setKnob = (key: keyof ChannelMisalignment, v: number) => setMis({ ...m, [ch]: { ...cur, [key]: v } })

  const times = useMemo(() => {
    if (!plan) return []
    const tau = plan.constants.tau_us
    const lo = tau
    const hi = Math.min(plan.duration_us + 0.5 * tau, plan.t_span_us[1] - 0.5 * tau)
    if (hi <= lo) return [lo]
    return [0.1, 0.4, 0.7, 1.0].map((f) => Number((lo + (hi - lo) * f).toFixed(3)))
  }, [plan])

  const run = async () => {
    if (!plan) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setAdvanced({ advSimStatus: 'loading', advSimError: null })
    const base = {
      plan: currentRequest(useStudio.getState()),
      times_us: times,
      grid: { half_x_um: 6.0, half_y_um: 6.0, n: 96, center: 'tracked' as const },
      xz: { nz: 24, z_half_um: 8.0 },
      model: { pupil_n: 256 as const, span_w_in: 2.5, order_model: 'j1' as const, comb_dft_points: 16, max_groups: 128, efficiency: null },
    }
    try {
      const mis = await api.advancedSimulate({ ...base, misalignment: m, reference: true }, ctrl.signal)
      if (ctrl.signal.aborted) return
      let ref: AdvancedSimulateResponse | null = mis
      if (!isAligned(m)) {
        ref = await api.advancedSimulate({ ...base, misalignment: {}, reference: false }, ctrl.signal)
        if (ctrl.signal.aborted) return
      }
      setAlignedRes(ref)
      setAdvanced({ advSim: mis, advSimStatus: 'ok' })
      setFrameIdx(Math.min(frameIdx, mis.frames.length - 1))
    } catch (e) {
      if (isAbort(e)) return
      setAdvanced({ advSimStatus: 'error', advSimError: errorMessage(e) })
    }
  }

  const res = adv.advSim
  const frame = res?.frames[frameIdx]
  const refFrame = aligned?.frames[frameIdx]
  const sharedPeak = useMemo(() => {
    if (!res) return 1
    let p = 0
    for (const f of res.frames) p = Math.max(p, f.xy.image.peak)
    if (aligned) for (const f of aligned.frames) p = Math.max(p, f.xy.image.peak)
    return p
  }, [res, aligned])

  return (
    <Panel
      title="Misalignment (FFT pupil engine)"
      sub="per-AOD aperture offset, sound-axis rotation, acoustic delay, 4f magnification, pointing tilt · J₁ thin-grating order model, comb-DFT line decomposition"
      actions={
        <>
          <button className="small" onClick={() => setMis({})} disabled={isAligned(m)}>
            reset all
          </button>
          <button className="primary small" onClick={() => void run()} disabled={!plan || adv.advSimStatus === 'loading'}>
            {adv.advSimStatus === 'loading' ? 'running…' : 'run'}
          </button>
        </>
      }
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: 16 }}>
        <div style={{ width: 340, flex: '0 0 auto' }}>
          <div className="tabs">
            {CHANNELS.map((c) => (
              <button key={c} className={ch === c ? 'active' : ''} style={{ color: ch === c ? CHANNEL_COLORS[c] : undefined }} onClick={() => setCh(c)}>
                {c}
                {m[c] && !isAligned({ [c]: m[c] }) ? ' •' : ''}
              </button>
            ))}
          </div>
          <div className="caption" style={{ marginBottom: 6 }}>
            {ch}: axis {CHANNEL_INFO[ch].axis}, sound {CHANNEL_INFO[ch].soundDir}, transducer on the {CHANNEL_INFO[ch].transducer}
          </div>
          <div className="sliders">
            {KNOBS.map((k) => (
              <label key={k.key} className="field" title={k.title}>
                <span>
                  {k.label} {k.unit && <span className="unit">[{k.unit}]</span>}
                </span>
                <input type="range" min={k.min} max={k.max} step={k.step} value={cur[k.key]} onChange={(e) => setKnob(k.key, parseFloat(e.target.value))} />
                <input type="number" step={k.step} value={cur[k.key]} onChange={(e) => Number.isFinite(parseFloat(e.target.value)) && setKnob(k.key, parseFloat(e.target.value))} />
              </label>
            ))}
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <button className="small" onClick={() => setMis({ ...m, [ch]: ALIGNED })}>
              reset {ch}
            </button>
            <span className="caption">
              run at t = {times.map((v) => fmt(v, 1)).join(', ')} µs · 96² grid · XZ nz = 24 — a run takes seconds (two calls when misaligned: the aligned
              reference is computed too)
            </span>
          </div>
          {adv.advSimStatus === 'error' && adv.advSimError && (
            <div style={{ marginTop: 8 }}>
              <ErrorBanner code="advanced/simulate" message={adv.advSimError} />
            </div>
          )}
          {!plan && <div className="caption" style={{ marginTop: 8 }}>Plan a trajectory in the basic workbench first.</div>}
          {!sim && plan && <div className="caption" style={{ marginTop: 8 }}>The basic simulation is still running; the engine does not need it, but its tracked plane is handy for comparison.</div>}
        </div>
        <div className="grow">
          {adv.advSimStatus === 'loading' && (
            <div className="row" style={{ marginBottom: 8 }}>
              <Spinner label="propagating the pupils (chirp-z transform per frequency group)…" />
            </div>
          )}
          {res && frame ? (
            <>
              <div className="row" style={{ marginBottom: 6 }}>
                {res.frames.map((f, i) => (
                  <button key={i} className={`small ${i === frameIdx ? 'active' : ''}`} onClick={() => setFrameIdx(i)}>
                    t = {fmt(f.time_us, 1)} µs
                  </button>
                ))}
                <Badge>{frame.n_groups} groups{frame.groups_dropped ? ` (${frame.groups_dropped} dropped)` : ''}</Badge>
                <Badge>plane Z {fmt(frame.plane_z_um, 2)} µm</Badge>
              </div>
              <div className="imgpair">
                <PackedView im={frame.xy} peak={sharedPeak} label="misaligned · XY (FFT engine)" />
                {refFrame ? <PackedView im={refFrame.xy} peak={sharedPeak} label="aligned reference · XY" /> : <div className="caption">no aligned frame</div>}
                {frame.xz && <PackedView im={frame.xz} peak={sharedPeak} label="misaligned · XZ" aspectH={120} />}
                {refFrame?.xz && <PackedView im={refFrame.xz} peak={sharedPeak} label="aligned reference · XZ" aspectH={120} />}
              </div>
              <MetricsTable frame={frame} res={res} />
              {res.notes.length > 0 && (
                <ul className="notes">
                  {res.notes.map((n, i) => (
                    <li key={i}>{n}</li>
                  ))}
                </ul>
              )}
            </>
          ) : (
            adv.advSimStatus !== 'loading' && <div className="caption">Set the knobs and press run. Intensities in this engine are in its own prefactor-dropped units; only ratios between images from the same call are meaningful.</div>
          )}
        </div>
      </div>
      <p className="caption" style={{ marginTop: 8 }}>
        Model assumptions: thin phase grating per AOD evaluated at its own retarded time on a 2-D pupil, J₁ law for the +1 order (reduces to aodl's
        weak-drive table at small C), fill mask scaled by the magnification, uncropped Gaussian input, groups by optical frequency sum add in intensity,
        common carrier-tilt pointing offset dropped and reported as static_offset. Waists are second-moment radii (2σ); z_best from a parabolic fit over
        the XZ/YZ scans.
      </p>
    </Panel>
  )
}

function PackedView({ im, peak, label, aspectH }: { im: ImageXY | ImageXZ; peak: number; label: string; aspectH?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  const values = useMemo(() => decodePacked(im.image), [im])
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const [ny, nx] = im.image.shape
    const width = cssWidth || canvas.clientWidth || 200
    const height = aspectH ?? Math.round((width * ny) / nx)
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
    for (let j = 0; j < ny; j++) {
      const src = (ny - 1 - j) * nx
      for (let i = 0; i < nx; i++) {
        const v = peak > 0 ? Math.min(1, values[src + i] / peak) : 0
        const s = Math.pow(v, GAMMA)
        const css = infernoCss(s)
        const q = (j * nx + i) * 4
        const parts = css.match(/\d+/g)!.map(Number)
        data.data[q] = parts[0]
        data.data[q + 1] = parts[1]
        data.data[q + 2] = parts[2]
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
    const xs = im.x_um
    ctx.fillText(`x ${fmt(xs[0], 1)}…${fmt(xs[xs.length - 1], 1)} µm · peak/shared ${fmt(im.image.peak / peak, 3)}`, width - 6, height - 6)
  }, [im, values, peak, label, aspectH, cssWidth])
  return <canvas ref={ref} className="plot" style={{ width: '100%' }} />
}

function MetricsTable({ frame, res }: { frame: AdvancedFrame; res: AdvancedSimulateResponse }) {
  const a = frame.metrics
  const r = frame.reference_metrics
  const row = (label: string, va: number | null, vr: number | null | undefined, digits = 3, unit = '') => (
    <tr>
      <td>{label}</td>
      <td className="num">{va === null ? '—' : `${fmt(va, digits)}${unit}`}</td>
      <td className="num dim">{vr === null || vr === undefined ? '—' : `${fmt(vr, digits)}${unit}`}</td>
      <td className="num">{va !== null && vr !== null && vr !== undefined ? fmtSigned(va - vr, digits) : ''}</td>
    </tr>
  )
  return (
    <div style={{ marginTop: 8 }}>
      <table className="data">
        <thead>
          <tr>
            <th>metric</th>
            <th className="num">misaligned</th>
            <th className="num">aligned</th>
            <th className="num">Δ</th>
          </tr>
        </thead>
        <tbody>
          {row('centroid x [µm]', a.x_um, r?.x_um, 3)}
          {row('centroid y [µm]', a.y_um, r?.y_um, 3)}
          {row('w_x (2σ) [µm]', a.wx_um, r?.wx_um, 3)}
          {row('w_y (2σ) [µm]', a.wy_um, r?.wy_um, 3)}
          {row('z_best x [µm]', a.z_best_x_um, r?.z_best_x_um, 2)}
          {row('z_best y [µm]', a.z_best_y_um, r?.z_best_y_um, 2)}
          {row('astigmatism Z_x − Z_y [µm]', a.astig_um, r?.astig_um, 2)}
          {row('peak (rel. to aligned)', a.peak_rel, r?.peak_rel, 3)}
          {row('power (rel. to aligned)', a.power_rel, r?.power_rel, 3)}
          {row('Strehl = peak / aligned peak', a.strehl, r?.strehl, 3)}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: 6, gap: 6 }}>
        <span className="small dim">static pointing offset dropped as alignment:</span>
        {CHANNELS.map((ch) => (
          <Badge key={ch} kind={`ch-${ch}`}>
            {ch} ({fmtSigned(res.static_offset_um[ch]?.x ?? 0, 2)}, {fmtSigned(res.static_offset_um[ch]?.y ?? 0, 2)}) µm
          </Badge>
        ))}
      </div>
    </div>
  )
}
