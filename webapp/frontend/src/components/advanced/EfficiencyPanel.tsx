/**
 * Bragg / TPM diffraction efficiency (DESIGN §1.6): η vs f for the first AOD (with the
 * deflection angle on a second axis and the band shaded), the A–B pair map with the ridge and
 * the band rectangle, and the efficiency along the current trajectory.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AlignedData } from 'uplot'
import uPlot from 'uplot'
import { api, errorMessage, isAbort } from '../../api/client'
import { CHANNELS } from '../../api/types'
import { CHANNEL_COLORS } from '../../channels'
import { fmt } from '../../physics/format'
import { useElementWidth } from '../../hooks/useCanvas'
import { deflectionMrad } from '../../physics/units'
import { currentRequest, useStudio } from '../../state/store'
import { CheckField, NumberField } from '../common/NumberField'
import { Badge, ErrorBanner, Panel, Skeleton, Spinner } from '../common/Panel'
import { UPlotChart } from '../metrics/UPlotChart'
import { AXIS, axisX, axisY } from '../metrics/uplotTheme'
import { infernoCss } from './colormaps'

export function EfficiencyPanel() {
  const hw = useStudio((s) => s.hardware)
  const plan = useStudio((s) => s.plan)
  const adv = useStudio((s) => s.advanced)
  const setAdvanced = useStudio((s) => s.setAdvanced)
  const setModel = useStudio((s) => s.setEfficiencyModel)
  const t = useStudio((s) => s.t_us)
  const setTime = useStudio((s) => s.setTime)
  const model = adv.efficiencyModel
  const abortRef = useRef<AbortController | null>(null)
  const [autoRun, setAutoRun] = useState(true)

  const run = useCallback(async () => {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setAdvanced({ efficiencyStatus: 'loading', efficiencyError: null, trajEffStatus: 'loading' })
    try {
      const eff = await api.efficiency({ hardware: hw, model, f_range_MHz: null, n: 241 }, ctrl.signal)
      if (ctrl.signal.aborted) return
      setAdvanced({ efficiency: eff, efficiencyStatus: 'ok' })
    } catch (e) {
      if (isAbort(e)) return
      setAdvanced({ efficiencyStatus: 'error', efficiencyError: errorMessage(e), trajEffStatus: 'idle' })
      return
    }
    if (!plan) {
      setAdvanced({ trajEffStatus: 'idle', trajEff: null })
      return
    }
    try {
      const te = await api.trajectoryEfficiency({ plan: currentRequest(useStudio.getState()), model, n: 300 }, ctrl.signal)
      if (ctrl.signal.aborted) return
      setAdvanced({ trajEff: te, trajEffStatus: 'ok' })
    } catch (e) {
      if (isAbort(e)) return
      setAdvanced({ trajEffStatus: 'error', trajEff: null, efficiencyError: errorMessage(e) })
    }
  }, [hw, model, plan, setAdvanced])

  // debounced auto-run on model / hardware / plan changes
  useEffect(() => {
    if (!autoRun) return
    const id = setTimeout(() => void run(), 500)
    return () => clearTimeout(id)
  }, [run, autoRun])

  const eff = adv.efficiency
  const curveOpts = useMemo(() => {
    const band = eff?.band_MHz ?? hw.band_MHz
    return {
      cursor: { drag: { x: false, y: false } },
      legend: { show: true },
      scales: { x: { time: false }, y: { range: [0, 1] as [number, number] } },
      axes: [
        axisX('f [MHz]'),
        axisY('η'),
        {
          ...AXIS,
          side: 0,
          scale: 'x',
          label: 'deflection Δθ = λ(f − f_center)/v [mrad]',
          size: 32,
          values: (_u: uPlot, splits: number[]) => splits.map((f) => fmt(deflectionMrad(f - hw.f_center_MHz, hw.wavelength_nm, hw.sound_speed_m_s), 2)),
        },
      ],
      hooks: {
        drawClear: [
          (u: uPlot) => {
            const ctx = u.ctx
            const x0 = u.valToPos(band[0], 'x', true)
            const x1 = u.valToPos(band[1], 'x', true)
            ctx.save()
            ctx.fillStyle = 'rgba(76,201,240,0.08)'
            ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height)
            ctx.restore()
          },
        ],
      },
      series: [
        { label: 'f' },
        { label: 'η first AOD (A)', stroke: '#4cc9f0', width: 1.6 },
        { label: 'η second AOD (B) at f_A = f_tpm', stroke: '#ffb703', width: 1.2, dash: [4, 3] },
      ],
    }
  }, [eff?.band_MHz, hw])
  const curveData = useMemo<AlignedData | null>(() => (eff ? [eff.f_MHz, eff.eta_first, eff.eta_second_at_center] : null), [eff])

  const te = adv.trajEff
  const trajOpts = useMemo(
    () => ({
      cursor: { drag: { x: false, y: false } },
      legend: { show: true },
      scales: { x: { time: false }, y: { range: [0, 1.02] as [number, number] } },
      axes: [axisX(), axisY('η')],
      series: [{ label: 't' }, ...CHANNELS.map((ch) => ({ label: ch, stroke: CHANNEL_COLORS[ch], width: 1 })), { label: 'product (relative trap power)', stroke: '#f2f5f8', width: 1.6 }],
    }),
    [],
  )
  const trajData = useMemo<AlignedData | null>(() => (te ? [te.t_us, ...CHANNELS.map((ch) => te.per_channel[ch]), te.product] : null), [te])

  return (
    <Panel
      title="Diffraction efficiency"
      sub="Kogelnik coupled-wave η = Γ² sin²√(Γ²+δ²)/(Γ²+δ²), δ = (πλL/2nv²)(f − f_tpm)²; the second AOD sees the first's deflection as an additive mismatch"
      actions={
        <>
          <CheckField label="auto-run" checked={autoRun} onChange={setAutoRun} />
          <button className="primary small" onClick={() => void run()} disabled={adv.efficiencyStatus === 'loading'}>
            {adv.efficiencyStatus === 'loading' ? 'computing…' : 'compute'}
          </button>
        </>
      }
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: 16 }}>
        <div style={{ width: 300, flex: '0 0 auto' }}>
          <NumberField label="interaction length L" unit="mm" value={model.interaction_length_mm} min={0.1} step={0.5} onChange={(v) => setModel({ interaction_length_mm: v })} />
          <CheckField label={`f_tpm = f_center (${hw.f_center_MHz} MHz)`} checked={model.f_tpm_MHz === null} onChange={(v) => setModel({ f_tpm_MHz: v ? null : hw.f_center_MHz })} />
          {model.f_tpm_MHz !== null && <NumberField label="f_tpm" unit="MHz" value={model.f_tpm_MHz} min={1} onChange={(v) => setModel({ f_tpm_MHz: v })} title="tangential-phase-matching frequency: a crystal-cut parameter" />}
          <NumberField label="refractive index n" value={model.refractive_index} min={1} step={0.01} onChange={(v) => setModel({ refractive_index: v })} />
          <NumberField label="peak efficiency η_peak = sin²Γ" value={model.peak_efficiency} min={0.01} max={1} step={0.05} onChange={(v) => setModel({ peak_efficiency: v })} />
          <NumberField
            label="angular coupling γ"
            value={model.angular_coupling}
            min={-1}
            max={1}
            step={0.1}
            onChange={(v) => setModel({ angular_coupling: v })}
            title="signed fraction of A's deflection reaching B's Bragg plane: |γ| = 1 ideal 4f relay with plane acoustic waves; 0.2 reproduces the paper's 'a few times narrower' first-AOD band at L = 4 mm"
          />
          {eff && (
            <div className="row" style={{ marginTop: 8, gap: 6 }}>
              <Badge kind="accent">3 dB bandwidth {fmt(eff.bandwidth_3dB_MHz, 1)} MHz</Badge>
              <Badge kind="warn">first AOD usable {fmt(eff.first_aod_usable_MHz, 1)} MHz</Badge>
              <Badge>f_tpm {fmt(eff.f_tpm_MHz, 1)} MHz</Badge>
            </div>
          )}
          {adv.efficiencyStatus === 'error' && adv.efficiencyError && (
            <div style={{ marginTop: 8 }}>
              <ErrorBanner code="advanced" message={adv.efficiencyError} />
            </div>
          )}
          {eff && <p className="caption" style={{ marginTop: 8 }}>{eff.assumptions}</p>}
        </div>
        <div className="grow stack" style={{ gap: 10 }}>
          {!eff ? (
            <Skeleton height={200} label={adv.efficiencyStatus === 'loading' ? 'computing the efficiency model…' : 'press compute'} />
          ) : (
            <>
              <div>
                <div className="small dim" style={{ marginBottom: 2 }}>
                  η(f) of a single AOD — band shaded; the first AOD's usable width is set by the pair map below, not by this curve
                </div>
                <UPlotChart opts={curveOpts} data={curveData!} height={190} />
              </div>
              <PairMap eff={eff} />
            </>
          )}
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <div className="row between small" style={{ marginBottom: 2 }}>
          <span className="dim">
            along this trajectory — per channel the amplitude²-weighted mean of η over the live tones at the beam-centre retarded time; B channels at the A
            channel's live frequency; the product is the predicted relative trap power
          </span>
          {te && <Badge kind={te.min_product < 0.5 ? 'danger' : te.min_product < 0.8 ? 'warn' : 'ok'}>min product {fmt(te.min_product, 3)}</Badge>}
        </div>
        {adv.trajEffStatus === 'loading' && <Spinner label="trajectory efficiency…" />}
        {te && trajData ? <UPlotChart opts={trajOpts} data={trajData} cursorX={t} onPick={setTime} height={160} /> : !plan ? <div className="caption">Plan a trajectory in the basic workbench first.</div> : null}
        {te && <p className="caption">{te.assumptions}</p>}
      </div>
    </Panel>
  )
}

function PairMap({ eff }: { eff: NonNullable<ReturnType<typeof useStudio.getState>['advanced']['efficiency']> }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const cssWidth = useElementWidth(ref)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = cssWidth || canvas.clientWidth || 400
    const height = 260
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(height * dpr)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, width, height)
    const pad = { left: 46, right: 70, top: 10, bottom: 30 }
    const fA = eff.pair.f_A_MHz
    const fB = eff.pair.f_B_MHz
    const m = fA.length
    const nB = fB.length
    const a0 = fA[0]
    const a1 = fA[m - 1]
    const b0 = fB[0]
    const b1 = fB[nB - 1]
    const X = (fb: number) => pad.left + ((fb - b0) / (b1 - b0)) * (width - pad.left - pad.right)
    const Y = (fa: number) => height - pad.bottom - ((fa - a0) / (a1 - a0)) * (height - pad.top - pad.bottom)
    let emax = 0
    for (const row of eff.pair.eta) for (const v of row) emax = Math.max(emax, v)
    const cw = (width - pad.left - pad.right) / (nB - 1)
    const ch = (height - pad.top - pad.bottom) / (m - 1)
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < nB; j++) {
        ctx.fillStyle = infernoCss(emax > 0 ? eff.pair.eta[i][j] / emax : 0)
        ctx.fillRect(X(fB[j]) - cw / 2, Y(fA[i]) - ch / 2, cw + 0.7, ch + 0.7)
      }
    }
    // band rectangle
    const [lo, hi] = eff.band_MHz
    ctx.strokeStyle = 'rgba(76,201,240,0.9)'
    ctx.setLineDash([4, 3])
    ctx.strokeRect(X(lo), Y(hi), X(hi) - X(lo), Y(lo) - Y(hi))
    ctx.setLineDash([])
    // ridge: f_B at max η for each f_A
    ctx.strokeStyle = '#f2f5f8'
    ctx.lineWidth = 1.2
    ctx.beginPath()
    eff.ridge.f_A_MHz.forEach((fa, i) => {
      const x = X(eff.ridge.f_B_at_max_MHz[i])
      const y = Y(fa)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
    // usable width in f_A (centred on f_tpm)
    const half = 0.5 * eff.first_aod_usable_MHz
    ctx.strokeStyle = '#ffb703'
    ctx.beginPath()
    ctx.moveTo(width - pad.right + 6, Y(eff.f_tpm_MHz - half))
    ctx.lineTo(width - pad.right + 6, Y(eff.f_tpm_MHz + half))
    ctx.stroke()
    ctx.fillStyle = '#ffb703'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(`A usable`, width - pad.right + 10, Y(eff.f_tpm_MHz) - 2)
    ctx.fillText(`${fmt(eff.first_aod_usable_MHz, 1)} MHz`, width - pad.right + 10, Y(eff.f_tpm_MHz) + 10)
    // axes
    ctx.fillStyle = '#9aa5b3'
    ctx.textAlign = 'center'
    for (const v of [b0, (b0 + b1) / 2, b1]) ctx.fillText(fmt(v, 0), X(v), height - 14)
    ctx.fillText('f_B [MHz] (second AOD)', (pad.left + width - pad.right) / 2, height - 3)
    ctx.textAlign = 'right'
    for (const v of [a0, (a0 + a1) / 2, a1]) ctx.fillText(fmt(v, 0), pad.left - 4, Y(v) + 3)
    ctx.save()
    ctx.translate(10, (pad.top + height - pad.bottom) / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('f_A [MHz] (first AOD)', 0, 0)
    ctx.restore()
    ctx.fillStyle = '#f2f5f8'
    ctx.textAlign = 'left'
    ctx.fillText(`η_pair(f_A, f_B) · max ${fmt(emax, 3)} · white: ridge (f_B re-matching A's deflection) · dashed: band`, pad.left + 4, pad.top + 11)
  }, [eff, cssWidth])
  return (
    <div>
      <div className="small dim" style={{ marginBottom: 2 }}>
        pair map — η_A(f_A)·η_B(f_B; f_A) with δ_B = C₂[(f_B − f_tpm)² − 2γ f_tpm (f_A − f_tpm)]: narrow in f_A because A's deflection detunes B
      </div>
      <canvas ref={ref} className="plot" style={{ height: 260 }} />
    </div>
  )
}
