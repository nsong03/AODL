/**
 * RF spectrum of the rendered drive around the cursor (API.md §6.6): Hann periodogram
 * normalised to 0 dB at the strongest line, band shaded; crest factor, PAPR and the peak
 * phase-modulation depth as badges.
 */
import { useMemo, useRef, useState } from 'react'
import type { AlignedData } from 'uplot'
import uPlot from 'uplot'
import { api, errorMessage, isAbort } from '../../api/client'
import { CHANNELS, type ChannelName } from '../../api/types'
import { CHANNEL_COLORS } from '../../channels'
import { fmt } from '../../physics/format'
import { currentRequest, useStudio } from '../../state/store'
import { NumberField, SelectField } from '../common/NumberField'
import { Badge, ErrorBanner, Panel } from '../common/Panel'
import { UPlotChart } from '../metrics/UPlotChart'
import { axisX, axisY } from '../metrics/uplotTheme'

export function SpectrumPanel() {
  const plan = useStudio((s) => s.plan)
  const hwBand = useStudio((s) => s.hardware.band_MHz)
  const t = useStudio((s) => s.t_us)
  const adv = useStudio((s) => s.advanced)
  const setAdvanced = useStudio((s) => s.setAdvanced)
  const [channel, setChannel] = useState<ChannelName>('Bx')
  const [window_us, setWindow] = useState(20)
  const [rate, setRate] = useState(625)
  const abortRef = useRef<AbortController | null>(null)

  const run = async () => {
    if (!plan) return
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setAdvanced({ spectrumStatus: 'loading', spectrumError: null })
    try {
      const sp = await api.spectrum({ plan: currentRequest(useStudio.getState()), channel, t_center_us: t, window_us, sample_rate_MSps: rate }, ctrl.signal)
      if (ctrl.signal.aborted) return
      setAdvanced({ spectrum: sp, spectrumStatus: 'ok' })
    } catch (e) {
      if (isAbort(e)) return
      setAdvanced({ spectrumStatus: 'error', spectrumError: errorMessage(e) })
    }
  }

  const sp = adv.spectrum
  const opts = useMemo(() => {
    const band = sp?.band_MHz ?? hwBand
    return {
      cursor: { drag: { x: false, y: false } },
      legend: { show: false },
      scales: { x: { time: false } },
      axes: [axisX('f [MHz]'), axisY('PSD [dB]')],
      hooks: {
        drawClear: [
          (u: uPlot) => {
            if (!band) return
            const x0 = u.valToPos(band[0], 'x', true)
            const x1 = u.valToPos(band[1], 'x', true)
            u.ctx.save()
            u.ctx.fillStyle = 'rgba(76,201,240,0.08)'
            u.ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height)
            u.ctx.restore()
          },
        ],
      },
      series: [{ label: 'f' }, { label: 'PSD', stroke: CHANNEL_COLORS[channel], width: 1 }],
    }
  }, [sp?.band_MHz, channel, hwBand])
  const data = useMemo<AlignedData | null>(() => (sp ? [sp.f_MHz, sp.psd_dB] : null), [sp])

  return (
    <Panel
      title="RF spectrum"
      sub="Hann periodogram of render_samples over a window around the cursor, 0 dB at the strongest line"
      actions={
        <button className="primary small" onClick={() => void run()} disabled={!plan || adv.spectrumStatus === 'loading'}>
          {adv.spectrumStatus === 'loading' ? 'computing…' : `compute at t = ${fmt(t, 1)} µs`}
        </button>
      }
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: 16 }}>
        <div style={{ width: 300, flex: '0 0 auto' }}>
          <SelectField label="channel" value={channel} options={CHANNELS} onChange={setChannel} />
          <NumberField label="window" unit="µs" value={window_us} min={1} step={5} onChange={setWindow} />
          <NumberField label="sample rate" unit="MS/s" value={rate} min={10} step={25} onChange={setRate} />
          {sp && (
            <div className="row" style={{ marginTop: 8, gap: 6 }}>
              <Badge>crest factor {fmt(sp.crest_factor, 2)}</Badge>
              <Badge>PAPR {fmt(sp.papr_dB, 2)} dB</Badge>
              <Badge kind={sp.peak_depth_rad > 1 ? 'warn' : ''}>peak depth {fmt(sp.peak_depth_rad, 2)} rad</Badge>
            </div>
          )}
          {adv.spectrumStatus === 'error' && adv.spectrumError && (
            <div style={{ marginTop: 8 }}>
              <ErrorBanner code="advanced/spectrum" message={adv.spectrumError} />
            </div>
          )}
          {sp && sp.notes.length > 0 && (
            <ul className="notes">
              {sp.notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          )}
        </div>
        <div className="grow">
          {sp && data ? <UPlotChart opts={opts} data={data} height={220} /> : <div className="caption">Move the cursor in the basic workbench to the time of interest, then compute. Crest factor = peak / rms of the real waveform; peak depth = C × max|V| / A_single_tone (how far into the thin-grating nonlinearity the drive goes).</div>}
        </div>
      </div>
      <p className="caption" style={{ marginTop: 8 }}>
        Model assumptions: the rendered AWG samples (carrier included) over the window only; no amplifier or transducer response.
      </p>
    </Panel>
  )
}
