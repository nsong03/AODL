/**
 * Requested vs simulated `x, y, z` (the simulated spot lags by τ/2; a dashed "expected at the
 * atoms" copy of the request shifted by +τ/2 is drawn unless `retard_compensate`), power per
 * spot (normalised to `power_ref`, spots matched frame to frame by nearest position), the
 * astigmatism `σ_astig` and the waists `w_x`, `w_y`; all share the time cursor.
 */
import { useMemo } from 'react'
import type { AlignedData } from 'uplot'
import type { PlanResponse, SimulateResponse } from '../../api/types'
import { fmt } from '../../physics/format'
import { trackSpots } from '../../physics/spots'
import { useStudio } from '../../state/store'
import { Panel, Skeleton } from '../common/Panel'
import { UPlotChart } from './UPlotChart'
import { AXIS, axisX, axisY } from './uplotTheme'

const SERIES_COLORS = ['#4cc9f0', '#8ac926', '#ffb703', '#f3a481', '#b388ff', '#4361ee', '#d6604d', '#92c5de', '#fddbc7', '#2166ac']

function interp(ts: number[], ys: number[], t: number): number {
  const n = ts.length
  if (n === 0) return NaN
  if (t <= ts[0]) return ys[0]
  if (t >= ts[n - 1]) return ys[n - 1]
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (ts[mid] <= t) lo = mid
    else hi = mid
  }
  const w = (t - ts[lo]) / (ts[hi] - ts[lo])
  return ys[lo] * (1 - w) + ys[hi] * w
}

type Col = (number | null)[] // uPlot needs null (not NaN) for gaps: NaN poisons its range computation

interface Series {
  t: number[]
  simX: Col
  simY: Col
  simZ: Col
  tracked: number[]
  reqX: number[]
  reqY: number[]
  reqZ: number[]
  expX: number[]
  expY: number[]
  expZ: number[]
  power: Col[] // per track
  powerSum: number[]
  sigma: Col
  wx: Col
  wy: Col
  nTracks: number
}

function buildSeries(plan: PlanResponse, sim: SimulateResponse): Series {
  const t = sim.times_us
  const b = sim.bounds_um
  const half = plan.constants.half_tau_us
  const tr = plan.trajectory
  const simX: Col = []
  const simY: Col = []
  const simZ: Col = []
  const sigma: Col = []
  const wx: Col = []
  const wy: Col = []
  for (const f of sim.frames) {
    let w = 0
    let sx = 0
    let sy = 0
    let sz = 0
    let ss = 0
    let swx = 0
    let swy = 0
    for (let k = 0; k < f.x_um.length; k++) {
      // shadow tweezers (off canvas) do not count toward the array centre
      if (f.x_um[k] < b.x0 || f.x_um[k] > b.x1 || f.y_um[k] < b.y0 || f.y_um[k] > b.y1) continue
      const p = f.power[k]
      w += p
      sx += p * f.x_um[k]
      sy += p * f.y_um[k]
      sz += p * f.z_um[k]
      ss += p * f.sigma_astig[k]
      swx += p * f.wx_um[k]
      swy += p * f.wy_um[k]
    }
    const ok = w > 0
    simX.push(ok ? sx / w : null)
    simY.push(ok ? sy / w : null)
    simZ.push(ok ? sz / w : null)
    sigma.push(ok ? ss / w : null)
    wx.push(ok ? swx / w : null)
    wy.push(ok ? swy / w : null)
  }
  const reqX = t.map((ti) => interp(tr.t_us, tr.x_um, ti))
  const reqY = t.map((ti) => interp(tr.t_us, tr.y_um, ti))
  const reqZ = t.map((ti) => interp(tr.t_us, tr.z_um, ti))
  const expX = t.map((ti) => interp(tr.t_us, tr.x_um, ti - half))
  const expY = t.map((ti) => interp(tr.t_us, tr.y_um, ti - half))
  const expZ = t.map((ti) => interp(tr.t_us, tr.z_um, ti - half))
  const tracks = trackSpots(sim.frames, 1.5 * Math.max(sim.constants.waist0_um, 0.5))
  const ref = sim.power_ref > 0 ? sim.power_ref : 1
  const power: Col[] = tracks.map((tk) => tk.map((idx, i) => (idx >= 0 ? sim.frames[i].power[idx] / ref : null)))
  const powerSum = sim.frames.map((f) => f.power.reduce((a, p) => a + p, 0) / ref)
  return { t, simX, simY, simZ, tracked: sim.tracked_z_um, reqX, reqY, reqZ, expX, expY, expZ, power, powerSum, sigma, wx, wy, nTracks: tracks.length }
}

export function MetricsPlots() {
  const plan = useStudio((s) => s.plan)
  const sim = useStudio((s) => s.sim)
  const t = useStudio((s) => s.t_us)
  const setTime = useStudio((s) => s.setTime)
  const compensate = useStudio((s) => s.options.retard_compensate)
  const series = useMemo(() => (plan && sim ? buildSeries(plan, sim) : null), [plan, sim])

  const posOpts = useMemo(() => {
    const mk = (label: string, color: string) => ({
      cursor: { drag: { x: false, y: false } },
      legend: { show: true },
      scales: { x: { time: false } },
      axes: [axisX(), axisY(label)],
      series: [
        { label: 't' },
        { label: `${label} requested`, stroke: color, width: 1, dash: [2, 3] },
        { label: `${label} expected at the atoms (+τ/2)`, stroke: color, width: 1, dash: [6, 4], show: !compensate },
        { label: `${label} simulated`, stroke: '#f2f5f8', width: 1.6 },
      ],
    })
    return { x: mk('x [µm]', '#4cc9f0'), y: mk('y [µm]', '#8ac926'), z: mk('z [µm]', '#f3a481') }
  }, [compensate])

  const powerOpts = useMemo(() => {
    const n = series?.nTracks ?? 0
    const shown = Math.min(n, 12)
    return {
      cursor: { drag: { x: false, y: false } },
      legend: { show: shown <= 4 },
      scales: { x: { time: false } },
      axes: [axisX(), axisY('P / P_ref')],
      series: [
        { label: 't' },
        ...Array.from({ length: shown }, (_, k) => ({ label: `spot ${k + 1}`, stroke: SERIES_COLORS[k % SERIES_COLORS.length], width: 1 })),
        { label: 'Σ all spots', stroke: '#f2f5f8', width: 1.2, dash: [3, 3] },
      ],
    }
  }, [series?.nTracks])

  const astigOpts = useMemo(
    () => ({
      cursor: { drag: { x: false, y: false } },
      legend: { show: true },
      scales: {
        x: { time: false },
        // never auto-range onto 6-significant-digit rounding noise of an astigmatism-free drive
        y: { range: (_u: unknown, min: number, max: number) => [Math.min(min, -0.05), Math.max(max, 0.05)] as [number, number] },
        w: { range: (_u: unknown, _min: number, max: number) => [0, Math.max(max * 1.1, 0.1)] as [number, number] },
      },
      axes: [axisX(), axisY('σ_astig'), { ...AXIS, side: 1, scale: 'w', label: 'w [µm]', size: 44 }],
      series: [
        { label: 't' },
        { label: 'σ_astig = ΔF / z_R', stroke: '#b388ff', width: 1.4 },
        { label: 'w_x', stroke: '#4cc9f0', width: 1, scale: 'w' },
        { label: 'w_y', stroke: '#8ac926', width: 1, scale: 'w' },
      ],
    }),
    [],
  )

  const posData = useMemo(() => {
    if (!series) return null
    return {
      x: [series.t, series.reqX, series.expX, series.simX] as AlignedData,
      y: [series.t, series.reqY, series.expY, series.simY] as AlignedData,
      z: [series.t, series.reqZ, series.expZ, series.simZ] as AlignedData,
    }
  }, [series])
  const powerData = useMemo(() => {
    if (!series) return null
    const shown = Math.min(series.nTracks, 12)
    return [series.t, ...series.power.slice(0, shown), series.powerSum] as AlignedData
  }, [series])
  const astigData = useMemo(() => (series ? ([series.t, series.sigma, series.wx, series.wy] as AlignedData) : null), [series])

  if (!plan || !sim || !series || !posData || !powerData || !astigData) {
    return (
      <Panel title="Metrics">
        <Skeleton height={200} label="waiting for the simulation…" />
      </Panel>
    )
  }
  const lag = plan.constants.half_tau_us
  return (
    <Panel title="Metrics" sub={`requested vs simulated · the atoms lag the request by τ/2 = ${fmt(lag, 2)} µs${compensate ? ' (retard-compensated: no +τ/2 overlay)' : ''} · click a plot to move the cursor`}>
      <div className="stack" style={{ gap: 6 }}>
        <UPlotChart opts={posOpts.x} data={posData.x} cursorX={t} onPick={setTime} height={120} />
        <UPlotChart opts={posOpts.y} data={posData.y} cursorX={t} onPick={setTime} height={120} />
        <UPlotChart opts={posOpts.z} data={posData.z} cursorX={t} onPick={setTime} height={120} />
        <div className="caption">
          Simulated position = power-weighted centre of the on-canvas spots (the array centre); z = tracked best focus. Off-canvas shadow tweezers are
          excluded.
        </div>
        <UPlotChart opts={powerOpts} data={powerData} cursorX={t} onPick={setTime} height={130} />
        <div className="caption">
          Power per spot / P_ref (median over the movie), {series.nTracks} spot track{series.nTracks === 1 ? '' : 's'}
          {series.nTracks > 12 ? ' (first 12 drawn)' : ''}; spots matched frame to frame by nearest position. Pair-driven light is dark before τ/2 and
          rises until τ.
        </div>
        <UPlotChart opts={astigOpts} data={astigData} cursorX={t} onPick={setTime} height={130} />
        <div className="caption">σ_astig = ΔF / z_R (Table I; 0 for astigmatism-free control) and the 1/e² radii at best focus, power-weighted over the spots.</div>
      </div>
    </Panel>
  )
}
