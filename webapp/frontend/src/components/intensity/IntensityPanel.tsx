/**
 * The tweezer intensity panel (R2 addendum §F4), under the tweezer view of the basic workbench:
 *
 * - **Trap depth vs time**: per site-merged trap (spots within `0.5·w0` per frame, sites tracked
 *   across frames by position), the astigmatic-Gaussian peak `Σ 2P/(π w_x w_y)` normalised to
 *   the plateau (`physics/traps.ts`). One line per trap up to 25; beyond, the min / median /
 *   max envelopes over the lit sites are added.
 * - **Line cuts at the cursor** from `/api/frame`: fetched 250 ms after the cursor stops,
 *   immediately on pause, never during playback; cached by frame time and plane.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import type { AlignedData, Options } from 'uplot'
import { api, errorMessage, isAbort } from '../../api/client'
import type { FrameResponse, SimulateResponse } from '../../api/types'
import { fmt } from '../../physics/format'
import type { TrapDepthSeries } from '../../physics/traps'
import { selectRequest, useStudio } from '../../state/store'
import { Panel, Skeleton } from '../common/Panel'
import { UPlotChart } from '../metrics/UPlotChart'
import { axisX, axisY } from '../metrics/uplotTheme'
import { simTrapDepths } from './depthCache'
import { LineCuts } from './LineCuts'

export const TRAP_DEPTH_LABEL = 'Gaussian approximation of the closed-form metrics: exact for an unwindowed single-term trap'
const MAX_LINES = 25
const COLORS = ['#4cc9f0', '#8ac926', '#ffb703', '#f3a481', '#b388ff', '#4361ee', '#d6604d', '#92c5de', '#fddbc7', '#2166ac', '#06d6a0', '#ef476f']
const DEBOUNCE_MS = 250

interface DepthPlot {
  opts: Omit<Options, 'width' | 'height'>
  data: AlignedData
  nTracks: number
  shown: number
}

function buildDepthPlot(sim: SimulateResponse, d: TrapDepthSeries): DepthPlot {
  const n = sim.times_us.length
  // strongest tracks first (by their median over the frames where they are lit)
  const rank = d.depth.map((col, k) => {
    const lit = Array.from(col).filter((v) => v > 0)
    lit.sort((a, b) => a - b)
    return { k, m: lit.length ? lit[lit.length >> 1] : 0 }
  })
  rank.sort((a, b) => b.m - a.m)
  const shown = Math.min(rank.length, MAX_LINES)
  const cols: (number | null)[][] = rank.slice(0, shown).map(({ k }) => Array.from(d.depth[k]))
  const series: Options['series'] = [{ label: 't' }]
  rank.slice(0, shown).forEach(({ k, m }, q) => {
    const w = d.where[k]
    series.push({
      label: `(${fmt(w.x_um, 1)}, ${fmt(w.y_um, 1)}) µm`,
      stroke: COLORS[q % COLORS.length],
      width: m >= 0.5 ? 1.2 : 0.8,
      dash: m >= 0.5 ? undefined : [3, 3],
    })
  })
  if (rank.length > MAX_LINES) {
    const lo: (number | null)[] = []
    const med: (number | null)[] = []
    const hi: (number | null)[] = []
    for (let i = 0; i < n; i++) {
      const v = d.depth.map((c) => c[i]).filter((x) => x >= 0.05)
      v.sort((a, b) => a - b)
      lo.push(v.length ? v[0] : null)
      med.push(v.length ? v[v.length >> 1] : null)
      hi.push(v.length ? v[v.length - 1] : null)
    }
    cols.push(lo, med, hi)
    series.push(
      { label: 'min over lit sites', stroke: '#f2f5f8', width: 1.4, dash: [2, 2] },
      { label: 'median over lit sites', stroke: '#f2f5f8', width: 1.6 },
      { label: 'max over lit sites', stroke: '#f2f5f8', width: 1.4, dash: [2, 2] },
    )
  }
  const opts: Omit<Options, 'width' | 'height'> = {
    cursor: { drag: { x: false, y: false } },
    legend: { show: shown <= 6 },
    scales: { x: { time: false }, y: { range: (_u: unknown, _min: number, max: number) => [0, Math.max(1.15, max * 1.05)] as [number, number] } },
    axes: [axisX(), axisY('depth / plateau')],
    series,
  }
  return { opts, data: [sim.times_us, ...cols] as AlignedData, nTracks: rank.length, shown }
}

/** Closed-form frame for the cuts at the cursor (module doc). Reuses the exact-frame view's response when it matches. */
function useCutFrame(sim: SimulateResponse | null): { frame: FrameResponse | null; loading: boolean; error: string | null } {
  const t = useStudio((s) => s.t_us)
  const playing = useStudio((s) => s.playing)
  const viewMode = useStudio((s) => s.viewMode)
  const exactFrame = useStudio((s) => s.exactFrame)
  const [frame, setFrame] = useState<FrameResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cache = useRef(new Map<string, FrameResponse>())
  const abort = useRef<AbortController | null>(null)
  const wasPlaying = useRef(playing)

  useEffect(() => {
    cache.current.clear()
    setFrame(null)
    setError(null)
  }, [sim])

  useEffect(() => {
    const justPaused = wasPlaying.current && !playing
    wasPlaying.current = playing
    if (!sim || playing || sim.times_us.length === 0) return
    const tc = Math.min(Math.max(t, sim.times_us[0]), sim.times_us[sim.times_us.length - 1])
    const key = `${viewMode}|${tc.toFixed(4)}`
    const hit = cache.current.get(key)
    if (hit) {
      setFrame(hit)
      setLoading(false)
      return
    }
    if (exactFrame && exactFrame.plan_id === sim.plan_id && Math.abs(exactFrame.time_us - tc) < 1e-4 && exactFrame.cuts.i_y) {
      cache.current.set(key, exactFrame)
      setFrame(exactFrame)
      return
    }
    const run = async () => {
      abort.current?.abort()
      const ctrl = new AbortController()
      abort.current = ctrl
      setLoading(true)
      const b = sim.bounds_um
      const spanX = b.x1 - b.x0
      const spanY = b.y1 - b.y0
      const long = 256
      const nx = spanX >= spanY ? long : Math.max(24, Math.round((long * spanX) / spanY))
      const ny = spanX >= spanY ? Math.max(24, Math.round((long * spanY) / spanX)) : long
      try {
        const res = await api.frame(
          {
            plan: selectRequest(useStudio.getState()),
            time_us: tc,
            plane: viewMode === 'tracked' ? 'tracked' : 'fixed',
            grid: { x0_um: b.x0, x1_um: b.x1, nx, y0_um: b.y0, y1_um: b.y1, ny },
            xz: null,
            movie_mixing_order: null,
          },
          ctrl.signal,
        )
        if (ctrl.signal.aborted) return
        cache.current.set(key, res)
        setFrame(res)
        setError(null)
        setLoading(false)
      } catch (e) {
        if (isAbort(e)) return
        setError(errorMessage(e))
        setLoading(false)
      }
    }
    if (justPaused) {
      void run()
      return
    }
    const timer = setTimeout(() => void run(), DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [sim, t, playing, viewMode, exactFrame])

  useEffect(() => () => abort.current?.abort(), [])
  return { frame, loading, error }
}

export function IntensityPanel() {
  const sim = useStudio((s) => s.sim)
  const t = useStudio((s) => s.t_us)
  const setTime = useStudio((s) => s.setTime)
  const playing = useStudio((s) => s.playing)
  const depths = useMemo(() => (sim ? simTrapDepths(sim) : null), [sim])
  const plot = useMemo(() => (sim && depths ? buildDepthPlot(sim, depths) : null), [sim, depths])
  const cut = useCutFrame(sim)

  if (!sim || !depths || !plot) {
    return (
      <Panel title="Trap intensity">
        <Skeleton height={160} label="waiting for the simulation…" />
      </Panel>
    )
  }
  const stale = cut.frame && Math.abs(cut.frame.time_us - t) > 1e-4
  return (
    <Panel title="Trap intensity" sub="trap depth vs time per trap · closed-form line cuts through the brightest trap at the cursor">
      <div className="stack" style={{ gap: 6 }}>
        <UPlotChart opts={plot.opts} data={plot.data} cursorX={t} onPick={setTime} height={150} />
        <div className="caption">
          <strong>{TRAP_DEPTH_LABEL}.</strong> Per trap Σ 2P/(π w_x w_y) over the spots within 0.5 w0 of each other (coincident hand-over rungs add), sites
          tracked through the movie by position, normalised to the plateau (median over frames of the brightest trap,{' '}
          {(depths.plateau * 1e12).toExponential(3)} per m² on the intensity_frame scale). {plot.nTracks} trap track{plot.nTracks === 1 ? '' : 's'} above 5 %
          {plot.nTracks > plot.shown ? ` (strongest ${plot.shown} drawn, with min / median / max envelopes over all lit sites)` : ''}; dashed lines are
          sites whose median depth is below ½ (shadow tweezers, hard-switched extra columns).
        </div>
        <div className="row between">
          <strong className="small">Line cuts at the cursor</strong>
          <span className="caption">
            {playing ? 'paused cuts: refreshed when playback stops' : cut.loading ? 'fetching /api/frame…' : stale ? `shown at t = ${fmt(cut.frame!.time_us, 1)} µs` : ''}
          </span>
        </div>
        {cut.error && (
          <div className="banner">
            <span className="code">frame</span>
            <span>{cut.error}</span>
          </div>
        )}
        {cut.frame ? (
          <LineCuts frame={cut.frame} plateau={depths.plateau} w0={sim.constants.waist0_um} zR={sim.constants.rayleigh_um} />
        ) : (
          !cut.error && <Skeleton height={120} label={playing ? 'pause to fetch the closed-form cuts' : 'fetching the closed-form cuts…'} />
        )}
      </div>
    </Panel>
  )
}
