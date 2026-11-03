/**
 * Trap power vs cursor time `t` for one tutorial column, three curves on shared axes:
 *  1. ideal trap depth from `/api/simulate` (closed-form engine, flat diffraction efficiency),
 *  2. hard-band-mask in-band fraction of the live tone amplitude² at `t − τ/2` (from the IR),
 *  3. efficiency-weighted trap power from `/api/advanced/trajectory_efficiency` (Bragg/TPM
 *     model), normalised to its own maximum over the drive.
 * Hand-overs (amber) and the first band exit (red) are drawn at the time the atoms see them.
 */
import { useMemo } from 'react'
import type uPlot from 'uplot'
import type { Options } from 'uplot'
import type { FadeEvent, PlanResponse } from '../../api/types'
import { UPlotChart } from '../metrics/UPlotChart'
import { axisX, axisY } from '../metrics/uplotTheme'
import { useTutorialCursor } from './cursor'
import { TRAP_COLORS, Y_MAX, type TrapSeries } from './trapSeries'
import type { BandExit } from './tutorialMath'

function markers(fades: FadeEvent[], half: number, exit: BandExit | null) {
  return (u: uPlot) => {
    const ctx = u.ctx
    const { top, height } = u.bbox
    ctx.save()
    ctx.lineWidth = Math.max(1, devicePixelRatio)
    ctx.strokeStyle = 'rgba(255,183,3,0.55)'
    for (const e of fades) {
      const x = u.valToPos(e.time_us + half, 'x', true)
      ctx.beginPath()
      ctx.moveTo(x, top)
      ctx.lineTo(x, top + height)
      ctx.stroke()
    }
    if (exit) {
      ctx.strokeStyle = '#ef6f6c'
      ctx.setLineDash([5, 4])
      const x = u.valToPos(exit.t_atoms_us, 'x', true)
      ctx.beginPath()
      ctx.moveTo(x, top)
      ctx.lineTo(x, top + height)
      ctx.stroke()
    }
    ctx.restore()
  }
}

export function TrapPowerChart({ plan, series, fades, exit }: { plan: PlanResponse; series: TrapSeries; fades: FadeEvent[]; exit: BandExit | null }) {
  const t = useTutorialCursor((s) => s.t_us)
  const setTime = useTutorialCursor((s) => s.setTime)
  const pause = useTutorialCursor((s) => s.pause)
  const half = plan.constants.half_tau_us
  const [t0, t1] = plan.t_span_us
  const opts = useMemo<Omit<Options, 'width' | 'height'>>(
    () => ({
      cursor: { drag: { x: false, y: false } },
      legend: { show: true },
      scales: { x: { time: false, range: [t0, t1] }, y: { range: [0, Y_MAX] } },
      axes: [axisX('t [µs] (cursor; atoms see the drive at t − τ/2)'), axisY('relative trap power')],
      series: [
        { label: 't' },
        { label: 'ideal trap depth (closed form, flat η)', stroke: TRAP_COLORS.ideal, width: 1.8, spanGaps: true },
        { label: 'in-band fraction (hard mask, IR)', stroke: TRAP_COLORS.inBand, width: 1.4, dash: [6, 3], spanGaps: true },
        { label: 'η-weighted power (Bragg/TPM model)', stroke: TRAP_COLORS.eff, width: 1.6, spanGaps: true },
      ],
      hooks: { draw: [markers(fades, half, exit)] },
    }),
    [t0, t1, fades, half, exit],
  )
  return (
    <UPlotChart
      opts={opts}
      data={series.data}
      height={210}
      cursorX={t}
      onPick={(x) => {
        pause()
        setTime(Math.min(t1, Math.max(t0, x)))
      }}
    />
  )
}
