/** Per-column derived quantities of the tutorial, all computed from the API responses. */
import { CHANNELS, type ChannelName, type FadeEvent, type PlanResponse, type SimulateResponse, type TrajectoryEfficiencyResponse } from '../../api/types'
import { requestedAt } from '../../physics/spots'
import { buildTrapSeries, type TrapSeries } from './trapSeries'
import { firstBandExit, handOvers, ladderSpacing, trapDepthSeries, type BandExit } from './tutorialMath'

export interface Derived {
  exit: BandExit | null
  fades: FadeEvent[]
  hand: ReturnType<typeof handOvers>
  spacing: { x: number | null; y: number | null }
  minMargin: { ch: ChannelName; margin: number; f_min: number; f_max: number }
  /** min ideal trap depth over lit frames (t ≥ τ/2) */
  idealMin: number | null
  /** brightest shadow (spot farther than w0 from the site) relative to power_ref, and its offset */
  shadowMax: { rel: number; offset_um: number } | null
  trap: TrapSeries | null
  /** η-weighted (normalised) value after the band exit: max over samples later than the exit + τ */
  effAfterExit: number | null
  inBandAfterExit: number | null
}

export function derive(plan: PlanResponse, sim: SimulateResponse | null, eff: TrajectoryEfficiencyResponse | null): Derived {
  const exit = firstBandExit(plan.waveform_ir)
  const hand = handOvers(plan.fade_events)
  let minMargin: Derived['minMargin'] | null = null
  for (const ch of CHANNELS) {
    const b = plan.band_usage[ch]
    if (!minMargin || b.margin_MHz < minMargin.margin) minMargin = { ch, margin: b.margin_MHz, f_min: b.f_min_MHz, f_max: b.f_max_MHz }
  }
  let idealMin: number | null = null
  let shadowMax: Derived['shadowMax'] = null
  if (sim) {
    const d = trapDepthSeries(plan, sim)
    for (const v of d.depth) if (v !== null && (idealMin === null || v < idealMin)) idealMin = v
    const w0 = sim.constants.waist0_um
    sim.frames.forEach((f, i) => {
      const site = requestedAt(plan.trajectory, sim.times_us[i] - plan.constants.half_tau_us)
      for (let k = 0; k < f.x_um.length; k++) {
        const r = Math.hypot(f.x_um[k] - site.x, f.y_um[k] - site.y)
        if (r <= w0) continue
        const rel = f.power[k] / sim.power_ref
        if (!shadowMax || rel > shadowMax.rel) shadowMax = { rel, offset_um: r }
      }
    })
  }
  const trap = sim ? buildTrapSeries(plan, sim, eff, exit) : null
  let effAfterExit: number | null = null
  let inBandAfterExit: number | null = null
  if (trap && exit) {
    // the plateau after the exit, up to the start of the final lowering seam (+ τ/2)
    const seams = plan.seams_us
    const tEnd = (seams.length >= 2 ? seams[seams.length - 2] : plan.duration_us) + plan.constants.half_tau_us
    const [grid, , inb, ef] = trap.data as [number[], unknown, (number | null)[], (number | null)[]]
    for (let i = 0; i < grid.length; i++) {
      if (grid[i] < exit.t_atoms_us + plan.constants.tau_us || grid[i] > tEnd) continue
      const a = ef[i]
      if (a !== null && a !== undefined) effAfterExit = effAfterExit === null ? a : Math.max(effAfterExit, a)
      const b = inb[i]
      if (b !== null && b !== undefined) inBandAfterExit = inBandAfterExit === null ? b : Math.max(inBandAfterExit, b)
    }
  }
  return { exit, fades: hand.events, hand, spacing: ladderSpacing(plan.waveform_ir), minMargin: minMargin!, idealMin, shadowMax, trap, effAfterExit, inBandAfterExit }
}
