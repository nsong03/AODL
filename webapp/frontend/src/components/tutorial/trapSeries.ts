/** Data of the tutorial's trap-power chart, on one union time grid (see `TrapPowerChart`). */
import type { AlignedData } from 'uplot'
import type { PlanResponse, SimulateResponse, TrajectoryEfficiencyResponse } from '../../api/types'
import { inBandFraction, onGrid, trapDepthSeries, unionGrid, type BandExit } from './tutorialMath'

export const TRAP_COLORS = { ideal: '#f2f5f8', inBand: '#ffb703', eff: '#4cc9f0' }
export const Y_MAX = 1.15
const N_BAND = 900

export interface TrapSeries {
  data: AlignedData
  effMax: number | null
}

export function buildTrapSeries(plan: PlanResponse, sim: SimulateResponse, eff: TrajectoryEfficiencyResponse | null, exit: BandExit | null): TrapSeries {
  const half = plan.constants.half_tau_us
  const [t0, t1] = plan.t_span_us
  const depth = trapDepthSeries(plan, sim)
  // in-band fraction on a dense grid of cursor times (plus both sides of the exit for a crisp step)
  const tb: number[] = []
  for (let i = 0; i < N_BAND; i++) tb.push(half + ((t1 - half) * i) / (N_BAND - 1))
  if (exit) tb.push(exit.t_atoms_us - 1e-3, exit.t_atoms_us + 1e-3)
  tb.sort((a, b) => a - b)
  const fb = tb.map((t) => inBandFraction(plan.waveform_ir, t - half).product)
  let effMax: number | null = null
  let te: number[] = []
  let fe: (number | null)[] = []
  if (eff) {
    effMax = Math.max(...eff.product)
    te = eff.t_us
    fe = eff.product.map((v) => (effMax && effMax > 0 ? v / effMax : null))
  }
  const grid = unionGrid([depth.t, tb, te, [t0, t1]])
  const data: AlignedData = [grid, onGrid(grid, depth.t, depth.depth), onGrid(grid, tb, fb), onGrid(grid, te, fe)]
  return { data, effMax }
}

