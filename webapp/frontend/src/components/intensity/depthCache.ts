/** `trapDepths` of a simulation, computed once per response object (shared by the intensity panel and the exact view). */
import type { SimulateResponse } from '../../api/types'
import { trapDepths, type TrapDepthSeries } from '../../physics/traps'

const depthCache = new WeakMap<SimulateResponse, TrapDepthSeries>()

export function simTrapDepths(sim: SimulateResponse): TrapDepthSeries {
  let d = depthCache.get(sim)
  if (!d) {
    d = trapDepths(sim.frames, sim.constants.waist0_um)
    depthCache.set(sim, d)
  }
  return d
}
