/**
 * One tutorial column's movie: XY at the tracked focus and the XZ slice (astigmatic Gaussian
 * approximation of the closed-form spot metrics, `XyView` / `XzView`), at the shared cursor,
 * with the shadow-tweezer badge. Both columns share one brightness scale (`peak`) so the
 * comparison is honest. A cursor-local wrapper: `TweezerView` itself is bound to the workbench
 * store's single plan.
 */
import { useMemo } from 'react'
import type { PlanResponse, SimulateResponse } from '../../api/types'
import { fmt } from '../../physics/format'
import { brightest, dedupeFades, offCanvas, requestedAt, spotRow, spotsAt } from '../../physics/spots'
import type { ViewScale } from '../tweezer/view'
import { XyView } from '../tweezer/XyView'
import { XzView } from '../tweezer/XzView'
import { useTutorialCursor } from './cursor'

export function TutorialTweezer({ plan, sim, scale, peak }: { plan: PlanResponse; sim: SimulateResponse; scale: ViewScale; peak: number }) {
  const t = useTutorialCursor((s) => s.t_us)
  const frame = useMemo(() => spotsAt(sim, t), [sim, t])
  const spots = frame.spots
  const planeZ = frame.tracked_z
  const half = plan.constants.half_tau_us
  const reqNow = requestedAt(plan.trajectory, t - half)
  const hidden = offCanvas(spots, scale.bounds)
  const fades = dedupeFades(plan.fade_events).filter((e) => Math.abs(e.time_us - (t - half)) <= 0.6 * plan.constants.tau_us)
  const top = brightest(spots)
  const shadowOffsets = Array.from(new Set(hidden.map((s) => fmt(Math.hypot(s.x_um - reqNow.x, s.y_um - reqNow.y), 1))))
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row" style={{ gap: 6, flexWrap: 'wrap', minHeight: 22 }}>
        {t < half && <span className="badge">dark before τ/2 = {fmt(half, 2)} µs</span>}
        {hidden.length > 0 && (
          <span className="badge warn" title={hidden.map((s) => `x ${fmt(s.x_um, 1)}, y ${fmt(s.y_um, 1)} µm, P/P_ref ${fmt(s.power / sim.power_ref, 2)}`).join('\n')}>
            {hidden.length} shadow tweezer{hidden.length === 1 ? '' : 's'} off canvas at ±{shadowOffsets.join(' / ±')} µm
            {fades.length > 0 ? ` (${fades.map((e) => e.axis).join(', ')} hand-over)` : ''}
          </span>
        )}
        {hidden.length === 0 && fades.length > 0 && <span className="badge warn">hand-over in progress ({fades.map((e) => e.axis).join(', ')})</span>}
        {top && (
          <span className="badge">
            brightest P/P_ref {fmt(top.power / sim.power_ref, 2)} · z {fmt(top.z_um, 2)} µm · {spots.length} spot{spots.length === 1 ? '' : 's'}
          </span>
        )}
      </div>
      <div style={{ width: '100%', maxWidth: 360, margin: '0 auto' }}>
        <XyView spots={spots} planeZ={planeZ} scale={scale} peak={peak} requested={reqNow} longSide={240} label="XY @ tracked focus" />
      </div>
      <XzView spots={spots} rowY={spotRow(spots)} planeZ={planeZ} scale={scale} peak={peak} label="XZ slice · Gaussian approx." />
    </div>
  )
}
