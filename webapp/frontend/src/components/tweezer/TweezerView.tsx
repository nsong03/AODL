import { useEffect, useMemo, useRef } from 'react'
import { fmt } from '../../physics/format'
import { brightest, offCanvas, requestedAt, spotRow, spotsAt } from '../../physics/spots'
import { useStudio } from '../../state/store'
import { Panel, Skeleton } from '../common/Panel'
import { ColorBar } from './ColorBar'
import { ExactXy, ExactXz } from './ExactFrame'
import { IsoView } from './IsoView'
import { Playback } from './Playback'
import { viewScale } from './view'
import { XyView } from './XyView'
import { XzView } from './XzView'

export function TweezerView() {
  const plan = useStudio((s) => s.plan)
  const sim = useStudio((s) => s.sim)
  const simStatus = useStudio((s) => s.simStatus)
  const simError = useStudio((s) => s.simError)
  const t = useStudio((s) => s.t_us)
  const playing = useStudio((s) => s.playing)
  const viewMode = useStudio((s) => s.viewMode)
  const setViewMode = useStudio((s) => s.setViewMode)
  const showExact = useStudio((s) => s.showExact)
  const setShowExact = useStudio((s) => s.setShowExact)
  const exactFrame = useStudio((s) => s.exactFrame)
  const exactStatus = useStudio((s) => s.exactStatus)
  const exactError = useStudio((s) => s.exactError)
  const fetchExact = useStudio((s) => s.fetchExact)
  const planStatus = useStudio((s) => s.planStatus)
  const planError = useStudio((s) => s.planError)

  const scale = useMemo(() => (sim ? viewScale(sim) : null), [sim])
  const frame = useMemo(() => (sim ? spotsAt(sim, t) : null), [sim, t])

  // refetch the exact frame when the cursor settles (not during playback)
  const exactTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (!showExact || playing || !sim) return
    if (exactTimer.current) clearTimeout(exactTimer.current)
    exactTimer.current = setTimeout(() => void fetchExact(t), 350)
    return () => {
      if (exactTimer.current) clearTimeout(exactTimer.current)
    }
  }, [showExact, playing, t, sim, fetchExact])

  if (!plan) {
    return (
      <Panel title="Tweezers">
        {planStatus === 'error' && planError ? (
          <div className="caption">The plan was refused ({planError.split('\n')[0]}); fix the request on the left to get a movie.</div>
        ) : (
          <Skeleton height={320} label={planStatus === 'loading' ? 'planning…' : 'waiting for a plan…'} />
        )}
      </Panel>
    )
  }
  if (!sim || !scale || !frame) {
    return (
      <Panel title="Tweezers" sub="astigmatic Gaussian approximation of the closed-form spot metrics">
        {simStatus === 'error' ? <div className="banner"><span className="code">simulate</span><span>{simError}</span></div> : <Skeleton height={320} label="simulating…" />}
        <div style={{ marginTop: 10 }}>
          <Playback />
        </div>
      </Panel>
    )
  }

  const spots = frame.spots
  const planeZ = viewMode === 'tracked' ? frame.tracked_z : 0
  const peak = viewMode === 'tracked' ? scale.peakTracked : scale.peakFixed
  const rowY = spotRow(spots)
  const reqNow = requestedAt(plan.trajectory, t)
  const hidden = offCanvas(spots, scale.bounds)
  const tc = t - plan.constants.half_tau_us
  const fades = plan.fade_events.filter((e) => Math.abs(e.time_us - tc) <= 0.6 * plan.constants.tau_us)
  const beforeHalfTau = t < plan.constants.half_tau_us
  const top = brightest(spots)
  const aspect = (scale.bounds.y1 - scale.bounds.y0) / (scale.bounds.x1 - scale.bounds.x0)

  return (
    <Panel
      title="Tweezers"
      sub={
        <>
          <strong>astigmatic Gaussian approximation</strong> of the closed-form spot metrics (positions, Z, ΔF, power exact; shape approximate) · {spots.length} spot
          {spots.length === 1 ? '' : 's'}
          {sim.notes.length > 0 && (
            <>
              {' '}
              · <span title={sim.notes.join('\n')}>{sim.notes.length} note{sim.notes.length === 1 ? '' : 's'}</span>
            </>
          )}
        </>
      }
      actions={
        <>
          <span className="tabs" style={{ border: 'none', margin: 0 }}>
            <button className={`small ${viewMode === 'tracked' ? 'active' : ''}`} onClick={() => setViewMode('tracked')} title="XY plane follows the power-weighted best focus">
              tracked plane
            </button>
            <button className={`small ${viewMode === 'fixed' ? 'active' : ''}`} onClick={() => setViewMode('fixed')} title="XY plane pinned at the static focal plane Z = 0 (a real camera)">
              fixed Z = 0
            </button>
          </span>
          <button className={`small ${showExact ? 'active' : ''}`} onClick={() => setShowExact(!showExact)} title="fetch the closed-form Eq. S11 frame at the cursor from /api/frame and show it beside the approximation on the same colour scale">
            exact frame
          </button>
        </>
      }
    >
      <div className="row" style={{ marginBottom: 6, gap: 6 }}>
        {beforeHalfTau && <span className="badge">dark before τ/2: both wavefronts of a pair must reach the beam</span>}
        {hidden.length > 0 && (
          <span className="badge warn" title={hidden.map((s) => `x ${fmt(s.x_um, 1)}, y ${fmt(s.y_um, 1)} µm, P/P_ref ${fmt(s.power / sim.power_ref, 2)}`).join('\n')}>
            {hidden.length} shadow tweezer{hidden.length === 1 ? '' : 's'} off canvas
            {fades.length > 0 ? ` at ±${fades.map((e) => fmt(e.shadow_um, 1)).filter((v, i, a) => a.indexOf(v) === i).join(' / ±')} µm (${fades.map((e) => e.axis).join(', ')} fade)` : ` at ${hidden.slice(0, 2).map((s) => `(${fmt(s.x_um, 0)}, ${fmt(s.y_um, 0)})`).join(', ')} µm`}
          </span>
        )}
        {fades.length > 0 && hidden.length === 0 && <span className="badge warn">fade hand-over in progress ({fades.map((e) => e.axis).join(', ')})</span>}
        {top && (
          <span className="badge">
            brightest: P/P_ref {fmt(top.power / sim.power_ref, 2)} · z {fmt(top.z_um, 2)} µm
          </span>
        )}
      </div>
      <div className="row" style={{ alignItems: 'stretch', gap: 8 }}>
        <div className="grow">
          <div className={showExact ? 'view-grid' : ''}>
            <div className="view-cell">
              <XyView spots={spots} planeZ={planeZ} scale={scale} peak={peak} requested={reqNow} label={viewMode === 'tracked' ? 'XY @ tracked focus · Gaussian approx.' : 'XY @ fixed Z = 0 · Gaussian approx.'} />
            </div>
            {showExact && (
              <div className="view-cell">
                {exactFrame ? (
                  <ExactXy frame={exactFrame} scale={scale} peak={peak} aspect={aspect} />
                ) : exactStatus === 'error' ? (
                  <div className="banner"><span className="code">frame</span><span>{exactError}</span></div>
                ) : (
                  <Skeleton height={200} label="fetching the closed-form frame…" />
                )}
                {exactFrame && (exactStatus === 'loading' || Math.abs(exactFrame.time_us - t) > 1e-6) && (
                  <div className="caption">{exactStatus === 'loading' ? 'updating…' : `shown at t = ${fmt(exactFrame.time_us, 1)} µs (pause to refresh)`}</div>
                )}
              </div>
            )}
          </div>
          <div className={showExact ? 'view-grid' : ''} style={{ marginTop: 8 }}>
            <div className="view-cell">
              <XzView spots={spots} rowY={rowY} planeZ={planeZ} scale={scale} peak={peak} label="XZ slice · Gaussian approx." />
            </div>
            {showExact && exactFrame && (
              <div className="view-cell">
                <ExactXz frame={exactFrame} scale={scale} peak={peak} />
              </div>
            )}
          </div>
          <div style={{ marginTop: 8 }}>
            <IsoView spots={spots} scale={scale} trajectory={plan.trajectory} requested={reqNow} powerRef={sim.power_ref} />
          </div>
        </div>
        <ColorBar zMax={scale.zMax} height={240} />
      </div>
      <div className="caption" style={{ marginTop: 6 }}>
        Hue = each spot's own best-focus Z (RdBu, white at Z = 0); brightness = intensity on one global scale over the whole movie (a dimming spot looks
        dimmer). The approximation drops the aperture fill edge, acoustic irising and intra-cluster interference — compare with the exact frame.
      </div>
      <div style={{ marginTop: 10 }}>
        <Playback />
      </div>
    </Panel>
  )
}
