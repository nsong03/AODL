import { useStudio } from '../../state/store'
import { fmt } from '../../physics/format'
import { Panel, Skeleton } from '../common/Panel'
import { BandUsage } from './BandUsage'

export function ReportCard() {
  const plan = useStudio((s) => s.plan)
  const status = useStudio((s) => s.planStatus)
  const hw = useStudio((s) => s.hardware)
  const setTime = useStudio((s) => s.setTime)
  if (!plan) {
    return (
      <Panel title="Report">
        {status === 'loading' ? <Skeleton height={160} label="planning…" /> : <div className="caption">No plan yet.</div>}
      </Panel>
    )
  }
  const zb = plan.z_budget
  return (
    <Panel
      title="Report"
      sub={
        <>
          plan <code>{plan.plan_id}</code>
          {status === 'loading' && <span className="spinner" style={{ marginLeft: 8 }} />}
        </>
      }
      actions={
        <span className={`badge ${plan.mode === 'shepard' ? 'warn' : 'ok'}`}>{plan.mode === 'shepard' ? 'fading Shepard (Eqs. S24–S28)' : 'Eq. S19'}</span>
      }
    >
      <div className="caption" style={{ marginBottom: 6 }}>
        {plan.description}
      </div>
      <BandUsage usage={plan.band_usage} band={hw.band_MHz} toneCounts={plan.tone_counts} />
      <dl className="kv" style={{ marginTop: 8 }}>
        <dt>trajectory</dt>
        <dd>
          {fmt(plan.duration_us, 1)} µs · drive span {fmt(plan.t_span_us[0], 1)}–{fmt(plan.t_span_us[1], 1)} µs
        </dd>
        <dt>Eq. 1 axial budget</dt>
        <dd>
          ∫Z dt = {zb.requested_m_s.toExponential(2)} m·s {zb.ceiling_m_s === null ? '(no ceiling)' : `/ ${zb.ceiling_m_s.toExponential(2)} m·s`} ·{' '}
          <span style={{ color: zb.fraction > 1 ? 'var(--warn)' : undefined }}>{fmt(100 * zb.fraction, 0)} %</span>
        </dd>
        <dt>retardation</dt>
        <dd>
          τ = {fmt(plan.constants.tau_us, 2)} µs · atoms lag the drive by τ/2 = {fmt(plan.constants.half_tau_us, 2)} µs
        </dd>
      </dl>
      {plan.fade_events.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="small dim">
            fade schedule — {plan.fade_events.length} hand-over{plan.fade_events.length === 1 ? '' : 's'} (shadow tweezers at ±(λF/v)·Δf mid-fade):
          </div>
          <div className="row" style={{ gap: 4, marginTop: 4 }}>
            {plan.fade_events.slice(0, 24).map((e, i) => (
              <button key={i} className="badge" style={{ cursor: 'pointer' }} onClick={() => setTime(e.time_us + plan.constants.half_tau_us)} title={`jump the cursor to the hand-over as seen at the atoms (t = ${fmt(e.time_us, 1)} + τ/2 µs)`}>
                {e.axis} @ {fmt(e.time_us, 1)} µs · ±{fmt(e.shadow_um, 1)} µm
              </button>
            ))}
            {plan.fade_events.length > 24 && <span className="caption">+{plan.fade_events.length - 24} more</span>}
          </div>
        </div>
      )}
      {plan.notes.length > 0 && (
        <ul className="notes">
          {plan.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
      <details style={{ marginTop: 8 }}>
        <summary className="small dim" style={{ cursor: 'pointer' }}>
          PlanReport.summary()
        </summary>
        <pre className="subpanel" style={{ marginTop: 6, maxHeight: 260, overflow: 'auto' }}>
          {plan.summary_text}
        </pre>
      </details>
    </Panel>
  )
}
