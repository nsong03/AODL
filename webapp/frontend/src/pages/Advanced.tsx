import { useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { EfficiencyPanel } from '../components/advanced/EfficiencyPanel'
import { MisalignmentPanel } from '../components/advanced/MisalignmentPanel'
import { SpectrumPanel } from '../components/advanced/SpectrumPanel'
import { ErrorBanner } from '../components/common/Panel'
import { useStudio } from '../state/store'

export default function Advanced() {
  const [params] = useSearchParams()
  const example = params.get('example')
  const loadExample = useStudio((s) => s.loadExample)
  const requestPlan = useStudio((s) => s.requestPlan)
  const plan = useStudio((s) => s.plan)
  const planError = useStudio((s) => s.planError)
  const planStatus = useStudio((s) => s.planStatus)
  const loaded = useRef<string | null>(null)
  useEffect(() => {
    const key = example ?? '(current)'
    if (loaded.current === key) return
    loaded.current = key
    if (example) {
      void loadExample(example).then((ok) => {
        if (!ok && !useStudio.getState().plan) void requestPlan()
      })
    } else if (!useStudio.getState().plan) void requestPlan()
  }, [example, loadExample, requestPlan])

  return (
    <div className="page">
      <div className="banner info" style={{ marginBottom: 12 }}>
        <span className="code">model</span>
        <span>
          The advanced engine (<code>aodl_advanced</code>) is separate FFT code: a plane-wave single-L Kogelnik efficiency model and a chirp-z pupil
          propagation with per-AOD misalignment. Every number here is a <strong>model prediction with user-set parameters, not a measurement</strong>; the
          paper's Fig. S1c/S8 maps are the calibration target. The trajectory is the one from the basic workbench
          {plan ? ` (plan ${plan.plan_id}, ${plan.mode}).` : '.'}
        </span>
      </div>
      {planStatus === 'error' && planError && (
        <div style={{ marginBottom: 12 }}>
          <ErrorBanner code="plan" message={planError} />
        </div>
      )}
      <div className="stack">
        <EfficiencyPanel />
        <MisalignmentPanel />
        <SpectrumPanel />
      </div>
    </div>
  )
}
