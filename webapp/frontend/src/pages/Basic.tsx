import { useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { AperturePanel } from '../components/aperture/AperturePanel'
import { CodePanel } from '../components/code/CodePanel'
import { ErrorBanner } from '../components/common/Panel'
import { ArrayPanel } from '../components/editor/ArrayPanel'
import { HardwarePanel } from '../components/editor/HardwarePanel'
import { MoveList } from '../components/editor/MoveList'
import { OptionsPanel } from '../components/editor/OptionsPanel'
import { IntensityPanel } from '../components/intensity/IntensityPanel'
import { MetricsPlots } from '../components/metrics/MetricsPlots'
import { ReportCard } from '../components/report/ReportCard'
import { RfPanel } from '../components/rf/RfPanel'
import { TweezerView } from '../components/tweezer/TweezerView'
import { useStudio } from '../state/store'

export default function Basic() {
  const [params] = useSearchParams()
  const example = params.get('example')
  const loadExample = useStudio((s) => s.loadExample)
  const requestPlan = useStudio((s) => s.requestPlan)
  const planError = useStudio((s) => s.planError)
  const planErrorCode = useStudio((s) => s.planErrorCode)
  const planStatus = useStudio((s) => s.planStatus)
  const plan = useStudio((s) => s.plan)
  const loaded = useRef<string | null>(null)

  // load the example named in the URL once (or plan the current request)
  useEffect(() => {
    const key = example ?? '(current)'
    if (loaded.current === key) return
    loaded.current = key
    if (example) {
      void loadExample(example).then((ok) => {
        if (!ok && !useStudio.getState().plan) void requestPlan()
      })
    } else if (!useStudio.getState().plan) {
      void requestPlan()
    }
  }, [example, loadExample, requestPlan])

  // keyboard: space play/pause, ←/→ step
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable)) return
      const s = useStudio.getState()
      if (e.code === 'Space') {
        e.preventDefault()
        s.togglePlay()
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        s.stepFrame(-1)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        s.stepFrame(1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="page">
      {planStatus === 'error' && planError && (
        <div style={{ marginBottom: 12 }}>
          <ErrorBanner code={planErrorCode ?? 'error'} message={`${planError}${plan ? ' — the previous plan stays on screen.' : ''}`} />
        </div>
      )}
      <div className="workbench">
        <div className="col left">
          <HardwarePanel />
          <ArrayPanel />
          <MoveList />
          <OptionsPanel />
          <div className="caption">
            Keyboard: <span className="kbd">space</span> play / pause · <span className="kbd">←</span> <span className="kbd">→</span> step one frame
          </div>
        </div>
        <div className="col centre">
          <TweezerView />
          <IntensityPanel />
        </div>
        <div className="col right">
          <ReportCard />
          <RfPanel />
          <AperturePanel />
          <MetricsPlots />
          <CodePanel />
        </div>
      </div>
    </div>
  )
}
