import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useStudio } from '../../state/store'
import { Skeleton } from '../common/Panel'

/** The three the brief names come first and are tagged "quick link". */
const QUICK = ['single_lift_hold', 'array_lift_hold', 'array_arbitrary_path']
const REQUIRED = ['pick_and_place', 'square_xy', 'sphere_loop', 'single_lift_hold', 'array_lift_hold', 'array_arbitrary_path', 'flagship_10x10']

export function ExampleCards() {
  const presets = useStudio((s) => s.presets)
  const status = useStudio((s) => s.presetsStatus)
  const loadPresets = useStudio((s) => s.loadPresets)
  useEffect(() => {
    void loadPresets()
  }, [loadPresets])

  if (!presets) {
    if (status === 'error') {
      return (
        <div className="cards">
          {REQUIRED.map((id) => (
            <Link key={id} to={`/basic?example=${id}`} className="card">
              <h4>{id.replace(/_/g, ' ')}</h4>
              <p className="faint">presets unavailable (backend not reachable) — the link still opens the workbench.</p>
            </Link>
          ))}
        </div>
      )
    }
    return (
      <div className="cards">
        {REQUIRED.map((id) => (
          <Skeleton key={id} height={120} />
        ))}
      </div>
    )
  }
  const ordered = [...presets.examples].sort((a, b) => {
    const qa = QUICK.indexOf(a.id)
    const qb = QUICK.indexOf(b.id)
    if (qa >= 0 && qb >= 0) return qa - qb
    if (qa >= 0) return -1
    if (qb >= 0) return 1
    return 0
  })
  return (
    <div className="cards">
      {ordered.map((ex) => {
        const r = ex.request
        const traps = r.array.mx * r.array.my
        return (
          <Link key={ex.id} to={`/basic?example=${ex.id}`} className="card">
            <h4>
              {ex.title} {QUICK.includes(ex.id) && <span className="badge accent">quick link</span>}
            </h4>
            <p>{ex.blurb}</p>
            <div className="tags">
              <span className="badge">
                {r.array.mx}×{r.array.my} {traps === 1 ? 'tweezer' : 'traps'}
              </span>
              <span className="badge">{r.moves.length} moves</span>
              {ex.tags.map((t) => (
                <span key={t} className={`badge ${t === 'slow' ? 'warn' : ''}`}>
                  {t}
                </span>
              ))}
            </div>
          </Link>
        )
      })}
    </div>
  )
}
