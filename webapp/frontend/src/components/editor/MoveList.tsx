import { useState } from 'react'
import type { MoveKind, MoveSpec, PathMove, ProfileName } from '../../api/types'
import { fmt } from '../../physics/format'
import { MOVE_KINDS } from '../../state/defaults'
import { useStudio } from '../../state/store'
import { NumberField, SelectField, CheckField } from '../common/NumberField'
import { Panel } from '../common/Panel'
import { PathGeneratorsModal } from './PathGenerators'
import { formatPoints, parsePoints, pathLength, type Vec3 } from './pathGenerators'

const PROFILES: ProfileName[] = ['min_jerk', 'constant_jerk', 'constant_accel', 'switching_constant_jerk', 'linear']

export function MoveList() {
  const moves = useStudio((s) => s.moves)
  const profiles = useStudio((s) => s.presets?.profiles) ?? PROFILES
  const addMove = useStudio((s) => s.addMove)
  const removeMove = useStudio((s) => s.removeMove)
  const reorderMove = useStudio((s) => s.reorderMove)
  const setMove = useStudio((s) => s.setMove)
  const replaceMove = useStudio((s) => s.replaceMove)
  const seams = useStudio((s) => s.plan?.seams_us)
  const [genFor, setGenFor] = useState<number | 'new' | null>(null)

  const total = moves.reduce((a, m) => a + m.duration_us, 0)

  return (
    <Panel
      title="Moves"
      sub={`${moves.length} moves · ${fmt(total, 1)} µs`}
      actions={
        <>
          {MOVE_KINDS.map((k) => (
            <button key={k} className="small" onClick={() => addMove(k)} title={`append a ${k} move`}>
              + {k}
            </button>
          ))}
        </>
      }
    >
      <div className="stack" style={{ gap: 6 }}>
        {moves.length === 0 && <div className="caption">Add a move — a plan needs at least one.</div>}
        {moves.map((m, i) => (
          <MoveCard
            key={i}
            i={i}
            move={m}
            profiles={profiles}
            seam={seams?.[i]}
            onPatch={(p) => setMove(i, p)}
            onReplace={(mv) => replaceMove(i, mv)}
            onRemove={() => removeMove(i)}
            onUp={() => reorderMove(i, i - 1)}
            onDown={() => reorderMove(i, i + 1)}
            onGenerate={() => setGenFor(i)}
            first={i === 0}
            last={i === moves.length - 1}
          />
        ))}
      </div>
      <p className="caption" style={{ marginTop: 8 }}>
        Moves chain rest-to-rest; a <code>path</code> is a clamped cubic spline through relative waypoints. Durations set the chirp the band
        must hold: a Z offset costs ~4.85 MHz/ms per µm on every channel.
      </p>
      {genFor !== null && (
        <PathGeneratorsModal
          onClose={() => setGenFor(null)}
          onApply={(points, closed) => {
            if (genFor === 'new') {
              addMove('path', { kind: 'path', points_um: points, duration_us: Math.max(100, 10 * pathLength(points, closed)), closed })
            } else {
              const cur = moves[genFor]
              replaceMove(genFor, { kind: 'path', points_um: points, duration_us: cur.duration_us, closed })
            }
            setGenFor(null)
          }}
        />
      )}
    </Panel>
  )
}

function MoveCard({
  i,
  move,
  profiles,
  seam,
  onPatch,
  onReplace,
  onRemove,
  onUp,
  onDown,
  onGenerate,
  first,
  last,
}: {
  i: number
  move: MoveSpec
  profiles: readonly ProfileName[]
  seam: number | undefined
  onPatch: (p: Partial<MoveSpec>) => void
  onReplace: (m: MoveSpec) => void
  onRemove: () => void
  onUp: () => void
  onDown: () => void
  onGenerate: () => void
  first: boolean
  last: boolean
}) {
  return (
    <div className="move-card">
      <header>
        <span className="idx">{i + 1}</span>
        <span className="kind">{move.kind}</span>
        {seam !== undefined && <span className="caption">starts at {fmt(seam, 1)} µs</span>}
        <span className="grow" />
        <button className="ghost small" disabled={first} onClick={onUp} title="move up">
          ↑
        </button>
        <button className="ghost small" disabled={last} onClick={onDown} title="move down">
          ↓
        </button>
        <button className="ghost small" onClick={onRemove} title="remove">
          ✕
        </button>
      </header>
      <div className="fields">
        {move.kind === 'lift' && (
          <>
            <NumberField label="Δz" unit="µm" value={move.dz_um} onChange={(v) => onPatch({ dz_um: v } as Partial<MoveSpec>)} />
            <NumberField label="duration" unit="µs" value={move.duration_us} min={0.01} onChange={(v) => onPatch({ duration_us: v })} />
          </>
        )}
        {move.kind === 'translate' && (
          <>
            <NumberField label="Δx" unit="µm" value={move.dx_um} onChange={(v) => onPatch({ dx_um: v } as Partial<MoveSpec>)} />
            <NumberField label="Δy" unit="µm" value={move.dy_um} onChange={(v) => onPatch({ dy_um: v } as Partial<MoveSpec>)} />
            <NumberField label="duration" unit="µs" value={move.duration_us} min={0.01} onChange={(v) => onPatch({ duration_us: v })} />
          </>
        )}
        {move.kind === 'hold' && <NumberField label="duration" unit="µs" value={move.duration_us} min={0.01} onChange={(v) => onPatch({ duration_us: v })} />}
        {(move.kind === 'lift' || move.kind === 'translate') && (
          <SelectField label="profile" value={move.profile} options={profiles} onChange={(v) => onPatch({ profile: v } as Partial<MoveSpec>)} />
        )}
        {move.kind === 'path' && (
          <>
            <NumberField label="duration" unit="µs" value={move.duration_us} min={0.01} onChange={(v) => onPatch({ duration_us: v })} />
            <CheckField label="closed (return to start)" checked={move.closed} onChange={(v) => onPatch({ closed: v } as Partial<MoveSpec>)} />
          </>
        )}
      </div>
      {move.kind === 'path' && <PathEditor move={move} onReplace={onReplace} onGenerate={onGenerate} />}
    </div>
  )
}

function PathEditor({ move, onReplace, onGenerate }: { move: PathMove; onReplace: (m: MoveSpec) => void; onGenerate: () => void }) {
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState('')
  const [bad, setBad] = useState(false)
  const len = pathLength(move.points_um as Vec3[], move.closed)
  return (
    <div style={{ marginTop: 4 }}>
      <div className="row between">
        <span className="caption">
          {move.points_um.length} waypoints · chord {fmt(len, 1)} µm · mean speed {fmt((len / move.duration_us) * 1e3, 1)} mm/s
        </span>
        <span className="row" style={{ gap: 4 }}>
          <button className="small" onClick={onGenerate}>
            generate…
          </button>
          <button
            className="small"
            onClick={() => {
              if (!editing) {
                setText(formatPoints(move.points_um as Vec3[]))
                setBad(false)
              }
              setEditing(!editing)
            }}
          >
            {editing ? 'done' : 'edit points'}
          </button>
        </span>
      </div>
      {editing && (
        <div style={{ marginTop: 4 }}>
          <textarea
            className="mono"
            rows={Math.min(10, move.points_um.length + 1)}
            style={{ width: '100%', background: 'var(--bg-elev)', border: `1px solid ${bad ? 'var(--danger)' : 'var(--border-strong)'}`, borderRadius: 5, padding: 4 }}
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              const pts = parsePoints(e.target.value)
              if (pts && pts.length >= 1) {
                setBad(false)
                onReplace({ ...move, points_um: pts })
              } else setBad(true)
            }}
            spellCheck={false}
            aria-label="waypoints, one 'dx, dy, dz' triple per line, µm"
          />
          <div className="caption">one waypoint per line: dx, dy, dz [µm], relative to the path start</div>
        </div>
      )}
    </div>
  )
}

export type { MoveKind }
