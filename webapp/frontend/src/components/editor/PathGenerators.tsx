import { useState } from 'react'
import { NumberField, SelectField } from '../common/NumberField'
import { circlePath, helixPath, sphereSpiralPath, squarePath, type Plane, type Vec3 } from './pathGenerators'

type Gen = 'square' | 'circle' | 'sphere' | 'helix'

export function PathGeneratorsModal({ onClose, onApply }: { onClose: () => void; onApply: (points: Vec3[], closed: boolean) => void }) {
  const [gen, setGen] = useState<Gen>('circle')
  const [plane, setPlane] = useState<Plane>('xy')
  const [side, setSide] = useState(15)
  const [perSide, setPerSide] = useState(3)
  const [radius, setRadius] = useState(8)
  const [turns, setTurns] = useState(1)
  const [points, setPoints] = useState(24)
  const [pitch, setPitch] = useState(4)

  const preview = (() => {
    switch (gen) {
      case 'square':
        return { points: squarePath(side, plane, perSide), closed: true }
      case 'circle':
        return circlePath(radius, plane, turns, points)
      case 'sphere':
        return { points: sphereSpiralPath(radius, points), closed: true }
      case 'helix':
        return { points: helixPath(radius, pitch, turns, points), closed: false }
    }
  })()

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Path generators">
        <div className="row between" style={{ marginBottom: 8 }}>
          <h2>Path generator</h2>
          <button className="ghost small" onClick={onClose}>
            close
          </button>
        </div>
        <p className="caption">
          Waypoints are displacements relative to where the path starts; the position law is a clamped cubic spline (rest-to-rest, C² inside),
          knots proportional to chord length. <code>closed</code> appends a return to the start.
        </p>
        <div className="tabs">
          {(['square', 'circle', 'sphere', 'helix'] as Gen[]).map((g) => (
            <button key={g} className={gen === g ? 'active' : ''} onClick={() => setGen(g)}>
              {g === 'sphere' ? 'sphere spiral' : g}
            </button>
          ))}
        </div>
        {gen === 'square' && (
          <>
            <NumberField label="side" unit="µm" value={side} min={0.1} onChange={setSide} />
            <SelectField label="plane" value={plane} options={['xy', 'xz', 'yz']} onChange={setPlane} />
            <NumberField label="points per side" value={perSide} min={1} step={1} integer onChange={setPerSide} />
          </>
        )}
        {gen === 'circle' && (
          <>
            <NumberField label="radius" unit="µm" value={radius} min={0.1} onChange={setRadius} />
            <SelectField label="plane" value={plane} options={['xy', 'xz', 'yz']} onChange={setPlane} />
            <NumberField label="turns" value={turns} min={0.1} step={0.5} onChange={setTurns} />
            <NumberField label="points" value={points} min={3} step={1} integer onChange={setPoints} />
          </>
        )}
        {gen === 'sphere' && (
          <>
            <NumberField label="radius" unit="µm" value={radius} min={0.1} onChange={setRadius} />
            <NumberField label="points" value={points} min={4} step={1} integer onChange={setPoints} />
            <p className="caption">Sphere centred at (0, 0, r): start at the south pole, θ 0→π, φ = 3θ, closed — the `sphere_loop` example.</p>
          </>
        )}
        {gen === 'helix' && (
          <>
            <NumberField label="radius" unit="µm" value={radius} min={0.1} onChange={setRadius} />
            <NumberField label="pitch per turn" unit="µm" value={pitch} onChange={setPitch} />
            <NumberField label="turns" value={turns} min={0.1} step={0.5} onChange={setTurns} />
            <NumberField label="points" value={points} min={4} step={1} integer onChange={setPoints} />
          </>
        )}
        <div className="subpanel" style={{ marginTop: 8 }}>
          <div className="caption">
            {preview.points.length} waypoints · {preview.closed ? 'closed' : 'open'} · extent{' '}
            {(['x', 'y', 'z'] as const).map((ax, k) => {
              const vals = preview.points.map((p) => p[k])
              return `${ax} ${Math.min(0, ...vals).toFixed(1)}…${Math.max(0, ...vals).toFixed(1)}`
            }).join(' · ')}{' '}
            µm
          </div>
          <PathPreview points={preview.points} />
        </div>
        <div className="row" style={{ justifyContent: 'flex-end', marginTop: 10 }}>
          <button className="primary" onClick={() => onApply(preview.points, preview.closed)}>
            use these waypoints
          </button>
        </div>
      </div>
    </div>
  )
}

/** Tiny oblique preview of the waypoints. */
function PathPreview({ points }: { points: Vec3[] }) {
  const w = 220
  const h = 120
  const all: Vec3[] = [[0, 0, 0], ...points]
  const proj = (p: Vec3) => ({ x: p[0] + 0.45 * p[1], y: -p[2] - 0.3 * p[1] })
  const pp = all.map(proj)
  const xs = pp.map((p) => p.x)
  const ys = pp.map((p) => p.y)
  const x0 = Math.min(...xs)
  const x1 = Math.max(...xs)
  const y0 = Math.min(...ys)
  const y1 = Math.max(...ys)
  const s = 0.85 * Math.min(w / Math.max(1e-6, x1 - x0), h / Math.max(1e-6, y1 - y0))
  const tx = (x: number) => w / 2 + (x - (x0 + x1) / 2) * s
  const ty = (y: number) => h / 2 + (y - (y0 + y1) / 2) * s
  return (
    <svg width={w} height={h} style={{ display: 'block', margin: '6px auto 0' }}>
      <polyline points={pp.map((p) => `${tx(p.x)},${ty(p.y)}`).join(' ')} fill="none" stroke="#4cc9f0" strokeWidth={1.2} />
      {pp.map((p, i) => (
        <circle key={i} cx={tx(p.x)} cy={ty(p.y)} r={i === 0 ? 3 : 1.8} fill={i === 0 ? '#f2f5f8' : '#4cc9f0'} />
      ))}
    </svg>
  )
}
