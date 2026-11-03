/**
 * Inline SVG of the four-AOD stack (arXiv:2510.11451 Fig. 1 / `docs/PLAN.md` §1.1), drawn as
 * an oblique projection so both acoustic axes are visible: the optical axis z runs left →
 * right, x is up, +y comes out of the page toward the viewer (down-left). Each crystal is a
 * slab in its channel colour with its transducer on the face the sound starts from
 * (`aodl.device.conventions.CHANNEL_GEOMETRY`: Ax sound −x ⇒ transducer on the +x face, Bx +x
 * ⇒ −x face, Ay −y ⇒ +y (front) face, By +y ⇒ −y (back) face), acoustic wavefronts fading away
 * from it, and an arrow for the sound direction.
 */
import type { ChannelName } from '../../api/types'
import { CHANNEL_COLORS } from '../../channels'

const AXIS_Y = 192 // optical axis (CSS px in the 900×360 viewBox)
const W = 46 // slab extent along z
const H = 96 // slab extent along x (aperture D)
const OX = 30 // depth offset of the back face (−y) relative to the front face
const OY = -18

interface Slab {
  name: ChannelName
  cx: number
  sound: '−x' | '+x' | '−y' | '+y'
  role: string
  tone: string
}

const SLABS: Slab[] = [
  { name: 'Ax', cx: 160, sound: '−x', role: 'first x deflector', tone: 'single tone · lateral X' },
  { name: 'Bx', cx: 420, sound: '+x', role: 'second x deflector', tone: 'column ladder (Δf_x)' },
  { name: 'Ay', cx: 520, sound: '−y', role: 'first y deflector', tone: 'single tone · lateral Y' },
  { name: 'By', cx: 610, sound: '+y', role: 'second y deflector', tone: 'row ladder (Δf_y)' },
]

function slabOrigin(cx: number): { x0: number; y0: number } {
  // the slab's centre (mid-depth) sits on the optical axis
  return { x0: cx - W / 2 - OX / 2, y0: AXIS_Y - H / 2 - OY / 2 }
}

function Crystal({ slab }: { slab: Slab }) {
  const color = CHANNEL_COLORS[slab.name]
  const { x0, y0 } = slabOrigin(slab.cx)
  const front = `${x0},${y0} ${x0 + W},${y0} ${x0 + W},${y0 + H} ${x0},${y0 + H}`
  const top = `${x0},${y0} ${x0 + OX},${y0 + OY} ${x0 + W + OX},${y0 + OY} ${x0 + W},${y0}`
  const side = `${x0 + W},${y0} ${x0 + W + OX},${y0 + OY} ${x0 + W + OX},${y0 + H + OY} ${x0 + W},${y0 + H}`
  const alongX = slab.sound === '−x' || slab.sound === '+x'
  const fromTop = slab.sound === '−x' // transducer on the +x (top) face, sound travels down
  const fromFront = slab.sound === '−y' // transducer on the +y (front) face, sound travels into the page

  const wavefronts: JSX.Element[] = []
  const n = 7
  for (let i = 1; i <= n; i++) {
    const f = i / (n + 1)
    const opacity = 0.85 * (1 - 0.8 * f) // fades with distance from the transducer
    if (alongX) {
      // planes of constant x → horizontal lines across the front face (and onto the side face)
      const yy = fromTop ? y0 + f * H : y0 + H - f * H
      wavefronts.push(
        <polyline
          key={i}
          points={`${x0 + 2},${yy} ${x0 + W},${yy} ${x0 + W + OX},${yy + OY}`}
          fill="none"
          stroke={color}
          strokeWidth={1}
          opacity={opacity}
        />,
      )
    } else {
      // planes of constant y → rectangles parallel to the front face, stacked in depth
      const d = fromFront ? f : 1 - f
      const dx = d * OX
      const dy = d * OY
      wavefronts.push(
        <rect
          key={i}
          x={x0 + dx + 3}
          y={y0 + dy + 3}
          width={W - 6}
          height={H - 6}
          fill="none"
          stroke={color}
          strokeWidth={1}
          opacity={opacity}
        />,
      )
    }
  }

  // transducer plate + sound arrow
  let transducer: JSX.Element
  let arrow: JSX.Element
  const ax = x0 + W / 2
  if (alongX) {
    transducer = fromTop ? (
      <polygon points={`${x0},${y0} ${x0 + OX},${y0 + OY} ${x0 + W + OX},${y0 + OY} ${x0 + W},${y0}`} fill="#f2f5f8" opacity={0.9} />
    ) : (
      <rect x={x0} y={y0 + H - 4} width={W} height={5} fill="#f2f5f8" opacity={0.9} />
    )
    const y1 = fromTop ? y0 + 10 : y0 + H - 10
    const y2 = fromTop ? y0 + H - 14 : y0 + 14
    arrow = <line x1={ax} y1={y1} x2={ax} y2={y2} stroke="#f2f5f8" strokeWidth={2} markerEnd="url(#arrow)" />
  } else {
    transducer = fromFront ? (
      <rect x={x0} y={y0} width={W} height={H} fill="none" stroke="#f2f5f8" strokeWidth={3} opacity={0.9} />
    ) : (
      <polygon
        points={`${x0 + OX},${y0 + OY} ${x0 + W + OX},${y0 + OY} ${x0 + W + OX},${y0 + H + OY} ${x0 + OX},${y0 + H + OY}`}
        fill="none"
        stroke="#f2f5f8"
        strokeWidth={3}
        strokeDasharray="4 3"
        opacity={0.9}
      />
    )
    const cxm = ax
    const cym = y0 + H / 2
    const p1 = fromFront ? { x: cxm - 4, y: cym + 8 } : { x: cxm + OX - 4, y: cym + OY + 8 }
    const p2 = fromFront ? { x: cxm + OX - 10, y: cym + OY + 14 } : { x: cxm + 2, y: cym + 2 }
    arrow = <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#f2f5f8" strokeWidth={2} markerEnd="url(#arrow)" />
  }

  const labelY = y0 + OY - 22
  return (
    <g>
      <polygon points={top} fill={color} opacity={0.35} stroke={color} strokeWidth={1} />
      <polygon points={side} fill={color} opacity={0.22} stroke={color} strokeWidth={1} />
      <polygon points={front} fill={color} opacity={0.16} stroke={color} strokeWidth={1.4} />
      {wavefronts}
      {transducer}
      {arrow}
      <text x={slab.cx} y={labelY} textAnchor="middle" fill={color} fontSize={14} fontWeight={700}>
        {slab.name}
      </text>
      <text x={slab.cx} y={labelY + 13} textAnchor="middle" fill="#d7dde5" fontSize={10}>
        sound {slab.sound}
      </text>
      <text x={slab.cx} y={y0 + H + 20} textAnchor="middle" fill="#d7dde5" fontSize={10}>
        {slab.role}
      </text>
      <text x={slab.cx} y={y0 + H + 32} textAnchor="middle" fill="#9aa5b3" fontSize={9.5}>
        {slab.tone}
      </text>
    </g>
  )
}

function Lens({ cx, h, label, strong }: { cx: number; h: number; label?: string; strong?: boolean }) {
  const r = strong ? 14 : 9
  return (
    <g>
      <path
        d={`M ${cx} ${AXIS_Y - h / 2} Q ${cx + r} ${AXIS_Y} ${cx} ${AXIS_Y + h / 2} Q ${cx - r} ${AXIS_Y} ${cx} ${AXIS_Y - h / 2} Z`}
        fill="#b388ff"
        opacity={0.35}
        stroke="#b388ff"
        strokeWidth={1.2}
      />
      {label && (
        <text x={cx} y={AXIS_Y + h / 2 + 14} textAnchor="middle" fill="#9aa5b3" fontSize={10}>
          {label}
        </text>
      )}
    </g>
  )
}

export function SetupDiagram() {
  const beamHalf = 13 // ±w_in
  const beam = '#ff6b6b'
  return (
    <svg className="diagram" viewBox="0 0 900 360" role="img" aria-labelledby="setup-title setup-desc">
      <title id="setup-title">The four-AOD acousto-optic deflector lens</title>
      <desc id="setup-desc">
        Laser, first x deflector Ax (sound toward −x, transducer on the +x face), a 4f relay, second x deflector Bx (sound +x), first y
        deflector Ay (sound −y), second y deflector By (sound +y), objective, focal volume with the trap trajectory.
      </desc>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#f2f5f8" />
        </marker>
        <marker id="arrow-dim" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa5b3" />
        </marker>
        <radialGradient id="trap">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="1" />
          <stop offset="40%" stopColor="#f3a481" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#f3a481" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="beamfade" x1="0" x2="1">
          <stop offset="0" stopColor={beam} stopOpacity="0.28" />
          <stop offset="1" stopColor={beam} stopOpacity="0.1" />
        </linearGradient>
      </defs>

      {/* optical axis */}
      <line x1={70} y1={AXIS_Y} x2={860} y2={AXIS_Y} stroke="#3a4352" strokeWidth={1} strokeDasharray="3 5" />

      {/* beam: collimated through the stack, a 4f relay between Ax and Bx, converging after the objective */}
      <rect x={70} y={AXIS_Y - beamHalf} width={125} height={2 * beamHalf} fill="url(#beamfade)" />
      <polygon points={`195,${AXIS_Y - beamHalf} 195,${AXIS_Y + beamHalf} 290,${AXIS_Y + 2} 290,${AXIS_Y - 2}`} fill={beam} opacity={0.18} />
      <polygon points={`290,${AXIS_Y - 2} 290,${AXIS_Y + 2} 385,${AXIS_Y + beamHalf} 385,${AXIS_Y - beamHalf}`} fill={beam} opacity={0.18} />
      <rect x={385} y={AXIS_Y - beamHalf} width={330} height={2 * beamHalf} fill={beam} opacity={0.15} />
      <polygon points={`715,${AXIS_Y - beamHalf} 715,${AXIS_Y + beamHalf} 805,${AXIS_Y + 1.5} 805,${AXIS_Y - 1.5}`} fill={beam} opacity={0.22} />

      {/* laser */}
      <rect x={18} y={AXIS_Y - 16} width={52} height={32} rx={3} fill="#1b2130" stroke="#5a6472" />
      <text x={44} y={AXIS_Y + 4} textAnchor="middle" fill="#f2f5f8" fontSize={11}>
        laser
      </text>
      <text x={44} y={AXIS_Y + 30} textAnchor="middle" fill="#9aa5b3" fontSize={9.5}>
        Gaussian, w_in
      </text>

      {/* 4f relay */}
      <Lens cx={243} h={74} />
      <Lens cx={337} h={74} />
      <line x1={243} y1={AXIS_Y - 46} x2={337} y2={AXIS_Y - 46} stroke="#9aa5b3" strokeWidth={1} markerStart="url(#arrow-dim)" markerEnd="url(#arrow-dim)" />
      <text x={290} y={AXIS_Y - 52} textAnchor="middle" fill="#d7dde5" fontSize={10}>
        4f relay (mag. −1)
      </text>
      <text x={290} y={AXIS_Y - 63} textAnchor="middle" fill="#9aa5b3" fontSize={9.5}>
        images Ax onto Bx
      </text>
      <text x={290} y={AXIS_Y + 48} textAnchor="middle" fill="#9aa5b3" fontSize={9.5}>
        A's deflection → B's incidence angle
      </text>

      {SLABS.map((s) => (
        <Crystal key={s.name} slab={s} />
      ))}

      {/* objective */}
      <Lens cx={715} h={110} strong />
      <text x={715} y={AXIS_Y - 66} textAnchor="middle" fill="#d7dde5" fontSize={11}>
        objective
      </text>
      <text x={715} y={AXIS_Y - 54} textAnchor="middle" fill="#9aa5b3" fontSize={9.5}>
        focal length F
      </text>

      {/* focal volume: oblique box with the trap trail */}
      <g transform="translate(805 0)">
        <polygon points={`0,${AXIS_Y - 34} 0,${AXIS_Y + 34} 48,${AXIS_Y + 34} 48,${AXIS_Y - 34}`} fill="none" stroke="#5a6472" strokeWidth={1} />
        <polygon points={`0,${AXIS_Y - 34} 22,${AXIS_Y - 48} 70,${AXIS_Y - 48} 48,${AXIS_Y - 34}`} fill="none" stroke="#5a6472" strokeWidth={1} />
        <polygon points={`48,${AXIS_Y - 34} 70,${AXIS_Y - 48} 70,${AXIS_Y + 20} 48,${AXIS_Y + 34}`} fill="none" stroke="#5a6472" strokeWidth={1} />
        {/* trail: lift, translate, lower */}
        <path d={`M 8 ${AXIS_Y + 20} C 8 ${AXIS_Y + 4}, 14 ${AXIS_Y - 6}, 20 ${AXIS_Y - 8} S 44 ${AXIS_Y - 14}, 50 ${AXIS_Y - 10} S 58 ${AXIS_Y + 4}, 58 ${AXIS_Y + 12}`} fill="none" stroke="#4393c3" strokeWidth={1.4} strokeDasharray="2 2" />
        <circle cx={8} cy={AXIS_Y + 20} r={2} fill="#4393c3" opacity={0.6} />
        <circle cx={58} cy={AXIS_Y + 12} r={2} fill="#d6604d" opacity={0.6} />
        <circle cx={36} cy={AXIS_Y - 11} r={9} fill="url(#trap)" />
        <circle cx={36} cy={AXIS_Y - 11} r={2.2} fill="#ffffff" />
        <text x={30} y={AXIS_Y + 50} textAnchor="middle" fill="#d7dde5" fontSize={10}>
          focal volume
        </text>
        <text x={30} y={AXIS_Y + 62} textAnchor="middle" fill="#9aa5b3" fontSize={9.5}>
          X, Y from Δf · Z̄ from chirps
        </text>
        <text x={30} y={AXIS_Y + 74} textAnchor="middle" fill="#9aa5b3" fontSize={9.5}>
          trail: lift → carry → lower
        </text>
      </g>

      {/* axis triad */}
      <g transform="translate(40 318)">
        <line x1={0} y1={0} x2={34} y2={0} stroke="#9aa5b3" strokeWidth={1.2} markerEnd="url(#arrow-dim)" />
        <line x1={0} y1={0} x2={0} y2={-30} stroke="#9aa5b3" strokeWidth={1.2} markerEnd="url(#arrow-dim)" />
        <line x1={0} y1={0} x2={-20} y2={13} stroke="#9aa5b3" strokeWidth={1.2} markerEnd="url(#arrow-dim)" />
        <text x={38} y={4} fill="#d7dde5" fontSize={10}>
          z (optical axis)
        </text>
        <text x={4} y={-24} fill="#d7dde5" fontSize={10}>
          x
        </text>
        <text x={-32} y={24} fill="#d7dde5" fontSize={10}>
          +y
        </text>
      </g>

      {/* legend */}
      <g transform="translate(250 300)" fontSize={10} fill="#d7dde5">
        <rect x={0} y={0} width={14} height={5} fill="#f2f5f8" />
        <text x={20} y={6}>
          transducer (sound source face)
        </text>
        <line x1={190} y1={3} x2={204} y2={3} stroke="#8ac926" strokeWidth={1} opacity={0.8} />
        <line x1={190} y1={7} x2={204} y2={7} stroke="#8ac926" strokeWidth={1} opacity={0.35} />
        <text x={210} y={9}>
          acoustic wavefronts (fade with travel)
        </text>
        <rect x={0} y={16} width={14} height={8} fill={beam} opacity={0.3} />
        <text x={20} y={24}>
          beam, ±w_in (uncropped Gaussian)
        </text>
        <line x1={190} y1={20} x2={204} y2={20} stroke="#f2f5f8" strokeWidth={2} markerEnd="url(#arrow)" />
        <text x={210} y={24}>
          sound direction (counter-propagating in each pair)
        </text>
        <text x={0} y={42} fill="#9aa5b3" fontSize={9.5}>
          A/B pairs: deflections subtract (X ∝ f_Bx − f_Ax), chirp lenses add (Z̄ ∝ Σ ḟ): the lens sign does not depend on the sound direction.
        </text>
      </g>
    </svg>
  )
}
