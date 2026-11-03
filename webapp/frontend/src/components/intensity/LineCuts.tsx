/**
 * Line cuts `I(x)`, `I(y)`, `I(z)` through the brightest trap of an `/api/frame` response: the
 * closed-form Eq. S11 cuts (solid) with the astigmatic Gaussian approximation of the same
 * frame's spots dashed on top, both in units of the movie's plateau trap depth, plus the
 * `exact_peak`, `gauss_peak` badges and their ratio (API.md §5.2, R2 addendum §F4).
 */
import { useMemo } from 'react'
import type { AlignedData, Options } from 'uplot'
import type { FrameResponse } from '../../api/types'
import { gaussCutX, gaussCutY, gaussCutZ, gaussPeakOfBrightest, nearestIndex } from '../../physics/cuts'
import { fmt } from '../../physics/format'
import type { SpotLike } from '../../physics/gaussianBeam'
import { PER_M2_TO_PER_UM2 } from '../../physics/units'
import { UPlotChart } from '../metrics/UPlotChart'
import { axisX, axisY } from '../metrics/uplotTheme'

export const EXACT_LABEL = 'closed form (Eq. S11)'
export const GAUSS_LABEL = 'Gaussian approx'

interface CutSet {
  x: AlignedData
  y: AlignedData | null
  z: AlignedData
  exactPeak: number
  gaussPeak: number
  exactFromServer: boolean
  gaussFromServer: boolean
}

function buildCuts(frame: FrameResponse, plateau: number, w0: number, zR: number): CutSet {
  const toRel = PER_M2_TO_PER_UM2 / plateau // closed form: per m² → plateau units
  const spots: SpotLike[] = frame.spots
  const z = frame.plane_z_um
  const c = frame.cuts
  // the closed-form row/column are XY-grid lines nearest the brightest spot (API.md §5.2): evaluate the approximation on the same lines
  const rowY = frame.xy.y_um.length ? frame.xy.y_um[nearestIndex(frame.xy.y_um, c.through.y_um)] : c.through.y_um
  const colX = frame.xy.x_um.length ? frame.xy.x_um[nearestIndex(frame.xy.x_um, c.through.x_um)] : c.through.x_um
  const x: AlignedData = [c.x_um, c.i_x.map((v) => v * toRel), gaussCutX(spots, c.x_um, rowY, z, w0, zR).map((v) => v / plateau)]
  let y: AlignedData | null = null
  if (c.i_y && c.i_y.length) {
    const ys = c.y_um && c.y_um.length === c.i_y.length ? c.y_um : frame.xy.y_um
    if (ys.length === c.i_y.length) y = [ys, c.i_y.map((v) => v * toRel), gaussCutY(spots, ys, colX, z, w0, zR).map((v) => v / plateau)]
  }
  const zc: AlignedData = [c.z_um, c.i_z.map((v) => v * toRel), gaussCutZ(spots, c.z_um, c.through.x_um, c.through.y_um, w0, zR).map((v) => v / plateau)]
  const exactFromServer = typeof frame.exact_peak === 'number'
  const gaussFromServer = typeof frame.gauss_peak === 'number'
  const exactPeak = (exactFromServer ? frame.exact_peak! : frame.xy.image.peak) * toRel
  const gaussPeak = gaussFromServer ? frame.gauss_peak! * toRel : gaussPeakOfBrightest(frame.spots, w0) / plateau
  return { x, y, z: zc, exactPeak, gaussPeak, exactFromServer, gaussFromServer }
}

function cutOpts(axis: string): Omit<Options, 'width' | 'height'> {
  return {
    cursor: { drag: { x: false, y: false } },
    legend: { show: false },
    scales: { x: { time: false }, y: { range: (_u: unknown, _min: number, max: number) => [0, Math.max(max * 1.05, 1e-3)] as [number, number] } },
    axes: [axisX(`${axis} [µm]`), axisY(`I(${axis}) / plateau`, 40)],
    series: [{ label: axis }, { label: EXACT_LABEL, stroke: '#f2f5f8', width: 1.4 }, { label: GAUSS_LABEL, stroke: '#f3a481', width: 1.2, dash: [5, 4] }],
  }
}

/** The three cuts side by side (stacked on narrow columns) with the peak badges. */
export function LineCuts({ frame, plateau, w0, zR, compact = false }: { frame: FrameResponse; plateau: number; w0: number; zR: number; compact?: boolean }) {
  const cuts = useMemo(() => (plateau > 0 ? buildCuts(frame, plateau, w0, zR) : null), [frame, plateau, w0, zR])
  const opts = useMemo(() => ({ x: cutOpts('x'), y: cutOpts('y'), z: cutOpts('z') }), [])
  if (!cuts) return <div className="caption">no plateau yet (the movie is dark) — cuts are shown in units of the plateau trap depth.</div>
  const h = compact ? 100 : 120
  const ratio = cuts.gaussPeak > 0 ? cuts.exactPeak / cuts.gaussPeak : NaN
  return (
    <div className="stack" style={{ gap: 4 }}>
      <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
        <span className="badge" title={cuts.exactFromServer ? 'FrameResponse.exact_peak: max of the closed-form XY image' : 'max of the closed-form XY image (PackedImage.peak; this backend does not send exact_peak)'}>
          exact peak {fmt(cuts.exactPeak, 3)}
        </span>
        <span className="badge" title={cuts.gaussFromServer ? 'FrameResponse.gauss_peak: Σ 2P/(π w_x w_y) over the spots coincident with the brightest' : 'Σ 2P/(π w_x w_y) over the spots coincident with the brightest, computed here (this backend does not send gauss_peak)'}>
          Gaussian peak {fmt(cuts.gaussPeak, 3)}
        </span>
        <span className={`badge ${Number.isFinite(ratio) && Math.abs(ratio - 1) > 0.1 ? 'warn' : ''}`}>exact / Gaussian {Number.isFinite(ratio) ? fmt(ratio, 3) : '—'}</span>
        <span className="caption">
          t = {fmt(frame.time_us, 1)} µs · plane Z = {fmt(frame.plane_z_um, 2)} µm · through ({fmt(frame.cuts.through.x_um, 2)}, {fmt(frame.cuts.through.y_um, 2)}) µm
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: compact ? '1fr' : 'repeat(auto-fit, minmax(180px, 1fr))', gap: 6 }}>
        <UPlotChart opts={opts.x} data={cuts.x} height={h} />
        {cuts.y ? <UPlotChart opts={opts.y} data={cuts.y} height={h} /> : <div className="caption">I(y) needs the round-2 backend (<code>cuts.i_y</code>).</div>}
        <UPlotChart opts={opts.z} data={cuts.z} height={h} />
      </div>
      <div className="caption">
        <span style={{ color: '#f2f5f8' }}>━ {EXACT_LABEL}</span> · <span style={{ color: '#f3a481' }}>╍ {GAUSS_LABEL}</span> — in units of the plateau trap depth (the
        brightest trap's Gaussian peak, median over the movie). Peaks: exact = max of the closed-form XY image; Gaussian = Σ 2P/(π w_x w_y) over the spots
        coincident with the brightest.
      </div>
    </div>
  )
}
