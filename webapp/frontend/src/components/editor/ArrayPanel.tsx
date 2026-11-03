import { useStudio } from '../../state/store'
import { fmt } from '../../physics/format'
import { umPerMHz } from '../../physics/units'
import { NumberField } from '../common/NumberField'
import { Panel } from '../common/Panel'

export function ArrayPanel() {
  const arr = useStudio((s) => s.array)
  const hw = useStudio((s) => s.hardware)
  const setArray = useStudio((s) => s.setArray)
  const scale = useStudio((s) => s.plan?.constants.um_per_MHz) ?? umPerMHz(hw.wavelength_nm, hw.focal_length_mm, hw.sound_speed_m_s)
  const traps = arr.mx * arr.my
  return (
    <Panel title="Array" sub="tone ladders on Bx (columns) and By (rows)">
      <div className="row" style={{ gap: 10 }}>
        <div className="grow">
          <NumberField label="m_x" value={arr.mx} min={1} step={1} integer onChange={(v) => setArray({ mx: Math.max(1, Math.round(v)) })} />
          <NumberField label="Δf_x" unit="MHz" value={arr.delta_f_x_MHz} step={0.1} onChange={(v) => setArray({ delta_f_x_MHz: v })} />
        </div>
        <div className="grow">
          <NumberField label="m_y" value={arr.my} min={1} step={1} integer onChange={(v) => setArray({ my: Math.max(1, Math.round(v)) })} />
          <NumberField label="Δf_y" unit="MHz" value={arr.delta_f_y_MHz} step={0.1} onChange={(v) => setArray({ delta_f_y_MHz: v })} />
        </div>
      </div>
      <dl className="kv" style={{ marginTop: 6 }}>
        <dt>traps</dt>
        <dd>
          {traps} {traps > 36 && <span className="badge warn">movie at 48 frames</span>}
        </dd>
        <dt>pitch x = (λF/v)·Δf_x</dt>
        <dd>{arr.mx > 1 ? `${fmt(scale * arr.delta_f_x_MHz, 2)} µm` : '— (single column)'}</dd>
        <dt>pitch y = (λF/v)·Δf_y</dt>
        <dd>{arr.my > 1 ? `${fmt(scale * arr.delta_f_y_MHz, 2)} µm` : '— (single row)'}</dd>
        <dt>lateral extent</dt>
        <dd>
          {fmt((arr.mx - 1) * scale * arr.delta_f_x_MHz, 1)} × {fmt((arr.my - 1) * scale * arr.delta_f_y_MHz, 1)} µm
        </dd>
      </dl>
      <p className="caption" style={{ marginTop: 6 }}>
        A multi-tone axis needs a non-zero spacing. The ladder width <code>(m−1)·Δf</code> comes out of the band before any lateral move or
        Z chirp does.
      </p>
    </Panel>
  )
}
