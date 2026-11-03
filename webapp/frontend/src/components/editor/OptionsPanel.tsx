import { useStudio } from '../../state/store'
import { CheckField, NumberField, SelectField } from '../common/NumberField'
import { Panel } from '../common/Panel'

export function OptionsPanel() {
  const o = useStudio((s) => s.options)
  const set = useStudio((s) => s.setOptions)
  const arr = useStudio((s) => s.array)
  const mode = useStudio((s) => s.plan?.mode)
  const tau = useStudio((s) => s.plan?.constants.tau_us)
  const shepardOn = mode === 'shepard'
  const dfx = o.shepard_delta_f_MHz?.[0] ?? (arr.mx > 1 ? arr.delta_f_x_MHz : 8)
  const dfy = o.shepard_delta_f_MHz?.[1] ?? (arr.my > 1 ? arr.delta_f_y_MHz : 8)
  return (
    <Panel title="Synthesis options" sub="plan_motion(...)">
      <SelectField
        label="shepard"
        value={o.shepard}
        options={[
          { value: 'auto', label: 'auto (Eq. S19, ladders only if the band check fails)' },
          { value: 'never', label: 'never (insist on Eq. S19)' },
          { value: 'always', label: 'always (ShepardConfig)' },
        ]}
        onChange={(v) => set({ shepard: v, shepard_delta_f_MHz: v === 'always' ? [dfx, dfy] : null })}
        wide
      />
      {o.shepard === 'always' && (
        <>
          <NumberField label="Shepard Δf_x" unit="MHz" value={dfx} min={0.01} onChange={(v) => set({ shepard_delta_f_MHz: [v, dfy] })} title="must equal the array spacing on an array axis" />
          <NumberField label="Shepard Δf_y" unit="MHz" value={dfy} min={0.01} onChange={(v) => set({ shepard_delta_f_MHz: [dfx, v] })} />
        </>
      )}
      <NumberField label="fade duty η" value={o.eta} min={0.01} max={0.5} step={0.05} onChange={(v) => set({ eta: v })} title="Eqs. S26/S27 window duty, (0, 0.5]" />
      <SelectField label="phases" value={o.phases} options={['schroeder', 'zero', 'random']} onChange={(v) => set({ phases: v })} />
      <NumberField label="phase seed" value={o.phase_seed} step={1} integer disabled={o.phases !== 'random'} onChange={(v) => set({ phase_seed: Math.round(v) })} title="numpy.random.default_rng(seed) for random phases" />
      <NumberField
        label="amplitude"
        value={o.amp}
        min={0}
        max={1}
        step={0.05}
        disabled={shepardOn}
        onChange={(v) => set({ amp: v })}
        title={shepardOn ? 'greyed out: amp ≠ 1 on a fading-Shepard drive cannot be serialised (FadeZoneEnvelope.amp has no schema slot)' : 'relative drive amplitude'}
      />
      {shepardOn && <div className="caption">amp is fixed at the current plan: a Shepard drive with amp ≠ 1 has no NPZ representation.</div>}
      <CheckField label="retard-compensate (evaluate the request at t + τ/2)" checked={o.retard_compensate} onChange={(v) => set({ retard_compensate: v })} />
      <SelectField
        label="f_z bias"
        value={o.f_z_bias}
        options={[
          { value: 'none', label: 'none (walk starts at the carrier)' },
          { value: 'auto', label: 'auto (centre the walk: ×2 budget)' },
        ]}
        onChange={(v) => set({ f_z_bias: v })}
        wide
      />
      <CheckField
        label={`hold tail t_pad: aodl default${tau ? ` (2τ = ${(2 * tau).toFixed(1)} µs)` : ' (2τ)'}`}
        checked={o.t_pad_us === null}
        onChange={(v) => set({ t_pad_us: v ? null : 0 })}
      />
      {o.t_pad_us !== null && (
        <NumberField label="t_pad" unit="µs" value={o.t_pad_us} min={0} onChange={(v) => set({ t_pad_us: v })} title="0 means the final rest position is never observable" />
      )}
      <CheckField label="check band (off: plot infeasible drives)" checked={o.check_band} onChange={(v) => set({ check_band: v })} />
    </Panel>
  )
}
