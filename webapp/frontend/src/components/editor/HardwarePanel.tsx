import { useStudio } from '../../state/store'
import { FALLBACK_HARDWARE_PRESETS } from '../../state/defaults'
import { fmt } from '../../physics/format'
import { tauUs, umPerMHz } from '../../physics/units'
import { NumberField, SelectField } from '../common/NumberField'
import { Panel } from '../common/Panel'

export function HardwarePanel() {
  const hw = useStudio((s) => s.hardware)
  const preset = useStudio((s) => s.hardwarePreset)
  const presets = useStudio((s) => s.presets)
  const setHardware = useStudio((s) => s.setHardware)
  const setPreset = useStudio((s) => s.setHardwarePreset)
  const constants = useStudio((s) => s.plan?.constants)
  const names = Object.keys(presets?.hardware_presets ?? FALLBACK_HARDWARE_PRESETS)
  const options = [...names.map((n) => ({ value: n, label: n })), { value: 'custom', label: 'custom' }]

  // live derived numbers (the plan's constants win once a plan exists)
  const tau = constants?.tau_us ?? tauUs(hw.aperture_mm, hw.sound_speed_m_s)
  const scale = constants?.um_per_MHz ?? umPerMHz(hw.wavelength_nm, hw.focal_length_mm, hw.sound_speed_m_s)
  const w0 = constants?.waist0_um ?? (hw.wavelength_nm * 1e-9 * hw.focal_length_mm * 1e-3) / (Math.PI * hw.w_in_mm * 1e-3) / 1e-6
  const zR = constants?.rayleigh_um ?? (Math.PI * (w0 * 1e-6) ** 2) / (hw.wavelength_nm * 1e-9) / 1e-6

  return (
    <Panel title="Hardware" sub="AODLParams — identical AODParams on the four channels">
      <SelectField label="preset" value={preset} options={options} onChange={(v) => v !== 'custom' && setPreset(v)} />
      <NumberField label="wavelength" unit="nm" value={hw.wavelength_nm} min={1} onChange={(v) => setHardware({ wavelength_nm: v })} />
      <NumberField label="focal length F" unit="mm" value={hw.focal_length_mm} min={0.01} onChange={(v) => setHardware({ focal_length_mm: v })} />
      <NumberField label="input radius w_in" unit="mm" value={hw.w_in_mm} min={0.01} onChange={(v) => setHardware({ w_in_mm: v })} />
      <NumberField label="sound speed v" unit="m/s" value={hw.sound_speed_m_s} min={1} onChange={(v) => setHardware({ sound_speed_m_s: v })} />
      <NumberField label="aperture D" unit="mm" value={hw.aperture_mm} min={0.01} onChange={(v) => setHardware({ aperture_mm: v })} />
      <NumberField label="f_center" unit="MHz" value={hw.f_center_MHz} min={0.1} onChange={(v) => setHardware({ f_center_MHz: v })} />
      <NumberField label="band low" unit="MHz" value={hw.band_MHz[0]} min={0.1} onChange={(v) => setHardware({ band_MHz: [v, hw.band_MHz[1]] })} />
      <NumberField label="band high" unit="MHz" value={hw.band_MHz[1]} min={0.1} onChange={(v) => setHardware({ band_MHz: [hw.band_MHz[0], v] })} />
      <NumberField label="drive strength C" value={hw.drive_strength} min={0} step={0.05} onChange={(v) => setHardware({ drive_strength: v })} title="phase-modulation depth of the thin grating (Eq. S1)" />
      <SelectField
        label="mixing order"
        value={String(hw.mixing_order) as '1' | '3'}
        options={[
          { value: '1', label: '1 (fundamentals only)' },
          { value: '3', label: '3 (compression + IM3)' },
        ]}
        onChange={(v) => setHardware({ mixing_order: v === '3' ? 3 : 1 })}
      />
      <dl className="kv" style={{ marginTop: 8 }}>
        <dt>τ = D/v</dt>
        <dd>{fmt(tau, 2)} µs</dd>
        <dt>λF/v</dt>
        <dd>{fmt(scale, 2)} µm/MHz</dd>
        <dt>w₀ = λF/(π w_in)</dt>
        <dd>{fmt(w0, 3)} µm</dd>
        <dt>z_R</dt>
        <dd>{fmt(zR, 2)} µm</dd>
        {constants && (
          <>
            <dt>Eq. 1 hold @ 10 µm</dt>
            <dd>{fmt(constants.eq1_hold_10um_us, 0)} µs</dd>
            <dt>chirp per µm of Z</dt>
            <dd>{fmt(constants.chirp_per_um_MHz_per_ms, 3)} MHz/ms</dd>
          </>
        )}
      </dl>
    </Panel>
  )
}
