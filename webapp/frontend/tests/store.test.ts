import { beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_REQUEST, newMove } from '../src/state/defaults'
import { currentRequest, simFrameCount, useStudio } from '../src/state/store'

// network is stubbed: these tests exercise the editing / cursor logic only
vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))

describe('store editing actions', () => {
  beforeEach(() => {
    useStudio.setState({ ...useStudio.getInitialState(), moves: DEFAULT_REQUEST.moves.map((m) => ({ ...m })) })
  })
  it('add / patch / reorder / remove moves', () => {
    const s = useStudio.getState()
    s.addMove('hold')
    expect(useStudio.getState().moves.length).toBe(4)
    expect(useStudio.getState().moves[3]).toEqual(newMove('hold'))
    s.setMove(3, { duration_us: 250 })
    expect(useStudio.getState().moves[3].duration_us).toBe(250)
    s.reorderMove(3, 0)
    expect(useStudio.getState().moves[0].kind).toBe('hold')
    s.reorderMove(0, 99) // out of range: no-op
    expect(useStudio.getState().moves[0].kind).toBe('hold')
    s.removeMove(0)
    expect(useStudio.getState().moves.length).toBe(3)
    expect(useStudio.getState().moves.map((m) => m.kind)).toEqual(['lift', 'translate', 'lift'])
  })
  it('currentRequest mirrors the editable fields and setOptions patches them', () => {
    useStudio.getState().setOptions({ shepard: 'always', shepard_delta_f_MHz: [8, 8] })
    const req = currentRequest(useStudio.getState())
    expect(req.options.shepard).toBe('always')
    expect(req.options.shepard_delta_f_MHz).toEqual([8, 8])
    expect(req.hardware).toEqual(DEFAULT_REQUEST.hardware)
  })
  it('setHardware marks the preset custom', () => {
    useStudio.getState().setHardware({ wavelength_nm: 808 })
    expect(useStudio.getState().hardwarePreset).toBe('custom')
    expect(useStudio.getState().hardware.wavelength_nm).toBe(808)
  })
  it('setTime clamps to the plan span; setSpeed is sticky', () => {
    const s = useStudio.getState()
    s.setTime(-5)
    expect(useStudio.getState().t_us).toBe(0)
    s.setSpeed(42)
    expect(useStudio.getState().speed_us_per_s).toBe(42)
    expect(useStudio.getState().speedTouched).toBe(true)
    s.togglePlay()
    expect(useStudio.getState().playing).toBe(true)
    s.pause()
    expect(useStudio.getState().playing).toBe(false)
  })
  it('simFrameCount: 80 frames, 48 beyond 36 traps', () => {
    expect(simFrameCount({ mx: 4, my: 4, delta_f_x_MHz: 1, delta_f_y_MHz: 1.3 })).toBe(80)
    expect(simFrameCount({ mx: 10, my: 10, delta_f_x_MHz: 1, delta_f_y_MHz: 1.3 })).toBe(48)
  })
})
