import { describe, expect, it } from 'vitest'
import type { ChannelIR, ChannelName, Envelope, WaveformIR } from '../src/api/types'
import { firstBandExit, inBandFraction, onGrid, unionGrid } from '../src/components/tutorial/tutorialMath'

const TAU = 11.5

/** One linear-chirp tone `f(t) = f0 + slope·t` [MHz detuning] on [0, T] (single segment). */
function chan(f0: number, slope: number, T: number, env: Envelope = { kind: 'constant', amp: 1 }, extra: ChannelIR['tones'] = []): ChannelIR {
  return {
    f_center_MHz: 100,
    tau_us: TAU,
    sound_sign: -1,
    axis: 'x',
    aperture_mm: 7.5,
    sound_speed_m_s: 650,
    band_MHz: [90, 110],
    tones: [{ freq: { breaks_us: [0, T], coeffs: [[f0, slope * T]] }, phase: { breaks_us: [0, T], coeffs: [[0]] }, phase0: 0, env }, ...extra],
  }
}

function ir(chans: Partial<Record<ChannelName, ChannelIR>>, T: number): WaveformIR {
  const still = chan(0, 0, T)
  return { channels: { Ax: chans.Ax ?? still, Bx: chans.Bx ?? still, Ay: chans.Ay ?? still, By: chans.By ?? still }, t_span_us: [0, T] }
}

describe('firstBandExit', () => {
  it('finds the crossing of a walking tone to bisection accuracy', () => {
    // 100 + 0.05 t leaves 110 MHz at t = 200 µs
    const x = firstBandExit(ir({ By: chan(0, 0.05, 400) }, 400))
    expect(x).not.toBeNull()
    expect(x!.channel).toBe('By')
    expect(x!.tone).toBe(0)
    expect(x!.edge_MHz).toBe(110)
    expect(x!.t_drive_us).toBeCloseTo(200, 3)
    expect(x!.t_atoms_us).toBeCloseTo(200 + TAU / 2, 3)
  })

  it('takes the earliest channel and the lower edge', () => {
    // Ax walks down: 100 − 0.1 t leaves 90 at t = 100; Bx walks up, leaves at 200
    const x = firstBandExit(ir({ Ax: chan(0, -0.1, 400), Bx: chan(0, 0.05, 400) }, 400))
    expect(x!.channel).toBe('Ax')
    expect(x!.edge_MHz).toBe(90)
    expect(x!.t_drive_us).toBeCloseTo(100, 3)
  })

  it('ignores tones whose envelope is zero while out of band', () => {
    // the walking tone is gated off before it reaches 110 MHz (t_off = 150 < 200)
    const gated: Envelope = { kind: 'smooth_on_off', t_on_us: 0, t_off_us: 150, ramp_us: 10 }
    expect(firstBandExit(ir({ Ax: chan(0, 0.05, 400, gated) }, 400))).toBeNull()
  })

  it('returns null for an in-band drive and t0 for a drive that starts outside', () => {
    expect(firstBandExit(ir({}, 100))).toBeNull()
    const x = firstBandExit(ir({ Ay: chan(15, 0, 100) }, 100))
    expect(x!.t_drive_us).toBe(0)
  })
})

describe('inBandFraction', () => {
  it('weights live tones by amplitude² and multiplies the channels', () => {
    // Ax: one in-band tone (A = 1) and one out-of-band tone (A = 0.5) → 1 / 1.25
    const extra = chan(15, 0, 100, { kind: 'constant', amp: 0.5 }).tones
    const w = ir({ Ax: chan(0, 0, 100, { kind: 'constant', amp: 1 }, extra) }, 100)
    const r = inBandFraction(w, 10)
    expect(r.per_channel.Ax).toBeCloseTo(0.8, 12)
    expect(r.per_channel.Bx).toBe(1)
    expect(r.product).toBeCloseTo(0.8, 12)
    const out = inBandFraction(ir({ By: chan(0, 0.05, 400) }, 400), 250)
    expect(out.product).toBe(0)
  })
})

describe('grid helpers', () => {
  it('merges grids and places samples', () => {
    const g = unionGrid([[0, 2, 4], [1, 2, 3]])
    expect(g).toEqual([0, 1, 2, 3, 4])
    expect(onGrid(g, [1, 3], [10, 30])).toEqual([null, 10, null, 30, null])
  })
})
