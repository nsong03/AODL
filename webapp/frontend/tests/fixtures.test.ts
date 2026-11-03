/**
 * Golden tests of the TypeScript evaluator against the fixtures W1 writes with
 * `webapp/scripts/gen_frontend_fixtures.py` (`aodl_webapp.ir.evaluate_ir` + `aperture_response`).
 * The suite skips (not fails) while a fixture file is absent so the frontend can be tested
 * before the backend lands; W4 runs everything end to end.
 */
import { describe, expect, it } from 'vitest'
import { CHANNELS, type ChannelName } from '../src/api/types'
import { apertureContent } from '../src/physics/aperture'
import { envA } from '../src/physics/envelope'
import { toneFreqMHz, tonePhaseRad, voltage } from '../src/physics/waveform'
import { hasFixture, loadFixture, maxAbs, type EvalFixture } from './helpers'

const FIXTURE_FILES = ['pick_and_place.json', 'array_lift_hold.json']

for (const file of FIXTURE_FILES) {
  describe.skipIf(!hasFixture(file))(`fixture ${file}`, () => {
    const fx: EvalFixture = hasFixture(file) ? loadFixture(file) : (null as unknown as EvalFixture)

    it('has the documented shape', () => {
      expect(fx.ir.t_span_us.length).toBe(2)
      for (const ch of CHANNELS) {
        expect(fx.ir.channels[ch].tones.length).toBeGreaterThan(0)
        expect(fx.eval[ch].f_MHz.length).toBe(fx.ir.channels[ch].tones.length)
        expect(fx.eval[ch].f_MHz[0].length).toBe(fx.t_us.length)
      }
      expect(Math.abs(fx.ir.channels.Ax.tau_us - fx.constants.tau_us)).toBeLessThan(1e-9)
    })

    for (const ch of CHANNELS as readonly ChannelName[]) {
      it(`${ch}: f, A and phase agree with evaluate_ir (1e-9 rel on the largest, A 1e-12 abs)`, () => {
        const chan = fx.ir.channels[ch]
        const ev = fx.eval[ch]
        const fScale = Math.max(1e-30, ...ev.f_MHz.map((row) => maxAbs(row)))
        const phScale = Math.max(1e-30, ...ev.phase_rad.map((row) => maxAbs(row)))
        let worstF = 0
        let worstA = 0
        let worstPh = 0
        chan.tones.forEach((tone, n) => {
          fx.t_us.forEach((t, i) => {
            worstF = Math.max(worstF, Math.abs(toneFreqMHz(tone, t) - ev.f_MHz[n][i]))
            worstA = Math.max(worstA, Math.abs(envA(tone.env, t) - ev.A[n][i]))
            worstPh = Math.max(worstPh, Math.abs(tonePhaseRad(tone, t) - ev.phase_rad[n][i]))
          })
        })
        expect(worstF / fScale).toBeLessThan(1e-9)
        expect(worstA).toBeLessThan(1e-12)
        expect(worstPh / phScale).toBeLessThan(1e-9)
      })

      it(`${ch}: V(t) on the sample times agrees to 1e-6`, () => {
        const chan = fx.ir.channels[ch]
        const v = voltage(chan, fx.t_us)
        const scale = Math.max(1, maxAbs(fx.eval[ch].V))
        let worst = 0
        fx.t_us.forEach((_, i) => (worst = Math.max(worst, Math.abs(v[i] - fx.eval[ch].V[i]))))
        expect(worst / scale).toBeLessThan(1e-6)
      })

      it(`${ch}: the ±0.2 µs voltage window (2001 points) agrees to 1e-6`, () => {
        const win = fx.voltage_window
        const ref = win[ch]
        if (!ref) return
        expect(win.t_us.length).toBe(2001)
        const v = voltage(fx.ir.channels[ch], win.t_us)
        const scale = Math.max(1, maxAbs(ref))
        let worst = 0
        for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(v[i] - ref[i]))
        expect(worst / scale).toBeLessThan(1e-6)
      })
    }

    it('aperture envelope, f_inst and fill mask agree with /api/aperture to 1e-6', () => {
      for (const snap of fx.aperture) {
        for (const ch of CHANNELS) {
          const ref = snap[ch]
          if (!ref) continue
          const n = ref.u_mm.length
          const got = apertureContent(fx.ir.channels[ch], snap.time_us, n)
          let worstU = 0
          let worstEnv = 0
          let worstF = 0
          let maskMismatch = 0
          let anyFilled = false
          for (let i = 0; i < n; i++) {
            worstU = Math.max(worstU, Math.abs(got.u_mm[i] - ref.u_mm[i]))
            worstEnv = Math.max(worstEnv, Math.abs(got.envelope[i] - ref.envelope[i]))
            const fr = ref.f_inst_MHz[i]
            const fg = got.f_inst_MHz[i]
            if (fr === null || Number.isNaN(fr)) {
              if (!Number.isNaN(fg)) worstF = Math.max(worstF, 1)
            } else {
              worstF = Math.max(worstF, Math.abs(fg - fr))
            }
            if ((got.filled[i] === 1) !== ref.filled[i]) maskMismatch++
            anyFilled ||= ref.filled[i]
          }
          expect(worstU).toBeLessThan(1e-9)
          expect(worstEnv).toBeLessThan(1e-6)
          expect(worstF).toBeLessThan(1e-6)
          expect(maskMismatch).toBe(0)
          if (snap.time_us < fx.constants.tau_us) {
            // partially filled: the transducer side carries content, the far side does not
            expect(anyFilled).toBe(true)
            expect(ref.filled.every(Boolean)).toBe(false)
          } else {
            expect(ref.filled.every(Boolean)).toBe(true)
          }
        }
      }
    })
  })
}

describe.skipIf(!hasFixture('examples.json'))('examples.json', () => {
  it('lists the seven required ids', async () => {
    const { loadExamples } = await import('./helpers')
    const ex = loadExamples()
    expect(ex).not.toBeNull()
    const ids = new Set(ex!.map((e) => e.id))
    for (const id of ['pick_and_place', 'square_xy', 'sphere_loop', 'single_lift_hold', 'array_lift_hold', 'array_arbitrary_path', 'flagship_10x10']) {
      expect(ids.has(id)).toBe(true)
    }
  })
})
