/**
 * Pure helpers of the fading-Shepard tutorial (design addendum F3): everything is evaluated
 * from the API responses (the waveform IR of API.md §4.1 and the spot frames of §5.1), never
 * from hard-coded numbers.
 *
 * Time bases: the IR is in *drive* time; what the atoms see at cursor time `t` is the drive at
 * the beam-centre retarded time `t − τ/2` (guide §3.4, DESIGN §1 item 3).
 */
import { CHANNELS, type ChannelName, type FadeEvent, type PlanRequest, type PlanResponse, type SimulateResponse, type WaveformIR } from '../../api/types'
import { envA } from '../../physics/envelope'
import { evalPoly } from '../../physics/piecewisePoly'
import { dedupeFades, requestedAt } from '../../physics/spots'

/** The first instant a live tone of the drive is outside its channel's usable band. */
export interface BandExit {
  /** drive time of the crossing [µs] */
  t_drive_us: number
  /** cursor time at which the beam centre sees it, `t_drive + τ/2` [µs] */
  t_atoms_us: number
  channel: ChannelName
  tone: number
  /** the band edge crossed [MHz] */
  edge_MHz: number
}

function outside(f: number, band: [number, number]): boolean {
  return f < band[0] || f > band[1]
}

/**
 * First drive time at which any tone with `A > 0` sits outside `band_MHz` (hard mask), found on
 * `n` samples over `span` (default the IR's `t_span_us`) and refined by bisection to `tol_us`.
 * Returns `null` when every live tone stays in band. The tone frequency is the rotating-frame
 * detuning of Eq. S2 plus `f_center`.
 */
export function firstBandExit(ir: WaveformIR, opts: { n?: number; span?: [number, number]; tol_us?: number } = {}): BandExit | null {
  const n = opts.n ?? 4000
  const [t0, t1] = opts.span ?? ir.t_span_us
  const tol = opts.tol_us ?? 1e-4
  let best: BandExit | null = null
  for (const ch of CHANNELS) {
    const chan = ir.channels[ch]
    if (!chan) continue
    const band = chan.band_MHz
    for (let k = 0; k < chan.tones.length; k++) {
      const tone = chan.tones[k]
      const bad = (t: number) => envA(tone.env, t) > 0 && outside(chan.f_center_MHz + evalPoly(tone.freq, t), band)
      let prev = t0
      for (let i = 0; i < n; i++) {
        const t = t0 + ((t1 - t0) * i) / (n - 1)
        if (best && t >= best.t_drive_us) break
        if (bad(t)) {
          let lo = prev
          let hi = t
          if (i === 0 || bad(lo)) {
            lo = hi
          } else {
            while (hi - lo > tol) {
              const mid = 0.5 * (lo + hi)
              if (bad(mid)) hi = mid
              else lo = mid
            }
          }
          const f = chan.f_center_MHz + evalPoly(tone.freq, hi)
          best = { t_drive_us: hi, t_atoms_us: hi + 0.5 * chan.tau_us, channel: ch, tone: k, edge_MHz: f > band[1] ? band[1] : band[0] }
          break
        }
        prev = t
      }
    }
  }
  return best
}

/**
 * Hard-band-mask in-band fraction of the live tone amplitude² at drive time `t_drive`:
 * per channel `Σ_{in band} A_n² / Σ A_n²`, and their product over the four channels (a trap
 * needs all four to diffract). `null` per channel when nothing is live.
 */
export function inBandFraction(ir: WaveformIR, t_drive: number): { product: number | null; per_channel: Record<ChannelName, number | null> } {
  const per = {} as Record<ChannelName, number | null>
  let product: number | null = 1
  for (const ch of CHANNELS) {
    const chan = ir.channels[ch]
    let tot = 0
    let inb = 0
    for (const tone of chan.tones) {
      const a = envA(tone.env, t_drive)
      if (a <= 0) continue
      const a2 = a * a
      tot += a2
      if (!outside(chan.f_center_MHz + evalPoly(tone.freq, t_drive), chan.band_MHz)) inb += a2
    }
    const v = tot > 0 ? inb / tot : null
    per[ch] = v
    product = v === null || product === null ? null : product * v
  }
  return { product, per_channel: per }
}

/**
 * Ideal (closed-form, flat-efficiency) trap depth at the requested site, per simulated frame:
 * `Σ 2P/(π w_x w_y)` over the spots within one waist `w0` of the requested lateral position at
 * `t − τ/2` (the coincident dying + rising rung pairs of a hand-over add, API.md §5.1; the
 * shadows at `±deflection_scale·Δf` are not at the site), normalised to the median over lit
 * frames. `null` before `τ/2` (both wavefronts must reach the beam first).
 */
export function trapDepthSeries(plan: PlanResponse, sim: SimulateResponse): { t: number[]; depth: (number | null)[]; plateau: number } {
  const half = plan.constants.half_tau_us
  const w0 = sim.constants.waist0_um
  const raw: (number | null)[] = sim.frames.map((f, i) => {
    const t = sim.times_us[i]
    if (t < half) return null
    const site = requestedAt(plan.trajectory, t - half)
    let s = 0
    for (let k = 0; k < f.x_um.length; k++) {
      if (Math.hypot(f.x_um[k] - site.x, f.y_um[k] - site.y) > w0) continue
      s += (2 * f.power[k]) / (Math.PI * f.wx_um[k] * f.wy_um[k])
    }
    return s
  })
  const lit = raw.filter((v): v is number => v !== null && v > 0).sort((a, b) => a - b)
  const plateau = lit.length ? lit[Math.floor(lit.length / 2)] : 1
  return { t: sim.times_us.slice(), depth: raw.map((v) => (v === null ? null : v / plateau)), plateau }
}

/** Total `hold` duration of a request [µs] (the tutorial's "hold asked for"). */
export function holdDuration(req: PlanRequest): number {
  return req.moves.reduce((a, m) => a + (m.kind === 'hold' ? m.duration_us : 0), 0)
}

/** Largest |dz| of any lift move of a request [µm]. */
export function liftHeight(req: PlanRequest): number {
  return req.moves.reduce((a, m) => Math.max(a, m.kind === 'lift' ? Math.abs(m.dz_um) : 0), 0)
}

/** Ladder spacing per axis from the IR's fade-zone envelopes (`null` for a plain drive). */
export function ladderSpacing(ir: WaveformIR): { x: number | null; y: number | null } {
  const pick = (ch: ChannelName) => {
    for (const tone of ir.channels[ch].tones) if (tone.env.kind === 'fade_zone') return tone.env.delta_f_MHz
    return null
  }
  return { x: pick('Ax') ?? pick('Bx'), y: pick('Ay') ?? pick('By') }
}

/** Hand-overs from the plan's fade timetable, deduplicated, counted per axis. */
export function handOvers(events: readonly FadeEvent[]): { events: FadeEvent[]; x: number; y: number; atStart: number } {
  const ev = dedupeFades(events)
  return { events: ev, x: ev.filter((e) => e.axis === 'x').length, y: ev.filter((e) => e.axis === 'y').length, atStart: ev.filter((e) => e.time_us <= 1e-9).length }
}

/** Sorted union of time grids with duplicates (within `eps`) merged. */
export function unionGrid(grids: readonly (readonly number[])[], eps = 1e-9): number[] {
  const all = grids.flatMap((g) => Array.from(g)).sort((a, b) => a - b)
  const out: number[] = []
  for (const v of all) if (!out.length || v - out[out.length - 1] > eps) out.push(v)
  return out
}

/** Place `(t, y)` samples on `grid` (exact matches within `eps`), `null` elsewhere. */
export function onGrid(grid: readonly number[], t: readonly number[], y: readonly (number | null)[], eps = 1e-9): (number | null)[] {
  const out: (number | null)[] = new Array(grid.length).fill(null)
  let j = 0
  for (let i = 0; i < t.length; i++) {
    while (j < grid.length && grid[j] < t[i] - eps) j++
    if (j < grid.length && Math.abs(grid[j] - t[i]) <= eps) out[j] = y[i]
  }
  return out
}

/** `|∫Z dt|` in µm·µs from the API's m·s. */
export const mSToUmUs = (v: number): number => v / 1e-12
