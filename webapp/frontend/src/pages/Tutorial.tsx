/**
 * `/tutorial/shepard` — the same held tweezer with and without fading-Shepard tones (design
 * addendum F3). Two independent request / plan / sim triples in page-local state, one shared
 * cursor and time bar. Every number on the page comes from the API responses.
 */
import { useEffect, useMemo } from 'react'
import { CHANNELS, type ExampleSpec, type PlanResponse } from '../api/types'
import { ErrorBanner, Panel, Skeleton } from '../components/common/Panel'
import { viewScale } from '../components/tweezer/view'
import { useTutorialCursor } from '../components/tutorial/cursor'
import { derive, type Derived } from '../components/tutorial/derive'
import { TrapPowerChart } from '../components/tutorial/TrapPowerChart'
import { TRAP_COLORS } from '../components/tutorial/trapSeries'
import { TutorialPlayback } from '../components/tutorial/TutorialPlayback'
import { TutorialRf } from '../components/tutorial/TutorialRf'
import { TutorialTweezer } from '../components/tutorial/TutorialTweezer'
import { holdDuration, liftHeight, mSToUmUs } from '../components/tutorial/tutorialMath'
import { N_FRAMES, TUTORIAL_IDS, useTutorialData, type Side, type SideData } from '../components/tutorial/useTutorialData'
import '../components/tutorial/tutorial.css'
import { fmt, fmtSigned } from '../physics/format'

const SIDES: Side[] = ['plain', 'shepard']
const HEAD: Record<Side, string> = { plain: 'without fading-Shepard', shepard: 'with fading-Shepard' }

export default function Tutorial() {
  const { presetsStatus, presetsError, sides } = useTutorialData()
  const derived = useMemo(() => {
    const out = {} as Record<Side, Derived | null>
    for (const s of SIDES) out[s] = sides[s].plan ? derive(sides[s].plan!, sides[s].sim, sides[s].eff) : null
    return out
  }, [sides])

  const plainPlan = sides.plain.plan
  const shepPlan = sides.shepard.plan
  const anyPlan = plainPlan ?? shepPlan

  // one speed for the shared cursor: the trajectory in 8 s
  useEffect(() => {
    if (anyPlan) useTutorialCursor.getState().setSpeed(anyPlan.duration_us / 8)
  }, [anyPlan])
  // start the cursor on the Shepard drive's first x hand-over as the atoms see it (shadows visible)
  const firstX = dShepFirstX(derived.shepard, shepPlan)
  useEffect(() => {
    if (firstX !== null && useTutorialCursor.getState().t_us === 0) useTutorialCursor.getState().setTime(firstX)
  }, [firstX])
  // stop playback when leaving the page
  useEffect(() => () => useTutorialCursor.getState().pause(), [])

  // shared RF frequency axis: both drives' live excursions (band_usage) and the band
  const ylim = useMemo<[number, number]>(() => {
    let lo = Infinity
    let hi = -Infinity
    for (const p of [plainPlan, shepPlan]) {
      if (!p) continue
      for (const ch of CHANNELS) {
        lo = Math.min(lo, p.band_usage[ch].f_min_MHz, p.waveform_ir.channels[ch].band_MHz[0])
        hi = Math.max(hi, p.band_usage[ch].f_max_MHz, p.waveform_ir.channels[ch].band_MHz[1])
      }
    }
    if (!Number.isFinite(lo)) return [0, 1]
    const m = 0.05 * (hi - lo)
    return [lo - m, hi + m]
  }, [plainPlan, shepPlan])

  // shared movie scale: both columns on one brightness scale
  const scales = useMemo(() => {
    const s = { plain: sides.plain.sim ? viewScale(sides.plain.sim) : null, shepard: sides.shepard.sim ? viewScale(sides.shepard.sim) : null }
    const peak = Math.max(s.plain?.peakTracked ?? 0, s.shepard?.peakTracked ?? 0)
    return { ...s, peak }
  }, [sides.plain.sim, sides.shepard.sim])

  const dPlain = derived.plain
  const dShep = derived.shepard

  return (
    <div className="page tutorial">
      <h2 style={{ marginBottom: 6 }}>Tutorial · holding a lifted tweezer with and without fading-Shepard tones</h2>
      {presetsStatus === 'waiting' && (
        <ErrorBanner info code="waiting" message={`The backend does not serve the tutorial examples yet (${TUTORIAL_IDS.plain}, ${TUTORIAL_IDS.shepard} in /api/presets); retrying every few seconds.`} />
      )}
      {presetsStatus === 'error' && presetsError && <ErrorBanner code="presets" message={`${presetsError} — retrying.`} />}

      <Intro plain={sides.plain} shepard={sides.shepard} dPlain={dPlain} />

      {anyPlan && (
        <div className="tut-sticky">
          <Panel className="tight">
            <TutorialPlayback plan={anyPlan} sim={sides.shepard.sim ?? sides.plain.sim} fades={dShep?.fades ?? []} exit={dPlain?.exit ?? null} />
          </Panel>
        </div>
      )}

      <Pair
        title="RF drives"
        sub="f_center + f_n(t) [MHz] vs drive time · opacity = envelope A_n · red shading = outside the usable band · amber ticks = hand-overs · dashed white = beam centre t − τ/2"
        render={(s) => {
          const plan = sides[s].plan
          const d = derived[s]
          if (!plan || !d) return <Skeleton height={340} label="planning…" />
          return (
            <>
              <TutorialRf plan={plan} ylim={ylim} exit={d.exit} fades={d.fades} />
              <div className="caption" style={{ marginTop: 6 }}>
                {d.exit ? (
                  <>
                    <strong style={{ color: '#ef6f6c' }}>
                      Leaves the band at t = {fmt(d.exit.t_atoms_us, 1)} µs
                    </strong>{' '}
                    at the beam centre: {d.exit.channel} tone {d.exit.tone} crosses {d.exit.edge_MHz} MHz at drive time {fmt(d.exit.t_drive_us, 1)} µs (+ τ/2 ={' '}
                    {fmt(plan.constants.half_tau_us, 2)} µs). Every channel co-chirps, peaking at {fmt(d.minMargin.f_max, 1)} MHz.
                  </>
                ) : (
                  <>
                    No live tone leaves the band: {plan.tone_counts.Ax} rungs per channel spaced Δf = {fmt(d.spacing.x, 1)} / {fmt(d.spacing.y, 1)} MHz (x / y) fade in and out
                    between {fmt(d.minMargin.f_min, 1)} and {fmt(d.minMargin.f_max, 1)} MHz; {d.hand.events.length} hand-overs (amber).
                  </>
                )}
              </div>
            </>
          )
        }}
      />

      <Pair
        title="Trap power vs time"
        sub={
          <span className="tut-legend">
            <span>
              <span className="sw" style={{ background: TRAP_COLORS.ideal }} />
              ideal trap depth, closed form
            </span>
            <span>
              <span className="sw" style={{ background: TRAP_COLORS.inBand }} />
              in-band fraction (hard mask)
            </span>
            <span>
              <span className="sw" style={{ background: TRAP_COLORS.eff }} />
              η-weighted power (model)
            </span>
          </span>
        }
        render={(s) => {
          const d = derived[s]
          const plan = sides[s].plan
          if (!plan || !d?.trap) return sides[s].error ? <ErrorBanner code="simulate" message={sides[s].error!} /> : <Skeleton height={230} label="simulating…" />
          return (
            <>
              <TrapPowerChart plan={plan} series={d.trap} fades={d.fades} exit={d.exit} />
              <div className="caption" style={{ marginTop: 4 }}>
                {s === 'plain' ? (
                  <>
                    The ideal trap depth stays at {fmt(d.idealMin, 2)}–1 throughout: <strong>the closed-form engine assumes flat diffraction efficiency</strong>, so it keeps
                    diffracting the plain drive's tones at up to {fmt(d.minMargin.f_max, 1)} MHz and the out-of-band trap stays bright. A real AOD does not: the hard in-band
                    fraction drops to {fmt(d.inBandAfterExit, 2)} and the model's η-weighted power to {fmt(d.effAfterExit, 3)} after the band exit (red dashed).
                  </>
                ) : (
                  <>
                    Through the {d.hand.events.length} hand-overs (amber, at drive time + τ/2) the ideal trap depth stays ≥ {fmt(d.idealMin, 2)} of its plateau and every live tone
                    stays in band (in-band fraction 1); the η-weighted power dips where the live rungs sit away from the TPM frequency but never collapses.
                  </>
                )}
                {sides[s].eff && (
                  <>
                    {' '}
                    η-weighted curve: <code>/api/advanced/trajectory_efficiency</code>, normalised to its maximum over the drive (raw η-product max {fmt(d.trap.effMax, 3)}, min{' '}
                    {fmt(sides[s].eff!.min_product, 3)}).
                  </>
                )}
                {sides[s].effError && <> · efficiency model unavailable: {sides[s].effError}</>}
              </div>
            </>
          )
        }}
      />
      <EngineNote eff={sides.shepard.eff?.assumptions ?? sides.plain.eff?.assumptions ?? null} fMax={dPlain?.minMargin.f_max ?? null} fCenter={plainPlan?.waveform_ir.channels.Ax.f_center_MHz ?? null} />

      <Pair
        title="Movies"
        sub={`XY at the tracked focus + XZ slice · astigmatic Gaussian approximation of /api/simulate (${N_FRAMES} frames) · both columns on one brightness scale`}
        render={(s) => {
          const sim = sides[s].sim
          const plan = sides[s].plan
          const sc = scales[s]
          if (!plan || !sim || !sc) return sides[s].error ? <ErrorBanner code="simulate" message={sides[s].error!} /> : <Skeleton height={300} label="simulating…" />
          return <TutorialTweezer plan={plan} sim={sim} scale={sc} peak={scales.peak} />
        }}
      />

      <Numbers plain={sides.plain} shepard={sides.shepard} derived={derived} />
      <Takeaways plain={sides.plain.plan} shepard={sides.shepard.plan} dPlain={dPlain} dShep={dShep} />
    </div>
  )
}

function dShepFirstX(d: Derived | null, plan: PlanResponse | null): number | null {
  if (!d || !plan) return null
  const e = d.fades.find((f) => f.axis === 'x' && f.time_us > 0)
  return e ? e.time_us + plan.constants.half_tau_us : null
}

function Pair({ title, sub, render }: { title: string; sub?: React.ReactNode; render: (s: Side) => React.ReactNode }) {
  return (
    <div className="tut-pair">
      {SIDES.map((s) => (
        <Panel
          key={s}
          title={
            <span className="tut-col-head">
              {title} <span className={`badge ${s === 'plain' ? 'danger' : 'ok'}`}>{HEAD[s]}</span>
            </span>
          }
          sub={sub}
        >
          {render(s)}
        </Panel>
      ))}
    </div>
  )
}

function Intro({ plain, shepard, dPlain }: { plain: SideData; shepard: SideData; dPlain: Derived | null }) {
  const p = plain.plan ?? shepard.plan
  const req = (plain.example ?? shepard.example)?.request
  return (
    <Panel className="tut-intro" title="Why a long hold needs fading-Shepard ladders">
      <p>
        Holding a tweezer off the focal plane costs a co-chirp on all four channels: the drive frequency walks as <code>f_Z(t) ∝ ∫Z dt</code> (Eq. S19), so a single
        tone held long enough must leave the AOD's band, and outside its band a real AOD stops diffracting. Fading-Shepard ladders (Eqs. S24–S28) replace each tone by
        rungs spaced Δf that fade in and out and hand the chirp from rung to rung, so the <em>live</em> tones never leave the band however long the hold.
      </p>
      {p && req ? (
        <p>
          <strong>The Eq. 1 budget:</strong> at this hardware a plain drive can hold <strong>{fmt(p.constants.eq1_hold_10um_us, 0)} µs at 10 µm</strong>; this
          trajectory lifts {fmt(liftHeight(req), 0)} µm and asks for a <strong>{fmt(holdDuration(req), 0)} µs</strong> hold — <code>|∫Z dt|</code> ={' '}
          {fmt(mSToUmUs(p.z_budget.requested_m_s), 0)} µm·µs, <code>z_budget.fraction</code> = <strong>{fmt(p.z_budget.fraction, 2)}</strong> of the plain ceiling.
          The left column forces the plain Eq. S19 drive anyway (<code>shepard = "never"</code>, <code>check_band = false</code>)
          {dPlain?.exit ? `, which leaves the band at t = ${fmt(dPlain.exit.t_atoms_us, 1)} µs` : ''}; the right column is the same request at default options, which{' '}
          <code>plan_motion</code> plans as <code>{shepard.plan?.mode ?? '…'}</code>.
        </p>
      ) : (
        <Skeleton height={40} label="loading the tutorial examples…" />
      )}
      {(plain.error || shepard.error) && <ErrorBanner code="error" message={[plain.error, shepard.error].filter(Boolean).join('\n')} />}
    </Panel>
  )
}

function EngineNote({ eff, fMax, fCenter }: { eff: string | null; fMax: number | null; fCenter: number | null }) {
  return (
    <div className="caption" style={{ marginTop: 8 }}>
      <strong>Why the two engines disagree out of band:</strong> the closed-form engine (<code>aodl.simulate</code>, Eq. S11) treats diffraction efficiency as flat
      across all frequencies, so it lights the plain drive's {fmt(fMax, 1)} MHz tone as brightly as one at the {fmt(fCenter, 0)} MHz centre; the in-band fraction applies the band as a hard mask, and the η-weighted curve
      is a <em>model</em> of the Bragg / tangential-phase-matching roll-off, not a measurement.
      {eff && (
        <details style={{ marginTop: 2 }}>
          <summary style={{ cursor: 'pointer' }}>efficiency model assumptions (from the API)</summary>
          <div style={{ marginTop: 2 }}>{eff}</div>
        </details>
      )}
    </div>
  )
}

function Numbers({ plain, shepard, derived }: { plain: SideData; shepard: SideData; derived: Record<Side, Derived | null> }) {
  const data: Record<Side, SideData> = { plain, shepard }
  type Row = { label: React.ReactNode; cell: (p: PlanResponse, d: Derived, s: Side) => React.ReactNode }
  const rows: Row[] = [
    { label: 'drive (plan.mode)', cell: (p) => <code>{p.mode}</code> },
    { label: 'tones per channel (Ax / Bx / Ay / By)', cell: (p) => CHANNELS.map((c) => p.tone_counts[c]).join(' / ') },
    {
      label: 'band margin (worst channel)',
      cell: (_p, d) => (
        <span className={d.minMargin.margin < 0 ? 'neg' : 'pos'}>
          {fmtSigned(d.minMargin.margin, 2)} MHz ({d.minMargin.ch}: {fmt(d.minMargin.f_min, 2)}–{fmt(d.minMargin.f_max, 2)} MHz)
        </span>
      ),
    },
    { label: 'first live tone out of band', cell: (_p, d) => (d.exit ? `t = ${fmt(d.exit.t_atoms_us, 1)} µs (drive ${fmt(d.exit.t_drive_us, 1)} µs, ${d.exit.channel})` : 'never') },
    {
      label: 'hand-overs (fade_events)',
      cell: (_p, d) =>
        d.hand.events.length === 0 ? '0' : `${d.hand.events.length} (x ${d.hand.x}, y ${d.hand.y}${d.hand.atStart ? `; ${d.hand.atStart} at t = 0` : ''})`,
    },
    {
      label: 'shadow offset ±(µm/MHz)·Δf',
      cell: (p, d) => {
        if (d.spacing.x === null && d.spacing.y === null) return 'no shadows (no ladder)'
        const k = p.constants.um_per_MHz
        const parts: string[] = []
        if (d.spacing.x !== null) parts.push(`x: ±${fmt(k, 1)} × ${fmt(d.spacing.x, 1)} = ±${fmt(k * d.spacing.x, 1)} µm`)
        if (d.spacing.y !== null) parts.push(`y: ±${fmt(k, 1)} × ${fmt(d.spacing.y, 1)} = ±${fmt(k * d.spacing.y, 1)} µm`)
        const rep = Array.from(new Set(d.hand.events.map((e) => fmt(e.shadow_um, 1))))
        return `${parts.join(' · ')}${rep.length ? ` (fade_events: ±${rep.join(' / ±')} µm)` : ''}${d.shadowMax ? ` · brightest shadow P/P_ref ${fmt(d.shadowMax.rel, 2)}` : ''}`
      },
    },
    {
      label: '|∫Z dt| requested vs ceiling',
      cell: (p) => {
        const z = p.z_budget
        const ceil = z.ceiling_m_s === null ? '∞ (Shepard: bounded live excursion)' : `${fmt(mSToUmUs(z.ceiling_m_s), 0)} µm·µs`
        return `${fmt(mSToUmUs(z.requested_m_s), 0)} µm·µs vs ${ceil} · ${fmt(z.fraction, 2)} × the plain Eq. 1 ceiling`
      },
    },
    { label: 'ideal trap depth, min (t ≥ τ/2)', cell: (_p, d) => fmt(d.idealMin, 3) },
    { label: 'η-product min (model)', cell: (_p, _d, s) => (data[s].eff ? fmt(data[s].eff!.min_product, 3) : '—') },
  ]
  return (
    <div style={{ marginTop: 12 }}>
      <Panel title="Numbers" sub="all from /api/plan, /api/simulate and /api/advanced/trajectory_efficiency">
        <div style={{ overflowX: 'auto' }}>
          <table className="tut-numbers">
            <thead>
              <tr>
                <th />
                {SIDES.map((s) => (
                  <th key={s}>
                    {HEAD[s]} <span className="faint">({exampleTitle(data[s].example)})</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <th>{r.label}</th>
                  {SIDES.map((s) => {
                    const p = data[s].plan
                    const d = derived[s]
                    return (
                      <td key={s} className="num">
                        {p && d ? r.cell(p, d, s) : '…'}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  )
}

function exampleTitle(ex: ExampleSpec | null): string {
  return ex ? ex.id : '…'
}

function Takeaways({ plain, shepard, dPlain, dShep }: { plain: PlanResponse | null; shepard: PlanResponse | null; dPlain: Derived | null; dShep: Derived | null }) {
  if (!plain || !shepard || !dPlain || !dShep) return null
  const shadow = dShep.hand.events[0]?.shadow_um
  return (
    <div style={{ marginTop: 12 }}>
      <Panel title="What to take away">
        <ul className="takeaways">
          <li>
            <strong>Band budget.</strong> A sustained Z offset is paid for in band: the plain drive needs {fmt(plain.z_budget.fraction, 2)}× the Eq. 1 ceiling, ends{' '}
            {fmt(-dPlain.minMargin.margin, 1)} MHz past the band edge and leaves the band at t = {dPlain.exit ? fmt(dPlain.exit.t_atoms_us, 1) : '—'} µs; the Shepard drive
            keeps a {fmtSigned(dShep.minMargin.margin, 1)} MHz margin with {shepard.tone_counts.Ax} rungs per channel.
          </li>
          <li>
            <strong>Hand-overs keep the summed power constant.</strong> During each of the {dShep.hand.events.length} hand-overs the dying and rising rungs share the trap
            with <code>p_A + p_B = 1</code>, so the ideal trap depth stays ≥ {fmt(dShep.idealMin, 2)} of its plateau.
          </li>
          <li>
            <strong>Shadows exist only while a pair hands over.</strong> Mid-fade the same lines make two shadow tweezers at ±{fmt(shadow, 1)} µm
            {dShep.shadowMax ? ` (up to ${fmt(dShep.shadowMax.rel, 2)} of a trap's power)` : ''}; schedule pick-ups on the plateaus between the hand-overs of{' '}
            <code>fade_events</code>, never during one.
          </li>
        </ul>
      </Panel>
    </div>
  )
}
