/**
 * Two tabs: the generated Python (copy / download .py / download the parametric .npz) and the
 * "waveform recipe" — every tone of the IR tabulated (channel, rung, frequency range over the
 * live span, segment count, envelope kind + parameters, phase0): the fade-in / fade-out design
 * made explicit.
 */
import { useMemo, useState } from 'react'
import { api, errorMessage } from '../../api/client'
import { CHANNELS, type ChannelName, type Envelope, type PlanResponse } from '../../api/types'
import { CHANNEL_COLORS } from '../../channels'
import { fmt } from '../../physics/format'
import { polyRangeSampled } from '../../physics/piecewisePoly'
import { useStudio } from '../../state/store'
import { liveTracks } from '../../physics/waveform'
import { Panel, Skeleton } from '../common/Panel'

interface RecipeRow {
  channel: ChannelName
  index: number
  fMin: number | null
  fMax: number | null
  liveFrom: number | null
  liveTo: number | null
  segments: number
  degree: number
  env: Envelope
  phase0: number
}

function envSummary(env: Envelope): string {
  switch (env.kind) {
    case 'constant':
      return `constant · amp ${fmt(env.amp, 2)}`
    case 'smooth_on_off':
      return `gate · on ${fmt(env.t_on_us, 1)} µs · off ${fmt(env.t_off_us, 1)} µs · ramp ${fmt(env.ramp_us, 1)} µs`
    case 'fade_zone':
      return `fade zone · Δf ${fmt(env.delta_f_MHz, 2)} MHz · η ${fmt(env.eta, 2)} · p ${fmt(env.p, 2)} · M ${env.m}${env.amp !== 1 ? ` · amp ${fmt(env.amp, 2)}` : ''}`
  }
}

function buildRecipe(plan: PlanResponse): RecipeRow[] {
  const rows: RecipeRow[] = []
  for (const ch of CHANNELS) {
    const chan = plan.waveform_ir.channels[ch]
    const tracks = liveTracks(chan, 800, plan.t_span_us)
    chan.tones.forEach((tone, i) => {
      const tr = tracks[i]
      let fMin: number | null = null
      let fMax: number | null = null
      if (tr.live) {
        fMin = tr.f_live_min
        fMax = tr.f_live_max
      } else if (tone.env.kind === 'constant' && tone.env.amp > 0) {
        const [a, b] = polyRangeSampled(tone.freq, plan.t_span_us[0], plan.t_span_us[1])
        fMin = chan.f_center_MHz + a
        fMax = chan.f_center_MHz + b
      }
      rows.push({
        channel: ch,
        index: i,
        fMin,
        fMax,
        liveFrom: tr.live ? tr.t_live_start : null,
        liveTo: tr.live ? tr.t_live_end : null,
        segments: tone.freq.coeffs.length,
        degree: (tone.freq.coeffs[0]?.length ?? 1) - 1,
        env: tone.env,
        phase0: tone.phase0,
      })
    })
  }
  return rows
}

type SortKey = 'channel' | 'index' | 'fMin' | 'liveFrom' | 'segments' | 'phase0'

export function CodePanel() {
  const plan = useStudio((s) => s.plan)
  const status = useStudio((s) => s.planStatus)
  const [tab, setTab] = useState<'python' | 'recipe'>('python')
  const [msg, setMsg] = useState<string | null>(null)
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'channel', dir: 1 })
  const [filter, setFilter] = useState<ChannelName | 'all'>('all')
  const [liveOnly, setLiveOnly] = useState(false)
  const recipe = useMemo(() => (plan ? buildRecipe(plan) : []), [plan])

  if (!plan) {
    return (
      <Panel title="Code">
        {status === 'loading' ? <Skeleton height={120} label="planning…" /> : <div className="caption">No plan yet.</div>}
      </Panel>
    )
  }

  const download = (name: string, blob: Blob) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(plan.code_python)
      setMsg('copied to the clipboard')
    } catch {
      setMsg('clipboard unavailable — select the text and copy')
    }
    setTimeout(() => setMsg(null), 2500)
  }

  const npz = async () => {
    setMsg('fetching the parametric NPZ…')
    try {
      const blob = await api.waveformNpz(plan.plan_id)
      download(`aodl_${plan.plan_id}.npz`, blob)
      setMsg(`aodl_${plan.plan_id}.npz (${(blob.size / 1024).toFixed(1)} kB, segment parameters, never samples)`)
    } catch (e) {
      setMsg(`NPZ refused: ${errorMessage(e)}`)
    }
  }

  const sorted = recipe
    .filter((r) => filter === 'all' || r.channel === filter)
    .filter((r) => !liveOnly || r.liveFrom !== null)
    .sort((a, b) => {
      const k = sort.key
      let d = 0
      if (k === 'channel') d = CHANNELS.indexOf(a.channel) - CHANNELS.indexOf(b.channel) || a.index - b.index
      else {
        const va = a[k] ?? Infinity
        const vb = b[k] ?? Infinity
        d = (va as number) - (vb as number)
      }
      return d * sort.dir
    })
  const th = (key: SortKey, label: string, num = false) => (
    <th className={num ? 'num' : ''} onClick={() => setSort((s) => ({ key, dir: s.key === key ? ((-s.dir) as 1 | -1) : 1 }))}>
      {label} {sort.key === key ? (sort.dir > 0 ? '▲' : '▼') : ''}
    </th>
  )
  const live = recipe.filter((r) => r.liveFrom !== null).length

  return (
    <Panel
      title="Code"
      sub={tab === 'python' ? 'runnable with plain aodl; the synthesis options are resolved explicitly' : `${recipe.length} tones, ${live} ever live · the fade-in / fade-out design made explicit`}
      actions={
        tab === 'python' ? (
          <>
            <button className="small" onClick={copy}>
              copy
            </button>
            <button className="small" onClick={() => download(`aodl_${plan.plan_id}.py`, new Blob([plan.code_python], { type: 'text/x-python' }))}>
              download .py
            </button>
            <button className="small" onClick={npz} title="GET /api/plan/{plan_id}/waveform.npz — WaveformSet.save (400 unsupported for a Shepard drive with amp ≠ 1)">
              download .npz
            </button>
          </>
        ) : (
          <>
            <select value={filter} onChange={(e) => setFilter(e.target.value as ChannelName | 'all')} style={{ width: 'auto' }}>
              <option value="all">all channels</option>
              {CHANNELS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <label className="check" style={{ margin: 0 }}>
              <input type="checkbox" checked={liveOnly} onChange={(e) => setLiveOnly(e.target.checked)} /> live only
            </label>
          </>
        )
      }
    >
      <div className="tabs">
        <button className={tab === 'python' ? 'active' : ''} onClick={() => setTab('python')}>
          Python
        </button>
        <button className={tab === 'recipe' ? 'active' : ''} onClick={() => setTab('recipe')}>
          Waveform recipe
        </button>
      </div>
      {msg && <div className="caption" style={{ marginBottom: 6, color: msg.startsWith('NPZ refused') ? 'var(--warn)' : undefined }}>{msg}</div>}
      {tab === 'python' ? (
        <pre className="subpanel" style={{ maxHeight: 420, overflow: 'auto' }}>
          {plan.code_python}
        </pre>
      ) : (
        <div style={{ overflow: 'auto', maxHeight: 420 }}>
          <table className="data">
            <thead>
              <tr>
                {th('channel', 'channel')}
                {th('index', 'tone', true)}
                {th('fMin', 'f range live [MHz]', true)}
                {th('liveFrom', 'live span [µs]', true)}
                {th('segments', 'segments', true)}
                <th>envelope</th>
                {th('phase0', 'phase0 [rad]', true)}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={`${r.channel}-${r.index}`} style={{ opacity: r.liveFrom === null && r.env.kind !== 'constant' ? 0.5 : 1 }}>
                  <td style={{ color: CHANNEL_COLORS[r.channel], fontWeight: 600 }}>{r.channel}</td>
                  <td className="num">{r.index}</td>
                  <td className="num">{r.fMin === null ? 'never live' : `${fmt(r.fMin, 3)} – ${fmt(r.fMax, 3)}`}</td>
                  <td className="num">{r.liveFrom === null ? '—' : `${fmt(r.liveFrom, 1)} – ${fmt(r.liveTo, 1)}`}</td>
                  <td className="num">
                    {r.segments} · deg {r.degree}
                  </td>
                  <td>{envSummary(r.env)}</td>
                  <td className="num">{fmt(r.phase0, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="caption" style={{ marginTop: 6 }}>
            Frequencies are absolute (f_center + detuning); a fade-zone rung is live while its fade coordinate |g| &lt; (M + η)Δf/2 and at full amplitude
            inside (M − η)Δf/2 (Eqs. S26/S27). Rungs that never come alive over the programmed span are listed greyed.
          </div>
        </div>
      )}
    </Panel>
  )
}
