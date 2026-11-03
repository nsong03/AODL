/**
 * Loads the two tutorial examples of API.md §2 (`tutorial_hold_plain`, `tutorial_hold_shepard`)
 * from `/api/presets` (polling until the backend serves them), then plans, simulates (80 frames)
 * and fetches `/api/advanced/trajectory_efficiency` (default `EfficiencyModel`) for each, in
 * local state: two independent request / plan / sim triples.
 */
import { useEffect, useState } from 'react'
import { api, errorMessage, isAbort } from '../../api/client'
import type { ExampleSpec, PlanResponse, SimulateResponse, TrajectoryEfficiencyResponse } from '../../api/types'
import { DEFAULT_EFFICIENCY_MODEL } from '../../state/store'

export const TUTORIAL_IDS = { plain: 'tutorial_hold_plain', shepard: 'tutorial_hold_shepard' } as const
export type Side = keyof typeof TUTORIAL_IDS
export const N_FRAMES = 80
const POLL_MS = 4000

export interface SideData {
  example: ExampleSpec | null
  plan: PlanResponse | null
  sim: SimulateResponse | null
  eff: TrajectoryEfficiencyResponse | null
  error: string | null
  effError: string | null
}

const EMPTY: SideData = { example: null, plan: null, sim: null, eff: null, error: null, effError: null }

export interface TutorialData {
  presetsStatus: 'loading' | 'waiting' | 'ok' | 'error'
  presetsError: string | null
  sides: Record<Side, SideData>
}

export function useTutorialData(): TutorialData {
  const [presetsStatus, setPresetsStatus] = useState<TutorialData['presetsStatus']>('loading')
  const [presetsError, setPresetsError] = useState<string | null>(null)
  const [sides, setSides] = useState<Record<Side, SideData>>({ plain: EMPTY, shepard: EMPTY })

  useEffect(() => {
    const ctrl = new AbortController()
    let timer: ReturnType<typeof setTimeout> | null = null
    const patch = (side: Side, p: Partial<SideData>) => {
      if (!ctrl.signal.aborted) setSides((s) => ({ ...s, [side]: { ...s[side], ...p } }))
    }

    const runSide = async (side: Side, ex: ExampleSpec) => {
      patch(side, { example: ex })
      let plan: PlanResponse
      try {
        plan = await api.plan(ex.request, ctrl.signal)
        patch(side, { plan })
      } catch (e) {
        if (!isAbort(e)) patch(side, { error: `plan: ${errorMessage(e)}` })
        return
      }
      const sim = api
        .simulate({ plan: ex.request, n_frames: N_FRAMES, t_start_us: null, t_end_us: null, movie_mixing_order: null }, ctrl.signal)
        .then((s) => patch(side, { sim: s }))
        .catch((e) => !isAbort(e) && patch(side, { error: `simulate: ${errorMessage(e)}` }))
      const eff = api
        .trajectoryEfficiency({ plan: ex.request, model: DEFAULT_EFFICIENCY_MODEL, n: 300 }, ctrl.signal)
        .then((r) => patch(side, { eff: r }))
        .catch((e) => !isAbort(e) && patch(side, { effError: errorMessage(e) }))
      await Promise.all([sim, eff])
    }

    const load = async () => {
      try {
        const presets = await api.presets(ctrl.signal)
        const plain = presets.examples.find((e) => e.id === TUTORIAL_IDS.plain)
        const shepard = presets.examples.find((e) => e.id === TUTORIAL_IDS.shepard)
        if (!plain || !shepard) {
          setPresetsStatus('waiting')
          timer = setTimeout(() => void load(), POLL_MS)
          return
        }
        setPresetsStatus('ok')
        await Promise.all([runSide('plain', plain), runSide('shepard', shepard)])
      } catch (e) {
        if (isAbort(e)) return
        setPresetsStatus('error')
        setPresetsError(errorMessage(e))
        timer = setTimeout(() => void load(), POLL_MS)
      }
    }
    void load()
    return () => {
      ctrl.abort()
      if (timer) clearTimeout(timer)
    }
  }, [])

  return { presetsStatus, presetsError, sides }
}
