/**
 * Application state (zustand). The editable `PlanRequest` lives here together with the last
 * plan / simulate responses, the global time cursor and the playback settings. Network calls
 * are issued from the actions below: `requestPlan` is debounced 400 ms after any edit and
 * aborts an in-flight request; a successful plan triggers `requestSim`.
 */
import { create } from 'zustand'
import { api, errorMessage, isAbort } from '../api/client'
import type {
  AdvancedSimulateResponse,
  ArraySpec,
  EfficiencyModel,
  EfficiencyResponse,
  FrameResponse,
  HardwareSpec,
  Misalignment,
  MoveKind,
  MoveSpec,
  PlanOptions,
  PlanRequest,
  PlanResponse,
  PresetsResponse,
  SimulateResponse,
  SpectrumResponse,
  TrajectoryEfficiencyResponse,
} from '../api/types'
import { spotRow, spotsAt } from '../physics/spots'
import { DEFAULT_ARRAY, DEFAULT_HARDWARE, DEFAULT_MOVES, DEFAULT_OPTIONS, newMove } from './defaults'

export type Status = 'idle' | 'loading' | 'ok' | 'error'
export type ViewMode = 'tracked' | 'fixed'

export const DEFAULT_EFFICIENCY_MODEL: EfficiencyModel = {
  interaction_length_mm: 4.0,
  f_tpm_MHz: null,
  refractive_index: 2.26,
  peak_efficiency: 0.85,
  angular_coupling: 0.2,
}

export interface AdvancedState {
  efficiencyModel: EfficiencyModel
  efficiency: EfficiencyResponse | null
  efficiencyStatus: Status
  efficiencyError: string | null
  trajEff: TrajectoryEfficiencyResponse | null
  trajEffStatus: Status
  misalignment: Misalignment
  advSim: AdvancedSimulateResponse | null
  advSimStatus: Status
  advSimError: string | null
  spectrum: SpectrumResponse | null
  spectrumStatus: Status
  spectrumError: string | null
}

export interface StudioState {
  // editable request
  hardware: HardwareSpec
  array: ArraySpec
  moves: MoveSpec[]
  options: PlanOptions
  hardwarePreset: string
  presets: PresetsResponse | null
  presetsStatus: Status
  // responses
  plan: PlanResponse | null
  planStatus: Status
  planError: string | null
  planErrorCode: string | null
  sim: SimulateResponse | null
  simStatus: Status
  simError: string | null
  // cursor & playback
  t_us: number
  playing: boolean
  speed_us_per_s: number
  speedTouched: boolean
  viewMode: ViewMode
  showExact: boolean
  exactFrame: FrameResponse | null
  exactStatus: Status
  exactError: string | null
  advanced: AdvancedState

  // actions
  loadPresets: () => Promise<PresetsResponse | null>
  loadExample: (id: string) => Promise<boolean>
  setRequest: (req: PlanRequest) => void
  setHardware: (patch: Partial<HardwareSpec>) => void
  setHardwarePreset: (name: string) => void
  setArray: (patch: Partial<ArraySpec>) => void
  setOptions: (patch: Partial<PlanOptions>) => void
  setMove: (i: number, patch: Partial<MoveSpec>) => void
  replaceMove: (i: number, move: MoveSpec) => void
  addMove: (kind: MoveKind, move?: MoveSpec) => void
  removeMove: (i: number) => void
  reorderMove: (from: number, to: number) => void
  schedulePlan: () => void
  requestPlan: () => Promise<void>
  requestSim: () => Promise<void>
  fetchExact: (t_us: number) => Promise<void>
  setShowExact: (on: boolean) => void
  setTime: (t_us: number) => void
  stepFrame: (dir: 1 | -1) => void
  play: () => void
  pause: () => void
  togglePlay: () => void
  setSpeed: (v: number) => void
  setViewMode: (m: ViewMode) => void
  setAdvanced: (patch: Partial<AdvancedState>) => void
  setEfficiencyModel: (patch: Partial<EfficiencyModel>) => void
  setMisalignment: (m: Misalignment) => void
}

let planTimer: ReturnType<typeof setTimeout> | null = null
let planAbort: AbortController | null = null
let simAbort: AbortController | null = null
let exactAbort: AbortController | null = null

export const PLAN_DEBOUNCE_MS = 400

export function currentRequest(s: Pick<StudioState, 'hardware' | 'array' | 'moves' | 'options'>): PlanRequest {
  return { hardware: s.hardware, array: s.array, moves: s.moves, options: s.options }
}

/** Frames for a simulate call: 80, or 48 for more than 36 traps (W3 §3). */
export function simFrameCount(array: ArraySpec): number {
  return array.mx * array.my > 36 ? 48 : 80
}

export const useStudio = create<StudioState>()((set, get) => {
  const edited = () => {
    get().schedulePlan()
  }
  return {
    hardware: DEFAULT_HARDWARE,
    array: DEFAULT_ARRAY,
    moves: DEFAULT_MOVES,
    options: DEFAULT_OPTIONS,
    hardwarePreset: 'default_1030',
    presets: null,
    presetsStatus: 'idle',
    plan: null,
    planStatus: 'idle',
    planError: null,
    planErrorCode: null,
    sim: null,
    simStatus: 'idle',
    simError: null,
    t_us: -1,
    playing: false,
    speed_us_per_s: 140 / 8,
    speedTouched: false,
    viewMode: 'tracked',
    showExact: false,
    exactFrame: null,
    exactStatus: 'idle',
    exactError: null,
    advanced: {
      efficiencyModel: DEFAULT_EFFICIENCY_MODEL,
      efficiency: null,
      efficiencyStatus: 'idle',
      efficiencyError: null,
      trajEff: null,
      trajEffStatus: 'idle',
      misalignment: {},
      advSim: null,
      advSimStatus: 'idle',
      advSimError: null,
      spectrum: null,
      spectrumStatus: 'idle',
      spectrumError: null,
    },

    async loadPresets() {
      const have = get().presets
      if (have) return have
      if (get().presetsStatus === 'loading') return null
      set({ presetsStatus: 'loading' })
      try {
        const p = await api.presets()
        set({ presets: p, presetsStatus: 'ok' })
        return p
      } catch {
        set({ presetsStatus: 'error' })
        return null
      }
    },

    async loadExample(id) {
      const presets = get().presets ?? (await get().loadPresets())
      const ex = presets?.examples.find((e) => e.id === id)
      if (!ex) return false
      get().setRequest(ex.request)
      return true
    },

    setRequest(req) {
      const presets = get().presets
      let preset = 'custom'
      if (presets) {
        for (const [name, hw] of Object.entries(presets.hardware_presets)) {
          if (JSON.stringify(hw) === JSON.stringify(req.hardware)) preset = name
        }
      } else if (JSON.stringify(req.hardware) === JSON.stringify(DEFAULT_HARDWARE)) preset = 'default_1030'
      set({
        hardware: { ...req.hardware },
        array: { ...req.array },
        moves: req.moves.map((m) => ({ ...m })),
        options: { ...DEFAULT_OPTIONS, ...req.options },
        hardwarePreset: preset,
        t_us: -1, // "unset": requestPlan places the cursor at τ (aperture full) once the plan is in
        playing: false,
      })
      if (planTimer) clearTimeout(planTimer)
      planTimer = null
      void get().requestPlan()
    },

    setHardware(patch) {
      set((s) => ({ hardware: { ...s.hardware, ...patch }, hardwarePreset: 'custom' }))
      edited()
    },
    setHardwarePreset(name) {
      const presets = get().presets
      const hw = presets?.hardware_presets[name]
      if (!hw) return
      set({ hardware: { ...hw }, hardwarePreset: name })
      edited()
    },
    setArray(patch) {
      set((s) => ({ array: { ...s.array, ...patch } }))
      edited()
    },
    setOptions(patch) {
      set((s) => ({ options: { ...s.options, ...patch } }))
      edited()
    },
    setMove(i, patch) {
      set((s) => ({ moves: s.moves.map((m, k) => (k === i ? ({ ...m, ...patch } as MoveSpec) : m)) }))
      edited()
    },
    replaceMove(i, move) {
      set((s) => ({ moves: s.moves.map((m, k) => (k === i ? move : m)) }))
      edited()
    },
    addMove(kind, move) {
      set((s) => ({ moves: [...s.moves, move ?? newMove(kind)] }))
      edited()
    },
    removeMove(i) {
      set((s) => ({ moves: s.moves.filter((_, k) => k !== i) }))
      edited()
    },
    reorderMove(from, to) {
      set((s) => {
        if (from === to || from < 0 || to < 0 || from >= s.moves.length || to >= s.moves.length) return {}
        const moves = s.moves.slice()
        const [m] = moves.splice(from, 1)
        moves.splice(to, 0, m)
        return { moves }
      })
      edited()
    },

    schedulePlan() {
      if (planTimer) clearTimeout(planTimer)
      planTimer = setTimeout(() => {
        planTimer = null
        void get().requestPlan()
      }, PLAN_DEBOUNCE_MS)
    },

    async requestPlan() {
      if (planAbort) planAbort.abort()
      if (simAbort) simAbort.abort()
      if (exactAbort) exactAbort.abort()
      const ctrl = new AbortController()
      planAbort = ctrl
      const req = currentRequest(get())
      if (req.moves.length === 0) {
        set({ planStatus: 'error', planError: 'add at least one move', planErrorCode: 'invalid_spec' })
        return
      }
      set({ planStatus: 'loading' })
      try {
        const plan = await api.plan(req, ctrl.signal)
        if (ctrl.signal.aborted) return
        const s = get()
        const speed = s.speedTouched ? s.speed_us_per_s : plan.duration_us / 8
        const tMax = plan.t_span_us[1]
        const tStart = s.t_us < 0 ? Math.min(plan.constants.tau_us, tMax) : Math.min(Math.max(s.t_us, plan.t_span_us[0]), tMax)
        set({
          plan,
          planStatus: 'ok',
          planError: null,
          planErrorCode: null,
          speed_us_per_s: speed,
          t_us: tStart,
          exactFrame: null,
          exactStatus: 'idle',
        })
        void get().requestSim()
      } catch (e) {
        if (isAbort(e)) return
        set({
          planStatus: 'error',
          planError: errorMessage(e),
          planErrorCode: (e as { code?: string }).code ?? null,
        })
      }
    },

    async requestSim() {
      const plan = get().plan
      if (!plan) return
      if (simAbort) simAbort.abort()
      const ctrl = new AbortController()
      simAbort = ctrl
      set({ simStatus: 'loading' })
      try {
        const sim = await api.simulate(
          {
            plan: currentRequest(get()),
            n_frames: simFrameCount(get().array),
            t_start_us: null,
            t_end_us: null,
            movie_mixing_order: null,
          },
          ctrl.signal,
        )
        if (ctrl.signal.aborted) return
        if (sim.plan_id !== get().plan?.plan_id) return // a newer plan superseded this one
        set({ sim, simStatus: 'ok', simError: null })
        if (get().showExact) void get().fetchExact(get().t_us)
      } catch (e) {
        if (isAbort(e)) return
        set({ simStatus: 'error', simError: errorMessage(e) })
      }
    },

    async fetchExact(t_us) {
      const { plan, sim } = get()
      if (!plan || !sim) return
      if (exactAbort) exactAbort.abort()
      const ctrl = new AbortController()
      exactAbort = ctrl
      set({ exactStatus: 'loading' })
      const b = sim.bounds_um
      const spanX = b.x1 - b.x0
      const spanY = b.y1 - b.y0
      const long = 192
      const nx = spanX >= spanY ? long : Math.max(16, Math.round((long * spanX) / spanY))
      const ny = spanX >= spanY ? Math.max(16, Math.round((long * spanY) / spanX)) : long
      const zHalf = Math.max(Math.abs(b.z0), Math.abs(b.z1), 2 * sim.constants.rayleigh_um)
      const tClamped = Math.min(Math.max(t_us, sim.times_us[0] ?? 0), sim.times_us[sim.times_us.length - 1] ?? t_us)
      // cut the exact XZ through the same tweezer row the approximation uses (the backend's own
      // snap sees the uncut spot list, which can pick a ghost row of a fading ladder)
      const atCursor = spotsAt(sim, tClamped).spots
      const rowY = atCursor.length ? spotRow(atCursor) : null
      try {
        const frame = await api.frame(
          {
            plan: currentRequest(get()),
            time_us: tClamped,
            plane: get().viewMode === 'tracked' ? 'tracked' : 'fixed',
            grid: { x0_um: b.x0, x1_um: b.x1, nx, y0_um: b.y0, y1_um: b.y1, ny },
            xz: { nx: 128, nz: 72, z_half_um: zHalf, row_y_um: rowY },
            movie_mixing_order: null,
          },
          ctrl.signal,
        )
        if (ctrl.signal.aborted) return
        set({ exactFrame: frame, exactStatus: 'ok', exactError: null })
      } catch (e) {
        if (isAbort(e)) return
        set({ exactStatus: 'error', exactError: errorMessage(e) })
      }
    },

    setShowExact(on) {
      set({ showExact: on })
      if (on) void get().fetchExact(get().t_us)
    },

    setTime(t) {
      const plan = get().plan
      const lo = plan ? plan.t_span_us[0] : 0
      const hi = plan ? plan.t_span_us[1] : Infinity
      set({ t_us: Math.min(Math.max(t, lo), hi) })
    },
    stepFrame(dir) {
      const { sim, plan, t_us } = get()
      let dt = 1
      if (sim && sim.times_us.length > 1) dt = sim.times_us[1] - sim.times_us[0]
      else if (plan) dt = plan.duration_us / 80
      get().pause()
      get().setTime(t_us + dir * dt)
    },
    play() {
      set({ playing: true })
    },
    pause() {
      set({ playing: false })
    },
    togglePlay() {
      set((s) => ({ playing: !s.playing }))
    },
    setSpeed(v) {
      set({ speed_us_per_s: Math.max(0.01, v), speedTouched: true })
    },
    setViewMode(m) {
      set({ viewMode: m })
      if (get().showExact) void get().fetchExact(get().t_us)
    },
    setAdvanced(patch) {
      set((s) => ({ advanced: { ...s.advanced, ...patch } }))
    },
    setEfficiencyModel(patch) {
      set((s) => ({ advanced: { ...s.advanced, efficiencyModel: { ...s.advanced.efficiencyModel, ...patch } } }))
    },
    setMisalignment(m) {
      set((s) => ({ advanced: { ...s.advanced, misalignment: m } }))
    },
  }
})

/** Selector: the current `PlanRequest`. */
export const selectRequest = (s: StudioState): PlanRequest => currentRequest(s)
