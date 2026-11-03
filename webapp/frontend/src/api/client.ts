/**
 * Thin fetch wrappers over the `/api` routes of API.md. Every domain refusal (HTTP 400) is
 * surfaced as an `ApiError` carrying the backend's `code` and its human-readable `message`
 * verbatim; validation failures (422) are summarised from FastAPI's default body.
 */
import type {
  AdvancedSimulateRequest,
  AdvancedSimulateResponse,
  ApertureRequest,
  ApertureResponse,
  CodeResponse,
  EfficiencyRequest,
  EfficiencyResponse,
  ErrorBody,
  FrameRequest,
  FrameResponse,
  HealthResponse,
  PlanRequest,
  PlanResponse,
  PresetsResponse,
  SimulateRequest,
  SimulateResponse,
  SpectrumRequest,
  SpectrumResponse,
  TrajectoryEfficiencyRequest,
  TrajectoryEfficiencyResponse,
} from './types'

export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly detail: unknown
  constructor(code: string, message: string, status: number, detail: unknown = null) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.detail = detail
  }
}

export const API_BASE = '/api'

/** Parse JSON tolerating Python's `NaN`/`Infinity` literals (→ null). */
function parseLoose<T>(text: string): T {
  try {
    return JSON.parse(text) as T
  } catch {
    return JSON.parse(text.replace(/\b(NaN|-?Infinity)\b/g, 'null')) as T
  }
}

interface FastApiValidation {
  detail?: { loc?: (string | number)[]; msg?: string }[] | string
}

async function toError(res: Response): Promise<ApiError> {
  const text = await res.text()
  let body: unknown = null
  try {
    body = parseLoose(text)
  } catch {
    /* non-JSON error body */
  }
  const err = body as Partial<ErrorBody> | null
  if (err && typeof err === 'object' && err.error && typeof err.error.message === 'string') {
    return new ApiError(err.error.code, err.error.message, res.status, err.error.detail)
  }
  const val = body as FastApiValidation | null
  if (val && Array.isArray(val.detail)) {
    const msg = val.detail
      .map((d) => `${(d.loc ?? []).filter((p) => p !== 'body').join('.')}: ${d.msg ?? 'invalid'}`)
      .join('; ')
    return new ApiError('validation', msg || 'request validation failed', res.status, val.detail)
  }
  if (val && typeof val.detail === 'string') return new ApiError('http_error', val.detail, res.status)
  return new ApiError('http_error', text.slice(0, 300) || `HTTP ${res.status}`, res.status)
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) throw await toError(res)
  return parseLoose<T>(await res.text())
}

function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>('/health', { signal }),
  presets: (signal?: AbortSignal) => request<PresetsResponse>('/presets', { signal }),
  plan: (req: PlanRequest, signal?: AbortSignal) => post<PlanResponse>('/plan', req, signal),
  code: (req: PlanRequest, signal?: AbortSignal) => post<CodeResponse>('/code', req, signal),
  simulate: (req: SimulateRequest, signal?: AbortSignal) => post<SimulateResponse>('/simulate', req, signal),
  frame: (req: FrameRequest, signal?: AbortSignal) => post<FrameResponse>('/frame', req, signal),
  aperture: (req: ApertureRequest, signal?: AbortSignal) => post<ApertureResponse>('/aperture', req, signal),
  efficiency: (req: EfficiencyRequest, signal?: AbortSignal) => post<EfficiencyResponse>('/advanced/efficiency', req, signal),
  trajectoryEfficiency: (req: TrajectoryEfficiencyRequest, signal?: AbortSignal) =>
    post<TrajectoryEfficiencyResponse>('/advanced/trajectory_efficiency', req, signal),
  advancedSimulate: (req: AdvancedSimulateRequest, signal?: AbortSignal) =>
    post<AdvancedSimulateResponse>('/advanced/simulate', req, signal),
  spectrum: (req: SpectrumRequest, signal?: AbortSignal) => post<SpectrumResponse>('/advanced/spectrum', req, signal),
  waveformNpzUrl: (planId: string) => `${API_BASE}/plan/${planId}/waveform.npz`,
  /** Download the NPZ; throws `ApiError('unsupported' | 'plan_not_cached')` when refused. */
  async waveformNpz(planId: string): Promise<Blob> {
    const res = await fetch(api.waveformNpzUrl(planId))
    if (!res.ok) throw await toError(res)
    return res.blob()
  },
}

/** Decode an API.md §0.3 packed image into physical values (`peak · code / 65535`). */
export function decodePacked(img: { shape: [number, number]; peak: number; data: string }): Float32Array {
  const bin = atob(img.data)
  const n = bin.length >> 1
  const out = new Float32Array(n)
  const scale = img.peak / 65535
  for (let i = 0; i < n; i++) {
    const lo = bin.charCodeAt(2 * i)
    const hi = bin.charCodeAt(2 * i + 1)
    out[i] = (lo | (hi << 8)) * scale
  }
  return out
}

export function isAbort(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError'
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message
  if (e instanceof Error) return e.message
  return String(e)
}
