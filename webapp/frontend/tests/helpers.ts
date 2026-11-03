import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ChannelName, ExampleSpec, PlanConstants, WaveformIR } from '../src/api/types'

export const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures')

/** Golden fixture written by `webapp/scripts/gen_frontend_fixtures.py` (W1 work order §2). */
export interface EvalFixture {
  ir: WaveformIR
  constants: PlanConstants
  t_us: number[]
  eval: Record<ChannelName, { f_MHz: number[][]; A: number[][]; phase_rad: number[][]; V: number[] }>
  voltage_window: { t_us: number[] } & Partial<Record<ChannelName, number[]>>
  aperture: ({ time_us: number } & Partial<Record<ChannelName, { u_mm: number[]; envelope: number[]; f_inst_MHz: (number | null)[]; filled: boolean[] }>>)[]
}

export function fixturePath(name: string): string {
  return join(FIXTURES, name)
}

export function hasFixture(name: string): boolean {
  return existsSync(fixturePath(name))
}

/** Parse JSON tolerating Python's `NaN` / `Infinity` literals (mapped to null). */
export function parseLoose<T>(text: string): T {
  return JSON.parse(text.replace(/\b(NaN|-?Infinity)\b/g, 'null')) as T
}

export function loadFixture<T = EvalFixture>(name: string): T {
  return parseLoose<T>(readFileSync(fixturePath(name), 'utf8'))
}

export function loadExamples(): ExampleSpec[] | null {
  if (!hasFixture('examples.json')) return null
  const raw = loadFixture<unknown>('examples.json')
  if (Array.isArray(raw)) return raw as ExampleSpec[]
  const obj = raw as { examples?: ExampleSpec[] }
  return obj.examples ?? null
}

export function maxAbs(a: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i])
    if (v > m) m = v
  }
  return m
}
