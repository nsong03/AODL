/**
 * The tutorial page's own time cursor (local page state — the global workbench store keeps its
 * own cursor and plan). Both columns read the same `t` [µs, drive time].
 */
import { create } from 'zustand'

export interface TutorialCursor {
  t_us: number
  playing: boolean
  speed_us_per_s: number
  setTime: (t: number) => void
  setSpeed: (s: number) => void
  togglePlay: () => void
  pause: () => void
}

export const useTutorialCursor = create<TutorialCursor>((set) => ({
  t_us: 0,
  playing: false,
  speed_us_per_s: 90,
  setTime: (t) => set({ t_us: t }),
  setSpeed: (s) => set({ speed_us_per_s: Math.max(0.1, s) }),
  togglePlay: () => set((s) => ({ playing: !s.playing })),
  pause: () => set({ playing: false }),
}))
