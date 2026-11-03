import { useMemo } from 'react'
import { CHANNELS, type ChannelName, type PlanResponse } from '../../api/types'
import { liveTracks, type ToneTrack } from '../../physics/waveform'

export interface ChannelCache {
  tracks: ToneTrack[]
  liveLo: number
  liveHi: number
}

export function useRfTracks(plan: PlanResponse | null): Record<ChannelName, ChannelCache> | null {
  return useMemo(() => {
    if (!plan) return null
    const out = {} as Record<ChannelName, ChannelCache>
    for (const ch of CHANNELS) {
      const tracks = liveTracks(plan.waveform_ir.channels[ch], 1500, plan.t_span_us)
      let lo = Infinity
      let hi = -Infinity
      for (const tr of tracks) {
        if (!tr.live) continue
        lo = Math.min(lo, tr.f_live_min)
        hi = Math.max(hi, tr.f_live_max)
      }
      out[ch] = { tracks, liveLo: lo, liveHi: hi }
    }
    return out
  }, [plan])
}

