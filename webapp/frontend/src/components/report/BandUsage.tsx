import { CHANNELS, type BandUsage as BandUsageT, type ChannelName } from '../../api/types'
import { CHANNEL_COLORS } from '../../channels'
import { fmt } from '../../physics/format'

/** Per-channel band occupation bars: the live frequency range inside the usable band. */
export function BandUsage({ usage, band, toneCounts }: { usage: Record<ChannelName, BandUsageT>; band: [number, number]; toneCounts: Record<ChannelName, number> }) {
  const span = band[1] - band[0]
  return (
    <div className="stack" style={{ gap: 4 }}>
      {CHANNELS.map((ch) => {
        const u = usage[ch]
        const lo = Math.max(0, Math.min(1, (u.f_min_MHz - band[0]) / span))
        const hi = Math.max(0, Math.min(1, (u.f_max_MHz - band[0]) / span))
        const violates = u.margin_MHz < 0
        return (
          <div key={ch} className="row" style={{ gap: 8 }}>
            <span style={{ width: 24, color: CHANNEL_COLORS[ch], fontWeight: 600 }}>{ch}</span>
            <div
              className="grow"
              style={{ position: 'relative', height: 12, background: 'var(--bg-elev)', border: '1px solid var(--border)', borderRadius: 3 }}
              title={`${fmt(u.f_min_MHz, 3)} – ${fmt(u.f_max_MHz, 3)} MHz, margin ${fmt(u.margin_MHz, 2)} MHz`}
            >
              <div
                style={{
                  position: 'absolute',
                  left: `${100 * lo}%`,
                  width: `${Math.max(0.6, 100 * (hi - lo))}%`,
                  top: 1,
                  bottom: 1,
                  background: violates ? 'var(--danger)' : CHANNEL_COLORS[ch],
                  borderRadius: 2,
                  opacity: 0.9,
                }}
              />
            </div>
            <span className="small dim mono" style={{ width: 150, textAlign: 'right' }}>
              {fmt(u.f_min_MHz, 2)}–{fmt(u.f_max_MHz, 2)}
            </span>
            <span className={`small mono ${violates ? 'bright' : 'dim'}`} style={{ width: 70, textAlign: 'right', color: violates ? 'var(--danger)' : undefined }}>
              {fmt(u.margin_MHz, 2)} MHz
            </span>
            <span className="small faint" style={{ width: 54, textAlign: 'right' }}>
              {toneCounts[ch]} tone{toneCounts[ch] === 1 ? '' : 's'}
            </span>
          </div>
        )
      })}
      <div className="caption">
        band {band[0]}–{band[1]} MHz · bar = live frequency range (absolute RF), right: margin to the nearest band edge, tone count
      </div>
    </div>
  )
}
