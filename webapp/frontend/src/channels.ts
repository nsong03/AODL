import type { ChannelName } from './api/types'

/** `aodl.viz.style.CHANNEL_COLORS`: counter-propagating partners share a hue family. */
export const CHANNEL_COLORS: Record<ChannelName, string> = {
  Ax: '#4cc9f0',
  Bx: '#4361ee',
  Ay: '#8ac926',
  By: '#ffb703',
}

/** Table of `docs/guide.md` §3.1 / `aodl.device.conventions.CHANNEL_GEOMETRY`. */
export const CHANNEL_INFO: Record<ChannelName, { axis: 'x' | 'y'; soundSign: 1 | -1; soundDir: string; role: string; transducer: string }> = {
  Ax: { axis: 'x', soundSign: -1, soundDir: '−x', role: 'first x deflector · single tone / lateral', transducer: '+x face' },
  Bx: { axis: 'x', soundSign: +1, soundDir: '+x', role: 'second x deflector · column ladder', transducer: '−x face' },
  Ay: { axis: 'y', soundSign: -1, soundDir: '−y', role: 'first y deflector · single tone / lateral', transducer: '+y face' },
  By: { axis: 'y', soundSign: +1, soundDir: '+y', role: 'second y deflector · row ladder', transducer: '−y face' },
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

export function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
