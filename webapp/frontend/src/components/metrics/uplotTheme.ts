/** Dark-theme axis defaults. */
export const AXIS = {
  stroke: '#9aa5b3',
  grid: { stroke: '#1b2130', width: 1 },
  ticks: { stroke: '#2a303a', width: 1 },
  font: '10px sans-serif',
  labelFont: '10px sans-serif',
  labelSize: 14,
}

export function axisY(label: string, size = 44) {
  return { ...AXIS, label, size, labelGap: 2 }
}
export function axisX(label = 't [µs]') {
  return { ...AXIS, label, size: 32, labelGap: 2 }
}
