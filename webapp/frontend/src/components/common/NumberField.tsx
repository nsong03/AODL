import { useEffect, useState } from 'react'

/**
 * Numeric input that keeps the text while typing and commits a parsed number on every valid
 * change (so the debounced planner sees edits as they happen), re-syncing from the value when
 * it changes from outside.
 */
export function NumberField({
  label,
  value,
  onChange,
  unit,
  step,
  min,
  max,
  disabled,
  title,
  wide,
  integer,
}: {
  label: string
  value: number | null
  onChange: (v: number) => void
  unit?: string
  step?: number
  min?: number
  max?: number
  disabled?: boolean
  title?: string
  wide?: boolean
  integer?: boolean
}) {
  const [text, setText] = useState(value === null ? '' : String(value))
  useEffect(() => {
    setText((prev) => {
      const parsed = Number(prev)
      if (value !== null && prev.trim() !== '' && Math.abs(parsed - value) < 1e-12) return prev
      return value === null ? '' : String(value)
    })
  }, [value])
  return (
    <label className={`field ${wide ? 'wide' : ''}`} title={title}>
      <span>
        {label} {unit && <span className="unit">[{unit}]</span>}
      </span>
      <input
        type="number"
        value={text}
        step={step ?? 'any'}
        min={min}
        max={max}
        disabled={disabled}
        onChange={(e) => {
          const s = e.target.value
          setText(s)
          const v = integer ? parseInt(s, 10) : parseFloat(s)
          if (Number.isFinite(v)) onChange(v)
        }}
      />
    </label>
  )
}

export function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
  disabled,
  title,
  wide,
}: {
  label: string
  value: T
  options: readonly (T | { value: T; label: string })[]
  onChange: (v: T) => void
  disabled?: boolean
  title?: string
  wide?: boolean
}) {
  return (
    <label className={`field ${wide ? 'wide' : ''}`} title={title}>
      <span>{label}</span>
      <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => {
          const v = typeof o === 'string' ? o : o.value
          const l = typeof o === 'string' ? o : o.label
          return (
            <option key={v} value={v}>
              {l}
            </option>
          )
        })}
      </select>
    </label>
  )
}

export function CheckField({ label, checked, onChange, disabled, title }: { label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; title?: string }) {
  return (
    <label className="check" title={title}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}
