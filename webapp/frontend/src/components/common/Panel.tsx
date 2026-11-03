import type { ReactNode } from 'react'

export function Panel({
  title,
  sub,
  actions,
  children,
  className,
  id,
}: {
  title?: ReactNode
  sub?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  id?: string
}) {
  return (
    <section className={`panel ${className ?? ''}`} id={id}>
      {(title || actions) && (
        <header>
          <div>
            {title && <h3>{title}</h3>}
            {sub && <div className="sub">{sub}</div>}
          </div>
          {actions && <div className="row">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function Skeleton({ height = 80, label }: { height?: number; label?: string }) {
  return (
    <div className="skeleton" style={{ height }} aria-busy="true" aria-label={label ?? 'loading'}>
      {label && <div className="caption" style={{ padding: 8 }}>{label}</div>}
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="row small dim" style={{ gap: 6 }}>
      <span className="spinner" /> {label}
    </span>
  )
}

export function ErrorBanner({ code, message, info }: { code?: string | null; message: string; info?: boolean }) {
  return (
    <div className={`banner ${info ? 'info' : ''}`} role={info ? 'status' : 'alert'}>
      <span className="code">{code ?? (info ? 'note' : 'error')}</span>
      <span className="grow" style={{ whiteSpace: 'pre-wrap' }}>
        {message}
      </span>
    </div>
  )
}

export function Badge({ children, kind }: { children: ReactNode; kind?: string }) {
  return <span className={`badge ${kind ?? ''}`}>{children}</span>
}
