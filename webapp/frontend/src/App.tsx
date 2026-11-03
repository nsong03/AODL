import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { api } from './api/client'
import type { HealthResponse } from './api/types'
import { useStudio } from './state/store'

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null | 'down'>(null)
  const loadPresets = useStudio((s) => s.loadPresets)
  useEffect(() => {
    let alive = true
    api
      .health()
      .then((h) => alive && setHealth(h))
      .catch(() => alive && setHealth('down'))
    void loadPresets()
    return () => {
      alive = false
    }
  }, [loadPresets])
  return (
    <div className="app">
      <div className="topbar">
        <NavLink to="/" className="brand">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="6" fill="#0b0d12" stroke="#2a303a" />
            <rect x="6" y="9" width="4" height="14" rx="1" fill="#4cc9f0" />
            <rect x="12" y="9" width="4" height="14" rx="1" fill="#4361ee" />
            <rect x="18" y="9" width="4" height="14" rx="1" fill="#8ac926" />
            <rect x="24" y="9" width="4" height="14" rx="1" fill="#ffb703" />
          </svg>
          AODL Studio
        </NavLink>
        <nav>
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/basic">Basic</NavLink>
          <NavLink to="/advanced">Advanced</NavLink>
          <NavLink to="/tutorial/shepard">Tutorial</NavLink>
        </nav>
        <div className="status" title="backend /api/health">
          {health === null && 'backend: …'}
          {health === 'down' && (
            <>
              backend: <span className="bad">unreachable</span> (start <code>uvicorn aodl_webapp.main:create_app --factory --port 8000</code>)
            </>
          )}
          {health && health !== 'down' && (
            <>
              backend <span className="ok">ok</span> · aodl {health.aodl_version} · studio {health.version}
            </>
          )}
        </div>
      </div>
      <Outlet />
    </div>
  )
}
