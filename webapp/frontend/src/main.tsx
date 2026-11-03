import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { viewScale } from './components/tweezer/view'
import * as physics from './physics'
import { router } from './router'
import { useStudio } from './state/store'
import './styles/app.css'

declare global {
  interface Window {
    /** Debug / verification hooks (W4's e2e checks read the store and re-run the renderer from here). */
    __aodl?: { store: typeof useStudio; physics: typeof physics; viewScale: typeof viewScale }
  }
}
window.__aodl = { store: useStudio, physics, viewScale }

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} future={{ v7_startTransition: true }} />
  </StrictMode>,
)
