import { Navigate, createBrowserRouter } from 'react-router-dom'
import App from './App'
import Advanced from './pages/Advanced'
import Basic from './pages/Basic'
import Landing from './pages/Landing'
import Tutorial from './pages/Tutorial'

export const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <App />,
      children: [
        { index: true, element: <Landing /> },
        { path: 'basic', element: <Basic /> },
        { path: 'advanced', element: <Advanced /> },
        { path: 'tutorial', element: <Navigate to="/tutorial/shepard" replace /> },
        { path: 'tutorial/shepard', element: <Tutorial /> },
      ],
    },
  ],
  // opt in to the v7 behaviours now so the console stays clean of future-flag warnings
  { future: { v7_relativeSplatPath: true, v7_fetcherPersist: true, v7_normalizeFormMethod: true, v7_partialHydration: true, v7_skipActionErrorRevalidation: true } },
)
