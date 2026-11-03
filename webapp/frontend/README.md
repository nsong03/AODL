# AODL Studio — frontend

Vite 6 + React 18 + TypeScript (strict), `react-router-dom` 6, `zustand`, `uplot`. Everything
else is hand-written canvas 2D. See `../docs/DESIGN.md` §4 and `../docs/workorders/W3-frontend.md`.

```bash
npm ci
npm run dev -- --port 5173     # proxies /api → http://localhost:8000 (override: AODL_API=http://host:port)
npm test -- --run              # vitest: physics vs the golden fixtures in tests/fixtures/, store, generators
npm run lint                   # eslint
npm run typecheck              # tsc -b
npm run build                  # tsc -b && vite build → dist/ (served by the backend at /)
```

Layout: `src/api` (types mirroring `../docs/API.md`, fetch client), `src/physics` (pure TS ports of
the `aodl` evaluators: piecewise polynomials, envelopes, voltage, aperture content, the astigmatic
Gaussian renderer, `zColor`), `src/state/store.ts` (zustand), `src/components/*`, `src/pages/*`.
