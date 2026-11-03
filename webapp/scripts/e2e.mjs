#!/usr/bin/env node
/**
 * AODL Studio end-to-end verification (W4 work order §2, DESIGN §5).
 *
 * Starts the backend (`uvicorn aodl_webapp.main:create_app --factory` on :8000, strict advanced
 * router) and a frontend server (`vite preview` of the built bundle on :4173 by default, the Vite
 * dev server with `--dev`, or the backend's own static mount with `--served`), drives the UI with
 * Playwright's bundled Chromium and
 *
 *   - loads `/`, `/basic?example=…` for four examples, `/tutorial/shepard` (plus the two tutorial
 *     examples in the workbench), `/advanced?example=pick_and_place` and the 10×10 flagship,
 *     asserting no console errors, page errors or failed requests;
 *   - round 2: during the 4×4 `array_lift_hold` hold no rendered trap moves more than 0.5·w0 per
 *     frame interval and never fewer than 16 traps are drawn; `pick_and_place` and
 *     `array_arbitrary_path` glide (no bright cross-fades); the tutorial's movie equals the
 *     workbench's for the same examples (same simulate body, same brightest spot at its cursor);
 *   - captures one screenshot per page into `webapp/docs/screenshots/`;
 *   - runs the DESIGN §5 numeric checks through the exposed `window.__aodl` hooks:
 *       (a) pick_and_place at t = 70 µs + τ/2: brightest pixel of the Gaussian approximation vs
 *           the exact `/api/frame` within 0.05 w0 (and the two planes agree);
 *       (b) the aperture panel's envelope / f_inst / fill mask / carrier zoom vs `/api/aperture`
 *           to 1e-6;
 *       (c) single_lift_hold at the first *x* hand-over + τ/2: two coincident main spots plus two
 *           shadows at ±10.3·Δf µm carrying half the main trap's power each (Eq. S31);
 *   - measures playback fps over 2 s on the 4×4 fading example and on the flagship;
 *   - checks the production path: the backend serves the SPA at `/` with a deep-link fallback.
 *
 * Usage:  node webapp/scripts/e2e.mjs [--dev | --served] [--keep] [--out results.json]
 * Requires: `pip install -e . -e webapp/backend[dev]`, `npm ci && npm run build` in webapp/frontend,
 * and `playwright` resolvable from Node (`PW_CHROMIUM` overrides the browser executable).
 */
import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { mkdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const { chromium } = require('playwright')

const HERE = path.dirname(fileURLToPath(import.meta.url))
const WEBAPP = path.resolve(HERE, '..')
const BACKEND = path.join(WEBAPP, 'backend')
const FRONTEND = path.join(WEBAPP, 'frontend')
const SHOTS = path.join(WEBAPP, 'docs', 'screenshots')
const API = 'http://127.0.0.1:8000'
const PREVIEW_PORT = 4173
const DEV_PORT = 5173

const args = process.argv.slice(2)
const flag = (name) => args.includes(name)
const opt = (name) => {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : null
}
const mode = flag('--dev') ? 'dev' : flag('--served') ? 'served' : 'preview'
const UI = mode === 'dev' ? `http://127.0.0.1:${DEV_PORT}` : mode === 'preview' ? `http://127.0.0.1:${PREVIEW_PORT}` : API
const PYTHON = process.env.PYTHON ?? 'python'

const results = { mode, pages: {}, checks: {}, timings: {}, failures: [] }
const fail = (what) => {
  results.failures.push(what)
  console.error(`  FAIL ${what}`)
}
const ok = (what) => console.log(`  ok   ${what}`)
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ------------------------------------------------------------------------------- servers
const children = []
function start(cmd, cmdArgs, cwd, env = {}) {
  const child = spawn(cmd, cmdArgs, { cwd, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] })
  child.stdout.on('data', (d) => process.env.E2E_VERBOSE && process.stdout.write(`[${path.basename(cmd)}] ${d}`))
  child.stderr.on('data', (d) => process.env.E2E_VERBOSE && process.stderr.write(`[${path.basename(cmd)}] ${d}`))
  children.push(child)
  return child
}
async function waitFor(url, timeoutMs, accept = (r) => r.ok) {
  const t0 = Date.now()
  let last = null
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(url)
      if (accept(r)) return r
      last = `status ${r.status}`
    } catch (e) {
      last = e.message
    }
    await sleep(250)
  }
  throw new Error(`timeout waiting for ${url}: ${last}`)
}
function stopAll() {
  for (const c of children) {
    try {
      c.kill('SIGTERM')
    } catch {
      /* already gone */
    }
  }
}
process.on('exit', stopAll)
process.on('SIGINT', () => {
  stopAll()
  process.exit(130)
})

async function api(route, body) {
  const r = await fetch(`${API}/api${route}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${route} -> ${r.status} ${await r.text()}`)
  return r.json()
}

// ------------------------------------------------------------------------------- browser
async function launch() {
  const base = { headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] }
  try {
    return await chromium.launch(base)
  } catch (e) {
    const executablePath = process.env.PW_CHROMIUM ?? '/opt/pw-browsers/chromium'
    console.log(`  default chromium launch failed (${e.message.split('\n')[0]}); retrying with ${executablePath}`)
    return chromium.launch({ ...base, executablePath })
  }
}

function watch(page, record) {
  page.on('console', (msg) => {
    if (msg.type() === 'error') record.consoleErrors.push(msg.text())
  })
  page.on('pageerror', (err) => record.pageErrors.push(String(err)))
  page.on('requestfailed', (req) => {
    const why = req.failure()?.errorText ?? ''
    // a fetch the app cancelled itself (AbortController: a newer cursor time superseded it, or the
    // page closed) is not a failure; it is listed separately so a flood of them stays visible
    if (why === 'net::ERR_ABORTED') (record.abortedRequests ??= []).push(`${req.method()} ${req.url()}`)
    else record.failedRequests.push(`${req.method()} ${req.url()} ${why}`)
  })
  page.on('response', (res) => {
    if (res.status() >= 400) record.failedRequests.push(`${res.request().method()} ${res.url()} -> ${res.status()}`)
  })
}

const storeReady = (p, status = 'ok') => p.waitForFunction((want) => {
  const s = window.__aodl?.store.getState()
  return !!s && s.planStatus === want && (want !== 'ok' || s.simStatus === 'ok')
}, status, { timeout: 120_000 })

async function loadPage(context, route, name, { waitStore = true, timeoutMs = 120_000 } = {}) {
  const record = { consoleErrors: [], pageErrors: [], failedRequests: [] }
  const page = await context.newPage()
  watch(page, record)
  const t0 = Date.now()
  await page.goto(`${UI}${route}`, { waitUntil: 'networkidle', timeout: timeoutMs })
  let tPlan = null
  let tSim = null
  if (waitStore) {
    await page.waitForFunction(() => window.__aodl?.store.getState().planStatus === 'ok', null, { timeout: timeoutMs })
    tPlan = Date.now() - t0
    await page.waitForFunction(() => window.__aodl?.store.getState().simStatus === 'ok', null, { timeout: timeoutMs })
    tSim = Date.now() - t0
    await page.waitForTimeout(400) // let the canvases paint
  }
  // per-route timings as the browser saw them (Resource Timing API; cold backend caches on
  // the first visit of an example)
  const api = await page.evaluate(() =>
    performance
      .getEntriesByType('resource')
      .filter((e) => /\/api\//.test(e.name))
      .map((e) => ({ route: new URL(e.name).pathname, ms: Math.round(e.duration), kB: Math.round((e.transferSize || e.encodedBodySize || 0) / 1024) })),
  )
  results.pages[name] = { route, ...record, ms_to_plan: tPlan, ms_to_sim: tSim, api }
  return { page, record }
}

function judge(name, record) {
  const bad = record.consoleErrors.length + record.pageErrors.length + record.failedRequests.length
  if (bad) fail(`${name}: ${record.consoleErrors.length} console errors, ${record.pageErrors.length} page errors, ${record.failedRequests.length} failed requests: ${[...record.consoleErrors, ...record.pageErrors, ...record.failedRequests].slice(0, 3).join(' | ')}`)
  else ok(`${name}: no console errors, no failed requests${record.abortedRequests?.length ? ` (${record.abortedRequests.length} superseded request(s) aborted by the app)` : ''}`)
}

async function shot(page, file) {
  await page.screenshot({ path: path.join(SHOTS, file), fullPage: false })
  console.log(`  shot ${file}`)
}

// ----------------------------------------------------------------------------------- checks
async function checkBrightest(page) {
  // (a) pick_and_place at t = 70 µs + τ/2: approximation vs exact brightest pixel
  const t = await page.evaluate(() => {
    const s = window.__aodl.store.getState()
    const t = 70 + s.plan.constants.half_tau_us
    s.pause()
    s.setTime(t)
    s.setShowExact(true)
    return t
  })
  await page.waitForFunction((want) => {
    const s = window.__aodl.store.getState()
    return s.exactStatus === 'ok' && s.exactFrame && Math.abs(s.exactFrame.time_us - want) < 1e-6
  }, t, { timeout: 60_000 })
  await page.waitForTimeout(300)
  const r = await page.evaluate(() => {
    const { store, physics, viewScale } = window.__aodl
    const s = store.getState()
    const sim = s.sim
    const ex = s.exactFrame
    const scale = viewScale(sim)
    const frame = physics.spotsAt(sim, s.t_us)
    const planeZ = s.viewMode === 'tracked' ? frame.tracked_z : 0
    const nx = ex.xy.x_um.length
    const ny = ex.xy.y_um.length
    const grid = { x0: ex.xy.x_um[0], x1: ex.xy.x_um[nx - 1], nx, y0: ex.xy.y_um[0], y1: ex.xy.y_um[ny - 1], ny }
    const rgb = physics.renderXY(frame.spots, grid, planeZ, scale.w0, scale.zR, scale.peakTracked, scale.zMax, 1)
    let best = -1
    let bi = 0
    for (let p = 0; p < nx * ny; p++) {
      const v = Math.max(rgb[3 * p], rgb[3 * p + 1], rgb[3 * p + 2])
      if (v > best) {
        best = v
        bi = p
      }
    }
    const ax = grid.x0 + (bi % nx) * ((grid.x1 - grid.x0) / (nx - 1))
    const ay = grid.y0 + Math.floor(bi / nx) * ((grid.y1 - grid.y0) / (ny - 1))
    // exact frame argmax (row 0 = y0)
    const bin = atob(ex.xy.image.data)
    let bestE = -1
    let be = 0
    for (let p = 0; p < nx * ny; p++) {
      const v = bin.charCodeAt(2 * p) | (bin.charCodeAt(2 * p + 1) << 8)
      if (v > bestE) {
        bestE = v
        be = p
      }
    }
    const exx = ex.xy.x_um[be % nx]
    const exy = ex.xy.y_um[Math.floor(be / nx)]
    const top = physics.brightest(frame.spots)
    const approxPeak = physics.spotPeak(top, planeZ, scale.w0, scale.zR)
    return {
      t_us: s.t_us,
      approx_px: [ax, ay],
      exact_px: [exx, exy],
      spot_centre: [top.x_um, top.y_um],
      dist_w0: Math.hypot(ax - exx, ay - exy) / scale.w0,
      plane_approx_um: planeZ,
      plane_exact_um: ex.plane_z_um,
      exact_peak_over_approx: (ex.xy.image.peak * physics.PER_M2_TO_PER_UM2) / approxPeak,
      pixel_um: (grid.x1 - grid.x0) / (nx - 1),
      w0: scale.w0,
    }
  })
  results.checks.brightest_pixel = r
  if (r.dist_w0 <= 0.05 && Math.abs(r.plane_approx_um - r.plane_exact_um) < 1e-3) ok(`(a) brightest pixel: approx (${r.approx_px.map((v) => v.toFixed(3))}) vs exact (${r.exact_px.map((v) => v.toFixed(3))}) µm → ${r.dist_w0.toFixed(4)} w0; planes ${r.plane_approx_um.toFixed(3)} / ${r.plane_exact_um.toFixed(3)} µm; exact peak / approx peak ${r.exact_peak_over_approx.toFixed(4)}`)
  else fail(`(a) brightest pixel distance ${r.dist_w0} w0 or plane mismatch ${r.plane_approx_um} vs ${r.plane_exact_um}`)
  await page.evaluate(() => window.__aodl.store.getState().setShowExact(false))
}

async function checkAperture(page, t_us) {
  // (b) frontend aperture physics vs /api/aperture at the cursor
  const fe = await page.evaluate((t) => {
    const { store, physics } = window.__aodl
    const s = store.getState()
    const out = {}
    for (const ch of ['Ax', 'Bx', 'Ay', 'By']) {
      const chan = s.plan.waveform_ir.channels[ch]
      const c = physics.apertureContent(chan, t, 512)
      const v = physics.apertureVoltage(chan, t, -0.05, 0.05, 4096)
      out[ch] = {
        u_mm: Array.from(c.u_mm),
        envelope: Array.from(c.envelope),
        f_inst_MHz: Array.from(c.f_inst_MHz, (x) => (Number.isNaN(x) ? null : x)),
        filled: Array.from(c.filled, (x) => x === 1),
        v_u_mm: Array.from(v.u_mm),
        v: Array.from(v.v),
      }
    }
    return { request: { hardware: s.hardware, array: s.array, moves: s.moves, options: s.options }, channels: out }
  }, t_us)
  const be = await api('/aperture', { plan: fe.request, time_us: t_us, n: 512, include_voltage: true, voltage_window_mm: null })
  const summary = {}
  let worst = { env: 0, f: 0, fill: 0, v: 0, u: 0 }
  for (const ch of ['Ax', 'Bx', 'Ay', 'By']) {
    const a = fe.channels[ch]
    const b = be.channels[ch]
    const envMax = Math.max(...b.envelope, 1e-300)
    let dEnv = 0
    let dF = 0
    let dFill = 0
    let dU = 0
    for (let i = 0; i < 512; i++) {
      dEnv = Math.max(dEnv, Math.abs(a.envelope[i] - b.envelope[i]) / envMax)
      dU = Math.max(dU, Math.abs(a.u_mm[i] - b.u_mm[i]))
      if (a.filled[i] !== b.filled[i]) dFill++
      const fa = a.f_inst_MHz[i]
      const fb = b.f_inst_MHz[i]
      if ((fa === null) !== (fb === null)) dFill++
      else if (fa !== null) dF = Math.max(dF, Math.abs(fa - fb))
    }
    const vMax = Math.max(...b.voltage.v.map(Math.abs), 1e-300)
    let dV = 0
    if (a.v.length !== b.voltage.v.length) dV = Infinity
    else for (let i = 0; i < a.v.length; i++) dV = Math.max(dV, Math.abs(a.v[i] - b.voltage.v[i]) / vMax, Math.abs(a.v_u_mm[i] - b.voltage.u_mm[i]))
    summary[ch] = { envelope_rel: dEnv, f_inst_MHz: dF, fill_mismatches: dFill, voltage_rel: dV, u_mm: dU, n_voltage: b.voltage.v.length, env_max: envMax }
    worst = { env: Math.max(worst.env, dEnv), f: Math.max(worst.f, dF), fill: worst.fill + dFill, v: Math.max(worst.v, dV), u: Math.max(worst.u, dU) }
  }
  results.checks.aperture = { t_us, per_channel: summary, worst }
  if (worst.env <= 1e-6 && worst.f <= 1e-6 && worst.fill === 0 && worst.v <= 1e-6 && worst.u <= 1e-9) ok(`(b) aperture at t = ${t_us.toFixed(3)} µs: envelope ${worst.env.toExponential(2)} rel, f_inst ${worst.f.toExponential(2)} MHz, fill mask exact, carrier V(u) ${worst.v.toExponential(2)} rel on ${summary.Ax.n_voltage} pts`)
  else fail(`(b) aperture mismatch ${JSON.stringify(worst)}`)
}

async function checkShadows(page) {
  // (c) single_lift_hold: first x hand-over + τ/2 → four spots
  const info = await page.evaluate(() => {
    const s = window.__aodl.store.getState()
    const ev = s.plan.fade_events.find((e) => e.axis === 'x')
    return { request: { hardware: s.hardware, array: s.array, moves: s.moves, options: s.options }, ev, half_tau: s.plan.constants.half_tau_us, first_event: s.plan.fade_events[0], n_events: s.plan.fade_events.length, power_ref: s.sim.power_ref }
  })
  const t = info.ev.time_us + info.half_tau
  const frame = await api('/frame', { plan: info.request, time_us: t, plane: 'tracked', grid: { x0_um: -4, x1_um: 4, nx: 8, y0_um: -4, y1_um: 4, ny: 8 }, xz: null, movie_mixing_order: null })
  const spots = frame.spots
  const main = spots.filter((s) => Math.abs(s.x_um) < 1e-3 && Math.abs(s.y_um) < 1e-3)
  const shadows = spots.filter((s) => Math.abs(s.x_um) >= 1e-3)
  const mainP = main.reduce((a, s) => a + s.power, 0)
  const r = {
    t_us: t,
    first_event: info.first_event,
    x_event: info.ev,
    n_spots: spots.length,
    main_spots: main.length,
    main_power_over_power_ref: mainP / info.power_ref,
    shadows: shadows.map((s) => ({ x_um: s.x_um, y_um: s.y_um, power_over_main: s.power / mainP, power_over_power_ref: s.power / info.power_ref, group: s.group, n_terms: s.n_terms })),
    coincident_groups: main.map((s) => s.group),
  }
  results.checks.shadow_pair = r
  const good =
    spots.length === 4 &&
    main.length === 2 &&
    shadows.length === 2 &&
    shadows.every((s) => Math.abs(Math.abs(s.x_um) - info.ev.shadow_um) < 1e-3 && Math.abs(s.power / mainP - 0.5) < 0.01)
  if (good) ok(`(c) shadows at t = ${t.toFixed(3)} µs: main = ${main.length} coincident spots (groups ${r.coincident_groups}), P_main/power_ref = ${r.main_power_over_power_ref.toFixed(3)}; shadows at x = ${shadows.map((s) => s.x_um.toFixed(1)).join(', ')} µm with P/P_main = ${shadows.map((s) => (s.power / mainP).toFixed(4)).join(', ')} (expected ±${info.ev.shadow_um.toFixed(1)}, 0.5 each)`)
  else fail(`(c) shadow pair unexpected: ${JSON.stringify(r)}`)
  // the UI at that instant: the badge must mention the off-canvas shadows
  await page.evaluate((tt) => {
    const s = window.__aodl.store.getState()
    s.pause()
    s.setTime(tt)
  }, t)
  await page.waitForTimeout(400)
  const badge = await page.locator('.badge.warn').allTextContents()
  results.checks.shadow_badge = badge
  if (badge.some((b) => /shadow tweezer/.test(b))) ok(`badge: "${badge.find((b) => /shadow/.test(b))}"`)
  else fail(`shadow badge missing at the hand-over (badges: ${JSON.stringify(badge)})`)
  return t
}

async function measureFps(page, seconds = 2) {
  await page.evaluate(() => window.__aodl.store.getState().play())
  const r = await page.evaluate(
    (ms) =>
      new Promise((resolve) => {
        const stamps = []
        const t0 = performance.now()
        const tick = (now) => {
          stamps.push(now)
          if (now - t0 < ms) requestAnimationFrame(tick)
          else {
            let worst = 0
            for (let i = 1; i < stamps.length; i++) worst = Math.max(worst, stamps[i] - stamps[i - 1])
            resolve({ frames: stamps.length - 1, seconds: (stamps[stamps.length - 1] - stamps[0]) / 1000, fps: ((stamps.length - 1) * 1000) / (stamps[stamps.length - 1] - stamps[0]), worst_gap_ms: worst })
          }
        }
        requestAnimationFrame(tick)
      }),
    seconds * 1000,
  )
  const extra = await page.evaluate(() => {
    const s = window.__aodl.store.getState()
    s.pause()
    const f = window.__aodl.physics.spotsAt(s.sim, s.t_us)
    return { spots_at_cursor: f.spots.length, n_frames: s.sim.frames.length, max_spots: Math.max(...s.sim.frames.map((fr) => fr.x_um.length)) }
  })
  return { ...r, ...extra }
}

async function checkMotion(page, label, t0, t1) {
  // Rendered motion between consecutive frames, through the same hooks the movie uses: inside one
  // frame interval `spotsAt` keeps one spot list (paired spots interpolate, unpaired ones fade),
  // so spot k just after frame i and just before frame i + 1 is the same rendered spot.
  // `max_moved_w0` = its largest displacement over an interval; `bright_cross_fades` = spots of
  // ≥ 20 % of the frame maximum that are not paired (they fade in place instead of moving);
  // `traps_appearing` / `traps_vanishing` = those with no spot of the other frame within
  // 0.5·w0 either (light switching on or off at a site, not a rung pair handing over).
  return page.evaluate(
    ({ t0, t1 }) => {
      const { store, physics } = window.__aodl
      const sim = store.getState().sim
      const w0 = sim.constants.waist0_um
      const T = sim.times_us
      let intervals = 0
      let maxMoved = 0
      let maxMovedAt = null
      let maxShift = 0
      let crossFades = 0
      let crossFadeIntervals = 0
      let appear = 0
      let vanish = 0
      const siteCounts = {}
      for (let i = 0; i + 1 < T.length; i++) {
        if (T[i] < t0 || T[i + 1] > t1) continue
        const A = physics.frameSpots(sim.frames[i])
        const B = physics.frameSpots(sim.frames[i + 1])
        if (!A.length || !B.length) continue
        intervals++
        const pmax = Math.max(...A.map((x) => x.power), ...B.map((x) => x.power))
        const pairing = physics.pairFrames(A, B, w0)
        maxShift = Math.max(maxShift, Math.hypot(pairing.shift[0], pairing.shift[1]) / w0)
        const eps = 1e-6 * (T[i + 1] - T[i])
        const s0 = physics.spotsAt(sim, T[i] + eps).spots
        const s1 = physics.spotsAt(sim, T[i + 1] - eps).spots
        for (let k = 0; k < Math.min(s0.length, s1.length); k++) {
          if (Math.max(s0[k].power, s1[k].power) < 0.05 * pmax) continue
          const d = Math.hypot(s1[k].x_um - s0[k].x_um, s1[k].y_um - s0[k].y_um) / w0
          if (d > maxMoved) {
            maxMoved = d
            maxMovedAt = T[i]
          }
        }
        let cf = 0
        const r = 0.5 * w0
        const [sx, sy] = pairing.shift
        const near = (list, x, y) => list.some((q) => Math.hypot(q.x_um - x, q.y_um - y) < r)
        A.forEach((x, k) => {
          if (pairing.pair[k] >= 0 || x.power < 0.2 * pmax) return
          cf++
          if (!near(B, x.x_um + sx, x.y_um + sy)) vanish++
        })
        B.forEach((x, j) => {
          if (pairing.paired[j] || x.power < 0.2 * pmax) return
          cf++
          if (!near(A, x.x_um - sx, x.y_um - sy)) appear++
        })
        crossFades += cf
        if (cf) crossFadeIntervals++
        const sites = physics.frameSites(sim.frames[i], w0)
        const dmax = Math.max(...sites.map((x) => x.depth))
        const n = sites.filter((x) => x.depth >= 0.25 * dmax).length
        siteCounts[n] = (siteCounts[n] ?? 0) + 1
      }
      return { t0, t1, frame_dt_us: T[1] - T[0], intervals, max_moved_w0: maxMoved, max_moved_at_us: maxMovedAt, max_rigid_shift_w0: maxShift, bright_cross_fades: crossFades, cross_fade_intervals: crossFadeIntervals, traps_appearing: appear, traps_vanishing: vanish, traps_per_frame: siteCounts }
    },
    { t0, t1 },
  ).then((r) => ({ label, ...r }))
}

// -------------------------------------------------------------------------------------- main
async function main() {
  mkdirSync(SHOTS, { recursive: true })
  console.log(`starting backend on :8000 (strict advanced router) and frontend mode "${mode}"`)
  start(PYTHON, ['-m', 'uvicorn', 'aodl_webapp.main:create_app', '--factory', '--port', '8000', '--host', '127.0.0.1'], BACKEND, { AODL_STUDIO_STRICT_ADVANCED: '1' })
  if (mode === 'preview') start(path.join(FRONTEND, 'node_modules', '.bin', 'vite'), ['preview', '--port', String(PREVIEW_PORT), '--strictPort', '--host', '127.0.0.1'], FRONTEND)
  if (mode === 'dev') start(path.join(FRONTEND, 'node_modules', '.bin', 'vite'), ['--port', String(DEV_PORT), '--strictPort', '--host', '127.0.0.1'], FRONTEND)
  const health = await (await waitFor(`${API}/api/health`, 60_000)).json()
  console.log(`  backend ${JSON.stringify(health)}`)
  await waitFor(`${UI}/`, 60_000)
  if (mode === 'preview' || mode === 'dev') await waitFor(`${UI}/api/health`, 60_000) // the proxy

  const browser = await launch()
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 })
  console.log(`  chromium ${browser.version()}`)

  // landing
  {
    const { page, record } = await loadPage(context, '/', 'landing', { waitStore: false })
    await page.waitForSelector('.badge', { timeout: 30_000 }) // example cards from /api/presets
    await page.waitForTimeout(500)
    await shot(page, 'landing.png')
    judge('landing', record)
    await page.close()
  }

  // basic: pick_and_place (+ checks a, b)
  {
    const { page, record } = await loadPage(context, '/basic?example=pick_and_place', 'basic_pick_and_place')
    await shot(page, 'basic_pick_and_place.png')
    await checkBrightest(page)
    const tCursor = await page.evaluate(() => window.__aodl.store.getState().t_us)
    await checkAperture(page, tCursor)
    // a second aperture instant while the aperture is still filling (0.6 τ)
    const tFill = await page.evaluate(() => 0.6 * window.__aodl.store.getState().plan.constants.tau_us)
    const saved = results.checks.aperture
    await checkAperture(page, tFill)
    results.checks.aperture_filling = results.checks.aperture
    results.checks.aperture = saved
    const glide = await checkMotion(page, 'pick_and_place', 0, 1e9)
    results.checks.motion_pick_and_place = glide
    if (glide.bright_cross_fades === 0) ok(`pick_and_place glides: ${glide.intervals} frame intervals of ${glide.frame_dt_us.toFixed(2)} µs, max rendered step ${glide.max_moved_w0.toFixed(3)} w0 (rigid shift ≤ ${glide.max_rigid_shift_w0.toFixed(3)} w0), 0 cross-fades`)
    else fail(`pick_and_place cross-fades: ${JSON.stringify(glide)}`)
    judge('basic_pick_and_place', record)
    await page.close()
  }

  // basic: single_lift_hold (+ check c, mid-fade screenshot)
  {
    const { page, record } = await loadPage(context, '/basic?example=single_lift_hold', 'basic_single_lift_hold')
    await checkShadows(page)
    await shot(page, 'basic_single_lift_hold_fade.png')
    judge('basic_single_lift_hold', record)
    await page.close()
  }

  // basic: array_lift_hold (+ fps)
  {
    const { page, record } = await loadPage(context, '/basic?example=array_lift_hold', 'basic_array_lift_hold')
    const notes = await page.evaluate(() => window.__aodl.store.getState().sim.notes)
    results.pages.basic_array_lift_hold.sim_notes = notes
    const fps = await measureFps(page, 2)
    results.checks.fps_array_lift_hold = fps
    if (fps.fps >= 30) ok(`playback 4×4: ${fps.fps.toFixed(1)} fps over ${fps.seconds.toFixed(2)} s (${fps.frames} frames, worst gap ${fps.worst_gap_ms.toFixed(1)} ms, up to ${fps.max_spots} spots/frame)`)
    else fail(`playback 4×4 only ${fps.fps.toFixed(1)} fps`)
    // during the hold: no rendered trap may move more than 0.5 w0 per frame interval
    const hold = await page.evaluate(() => {
      const s = window.__aodl.store.getState()
      const seams = s.plan.seams_us
      const ht = s.plan.constants.half_tau_us
      const evs = s.plan.fade_events.filter((e) => e.axis === 'x' && e.time_us > seams[1] && e.time_us < seams[2])
      const mid = evs[Math.floor(evs.length / 2)]
      return { t0: seams[1] + ht, t1: seams[2] + ht, handover: mid.time_us + ht, w0: s.sim.constants.waist0_um }
    })
    const motion = await checkMotion(page, 'array_lift_hold hold', hold.t0, hold.t1)
    results.checks.motion_array_lift_hold = motion
    const counts = Object.keys(motion.traps_per_frame).map(Number)
    if (motion.max_moved_w0 <= 0.5) ok(`4×4 hold [${hold.t0.toFixed(1)}, ${hold.t1.toFixed(1)}] µs: ${motion.intervals} intervals of ${motion.frame_dt_us.toFixed(2)} µs, max rendered step ${motion.max_moved_w0.toFixed(4)} w0 (≤ 0.5), ${motion.bright_cross_fades} spot cross-fades in place (${motion.traps_appearing} traps switching on, ${motion.traps_vanishing} off: the edge columns) in ${motion.cross_fade_intervals} intervals, traps per frame ${JSON.stringify(motion.traps_per_frame)}`)
    else fail(`4×4 hold: a rendered trap moved ${motion.max_moved_w0} w0 in one frame interval (at ${motion.max_moved_at_us} µs)`)
    if (Math.min(...counts) >= 16) ok(`4×4 hold: never fewer than 16 traps rendered (min ${Math.min(...counts)}, max ${Math.max(...counts)})`)
    else fail(`4×4 hold: only ${Math.min(...counts)} traps in some frame`)
    await page.evaluate((t) => {
      const s = window.__aodl.store.getState()
      s.pause()
      s.setTime(t)
    }, hold.handover)
    await page.waitForTimeout(1200) // line cuts fetch 250 ms after the cursor settles
    const atCursor = await page.evaluate(() => {
      const { store, physics } = window.__aodl
      const s = store.getState()
      const f = physics.spotsAt(s.sim, s.t_us)
      return { t_us: s.t_us, spots: f.spots.length }
    })
    results.checks.array_lift_hold_handover_cursor = atCursor
    await shot(page, 'basic_array_lift_hold.png')
    const panel = page.locator('section.panel', { has: page.locator('h3', { hasText: 'Trap intensity' }) })
    await panel.scrollIntoViewIfNeeded()
    await page.waitForTimeout(400)
    await panel.screenshot({ path: path.join(SHOTS, 'intensity_panel.png') })
    console.log('  shot intensity_panel.png')
    judge('basic_array_lift_hold', record)
    await page.close()
  }

  // basic: array_arbitrary_path
  {
    const { page, record } = await loadPage(context, '/basic?example=array_arbitrary_path', 'basic_array_arbitrary_path')
    // mid-path: show the tilted circle in flight
    await page.evaluate(() => {
      const s = window.__aodl.store.getState()
      s.setTime(50 + 150 + s.plan.constants.half_tau_us)
    })
    await page.waitForTimeout(400)
    await shot(page, 'basic_array_arbitrary_path.png')
    const glide = await checkMotion(page, 'array_arbitrary_path', 0, 1e9)
    results.checks.motion_array_arbitrary_path = glide
    if (glide.bright_cross_fades === 0) ok(`array_arbitrary_path glides: ${glide.intervals} intervals of ${glide.frame_dt_us.toFixed(2)} µs, max rendered step ${glide.max_moved_w0.toFixed(3)} w0 (rigid shift ≤ ${glide.max_rigid_shift_w0.toFixed(3)} w0), 0 cross-fades`)
    else fail(`array_arbitrary_path cross-fades: ${JSON.stringify(glide)}`)
    judge('basic_array_arbitrary_path', record)
    await page.close()
  }

  // tutorial: /tutorial/shepard — two drives side by side on the page-local cursor; compare its
  // movie with the workbench's for the same examples (same /api/simulate body, same spots)
  {
    const sims = {}
    const record = { consoleErrors: [], pageErrors: [], failedRequests: [] }
    const page = await context.newPage()
    watch(page, record)
    page.on('response', async (res) => {
      if (!/\/api\/simulate$/.test(res.url()) || res.status() !== 200) return
      try {
        const body = await res.json()
        sims[body.plan_id] = body
      } catch {
        /* navigation raced the body */
      }
    })
    const t0 = Date.now()
    await page.goto(`${UI}/tutorial/shepard`, { waitUntil: 'networkidle', timeout: 120_000 })
    await page.waitForFunction(() => [...document.querySelectorAll('.badge')].filter((b) => /brightest P\/P_ref/.test(b.textContent ?? '')).length === 2, null, { timeout: 120_000 })
    const tReady = Date.now() - t0
    await page.waitForTimeout(800)
    const badges = await page.locator('.badge', { hasText: 'brightest P/P_ref' }).allTextContents()
    const warn = await page.locator('.badge.warn').allTextContents()
    results.pages.tutorial = { route: '/tutorial/shepard', ...record, ms_ready: tReady, badges, warn }
    await shot(page, 'tutorial.png')
    // the redirect
    await page.goto(`${UI}/tutorial`, { waitUntil: 'networkidle' })
    const redirected = page.url()
    judge('tutorial', record)
    await page.close()
    // the same two examples in the workbench, at the tutorial's start cursor (the Shepard
    // drive's first x hand-over + τ/2)
    const presets = await api('/presets')
    const shepReq = presets.examples.find((e) => e.id === 'tutorial_hold_shepard').request
    const shepPlan = await api('/plan', shepReq)
    const tStart = shepPlan.fade_events.find((e) => e.axis === 'x').time_us + shepPlan.constants.half_tau_us
    const compare = {}
    const expected = []
    for (const id of ['tutorial_hold_plain', 'tutorial_hold_shepard']) {
      const { page: bp, record: br } = await loadPage(context, `/basic?example=${id}`, `basic_${id}`)
      const r = await bp.evaluate(
        ({ tutSims, t }) => {
          const { store, physics } = window.__aodl
          const s = store.getState()
          const f = physics.spotsAt(s.sim, t)
          const top = physics.brightest(f.spots)
          return { plan_id: s.sim.plan_id, sim_identical: JSON.stringify(s.sim) === JSON.stringify(tutSims[s.sim.plan_id] ?? null), n: f.spots.length, rel: top.power / s.sim.power_ref, z: top.z_um }
        },
        { tutSims: sims, t: tStart },
      )
      compare[id] = r
      const f2 = (v) => v.toFixed(2) // physics/format.ts fmt(x, 2)
      expected.push(`brightest P/P_ref ${f2(r.rel)} · z ${f2(r.z)} µm · ${r.n} spot${r.n === 1 ? '' : 's'}`)
      judge(`basic_${id}`, br)
      await bp.close()
    }
    const same = Object.values(compare).every((c) => c.sim_identical) && badges.length === 2 && badges.every((b, k) => b.trim() === expected[k])
    results.checks.tutorial_vs_workbench = { t_cursor_us: tStart, compare, tutorial_badges: badges, workbench_badges: expected, redirect: redirected }
    if (same) ok(`tutorial = workbench: identical /api/simulate bodies for both drives; at t = ${tStart.toFixed(3)} µs both render "${expected[0]}" | "${expected[1]}"`)
    else fail(`tutorial vs workbench differ: ${JSON.stringify(results.checks.tutorial_vs_workbench)}`)
    if (/\/tutorial\/shepard$/.test(redirected)) ok('/tutorial redirects to /tutorial/shepard')
    else fail(`/tutorial redirect landed on ${redirected}`)
    if (warn.some((w) => /shadow tweezer/.test(w))) ok(`tutorial start cursor shows the Shepard shadows: "${warn.find((w) => /shadow/.test(w))}"`)
    else fail(`tutorial: no shadow badge at the start cursor (${JSON.stringify(warn)})`)
  }

  // advanced: efficiency panel
  {
    const { page, record } = await loadPage(context, '/advanced?example=pick_and_place', 'advanced_pick_and_place')
    await page.waitForFunction(() => {
      const a = window.__aodl.store.getState().advanced
      return a.efficiencyStatus === 'ok' && a.trajEffStatus === 'ok'
    }, null, { timeout: 60_000 })
    await page.waitForTimeout(500)
    const eff = await page.evaluate(() => {
      const a = window.__aodl.store.getState().advanced
      return { bandwidth_3dB_MHz: a.efficiency.bandwidth_3dB_MHz, first_aod_usable_MHz: a.efficiency.first_aod_usable_MHz, min_product: a.trajEff.min_product }
    })
    results.checks.advanced_efficiency = eff
    ok(`advanced efficiency: 3 dB ${eff.bandwidth_3dB_MHz.toFixed(2)} MHz, first-AOD usable ${eff.first_aod_usable_MHz.toFixed(2)} MHz, trajectory min product ${eff.min_product.toFixed(4)}`)
    await shot(page, 'advanced_efficiency.png')
    judge('advanced_pick_and_place', record)
    await page.close()
  }

  // flagship 10×10 in the browser
  {
    const { page, record } = await loadPage(context, '/basic?example=flagship_10x10', 'basic_flagship_10x10', { timeoutMs: 180_000 })
    const meta = await page.evaluate(() => {
      const s = window.__aodl.store.getState()
      return { mode: s.plan.mode, notes: s.sim.notes, n_frames: s.sim.frames.length, max_spots: Math.max(...s.sim.frames.map((f) => f.x_um.length)), tone_counts: s.plan.tone_counts }
    })
    results.pages.basic_flagship_10x10 = { ...results.pages.basic_flagship_10x10, ...meta }
    const order1 = meta.notes.some((n) => /mixing order 1/.test(n))
    const routeMs = (p) => results.pages.basic_flagship_10x10.api.filter((e) => e.route.endsWith(p)).map((e) => `${e.ms} ms / ${e.kB} kB`).join(', ')
    if (meta.mode === 'shepard' && order1) ok(`flagship: ${meta.mode}; movie ready ${results.pages.basic_flagship_10x10.ms_to_sim} ms after navigation; /api/plan ${routeMs('/plan')}; /api/simulate ${routeMs('/simulate')} (${meta.n_frames} frames, ≤ ${meta.max_spots} spots/frame, order-1 fallback noted)`)
    else fail(`flagship unexpected: ${JSON.stringify(meta)}`)
    const fps = await measureFps(page, 2)
    results.checks.fps_flagship = fps
    if (fps.fps >= 30) ok(`playback flagship: ${fps.fps.toFixed(1)} fps (worst gap ${fps.worst_gap_ms.toFixed(1)} ms, ${fps.spots_at_cursor} spots at the cursor)`)
    else fail(`playback flagship only ${fps.fps.toFixed(1)} fps`)
    await page.evaluate(() => {
      const s = window.__aodl.store.getState()
      s.setTime(150 + 125 + s.plan.constants.half_tau_us) // mid-traverse
    })
    await page.waitForTimeout(500)
    await shot(page, 'basic_flagship_10x10.png')
    judge('basic_flagship_10x10', record)
    await page.close()
  }

  // production path: the backend serves the SPA with a deep-link fallback (hard reload)
  {
    const record = { consoleErrors: [], pageErrors: [], failedRequests: [] }
    const page = await context.newPage()
    watch(page, record)
    await page.goto(`${API}/basic?example=pick_and_place`, { waitUntil: 'networkidle' })
    await storeReady(page)
    await page.reload({ waitUntil: 'networkidle' })
    await storeReady(page)
    const title = await page.title()
    const url = page.url()
    const asset = await page.evaluate(async () => {
      const src = [...document.scripts].map((s) => s.src).find((s) => /\/assets\//.test(s))
      const r = await fetch(src)
      return { src, status: r.status, type: r.headers.get('content-type') }
    })
    const missing = await fetch(`${API}/no/such/route`)
    const missingType = missing.headers.get('content-type') ?? ''
    const badApi = (await fetch(`${API}/api/no-such-route`)).status
    const badAsset = (await fetch(`${API}/assets/no-such-file.js`)).status
    results.checks.spa_fallback = { title, url, asset, unknown_route_status: missing.status, unknown_route_type: missingType, unknown_api_status: badApi, missing_asset_status: badAsset }
    if (title === 'AODL Studio' && /\/basic\?example=pick_and_place$/.test(url) && asset.status === 200 && /javascript/.test(asset.type ?? '') && missing.status === 200 && /text\/html/.test(missingType) && badApi === 404 && badAsset === 404) ok(`SPA served by uvicorn at /: deep link survives a hard reload (${url}), assets as files (${asset.type}), unknown routes → index.html, unknown /api/* and missing /assets/* → 404`)
    else fail(`SPA fallback: ${JSON.stringify(results.checks.spa_fallback)}`)
    judge('served_spa_deep_link', record)
    await page.close()
  }

  await browser.close()
  const out = opt('--out')
  if (out) writeFileSync(out, JSON.stringify(results, null, 2))
  console.log('\nRESULTS ' + JSON.stringify(results))
  if (!flag('--keep')) stopAll()
  if (results.failures.length) {
    console.error(`\n${results.failures.length} check(s) failed`)
    process.exitCode = 1
  } else console.log('\nall e2e checks passed')
}

main().catch((e) => {
  console.error(e)
  stopAll()
  process.exit(1)
})
