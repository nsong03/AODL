# WO-04 — Field assembly: analytic focal fields, frequency grouping, spot metrics

**Role:** implementation agent, Wave B (parallel with WO-02/WO-03). **No git.**
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1.3, `docs/ARCHITECTURE.md` §3, then this.
WO-01 is merged (`field/gaussian.py`, `field/reference.py`, params). WO-03's `TermArray`
interface is frozen in `docs/workorders/WO-03-device.md` §3 — code against it; for your
tests, construct synthetic TermArrays yourself (do NOT import `device/` — that keeps the
two Wave-B tracks independent; integration happens in WO-05). If the dataclass isn't
importable at your runtime, define a structurally identical local test fixture and note it.

## Owned files

```
src/aodl/field/focal.py  src/aodl/field/measure.py
tests/test_focal.py  tests/test_measure.py
```

## 1. `field/focal.py`

Per-axis mapping (pinned by WO-01 `test_focal_geometry`): for image coordinate (X, Z_S11):

```
a_axis = 1/w_in² − i·(θ2_axis − k·Z_S11/(2F²))
b_axis = i·(θ1_axis − k·X/F)
field_axis = α0·I0(a,b) + α1·I1(a,b) + α2·I2(a,b)      # gauss_moments
             (edge-windowed variants when the term carries a fill edge on that axis)
U_term(X, Y) = c · field_x · field_y                    # constant prefactors dropped
                                                        # (document: intensity-safe)
```

All Z arguments in the public API are **lab Z**; convert internally via
`conventions.Z_LAB_SIGN` (import the constant — it is the one allowed device import).

Public API (all vectorized over terms; complex128):

```python
def term_field(terms, optics, X, Y, z_lab)         # X,Y broadcastable arrays, scalar z_lab
def group_terms(terms, tol=10*kHz) -> list[slice/index arrays]   # cluster by df_opt
def intensity_frame(terms, optics, grid, z_lab) -> float[ny, nx]
    # grid: FrameGrid dataclass (x0, x1, nx, y0, y1, ny) in meters
    # Per frequency group: union bounding patch (see §patch), accumulate complex field on
    # patch, |·|², add into canvas. Cross-group beats average out (Supplement, interlaced
    # fading logic).
def intensity_slice_xz(terms, optics, x_axis, z_axis_lab, y0) -> float[nz, nx]
    # same grouping; evaluate at fixed Y = y0 over an (X, Z_lab) grid — the XZ panel
```

**Patch policy:** spot center per term: X_c = θ1x·F/k, Y_c = θ1y·F/k; per-axis intensity
1/e² radius at the evaluation plane: `w_eff = sqrt(2/Re(A))/ (k/F)`... derive from
`|field|² ∝ exp(−(k/F)²·Re(1/a)·ΔX²·½·…)` — concretely: the X-dependence of |field_x|² is
`exp(2·Re(b²/(4a)))` with b = i(θ1 − kX/F); expand to get the Gaussian radius
`w_eff = (F/k)·sqrt(2/Re(1/a))⁻¹`… **implement as**: `sigma² = Re(1/a)/2 · (F/k)²... `
Do the derivation carefully once, verify against `reference_field_separable`, and write it
as a helper `spot_params(terms, optics, z_lab) -> (Xc, Yc, wx, wy)` used by both the patch
logic and `measure.py`. Patch = union over group members of center ± 4·max(wx, wy),
clipped to grid; guard patches ≥ 3×3 px.

## 2. `field/measure.py`

Closed-form per-group metrics (no fitting where avoidable):

```python
@dataclass class SpotMetrics:
    x, y: float          # lab meters (amplitude-weighted over group fundamentals)
    z_lab: float         # best-focus lab Z:  mean of per-axis  Z_axis; per axis from
                         # θ2_axis = k·Z_S11/(2F²)  →  Z_axis_lab = Z_LAB_SIGN·2F²θ2/k
    delta_f: float       # astigmatic interval  Z_x_lab − Z_y_lab   (Table I ΔF)
    sigma_astig: float   # delta_f / rayleigh   (paper's σ_astig)
    wx, wy: float        # intensity 1/e² radii at z_lab (from spot_params)
    power: float         # Σ|c|² · (closed-form ∫∫|U|², or peak·wx·wy·π/2 — document choice)
    df_opt: float        # group frequency tag

def measure(terms, optics) -> list[SpotMetrics]        # one per frequency group
def track_z(metrics) -> float                          # power-weighted mean z_lab (movie
                                                       # auto-tracking plane)
```

## 3. Tests (synthetic TermArrays only)

`test_focal.py`
- Single term, θ1 = θ2 = 0, α = (1,0,0): `intensity_frame` peak at origin, waist =
  `optics.waist0` (fit 1D cuts, rel 1e-4); matches `reference_field_separable` frame on
  a 31×31 patch (rel 1e-6 after peak normalization).
- Term with θ1x ≠ 0, θ2y ≠ 0: peak at (θ1x·F/k, 0); evaluating `intensity_frame` at
  `z_lab = Z_LAB_SIGN·2F²θ2y/k` maximizes on-axis intensity vs neighboring z (scan);
  x/y waists differ off-focus (astigmatism visible).
- Two terms, same df_opt, same location, phases 0 and π: intensity_frame ≈ 0 (destructive —
  coherent within group). Same two terms with df_opt differing by 1 MHz: intensity =
  sum of individual intensities (incoherent across groups). Tol 1e-10 relative.
- Edge-windowed term (fill edge at u0 = −0.5·w_in on x): matches
  `reference_field_separable` with hard-edged pupil (rel 1e-3).
- `intensity_slice_xz`: for a chirped-like term (θ2 both axes equal), the slice's
  brightest Z tracks the predicted z_lab (2% of rayleigh).
- Patch accumulation == brute full-grid evaluation for a 3-group, 5-term random scene
  (exact to 1e-12 where patches cover, and patch coverage misses < 1e-6 of total power).

`test_measure.py`
- Random single terms: measure() returns the closed-form positions/z/ΔF used to build
  them (1e-9 — algebra identity); sigma_astig = ΔF/z_R.
- Astigmatic term (θ2x = −θ2y): z_lab ≈ 0, |delta_f| = |Z_x − Z_y| matches
  2·|Z_axis|; wx(z=0) > wy(z=0) ordering consistent with signs.
- `track_z`: power weighting verified with two groups of unequal |c|².

## Definition of done

Own tests green; `ruff check` clean on owned files; no git; report per ORCHESTRATION.md
(include the derived `spot_params` formula in the report so the architect can review it).
