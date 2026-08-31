# WO-01 — Core scaffold, PiecewisePoly, Gaussian integral kernel, reference integrator

**Role:** implementation agent, Wave A. **You are the wave closer**: when done, commit your
files and push (`git push -u origin claude/optical-tweezer-simulation-ik9m8f`).
**Read first:** `CLAUDE.md`, `docs/PLAN.md` §1–2, `docs/ARCHITECTURE.md`.

## Owned files

```
pyproject.toml  .gitignore
src/aodl/__init__.py            (create: version string + empty; assembly wires exports later)
src/aodl/units.py  src/aodl/params.py  src/aodl/poly.py
src/aodl/field/__init__.py  src/aodl/field/gaussian.py  src/aodl/field/reference.py
src/aodl/trajectory/__init__.py  src/aodl/waveform/__init__.py
src/aodl/device/__init__.py  src/aodl/viz/__init__.py     (empty package inits)
tests/conftest.py  tests/test_poly.py  tests/test_gaussian.py  tests/test_focal_geometry.py
```

## 1. Packaging

`pyproject.toml`: package `aodl`, `src` layout, Python ≥ 3.11; deps `numpy`, `scipy`,
`matplotlib`, `imageio`, `imageio-ffmpeg`; extra `dev = [pytest, ruff, mypy, jupyterlab,
nbmake]`. Ruff config: line-length 100, `select = ["E","F","W","I","NPY"]`. Pytest config:
`testpaths = ["tests"]`. `.gitignore`: standard Python + `*.mp4 *.gif *.npz
.ipynb_checkpoints outputs/`.

## 2. `units.py`

Constants only: `MHz=1e6, kHz=1e3, GHz=1e9, um=1e-6, mm=1e-3, nm=1e-9, us=1e-6, ms=1e-3`.

## 3. `params.py` (frozen dataclasses)

```python
@dataclass(frozen=True)
class AODParams:
    sound_speed: float      # v [m/s]
    aperture: float         # D [m], active aperture along sound axis
    f_center: float         # [Hz] rotating-frame carrier (Eq. S2)
    band: tuple[float,float]# usable ABSOLUTE band [Hz], e.g. (90 MHz, 110 MHz)
    drive_strength: float = 0.30   # C·A at unit envelope: peak phase modulation [rad] (Eq. S1)
    @property def transit_time(self) -> float: ...    # D / v

@dataclass(frozen=True)
class OpticsParams:
    wavelength: float       # λ [m]
    focal_length: float     # F [m] (effective objective)
    w_in: float             # input beam 1/e² intensity radius at AOD plane [m]
    @property def k(self): 2π/λ
    @property def waist0(self): λF/(π w_in)         # focal waist of uncropped Gaussian
    @property def rayleigh(self): π waist0²/λ

CHANNELS = ("Ax", "Bx", "Ay", "By")

@dataclass(frozen=True)
class AODLParams:
    optics: OpticsParams
    channels: dict[str, AODParams]        # keys exactly CHANNELS
    @property def deflection_scale(self): # λF/v [m per Hz of frequency difference], Table I
    @property def lens_scale(self):       # λF²/v² [m·s], Table I  (use channel Ax's v; assert all equal)

def default_1030() -> AODLParams   # v=650, D=7.5mm, f_center=100 MHz, band=(90,110) MHz,
                                   # λ=1030 nm, F=6.5 mm, w_in=2.0 mm
def paper_808() -> AODLParams      # same but λ=808 nm
```

## 4. `poly.py` — PiecewisePoly

Frozen dataclass: `breaks: (K+1,) float64` strictly increasing; `coeffs: (K, D+1)` where
segment k evaluates as `sum_j coeffs[k,j] * tau**j`, **tau = (t - breaks[k]) /
(breaks[k+1] - breaks[k]) ∈ [0,1]** (normalized local time — keeps coefficients O(1)).

API (all vectorized over `t`):
- `__call__(t)` — evaluate; **clamp-hold outside domain** (t < start → value at start;
  t > end → value at end). Document this.
- `derivative()` / `antiderivative(c0=0.0)` — exact; handle the 1/T (resp. T) Jacobian of
  normalized time; antiderivative must be continuous across breaks (accumulate segment
  integrals).
- Constructors: `constant(value, t0, t1)`, `from_segment_coeffs(breaks, coeffs)`,
  `concat(polys)` (contiguous domains), `poly.shift(dt)`, `poly.scale(s)` (values × s),
  `poly.offset(c)` (values + c), `p + q` (identical overall domain required; refine to the
  union of interior breaks), `degree` property, `domain` property.
- `MAX_DEGREE = 9` module constant (serialization padding; assert on construction).

## 5. `field/gaussian.py` — closed-form kernel

All complex-valued, vectorized (numpy broadcasting), `Re(a) > 0` asserted.

Full-line moments $I_n(a,b)=\int_{-\infty}^{\infty} u^n e^{-au^2+bu}\,du$:

- `I0 = sqrt(pi/a) * exp(b²/(4a))`
- `I1 = (b/(2a)) * I0`
- `I2 = (1/(2a) + b²/(4a²)) * I0`

Lower-edge moments $E_n(a,b,u_0)=\int_{u_0}^{\infty} u^n e^{-au^2+bu}\,du$ — **use the
numerically stable erfcx form**, never the naive product `exp(b²/4a)·erfc(w)`:

Let `w = sqrt(a)*u0 - b/(2*sqrt(a))` (principal sqrt), and `g0 = exp(-a*u0² + b*u0)`
(the integrand at the edge — bounded in all physical cases). Then, using
`erfcx(z) = wofz(1j*z)` (scipy.special.wofz):

- if `Re(w) >= 0`:  `E0 = 0.5*sqrt(pi/a) * erfcx(w) * g0`
- else (reflection `erfc(w) = 2 - erfc(-w)`):  `E0 = I0(a,b) - 0.5*sqrt(pi/a) * erfcx(-w) * g0`

(Identity check used here: `b²/(4a) - w² = -a·u0² + b·u0`.) Implement branchless via
`np.where` on `Re(w)`. Then:

- `E1 = g0/(2a) + (b/(2a))*E0`
- `E2 = u0*g0/(2a) + E0/(2a) + (b/(2a))*E1`

Upper-edge moments $F_n(a,b,u_1)=\int_{-\infty}^{u_1}$ by symmetry `u → -u`:
`F_n(a,b,u1) = (-1)^n * E_n(a, -b, -u1)`.

Public API: `gauss_moments(a, b) -> (I0,I1,I2)`, `gauss_moments_lower(a, b, u0)`,
`gauss_moments_upper(a, b, u1)`, and `erfcx_complex(z)`.

## 6. `field/reference.py` — quadrature reference (tests only)

Eq. S11 without approximations, brute force:

```python
def reference_field_separable(pupil_x, pupil_y, optics, X, Y, Z, n=8001, span=6.0):
    # pupil_x: callable u -> complex 1D pupil factor (includes input-beam Gaussian)
    # trapezoid over u ∈ [-span·w_in, span·w_in];
    # kernel per axis: exp(-1j*k/F * u*X) * exp(-1j*k*Z/(2F²) * u²)
    # returns field ∝ (1/(i λ F)) * Ix(X,Z) * Iy(Y,Z)   (omit e^{ikF} and the common
    # exp[ik(X²+Y²)/(2F)] curvature phase; document that intensities are unaffected)
def reference_field_2d(pupil, optics, X, Y, Z, n=801, span=6.0):
    # pupil: callable (x, y) -> complex; non-separable double sum, chunked over image points
```

## 7. Tests

`tests/conftest.py`: fixtures `params1030 = default_1030()`, rng seeded.

`test_poly.py`
- min-jerk quintic on [0,T]: build via `from_segment_coeffs` with coeffs
  `x_i + Δ·[0,0,0,10,-15,6]`; assert `p(0)=x_i, p(T)=x_f`, derivative zero at both ends,
  antiderivative(derivative(p)) − p = const to 1e-12 (relative).
- random piecewise (5 segments, degree ≤ 6): eval vs `numpy.polynomial` per-segment
  reference at 1000 points, 1e-12.
- clamp-hold semantics; `+` with break-union; `concat`; `shift`.
- phase-continuity: antiderivative of a discontinuous-derivative piecewise is continuous
  at every break (1e-12 × scale).

`test_gaussian.py`
- `I_n`, `E_n`, `F_n` vs `scipy.integrate.quad` (real/imag separately, generous limits) over
  ≥ 200 random draws: `a = 10^U(-2,2) · e^{iθ}, |θ|<80°` scaled so Re a>0;
  `b` complex with `|b| ≤ 10·sqrt(|a|)`; `u0 ∈ [-3,3]/sqrt(|a|)`. Rel. tol 1e-9.
- Stability: physical-scale case `a ≈ 2.5e5 − 3.7e5j m⁻²`, `b = i·1e6 m⁻¹`, u0 = ±2 mm —
  finite, matches quad; and an extreme `|b²/4a| ~ 700` case (naive form would overflow)
  returns finite values matching the identity `E0 + (F0 at same edge) = I0`.

`test_focal_geometry.py` — pins the S11 geometry mapping used by later waves:
for a synthetic separable pupil term
`pupil(u) = (α0+α1 u+α2 u²) · exp(-u²/w_in² + iθ2 u² + iθ1 u)` (random modest θ1, θ2, α):
analytic field via `gauss_moments` with **`a = 1/w_in² − i(θ2 − kZ/(2F²))`,
`b = i(θ1 − kX/F)`** equals `reference_field_separable` on a 21×21 (X,Z) patch to rel.
1e-8 (normalize by peak). Also: θ1=θ2=0 → fitted waist = `optics.waist0` (1e-6),
intensity peak at Z where `θ2 = kZ/(2F²)` for θ2≠0 (this pins the defocus sign that
`device/conventions.py` will reference).

## Definition of done

`pip install -e ".[dev]"` succeeds; `pytest` green; `ruff check src tests` clean;
committed (message: `M0: scaffold, piecewise polynomials, Gaussian integral kernel`,
plus the footer lines given in your dispatch instructions) and pushed. Report: file list,
test summary, any spec friction.
