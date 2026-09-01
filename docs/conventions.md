# Conventions — axes, signs, retarded time, units

Companion to [`PLAN.md`](PLAN.md) §1.1–1.3 and [`ARCHITECTURE.md`](ARCHITECTURE.md) §3.
Everything on this page is implemented in **`src/aodl/device/conventions.py`** and nowhere
else, and pinned by `tests/test_conventions.py`. No other module may hardcode a
sound-direction, diffraction-order or defocus sign: read the table and the helper functions.

Equation numbers `S#` refer to the Supplement of arXiv:2510.11451.

---

## 1. Units and coordinates

SI everywhere internally (Hz, m, s, rad); `aodl.units` constants exist only so boundary
code can be written `100 * MHz`. Frequencies in the waveform IR are **detunings from
`AODParams.f_center`** (rotating frame, Eq. S2) — the carrier is put back only in
`waveform/export.py` and in the `device/aod.py` aperture-window diagnostic.

| Symbol | Meaning |
|--------|---------|
| `u` | aperture coordinate along a channel's own axis, `u = 0` at the beam center [m] |
| `v` | acoustic velocity [m/s] |
| `D`, `tau = D/v` | active aperture [m] and acoustic transit time [s] |
| `t` | frame (observation) time [s]; `t'` a drive time |
| `t_c = t - tau/2` | drive time whose sample illuminates the beam center |
| `theta1`, `theta2` | pupil phase Taylor coefficients [rad/m], [rad/m²] |
| `X, Y` | lateral image coordinates [m] — lab and Eq. S11 agree |
| `Z_lab`, `Z_S11` | axial coordinate; `Z_lab = Z_LAB_SIGN * Z_S11`, see §6 |

**Axes.** `axis = 0` is x, `axis = 1` is y.

---

## 2. Channel geometry (Eq. S7)

```python
CHANNEL_GEOMETRY = {"Ax": (0, -1), "Bx": (0, +1), "Ay": (1, -1), "By": (1, +1)}
#                          axis  sound_sign
DIFFRACTION_ORDER = +1     # every channel
```

`sound_sign = s = +1` means the acoustic wave travels toward `+axis`, so its transducer sits
at `u = -D/2`; `s = -1` means it travels toward `-axis` from a transducer at `u = +D/2`.
The A/B members of a pair counter-propagate, which is what makes the Table I mapping of §5
come out with a *difference* of frequencies laterally and a *sum* of chirp rates axially.

---

## 3. Retarded phase and the beam-center Taylor expansion (Eqs. S4–S6)

The acoustic field at aperture coordinate `u` is the drive delayed by the travel time from
the transducer, so the channel imprints on the optical field the phase

```
-Phi(t_ret(u)),        Phi(t') = 2 pi f_center t' + phase(t'),      t_ret(u) = t_c - s u / v
```

(the leading minus sign is the `+1` order: the weak-drive expansion `exp(i C V) ~ 1 + i C V`
splits each cosine into `exp(+i Phi)` and `exp(-i Phi)`, and the order that *up*-shifts the
optical frequency keeps the latter — see §4).

Expanding about the beam center with `s² = 1`:

```
-Phi(t_c - s u / v) = -Phi(t_c)
                      + s (2 pi / v) [f_center + f(t_c)] u
                      -   (2 pi / (2 v^2)) fdot(t_c) u^2
                      + O(u^3)                                   <- coma, dropped in M1
```

Dropping the carrier — a constant phase plus a *common* tilt `s 2 pi f_center u / v` that
simply defines where the optical axis points — leaves the per-channel contributions

```
theta1 += s * 2 pi f(t_c) / v          [rad/m]     deflection
theta2 += -2 pi fdot(t_c) / (2 v^2)    [rad/m^2]   cylindrical chirp lens
```

**`theta2` does not depend on `s`.** That is the whole trick of the 3D-AODL (`PLAN.md`
§1.2): a counter-propagating pair driven with equal chirps cancels its deflection while its
lensing adds.

The amplitude envelope expands the same way (Eq. S5):

```
alpha(u) = A(t_c) - s (A'(t_c) / v) u + (A''(t_c) / (2 v^2)) u^2
```

— 0th order is intensity control, 1st is an amplitude tilt (carries `s`), 2nd is **acoustic
irising**. Products across channels are truncated back to degree 2.

*Bookkeeping note.* `device/aod.py` folds `A(t_c)` into the complex line amplitude
`amp = (i C / 2) A(t_c) exp(-i phase(t_c))` (Eq. S3), so `device/aodl.py` multiplies the
**normalized** shape `(1, -s (A'/A) / v, (A''/A) / (2 v^2))` instead, and a constant
envelope gives `alpha = (1, 0, 0)`. The physics is unchanged; only `A` is not counted twice.

---

## 4. Optical frequency bookkeeping

Each `+1` order shifts the light by `+(f_center + f)` in absolute terms. The `f_center`
part is common to every term and is ignored; a term's tag is

```
df_opt = sum over participating channels of f(t_c)      [Hz]
```

`field/focal.py` groups terms by `df_opt` (default tolerance `GROUP_TOL` = **1 kHz**, which
also caps a group's total width, not just the gap between neighbours — see
`focal.group_terms`): degenerate terms interfere coherently, distinct groups add in intensity
because their MHz beat notes average out over any camera or atomic timescale (`PLAN.md` §1.3).

---

## 5. Lateral position — Table I

Eq. S11 maps a pupil tilt to an image position as `X = theta1 F / k` (pinned by
`tests/test_focal_geometry.py`), so

```
X_spot = (F/k) * sum_{x channels} s * 2 pi f / v = deflection_scale * sum_{x channels} s * f
```

with `deflection_scale = lambda F / v`. With `s(Ax) = -1`, `s(Bx) = +1` this is exactly the
paper's `X = (lambda F / v) (f_Bx - f_Ax)`, and likewise `Y = (lambda F / v)(f_By - f_Ay)`.
Lateral image coordinates need no sign flip: lab `X, Y` *are* the Eq. S11 `X, Y`.

---

## 6. Axial sign — `Z_LAB_SIGN = -1`

This is the one place where the paper's reported quantity and a literal Eq. S11 evaluation
disagree in sign, so it is spelled out.

1. WO-01's `tests/test_focal_geometry.py` pins the Eq. S11 defocus mapping: a pupil
   curvature `theta2` puts the sharpest focus at

   ```
   Z_S11 = 2 F^2 theta2 / k          (equivalently  theta2 = k Z_S11 / (2 F^2))
   ```

2. §3 gives, per axis, `theta2_axis = -pi * sum_axis(fdot) / v^2`. Substituting:

   ```
   Z_S11_axis = -(lambda F^2 / v^2) * sum_axis(fdot) = -lens_scale * sum_axis(fdot)
   ```

3. The paper's Table I instead reports
   `Zbar = +(1/2)(lambda F^2 / v^2)(fdot_Ax + fdot_Bx + fdot_Ay + fdot_By)` — an up-chirp
   moves the tweezer *above* the static focal plane.

We adopt **Table I's sign as the lab axis**:

```
Z_lab = Z_LAB_SIGN * Z_S11,     Z_LAB_SIGN = -1
```

so that

```
Z_axis_lab = + lens_scale * sum_axis(fdot)
Zbar       = (Z_x_lab + Z_y_lab) / 2 = (1/2) lens_scale * sum_all(fdot)     <- Table I
Delta F    =  Z_x_lab - Z_y_lab      =       lens_scale * (x sum - y sum)   <- Table I
```

**Rule.** Every user-facing, trajectory and metrics `Z` is **lab Z**. `field/` converts on
the way in (`Z_S11 = Z_LAB_SIGN * Z_lab`) and nothing else touches the conversion.

*Worked single-channel case (the M1 acceptance test).* Driving only `Ay` with `fdot > 0`
gives `Z_y_lab = +lens_scale * fdot` and `Z_x_lab = 0`; in Table I's variables
`Zbar = lens_scale * fdot / 2` and `Delta F = -lens_scale * fdot`, and indeed
`Z_x = Zbar + DeltaF/2 = 0`, `Z_y = Zbar - DeltaF/2 = lens_scale * fdot`. One AOD makes a
cylindrical lens, i.e. a pure astigmat — exactly the M1 phenomenology.

---

## 7. Acoustic timing and the aperture fill window

The drive starts at `t = 0`. The sample emitted at drive time `t'` has travelled from the
transducer at `u = -s D/2`, so it sits where

```
t - (s u + D/2) / v = t'
```

Two consequences:

* **Beam-center retarded time** `t_c = t - tau/2` (set `u = 0`, `t' = t_c`). Every
  frequency, chirp rate, envelope and phase used by the device layer is evaluated at `t_c`,
  never at `t`. The difference is not cosmetic: a chirp at `fdot` puts the spot
  `deflection_scale * fdot * tau / 2` away from where the naive `t_c = t` would place it
  (about 3 µm ≈ 3 waists at the default hardware and 50 MHz/ms).
* **Fill window.** The aperture holds drive content exactly where

  ```
  s u <= v t - D/2
  ```

  i.e. on the half-line containing the transducer, bounded by the leading wavefront at

  ```
  u_edge = s (v t - D/2)
  ```

  The aperture is **fully filled for `t >= tau`**, and `device/aod.py:fill_edge` returns
  `None` there. Otherwise it returns `FillEdge(u_edge, side)` with

  | `sound_sign` | filled region | `side` | Gaussian moments to use |
  |---|---|---|---|
  | `+1` | `u <= u_edge` | `"upper"` | `gauss_moments_upper(a, b, u_edge)` |
  | `-1` | `u >= u_edge` | `"lower"` | `gauss_moments_lower(a, b, u_edge)` |

  The input beam itself is modelled as an **uncropped** Gaussian (`PLAN.md` decision 2), so
  this fill edge is the only aperture window appearing in the field integrals; the physical
  `|u| <= D/2` crop is deliberately not applied. `device/aod.py:aperture_window` is a
  diagnostic that *does* return the literal `[-D/2, +D/2]` waveform on the crystal (carrier
  included), for plotting the startup transient.

  **Counter-propagating pairs (two channels on one axis).** Each channel contributes its
  filled half-line; `device/aodl.py:_axis_interval` intersects them. For an Ax+Bx pair
  starting together at `t = 0` the intersection is `[D/2 − vt, vt − D/2]`: **empty for
  `t < tau/2`** (both waves must reach a point before the pair diffracts there), so a
  pair-driven tweezer is strictly dark until `tau/2`, then grows with window
  `FillWindow(lo, hi)` and `gauss_moments_window(a, b, lo, hi)` until full at `tau`.
  Empty intersection ⇒ the term is dropped (darkness is physics, not pruning — it does
  not count toward `pruned_power`).

---

## 8. Summary table

| Quantity | Convention | Owner |
|----------|-----------|-------|
| axis index | 0 = x, 1 = y | `AXIS_X`, `AXIS_Y` |
| sound direction | `+1` toward `+axis`, transducer at `-D/2` | `CHANNEL_GEOMETRY` |
| diffraction order | `+1`, imprints `-Phi(t_ret)`, up-shifts by `f_center + f` | `DIFFRACTION_ORDER` |
| tilt | `theta1 += s 2 pi f / v` | `theta1_contribution` |
| curvature | `theta2 += -pi fdot / v^2` | `theta2_contribution` |
| amplitude poly | `A - s (A'/v) u + (A''/2v²) u²` | `amplitude_poly` |
| optical frequency tag | `df_opt = sum f` | `device/aodl.py` |
| lateral | `X = theta1 F / k = deflection_scale * sum(s f)` | Eq. S11 mapping |
| axial | `Z_lab = -Z_S11`, `Z_axis_lab = +lens_scale * sum(fdot)` | `Z_LAB_SIGN` |
| retarded time | `t_c = t - tau/2`, `t_ret(u) = t_c - s u / v` | `beam_center_time`, `retarded_time` |
| fill | `s u <= v t - D/2`, full at `t >= tau` | `is_filled`, `fill_edge` |
