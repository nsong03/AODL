import { Link } from 'react-router-dom'
import { ExampleCards } from '../components/landing/ExampleCards'
import { ModeCards } from '../components/landing/ModeCards'
import { SetupDiagram } from '../components/landing/SetupDiagram'

export default function Landing() {
  return (
    <div className="landing">
      <div className="hero">
        <div>
          <h1>AODL Studio</h1>
          <p className="lede">
            A 3D trajectory for an atom array goes in; the four RF drives of a 3D acousto-optic deflector lens come out — with a preview of
            the tweezers they make.
          </p>
          <p className="caption">
            Physics: arXiv:2510.11451 through the <code>aodl</code> package (Eq. S19 synthesis, fading-Shepard ladders, closed-form Eq. S11
            fields). The movie is the astigmatic Gaussian approximation of the closed-form spot metrics; exact frames are one click away.
          </p>
          <div className="cta">
            <Link to="/basic?example=pick_and_place" className="primary">
              Open the workbench
            </Link>
            <Link to="/advanced">Advanced engine</Link>
          </div>
        </div>
        <div className="caption">
          <strong className="bright">How to read the diagram.</strong> The beam crosses four tellurium-dioxide crystals whose acoustic axes
          alternate x, x, y, y in counter-propagating pairs. A tone at frequency <code>f</code> deflects by <code>λ f / v</code>; a chirp{' '}
          <code>ḟ</code> adds a cylindrical lens. The sample illuminating the beam centre left its transducer <code>τ/2 = D/2v</code> earlier
          (5.8 µs at the default hardware), so the atom plane lags the drive — AODL Studio shows that lag rather than hiding it.
        </div>
      </div>

      <SetupDiagram />

      <div className="section">
        <h2>Table I — what the four frequencies control</h2>
        <div className="mapping">
          <div className="eq">
            <div className="f">X = (λF/v) · (f_Bx − f_Ax)</div>
            <div className="what">lateral x: the counter-propagating pair's deflections subtract — 10.3 µm per MHz at the default hardware</div>
          </div>
          <div className="eq">
            <div className="f">Y = (λF/v) · (f_By − f_Ay)</div>
            <div className="what">lateral y, likewise; an array is a ladder of tones spaced Δf on Bx (columns) and By (rows)</div>
          </div>
          <div className="eq">
            <div className="f">Z̄ = ½ · (λF²/v²) · (ḟ_Ax + ḟ_Bx + ḟ_Ay + ḟ_By)</div>
            <div className="what">axial: the chirp lenses add regardless of sound direction — 4.85 MHz/ms per µm of Z̄ on every channel</div>
          </div>
          <div className="eq">
            <div className="f">ΔF = (λF²/v²) · (ḟ_Ax + ḟ_Bx − ḟ_Ay − ḟ_By) = 0</div>
            <div className="what">astigmatism-free 3D control: equal chirps on both axes leave exactly three degrees of freedom (X, Y, Z̄)</div>
          </div>
        </div>
      </div>

      <div className="section">
        <h2>How it works</h2>
        <div className="cards">
          <div className="card">
            <h4>Deflection = frequency difference</h4>
            <p>
              Each AOD imprints the phase <code>−Φ(t − s·u/v)</code> of its retarded drive. The linear term is a tilt <code>s·2πf/v</code>; the
              A and B members of a pair have opposite <code>s</code>, so their tilts subtract and the trap moves only with{' '}
              <code>f_B − f_A</code>. Frequencies are detunings from the 100 MHz carrier (the rotating frame of Eq. S2).
            </p>
          </div>
          <div className="card">
            <h4>Lens = chirp sum</h4>
            <p>
              The quadratic term <code>−πḟ/v²</code> does not depend on the sound direction, so a common chirp on all four channels is a lens:
              the focus moves along Z̄ while <code>ΔF</code> stays zero. Holding Z off the focal plane costs a chirp that walks the frequency
              through the band — the Eq. 1 budget, 206 µs for 10 µm at the default hardware.
            </p>
          </div>
          <div className="card">
            <h4>Fading Shepard for sustained Z</h4>
            <p>
              Beyond the budget each tone becomes a ladder spaced Δf whose rungs fade in and out with a <code>cos^p</code> window as the
              common chirp walks past them (Eqs. S24–S28) — a Shepard tone: periodic in the walk, bounded in the band. Mid-fade, the same lines
              make two shadow tweezers at <code>±(λF/v)·Δf</code> with half a trap's power each; the fade schedule is reported.
            </p>
          </div>
        </div>
      </div>

      <div className="section">
        <h2>Modes</h2>
        <ModeCards />
      </div>

      <div className="section">
        <h2>Starter examples</h2>
        <p className="caption" style={{ marginBottom: 10 }}>
          Each opens the basic workbench with the trajectory loaded. The three quick links are the brief's: a single tweezer lifted and held
          (fading Shepard), a 4×4 array lifted and held, and a 3×3 array on an arbitrary tilted 3D path.
        </p>
        <ExampleCards />
      </div>
    </div>
  )
}
