import { Link } from 'react-router-dom'

export function ModeCards() {
  return (
    <div className="cards">
      <Link to="/basic" className="card">
        <h4>Basic mode</h4>
        <p>
          Edit the array and the moves, read the band usage and the Eq. 1 axial budget, scrub the RF drives with the acoustic window of each
          crystal, and play the tweezer movie (astigmatic Gaussian approximation) with exact Eq. S11 frames on demand. Download the generated
          Python and the parametric NPZ.
        </p>
        <div className="tags">
          <span className="badge">aodl.plan_motion</span>
          <span className="badge">aodl.simulate</span>
        </div>
      </Link>
      <Link to="/advanced" className="card">
        <h4>Advanced mode</h4>
        <p>
          A separate FFT pupil engine: Bragg / tangential-phase-matching efficiency of the first AOD and of the A–B pair, efficiency along the
          trajectory, per-AOD misalignment (offset, rotation, delay, 4f magnification, tilt) with Strehl, waists and astigmatism against the
          aligned reference, RF spectra and crest factor.
        </p>
        <div className="tags">
          <span className="badge">aodl_advanced</span>
          <span className="badge">model, not measurement</span>
        </div>
      </Link>
      <Link to="/tutorial/shepard" className="card">
        <h4>Tutorial · fading-Shepard tones</h4>
        <p>
          The same tweezer lifted and held past the Eq. 1 budget, side by side: a plain drive whose co-chirp walks out of the band, and fading-Shepard
          ladders that hand the chirp from rung to rung. RF tracks, trap power (ideal, in-band, efficiency model), shared-cursor movies and the shadow
          tweezers of each hand-over.
        </p>
        <div className="tags">
          <span className="badge">tutorial_hold_plain</span>
          <span className="badge">tutorial_hold_shepard</span>
        </div>
      </Link>
    </div>
  )
}
