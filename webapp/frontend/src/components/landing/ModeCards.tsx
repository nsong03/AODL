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
    </div>
  )
}
