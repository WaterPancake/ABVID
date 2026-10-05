"""Numeric components extracted from CAST/src/cast/synthetic.py."""
def fixtures():
    base = {"spacing_hz": [170.0]*3, "harmonic_weights": [0.6, .2, .1, .05, .03, .01, .005, .005],
            "noise_weights": [.04, .06, .1, .3, .25, .15, .06, .04], "harmonic_fraction": 1.0,
            "envelope_knots": [1.0]*5}
    return {
        "single_tone": {**base, "spacing_hz": [233.25]*3, "harmonic_weights": [1., 0, 0, 0, 0, 0, 0, 0]},
        "unrestricted_tone": {**base, "spacing_hz": [233.25]*3, "harmonic_weights": [1., 0, 0, 0, 0, 0, 0, 0]},
        "harmonic": base,
        "noise": {**base, "harmonic_fraction": 0.0},
        "changing": {**base, "spacing_hz": [130., 145., 170.], "envelope_knots": [.4, .9, 1.7, .8, .3]},
        "mixture": {**base, "spacing_hz": [105., 110., 100.], "harmonic_fraction": .55,
                    "envelope_knots": [.5, .9, 1.4, 1., .6]},
        "missing_fundamental": {**base, "spacing_hz": [95.]*3, "harmonic_weights": [0, .65, 0, .35, 0, 0, 0, 0]},
    }
