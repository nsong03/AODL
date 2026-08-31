"""Unit constants.

SI is used everywhere internally (Hz, m, s, rad).  These constants exist purely so that
boundary code can be written as ``100 * MHz`` or ``7.5 * mm`` instead of magic numbers;
never bake a unit conversion into physics code.

Every constant is "how many SI units one of these is", so multiplying converts *into* SI
(``2.0 * mm -> 0.002``) and dividing converts *out of* SI (``0.002 / mm -> 2.0``).
"""

# Frequency
kHz = 1e3
MHz = 1e6
GHz = 1e9

# Length
nm = 1e-9
um = 1e-6
mm = 1e-3

# Time
us = 1e-6
ms = 1e-3

__all__ = ["GHz", "MHz", "kHz", "mm", "ms", "nm", "um", "us"]
