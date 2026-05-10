"""
turbulence.py -- Von Karman turbulence spectrum.

Implements the one-sided power spectral density (PSD) for vertical atmospheric
turbulence according to the Von Karman model (MIL-SPEC / ESDU standard):

    Phi(omega) = sigma^2 * (L / (pi*V))
                    * [1 + (8/3) * (1.339 * L*omega/V)^2]
                    / [1 + (1.339 * L*omega/V)^2]^(11/6)

This formulation gives a one-sided PSD in (m/s)^2 / (rad/s).
The variance satisfies:

    integral(Phi(omega), 0, inf) = sigma^2
"""

import numpy as np


def von_karman_psd(omega, V, L, sigma):
    """
    Compute the Von Karman one-sided PSD for vertical gust velocity.

    Parameters
    ----------
    omega : ndarray
        Angular frequency vector in rad/s.  Must be non-negative.
    V : float
        True airspeed in m/s.
    L : float
        Turbulence scale length in m (typically 762 m at cruise altitude).
    sigma : float
        Standard deviation of the turbulence velocity (m/s).

    Returns
    -------
    phi : ndarray, same shape as omega
        One-sided PSD in (m/s)^2 / (rad/s).

    Notes
    -----
    The model aircraft input is wind incidence alpha_w = w/V (rad), not wind
    speed w (m/s).  To obtain the incidence PSD divide by V^2:
        Phi_alpha(omega) = Phi_w(omega) / V^2

    The numerical constant 1.339 arises from matching the Von Karman spectrum
    to the Dryden rational approximation.

    Normalisation: integral(Phi(omega), 0, inf) = sigma^2.

    Examples
    --------
    >>> import numpy as np
    >>> omega = np.linspace(1e-3, 100, 10000)
    >>> phi = von_karman_psd(omega, V=247.6, L=762.0, sigma=21.58)
    >>> # Verify variance: integral should be close to sigma^2
    >>> variance = np.trapezoid(phi, omega)
    >>> abs(variance - 21.58**2) / 21.58**2 < 0.02
    True
    """
    Omega = 1.339 * L / V * omega
    phi = (sigma**2 * L / (np.pi * V)
           * (1 + (8 / 3) * Omega**2)
           / (1 + Omega**2) ** (11 / 6))
    return phi
