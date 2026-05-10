"""
transfer_functions.py -- Frequency-domain evaluation of LTI transfer functions.

Evaluates H(j*omega) = C * (j*omega*I - A)^-1 * B + D for arrays of
frequencies using an efficient linear-solve approach (avoids explicit
matrix inversion).
"""

import numpy as np
import scipy.linalg


def compute_tf(A, B, C, D, omega):
    """
    Evaluate the frequency-response matrix H(j*omega) at a vector of frequencies.

    Uses scipy.linalg.solve to compute (j*omega*I - A)^-1 * B for each omega,
    which is numerically more stable than explicit inversion.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        State matrix.
    B : ndarray, shape (n, m)
        Input matrix.
    C : ndarray, shape (p, n)
        Output matrix.
    D : ndarray, shape (p, m)
        Feed-through matrix.
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s at which to evaluate H.

    Returns
    -------
    TF : ndarray, shape (p, m, N), dtype complex
        Frequency-response tensor.  TF[i, j, k] is the complex gain
        from input j to output i at frequency omega[k].

    Examples
    --------
    >>> import numpy as np
    >>> A = -np.eye(2); B = np.ones((2, 1)); C = np.eye(2); D = np.zeros((2, 1))
    >>> TF = compute_tf(A, B, C, D, np.array([0.0, 1.0]))
    >>> TF.shape
    (2, 1, 2)
    """
    n_out = C.shape[0]
    n_in  = B.shape[1]
    n_freq= len(omega)
    TF = np.zeros((n_out, n_in, n_freq), dtype=complex)
    I  = np.eye(A.shape[0])
    for k, w in enumerate(omega):
        X = scipy.linalg.solve(1j * w * I - A, B)
        TF[:, :, k] = C @ X + D
    return TF


def bode_data(TF, freq_hz):
    """
    Extract magnitude (dB) and phase (deg) from a transfer function tensor.

    Parameters
    ----------
    TF : ndarray, shape (p, m, N), complex
        Frequency-response tensor from compute_tf().
    freq_hz : ndarray, shape (N,)
        Frequency vector in Hz (passed through to output).

    Returns
    -------
    magnitude_db : ndarray, shape (p, m, N)
        Magnitude in dB:  20 * log10(|H|).
    phase_deg : ndarray, shape (p, m, N)
        Phase in degrees.
    freq_hz : ndarray, shape (N,)
        Pass-through of the input frequency vector.
    """
    magnitude_db = 20 * np.log10(np.abs(TF) + 1e-300)
    phase_deg    = np.angle(TF, deg=True)
    return magnitude_db, phase_deg, freq_hz
