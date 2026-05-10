"""
modal.py -- Eigenvalue-based modal analysis of LTI state-space systems.

Provides freq_damp() which extracts natural frequencies and damping ratios
from the system matrix A, and classify_modes() which labels rigid-body,
flexible, and aerodynamic lag modes.
"""

import numpy as np


def freq_damp(A):
    """
    Compute natural frequencies and damping ratios from the system matrix A.

    Solves the eigenvalue problem lam = eig(A) and converts complex poles to
    modal parameters:

        wn = |lam|,   zeta = -Re(lam) / wn * 100  (percent),   f = wn / (2*pi)

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Square state matrix of an LTI system (continuous-time).

    Returns
    -------
    freq_hz : ndarray, shape (n,)
        Natural frequencies in Hz, sorted in ascending order.
    zeta : ndarray, shape (n,)
        Damping ratios in percent. NaN for zero-frequency (integrator) modes.
    eigenvalues : ndarray, shape (n,), complex
        Complex eigenvalues corresponding to the sorted frequencies.

    Notes
    -----
    Conjugate pairs appear as two identical rows (same |lam|) -- this is
    expected and consistent with the number of states.
    """
    eigenvalues = np.linalg.eig(A)[0]
    wn = np.abs(eigenvalues)
    with np.errstate(invalid='ignore', divide='ignore'):
        zeta = np.where(wn > 1e-10, -np.real(eigenvalues) / wn * 100, np.nan)
    freq_hz = wn / (2 * np.pi)
    idx = np.argsort(freq_hz)
    return freq_hz[idx], zeta[idx], eigenvalues[idx]


def classify_modes(freq_hz, zeta, rigid_body_thresh=0.05,
                   lag_zeta_thresh=95.0):
    """
    Classify eigenvalues into rigid-body, flexible, and aerodynamic-lag modes.

    Parameters
    ----------
    freq_hz : ndarray
        Natural frequencies in Hz (output of freq_damp()).
    zeta : ndarray
        Damping ratios in percent (output of freq_damp()).
    rigid_body_thresh : float, optional
        Frequency below which a mode is considered rigid-body (Hz).
        Default 0.05 Hz.
    lag_zeta_thresh : float, optional
        Damping ratio above which a mode is classified as an aerodynamic
        lag state (non-structural). Default 95 %.

    Returns
    -------
    labels : list of str
        Human-readable label for each eigenvalue in the input arrays.
    """
    labels = []
    flex_count = 0
    for f, z in zip(freq_hz, zeta):
        if f < rigid_body_thresh:
            labels.append("Rigid-body (integrator)")
        elif not np.isnan(z) and z >= lag_zeta_thresh:
            labels.append("Aero lag / actuator")
        elif f < rigid_body_thresh * 10:
            labels.append("Rigid-body")
        else:
            flex_count += 1
            labels.append(f"Flexible mode {flex_count}")
    return labels
