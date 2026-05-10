"""
psd_response.py -- Output PSD computation for a turbulence-excited LTI system.

Given the system transfer function H(j*omega) and an input PSD Phi_input(omega),
the output PSD for each channel is:

    S_y(omega) = |H(j*omega)|^2 * Phi_input(omega)

where the wind input column of H must already be divided by VTAS to convert
wind-speed PSD (m/s)^2/(rad/s) into wind-incidence PSD rad^2/(rad/s).
"""

import numpy as np


def output_psd(TF, phi_wind_speed, input_idx, VTAS):
    """
    Compute output PSD from a single turbulent wind-speed input.

    The wind model input is incidence alpha_w = w/V (rad), so the effective
    transfer function is H_alpha = H_w / V.  This function applies that
    scaling internally.

    Parameters
    ----------
    TF : ndarray, shape (n_out, n_in, n_freq), complex
        Full frequency-response tensor of the plant.
    phi_wind_speed : ndarray, shape (n_freq,)
        Von Karman one-sided PSD in (m/s)^2 / (rad/s).
    input_idx : int
        Column index of the wind-incidence input in TF.
    VTAS : float
        True airspeed in m/s -- used to convert wind speed to incidence.

    Returns
    -------
    S_y : ndarray, shape (n_out, n_freq)
        One-sided output PSD for each output channel.

    Notes
    -----
    S_y[i, k] = |TF[i, input_idx, k] / VTAS|^2 * phi_wind_speed[k]
    """
    H_incidence = TF[:, input_idx, :] / VTAS    # shape (n_out, n_freq)
    return np.abs(H_incidence) ** 2 * phi_wind_speed[np.newaxis, :]


def rms_from_psd(S_y, omega):
    """
    Compute the RMS value from a one-sided PSD via trapezoidal integration.

    Parameters
    ----------
    S_y : ndarray, shape (n_freq,) or (n_out, n_freq)
        One-sided PSD.  If 2-D, RMS is computed along the last axis.
    omega : ndarray, shape (n_freq,)
        Angular frequency vector in rad/s.

    Returns
    -------
    rms : float or ndarray
        Root mean square value(s).
    """
    return np.sqrt(np.trapezoid(S_y, omega, axis=-1))
