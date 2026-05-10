"""
control_law.py -- TLA (Torsion Load Alleviation) control law.

The TLA law uses the differential acceleration sensor (output 11: acc_F - acc_E)
to command symmetric deflection of inner and outer ailerons.

Transfer function:

    H_law(s) = (1/g) * K * H_filter(s) * 1/(tau*s + 1) * (pi/180)

where:
  - g        = 9.81 m/s^2  (converts acceleration sensor output to g-units)
  - K        = law gain (negative for load alleviation in positive feedback)
  - H_filter = Butterworth lowpass filter (2nd order, fc = 1.20 Hz)
  - tau      = 0.1 s  actuator time constant
  - pi/180   = converts command from degrees to radians

The feedback loop is positive (MATLAB feedback(..., +1)) so the
Nyquist critical point is +1, not -1.
"""

import numpy as np
import scipy.signal as sp_signal


def law_freq_response(Gain, num_c, den_c, tau, omega):
    """
    Evaluate the TLA law frequency response H_law(j*omega).

    Parameters
    ----------
    Gain : float
        Law gain K.  Use negative values to achieve load alleviation
        under positive-feedback convention.
    num_c : array-like
        Numerator polynomial coefficients of the shaping filter
        (descending powers of s).
    den_c : array-like
        Denominator polynomial coefficients of the shaping filter.
    tau : float
        Actuator first-order time constant in seconds.
    omega : ndarray
        Angular frequencies in rad/s at which to evaluate the law.

    Returns
    -------
    H : ndarray, complex, same shape as omega
        Complex frequency response of the full law.
    """
    s = 1j * omega
    H_filter   = np.polyval(num_c, s) / np.polyval(den_c, s)
    H_actuator = 1.0 / (tau * s + 1)
    return (1.0 / 9.81) * Gain * H_filter * H_actuator * (np.pi / 180)


def compute_margins(H_ol, omega, freq):
    """
    Compute gain and phase margins for a positive-feedback loop.

    In positive feedback the critical point is +1 (not -1).
    The gain margin is computed at positive real-axis crossings of the
    Nyquist locus; the phase margin is computed at the gain-crossover
    frequency where |L(j*omega)| = 1.

    Parameters
    ----------
    H_ol : ndarray, complex
        Open-loop frequency response L(j*omega).
    omega : ndarray
        Angular frequencies in rad/s (same length as H_ol; kept for
        API symmetry but not used internally).
    freq : ndarray
        Frequencies in Hz, same length as H_ol.

    Returns
    -------
    gm : float
        Gain margin.  inf if the Nyquist locus never crosses the
        positive real axis.
    pm : float
        Phase margin in degrees.  nan if |L(j*omega)| < 1 everywhere
        (unconditionally stable in gain).
    f_gm : float
        Frequency in Hz at the gain-margin crossing.  nan if none.
    f_pm : float
        Frequency in Hz at the phase-margin crossing.  nan if none.

    Notes
    -----
    When pm is nan the loop is unconditionally stable in phase because
    the gain never reaches 1 -- no phase requirement applies.
    """
    gm, f_gm = np.inf, np.nan

    # Find positive real-axis crossings (Im -> 0 with Re > 0)
    for idx in np.where(np.diff(np.sign(np.imag(H_ol))))[0]:
        re_cross = np.interp(0.0,
                             np.imag(H_ol[[idx, idx + 1]]),
                             np.real(H_ol[[idx, idx + 1]]))
        f_cross  = np.interp(0.0,
                             np.imag(H_ol[[idx, idx + 1]]),
                             freq[[idx, idx + 1]])
        if re_cross > 0:
            candidate = 1.0 / re_cross
            if candidate < gm:
                gm, f_gm = candidate, f_cross

    # Phase margin: last gain-crossover (|L| = 1)
    mag = np.abs(H_ol)
    pm, f_pm = np.nan, np.nan
    gc_indices = np.where(np.diff(np.sign(mag - 1.0)))[0]
    if len(gc_indices) > 0:
        idx = gc_indices[-1]
        f_pm = np.interp(1.0,
                         mag[[idx, idx + 1]][::-1],
                         freq[[idx, idx + 1]][::-1])
        pm   = np.interp(1.0,
                         mag[[idx, idx + 1]][::-1],
                         np.angle(H_ol[[idx, idx + 1]], deg=True)[::-1])
    return gm, pm, f_gm, f_pm


def build_closed_loop(A, B, C, D, Gain, num_c, den_c, tau,
                      sensor_idx, cs_indices, ext_input_slice):
    """
    Construct the closed-loop state-space matrices analytically.

    The law commands ailerons cs_indices based on sensor output
    sensor_idx.  External inputs (elevators + wind) are preserved.

    Parameters
    ----------
    A, B, C, D : ndarray
        Open-loop plant matrices (n x n, n x m, p x n, p x m).
    Gain : float
        Law gain (negative for load alleviation).
    num_c, den_c : array-like
        Filter polynomial coefficients.
    tau : float
        Actuator time constant (s).
    sensor_idx : int
        Output index (0-based) used as the feedback sensor.
    cs_indices : list of int
        Input indices (0-based) driven by the law command.
    ext_input_slice : slice or list
        Input indices kept as external inputs in the closed-loop model.

    Returns
    -------
    A_cl, B_cl, C_cl, D_cl : ndarray
        Closed-loop state-space matrices.
        States: n_plant + n_law;  Inputs: len(ext_input_slice);
        Outputs: p.
    """
    K_static = (1.0 / 9.81) * Gain * (np.pi / 180)
    num_law  = np.array(num_c) * K_static
    den_law  = np.polymul(den_c, [tau, 1.0])

    A_l, B_l, C_l, D_l = sp_signal.tf2ss(num_law, den_law)

    # Summed control-surface column (inner + outer ailerons)
    B_cs     = np.sum(B[:, cs_indices], axis=1, keepdims=True)
    D_cs     = np.sum(D[:, cs_indices], axis=1, keepdims=True)

    C_sen    = C[sensor_idx:sensor_idx + 1, :]
    D_cs_sen = float(np.sum(D[sensor_idx, cs_indices]))
    D_sen_ext= D[sensor_idx:sensor_idx + 1, ext_input_slice]

    A_cl = np.block([
        [A,               B_cs @ C_l],
        [B_l @ C_sen,     A_l + B_l * D_cs_sen * C_l]
    ])
    B_cl = np.block([
        [B[:, ext_input_slice]],
        [B_l @ D_sen_ext]
    ])
    C_cl = np.hstack([C, D_cs @ C_l])
    D_cl = D[:, ext_input_slice]

    return A_cl, B_cl, C_cl, D_cl
