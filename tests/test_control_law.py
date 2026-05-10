"""Tests for src/control_law.py -- law_freq_response and compute_margins."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.control_law import law_freq_response, compute_margins
from src.config import TAU


class TestLawFreqResponse:
    def _butter2_lowpass(self, fc_hz):
        from scipy.signal import butter
        wc = 2 * np.pi * fc_hz
        b, a = butter(2, wc, btype="low", analog=True)
        return b.tolist(), a.tolist()

    def test_output_shape(self):
        num, den = self._butter2_lowpass(1.2)
        omega = np.linspace(0.01, 30, 200)
        H = law_freq_response(-4.3, num, den, TAU, omega)
        assert H.shape == omega.shape

    def test_output_complex(self):
        num, den = self._butter2_lowpass(1.2)
        omega = np.array([1.0, 5.0])
        H = law_freq_response(-4.3, num, den, TAU, omega)
        assert np.iscomplexobj(H)

    def test_gain_sign_reflected(self):
        """Negating gain negates the frequency response."""
        num, den = self._butter2_lowpass(1.2)
        omega = np.array([0.5, 2.0])
        H_pos = law_freq_response(+1.0, num, den, TAU, omega)
        H_neg = law_freq_response(-1.0, num, den, TAU, omega)
        np.testing.assert_allclose(H_neg, -H_pos)

    def test_dc_magnitude_order_of_magnitude(self):
        """At very low freq, law gain should be small (< 1 for Gain=-4.3)."""
        num, den = self._butter2_lowpass(1.2)
        omega = np.array([1e-3])
        H = law_freq_response(-4.3, num, den, TAU, omega)
        assert np.abs(H[0]) < 1.0


class TestComputeMargins:
    def _make_loop(self, K):
        """Simple first-order open loop L(jw) = K / (jw + 1)."""
        omega = np.linspace(1e-4, 100, 50000)
        freq  = omega / (2 * np.pi)
        H_ol  = K / (1j * omega + 1.0)
        return H_ol, omega, freq

    def test_stable_large_gain_margin(self):
        """K=0.1 -> |L| < 1 everywhere -> GM = inf, PM = nan."""
        H_ol, omega, freq = self._make_loop(0.1)
        gm, pm, f_gm, f_pm = compute_margins(H_ol, omega, freq)
        assert gm == np.inf
        assert np.isnan(pm)

    def test_finite_gain_margin(self):
        """Loop with pure delay crosses positive real axis at omega = 2*pi.
        H = 0.3 * exp(-j*omega*tau), tau=1: Im(H)=0 and Re(H)=0.3>0 at omega=2*pi,
        so GM = 1/0.3 = 3.33.
        """
        omega = np.linspace(1e-3, 50, 200_000)
        freq  = omega / (2 * np.pi)
        tau   = 1.0
        H_ol  = 0.3 * np.exp(-1j * omega * tau)
        gm, pm, f_gm, _ = compute_margins(H_ol, omega, freq)
        assert np.isfinite(gm), f"Expected finite GM, got {gm}"
        assert gm > 1.0
        assert abs(gm - 1.0 / 0.3) / (1.0 / 0.3) < 0.05
