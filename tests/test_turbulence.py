"""Tests for src/turbulence.py — von_karman_psd."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.turbulence import von_karman_psd
from src.config import VTAS, L, USIGMA

OMEGA = np.linspace(1e-4, 500, 200_000)


class TestVonKarmanPSD:
    def test_output_positive(self):
        """PSD values must be non-negative."""
        phi = von_karman_psd(OMEGA, VTAS, L, USIGMA)
        assert np.all(phi >= 0)

    def test_output_shape(self):
        """Output shape must match input omega shape."""
        omega = np.linspace(0.01, 50, 300)
        phi = von_karman_psd(omega, VTAS, L, USIGMA)
        assert phi.shape == omega.shape

    def test_variance_within_2_percent(self):
        """Integral of one-sided PSD over omega should equal sigma^2 within 2 %.
        The VK spectrum satisfies integral(Phi(w), 0, inf) = sigma^2.
        Integration over [1e-4, 500] rad/s captures >99.4% of the total.
        """
        phi = von_karman_psd(OMEGA, VTAS, L, USIGMA)
        variance = np.trapezoid(phi, OMEGA)
        rel_err = abs(variance - USIGMA ** 2) / USIGMA ** 2
        assert rel_err < 0.02, f"Relative error {rel_err:.4f} exceeds 2 %"

    def test_high_frequency_decay(self):
        """PSD at high freq must be orders of magnitude below peak."""
        phi_lo  = von_karman_psd(np.array([0.1]),  VTAS, L, USIGMA)
        phi_hi  = von_karman_psd(np.array([100.0]), VTAS, L, USIGMA)
        assert phi_hi[0] < phi_lo[0] / 1000

    def test_sigma_scaling(self):
        """PSD scales as sigma^2."""
        omega = np.array([1.0, 5.0])
        phi1  = von_karman_psd(omega, VTAS, L, sigma=10.0)
        phi2  = von_karman_psd(omega, VTAS, L, sigma=20.0)
        np.testing.assert_allclose(phi2, phi1 * 4.0, rtol=1e-10)
