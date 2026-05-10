"""Tests for src/modal.py — freq_damp and classify_modes."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modal import freq_damp, classify_modes


class TestFreqDamp:
    def test_known_2x2_frequency(self):
        """2x2 companion system with ωₙ = 2π × 1.5 Hz."""
        wn   = 2 * np.pi * 1.5
        zeta = 0.05
        A = np.array([[-zeta * wn, -wn],
                      [wn,          -zeta * wn]])
        freq, z, _ = freq_damp(A)
        assert np.allclose(freq[freq > 0.01], 1.5, atol=0.02)

    def test_known_2x2_damping(self):
        wn   = 2 * np.pi * 2.0
        zeta = 0.10
        A = np.array([[-zeta * wn, -wn],
                      [wn,          -zeta * wn]])
        _, z, _ = freq_damp(A)
        valid = z[~np.isnan(z) & (np.abs(z) < 50)]
        assert np.allclose(valid, 10.0, atol=0.5)

    def test_sorted_ascending(self):
        """Output frequencies must be non-decreasing."""
        A = -np.diag([1.0, 5.0, 0.5, 20.0])
        freq, _, _ = freq_damp(A)
        assert np.all(np.diff(freq) >= 0)

    def test_output_lengths_match_n(self):
        n = 8
        A = -np.eye(n)
        freq, zeta, eigs = freq_damp(A)
        assert len(freq) == n
        assert len(zeta) == n
        assert len(eigs) == n

    def test_pure_real_negative_pole(self):
        """Pure real negative pole: ζ should be 100 %."""
        A = np.array([[-5.0]])
        freq, zeta, _ = freq_damp(A)
        assert np.allclose(zeta, 100.0, atol=0.01)

    def test_integrator_gives_nan_zeta(self):
        """Zero eigenvalue (pure integrator) should give NaN damping."""
        A = np.zeros((2, 2))
        _, zeta, _ = freq_damp(A)
        assert np.all(np.isnan(zeta) | (np.abs(zeta) < 1e-6))


class TestClassifyModes:
    def test_rigid_body_low_freq(self):
        freq = np.array([0.001, 0.02])
        zeta = np.array([np.nan, 100.0])
        labels = classify_modes(freq, zeta)
        assert all("Rigid" in l or "integrator" in l for l in labels)

    def test_lag_high_damping(self):
        freq  = np.array([2.0])
        zeta  = np.array([100.0])
        labels = classify_modes(freq, zeta)
        assert "lag" in labels[0].lower() or "actuator" in labels[0].lower()

    def test_flexible_mode_labeled(self):
        freq  = np.array([1.19, 2.02])
        zeta  = np.array([10.1, 5.1])
        labels = classify_modes(freq, zeta)
        assert all("Flexible" in l for l in labels)
