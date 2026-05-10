"""Tests for src/transfer_functions.py — compute_tf and bode_data."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transfer_functions import compute_tf, bode_data


class TestComputeTF:
    def test_static_gain_at_zero_frequency(self):
        """At omega=0: H(0) = D + C (-A)^-1 B  (DC gain)."""
        A = -5.0 * np.eye(2)
        B = np.ones((2, 1))
        C = np.eye(2)
        D = np.zeros((2, 1))
        TF = compute_tf(A, B, C, D, np.array([0.0]))
        expected = C @ np.linalg.solve(-A, B) + D
        np.testing.assert_allclose(np.real(TF[:, :, 0]), expected, atol=1e-10)

    def test_output_shape(self):
        n, m, p, N = 4, 2, 3, 50
        A = -np.eye(n)
        B = np.ones((n, m))
        C = np.ones((p, n))
        D = np.zeros((p, m))
        omega = np.linspace(0.1, 10, N)
        TF = compute_tf(A, B, C, D, omega)
        assert TF.shape == (p, m, N)

    def test_dtype_complex(self):
        A = -np.eye(2); B = np.eye(2); C = np.eye(2); D = np.zeros((2, 2))
        TF = compute_tf(A, B, C, D, np.array([1.0]))
        assert np.iscomplexobj(TF)

    def test_single_pole_magnitude(self):
        """First-order system H(s)=1/(s+a): |H(jw)| = 1/sqrt(w^2+a^2)."""
        a = 3.0
        A = np.array([[-a]]); B = np.array([[1.0]])
        C = np.array([[1.0]]); D = np.array([[0.0]])
        omega = np.array([1.0, 3.0, 10.0])
        TF = compute_tf(A, B, C, D, omega)
        expected = 1.0 / np.sqrt(omega**2 + a**2)
        np.testing.assert_allclose(np.abs(TF[0, 0, :]), expected, rtol=1e-10)


class TestBodeData:
    def test_shapes_preserved(self):
        A = -np.eye(3); B = np.eye(3); C = np.eye(3); D = np.zeros((3, 3))
        omega = np.linspace(0.1, 5, 40)
        freq  = omega / (2 * np.pi)
        TF    = compute_tf(A, B, C, D, omega)
        mag, phase, f = bode_data(TF, freq)
        assert mag.shape == TF.shape
        assert phase.shape == TF.shape
        np.testing.assert_array_equal(f, freq)
