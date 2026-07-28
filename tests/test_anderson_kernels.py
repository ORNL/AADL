"""Unit tests for the Anderson acceleration kernels.

These tests exercise ``anderson_qr_factorization`` and
``anderson_normal_equation`` on a small symmetric positive-definite linear
fixed point with a known answer, plus a handful of property checks.
"""

import os
import sys
import unittest

import torch

# Allow running directly from the tests directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AADL.anderson_acceleration import (
    anderson_normal_equation,
    anderson_qr_factorization,
    get_acceleration,
)


def _richardson_history(A, b, x0, omega, n_iters):
    """Generate a sequence of Richardson iterates as columns of a matrix."""
    x = x0.clone()
    cols = [x.clone()]
    for _ in range(n_iters):
        x = x + omega * (b - A @ x)
        cols.append(x.clone())
    return torch.stack(cols, dim=1)


def _spd_system(n=8, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=dtype)
    A = M @ M.t() + n * torch.eye(n, dtype=dtype)  # well-conditioned SPD
    x_star = torch.randn(n, generator=g, dtype=dtype)
    b = A @ x_star
    return A, b, x_star


class AndersonKernelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.A, self.b, self.x_star = _spd_system()
        # spectral radius of (I - omega A) < 1
        eig_max = torch.linalg.eigvalsh(self.A).max().item()
        self.omega = 1.0 / eig_max
        self.x0 = torch.zeros_like(self.b)

    # -- equivalence -------------------------------------------------------

    def test_kernels_agree(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        x_qr = anderson_qr_factorization(X)
        x_ne = anderson_normal_equation(X)
        self.assertTrue(
            torch.allclose(x_qr, x_ne, atol=1e-8, rtol=1e-6),
            f"qr vs normal-eqn mismatch: max diff {(x_qr - x_ne).abs().max().item():.2e}",
        )

    def test_acceleration_reduces_residual(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        last = X[:, -1]
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_acc = kernel(X)
            r_last = (self.b - self.A @ last).norm().item()
            r_acc = (self.b - self.A @ x_acc).norm().item()
            self.assertLess(r_acc, r_last, f"{kernel.__name__} did not reduce residual")

    def test_relaxation_one_is_default(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_default = kernel(X)
            x_explicit = kernel(X, relaxation=1.0)
            self.assertTrue(torch.allclose(x_default, x_explicit, atol=1e-12))

    def test_relaxation_interpolates(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_full = kernel(X, relaxation=1.0)
            x_half = kernel(X, relaxation=0.5)
            # Half relaxation must lie strictly between zero-relaxation
            # behavior and full Anderson; loosely: residual sits between.
            r_full = (self.b - self.A @ x_full).norm().item()
            r_half = (self.b - self.A @ x_half).norm().item()
            r_last = (self.b - self.A @ X[:, -1]).norm().item()
            self.assertLessEqual(r_full, r_half + 1e-8)
            self.assertLessEqual(r_half, r_last + 1e-8)

    # -- regularization ----------------------------------------------------

    def test_regularization_runs_and_is_close_to_unregularized(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_reg = kernel(X, regularization=1e-10)
            x_unreg = kernel(X, regularization=0.0)
            self.assertTrue(
                torch.allclose(x_reg, x_unreg, atol=1e-4, rtol=1e-4),
                f"{kernel.__name__} reg vs unreg differ by "
                f"{(x_reg - x_unreg).abs().max().item():.2e}",
            )

    def test_regularization_must_be_nonnegative(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        with self.assertRaises(AssertionError):
            anderson_qr_factorization(X, regularization=-1.0)
        with self.assertRaises(AssertionError):
            anderson_normal_equation(X, regularization=-1.0)

    # -- dtype/device preservation ----------------------------------------

    def test_dtype_preserved_float32(self):
        A = self.A.to(torch.float32)
        b = self.b.to(torch.float32)
        x0 = self.x0.to(torch.float32)
        omega = float(self.omega)
        X = _richardson_history(A, b, x0, omega, n_iters=6)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            for reg in (0.0, 1e-6):
                x_acc = kernel(X, regularization=reg)
                self.assertEqual(
                    x_acc.dtype, torch.float32,
                    f"{kernel.__name__} (reg={reg}) lost float32 dtype",
                )
                self.assertEqual(x_acc.device, X.device)

    def test_dtype_preserved_float64_with_regularization(self):
        # The pre-fix bug created torch.eye(...) in default float32; with a
        # float64 input this would have raised a dtype mismatch.
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_acc = kernel(X, regularization=1e-8)
            self.assertEqual(x_acc.dtype, torch.float64)

    # -- mixing-vector precision (dtype override) --------------------------

    def test_mixing_dtype_returns_original_dtype(self):
        # Input is float32; computing the mixing vector in float64 must still
        # return a float32 extrapolation.
        X = _richardson_history(
            self.A.to(torch.float32), self.b.to(torch.float32),
            self.x0.to(torch.float32), float(self.omega), n_iters=6,
        )
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            for dt in (torch.float64, "float64", torch.float32):
                x_acc = kernel(X, dtype=dt)
                self.assertEqual(
                    x_acc.dtype, torch.float32,
                    f"{kernel.__name__} (dtype={dt}) did not restore float32",
                )

    def test_mixing_dtype_high_precision_reduces_residual(self):
        # Computing the mixing vector in float64 from a float32 history should
        # still reduce the residual.
        A32, b32 = self.A.to(torch.float32), self.b.to(torch.float32)
        X = _richardson_history(A32, b32, self.x0.to(torch.float32),
                                float(self.omega), n_iters=6)
        r_last = (b32 - A32 @ X[:, -1]).norm().item()
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_acc = kernel(X, dtype=torch.float64)
            r_acc = (b32 - A32 @ x_acc).norm().item()
            self.assertLess(r_acc, r_last, f"{kernel.__name__} did not reduce residual")

    def test_mixing_dtype_invalid_raises(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        with self.assertRaises(ValueError):
            anderson_qr_factorization(X, dtype="not_a_dtype")

    def test_mixing_dtype_unsupported_low_precision_raises_clear_error(self):
        # bfloat16/float16 are not implemented by torch.linalg on CPU; the
        # kernels must surface a clear, actionable message rather than the
        # raw backend NotImplementedError.
        X = _richardson_history(
            self.A.to(torch.float32), self.b.to(torch.float32),
            self.x0.to(torch.float32), float(self.omega), n_iters=6,
        )
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            for dt in (torch.bfloat16, torch.float16):
                with self.assertRaises(NotImplementedError) as ctx:
                    kernel(X, dtype=dt)
                msg = str(ctx.exception)
                self.assertIn("float32", msg)
                self.assertIn("float64", msg)

    # -- input validation --------------------------------------------------
    def test_x_must_be_matrix(self):
        v = torch.randn(10)
        with self.assertRaises(AssertionError):
            anderson_qr_factorization(v)
        with self.assertRaises(AssertionError):
            anderson_normal_equation(v)

    # -- registry ----------------------------------------------------------

    def test_get_acceleration_known(self):
        self.assertIs(get_acceleration("anderson"), anderson_qr_factorization)
        self.assertIs(
            get_acceleration("anderson_normal_equation"), anderson_normal_equation
        )

    def test_get_acceleration_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_acceleration("not-a-real-kernel")


if __name__ == "__main__":
    unittest.main()
