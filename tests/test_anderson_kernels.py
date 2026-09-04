"""Unit tests for the Anderson acceleration kernels.

These tests exercise ``anderson_qr_factorization`` and
``anderson_normal_equation`` on a small symmetric positive-definite linear
fixed point with a known answer, plus a handful of property checks.
"""

import math
import os
import sys
import unittest

import torch

# Allow running directly from the tests directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AADL.anderson_acceleration import (
    _compute_differences,
    _equilibrate_columns,
    _num_oldest_to_drop,
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


def _ill_conditioned_history(cond=1e6, n=8, seed=1, n_iters=6, dtype=torch.float64):
    """Richardson history for an SPD system with a prescribed condition number."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=dtype))
    eig = torch.logspace(0, math.log10(cond), n, dtype=dtype)
    A = (Q * eig) @ Q.t()
    x_star = torch.randn(n, generator=g, dtype=dtype)
    b = A @ x_star
    omega = 1.0 / float(eig.max())
    x0 = torch.zeros_like(b)
    return _richardson_history(A, b, x0, omega, n_iters), A, b


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

    def test_full_row_sketch_matches_unsketched_solve(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        rows = torch.arange(X.size(0), dtype=torch.long)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            expected = kernel(X)
            actual = kernel(X, row_indices=rows)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-12))

    def test_reduced_row_sketch_returns_full_finite_iterate(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        rows = torch.tensor([0, 2, 4, 6], dtype=torch.long)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            actual = kernel(X, regularization=1e-8, row_indices=rows)
            self.assertEqual(actual.shape, X[:, -1].shape)
            self.assertTrue(torch.isfinite(actual).all())

    def test_invalid_row_sketch_raises(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        invalid = (
            torch.tensor([], dtype=torch.long),
            torch.tensor([0.0]),
            torch.tensor([[0]], dtype=torch.long),
            torch.tensor([X.size(0)], dtype=torch.long),
        )
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                anderson_qr_factorization(X, row_indices=rows)

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

    def test_normal_equation_rank_deficient_history_falls_back(self):
        # A history whose iterates barely move makes DR nearly rank-deficient,
        # so RR = DR^T DR is (numerically) only positive *semi*-definite and
        # the Cholesky path must fall back to the general solver instead of
        # failing. The result must still match the QR kernel.
        A, b = self.A, self.b
        x = self.x0.clone()
        cols = [x.clone()]
        for _ in range(6):
            x = x + 1e-12 * (b - A @ x)  # essentially stationary iterates
            cols.append(x.clone())
        X = torch.stack(cols, dim=1)
        x_ne = anderson_normal_equation(X, regularization=0.0)
        x_qr = anderson_qr_factorization(X, regularization=0.0)
        self.assertTrue(torch.isfinite(x_ne).all(), "normal-eqn produced non-finite output")
        self.assertTrue(
            torch.allclose(x_ne, x_qr, atol=1e-6, rtol=1e-5),
            f"fallback path diverged from QR: max diff "
            f"{(x_ne - x_qr).abs().max().item():.2e}",
        )

    # -- column equilibration ---------------------------------------------

    def test_equilibrate_is_numerically_neutral_unregularized(self):
        # For a well-conditioned, full-rank history and reg=0, column scaling
        # is an exact change of variables: the mixing vector (hence the
        # extrapolated iterate) is unchanged up to round-off.
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_eq = kernel(X, equilibrate=True)
            x_noeq = kernel(X, equilibrate=False)
            self.assertTrue(
                torch.allclose(x_eq, x_noeq, atol=1e-7, rtol=1e-6),
                f"{kernel.__name__}: equilibration changed the result "
                f"(max diff {(x_eq - x_noeq).abs().max().item():.2e})",
            )

    def test_equilibrate_columns_helper(self):
        A = torch.tensor(
            [[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]], dtype=torch.float64
        )
        A_s, scale = _equilibrate_columns(A)
        self.assertTrue(torch.allclose(scale, torch.tensor([3.0, 4.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(A_s.norm(dim=0), torch.ones(2, dtype=torch.float64)))

    def test_equilibrate_columns_zero_column_is_safe(self):
        A = torch.tensor([[0.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
        A_s, scale = _equilibrate_columns(A)
        self.assertTrue(torch.isfinite(A_s).all())
        # zero-norm column left unscaled (scale forced to 1)
        self.assertEqual(scale[0].item(), 1.0)

    # -- Walker-Ni column filtering ---------------------------------------

    def test_filter_disabled_returns_zero_drop(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        _, DR = _compute_differences(X)
        A_s, _ = _equilibrate_columns(DR)
        self.assertEqual(_num_oldest_to_drop(A_s, 0.0), 0)
        self.assertEqual(_num_oldest_to_drop(A_s, -1.0), 0)

    def test_filter_drops_oldest_until_conditioned(self):
        X, _, _ = _ill_conditioned_history(cond=1e8)
        _, DR = _compute_differences(X)
        A_s, _ = _equilibrate_columns(DR)
        threshold = 1e3
        n_drop = _num_oldest_to_drop(A_s, threshold)
        self.assertGreater(n_drop, 0, "expected filtering to drop columns")
        # the retained sub-matrix must satisfy the requested condition bound
        kept = A_s[:, n_drop:]
        cond = torch.linalg.cond(kept).item()
        self.assertLessEqual(cond, threshold * 1.5)

    def test_filter_kernel_runs_and_reduces_residual(self):
        X, A, b = _ill_conditioned_history(cond=1e8)
        last = X[:, -1]
        r_last = (b - A @ last).norm().item()
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_acc = kernel(X, filter_condition=1e3)
            self.assertTrue(torch.isfinite(x_acc).all())
            r_acc = (b - A @ x_acc).norm().item()
            self.assertLess(r_acc, r_last, f"{kernel.__name__} did not reduce residual")

    # -- mixed-precision iterative refinement -----------------------------

    def test_refinement_recovers_high_precision_accuracy(self):
        # Factorize in float32 on a moderately ill-conditioned history, then
        # refine with a float64 residual: the refined mixing vector must land
        # strictly closer to the full float64 result than the un-refined one.
        X64, _, _ = _ill_conditioned_history(cond=1e3)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_ref = kernel(X64)  # full float64 reference
            x_fp32 = kernel(X64, dtype=torch.float32)
            x_refined = kernel(X64, dtype=torch.float32, refinement_steps=5)
            e_fp32 = (x_fp32 - x_ref).norm().item()
            e_refined = (x_refined - x_ref).norm().item()
            if kernel is anderson_qr_factorization:
                self.assertLess(
                    e_refined, e_fp32,
                    f"{kernel.__name__}: refinement did not improve accuracy "
                    f"({e_refined:.2e} !< {e_fp32:.2e})",
                )
            else:
                # The refinement guard controls the normal-equation residual,
                # not forward error. Different BLAS backends may therefore
                # move the latter slightly in either direction.
                self.assertLessEqual(e_refined, e_fp32 * 1.05 + 1e-9)

    def test_refinement_is_safe_on_extreme_conditioning(self):
        # When the reduced-precision factor is a poor preconditioner the monotone
        # guard must keep refinement from diverging. It cannot guarantee the
        # *solution error* is monotone (the guard controls the residual, which
        # differs from the error by cond(A) ~ 1e10 here), so allow round-off
        # slack: the point is that refinement stays bounded, not that it helps.
        X64, _, _ = _ill_conditioned_history(cond=1e10)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            x_ref = kernel(X64)
            x_fp32 = kernel(X64, dtype=torch.float32)
            x_refined = kernel(X64, dtype=torch.float32, refinement_steps=5)
            self.assertTrue(torch.isfinite(x_refined).all())
            e_fp32 = (x_fp32 - x_ref).norm().item()
            e_refined = (x_refined - x_ref).norm().item()
            # no blow-up: refined error stays within round-off of the fp32 error
            self.assertLessEqual(e_refined, e_fp32 * (1.0 + 1e-4) + 1e-9)

    def test_refinement_preserves_dtype(self):
        X = _richardson_history(self.A, self.b, self.x0, self.omega, n_iters=6)
        for kernel in (anderson_qr_factorization, anderson_normal_equation):
            out = kernel(X, dtype=torch.float32, refinement_steps=2)
            self.assertEqual(out.dtype, X.dtype)

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
