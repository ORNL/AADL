"""Integration test for the public ``AADL.accelerate`` API.

Trains the same tiny linear-regression problem twice with vanilla SGD and
with SGD + Anderson acceleration and asserts that:

  * acceleration reaches a target loss in strictly fewer iterations, and
  * removing acceleration restores the original ``optimizer.step``.

Kept hermetic (no model_zoo, no DataLoader, no wrappers) so it runs as a
unit test in well under a second.
"""

import unittest

import torch

from AADL import accelerate, remove_acceleration


def _make_problem(n=64, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w_star = torch.randn(d, generator=g)
    y = X @ w_star
    return X, y, w_star


def _train(model, opt, X, y, *, target_loss, max_iters):
    """Run full-batch GD until the loss falls below ``target_loss``.

    Returns the iteration count (``max_iters`` if the target was not met).
    """
    loss_fn = torch.nn.MSELoss()
    for it in range(1, max_iters + 1):
        def closure():
            with torch.enable_grad():
                opt.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
            return loss
        loss = opt.step(closure)
        if loss is None:
            with torch.no_grad():
                loss = loss_fn(model(X), y)
        if float(loss) < target_loss:
            return it
    return max_iters


class TestAccelerateAPI(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.X, self.y, _ = _make_problem()

    def _fresh_model(self):
        torch.manual_seed(0)
        return torch.nn.Linear(self.X.size(1), 1, bias=False)

    def test_acceleration_speedup(self):
        target_loss = 1e-4
        max_iters = 2000

        baseline_model = self._fresh_model()
        baseline_opt = torch.optim.SGD(baseline_model.parameters(), lr=1e-2)
        baseline_iters = _train(
            baseline_model, baseline_opt, self.X, self.y.unsqueeze(1),
            target_loss=target_loss, max_iters=max_iters,
        )

        acc_model = self._fresh_model()
        acc_opt = torch.optim.SGD(acc_model.parameters(), lr=1e-2)
        accelerate(
            acc_opt,
            acceleration_type="anderson",
            relaxation=1.0,
            wait_iterations=1,
            history_depth=10,
            frequency=3,
            store_each_nth=1,
            reg_acc=1e-8,
        )
        acc_iters = _train(
            acc_model, acc_opt, self.X, self.y.unsqueeze(1),
            target_loss=target_loss, max_iters=max_iters,
        )

        self.assertLess(baseline_iters, max_iters,
                        "baseline failed to converge — test setup is too hard")
        self.assertLess(acc_iters, baseline_iters,
                        f"AAR did not speed up: baseline={baseline_iters}, "
                        f"accelerated={acc_iters}")

    def test_safeguard_rejects_divergent_candidate(self):
        # The safeguard must reject an accelerated candidate that is worse than
        # the plain optimizer step and revert to that plain step, comparing
        # against the *post-step* loss (not the stale pre-step loss).
        import AADL.anderson_acceleration as anderson_mod

        model = self._fresh_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        accelerate(
            opt, acceleration_type="anderson", wait_iterations=1,
            history_depth=6, frequency=1, store_each_nth=1,
        )
        loss_fn = torch.nn.MSELoss()
        X, y = self.X, self.y.unsqueeze(1)

        def closure():
            with torch.enable_grad():
                opt.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
            return loss

        # Warm up the history with the real kernel.
        for _ in range(6):
            opt.step(closure)
        pre = float(loss_fn(model(X), y))

        # Force a divergent extrapolation; the safeguard must reject it and keep
        # the plain optimizer step instead.
        def _divergent(Xhist, *args, **kwargs):
            return torch.full(
                (Xhist.size(0),), 1e6, dtype=Xhist.dtype, device=Xhist.device
            )

        original = anderson_mod.get_acceleration
        anderson_mod.get_acceleration = lambda acc_type: _divergent
        try:
            opt.step(closure)
        finally:
            anderson_mod.get_acceleration = original

        post = float(loss_fn(model(X), y))
        self.assertLess(post, 1e3, "divergent candidate was not rejected")
        self.assertLessEqual(
            post, pre + 1e-6,
            "safeguard let the loss increase above the plain optimizer step",
        )

    def test_safeguard_uses_post_step_baseline(self):
        # Regression test for the acceptance baseline: a candidate whose loss is
        # *better than the previous iterate but worse than the plain step* must
        # be REJECTED. Comparing against the stale pre-step loss (the old bug)
        # would wrongly accept it.
        import AADL.anderson_acceleration as anderson_mod
        from torch.nn.utils import parameters_to_vector

        model = self._fresh_model()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        accelerate(
            opt, acceleration_type="anderson", wait_iterations=1,
            history_depth=6, frequency=1, store_each_nth=1,
        )
        loss_fn = torch.nn.MSELoss()
        X, y = self.X, self.y.unsqueeze(1)

        def closure():
            with torch.enable_grad():
                opt.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
            return loss

        for _ in range(6):
            opt.step(closure)

        # Candidate = midpoint between the previous iterate (theta_t) and the
        # plain step (theta_{t+1}); for this convex problem its loss sits
        # strictly between the two, i.e. worse than the plain step.
        captured = {}

        def _between(Xhist, *args, **kwargs):
            plain = Xhist[:, -1].clone()
            cand = 0.5 * (Xhist[:, -2] + plain)
            captured["plain"] = plain
            captured["cand"] = cand.clone()
            return cand

        original = anderson_mod.get_acceleration
        anderson_mod.get_acceleration = lambda acc_type: _between
        try:
            opt.step(closure)
        finally:
            anderson_mod.get_acceleration = original

        final = parameters_to_vector(model.parameters()).detach()
        # The candidate must be rejected: params equal the plain step, not the
        # (worse) midpoint candidate.
        self.assertTrue(
            torch.allclose(final, captured["plain"], atol=1e-6),
            "safeguard did not revert to the plain step",
        )
        self.assertFalse(
            torch.allclose(final, captured["cand"], atol=1e-6),
            "safeguard wrongly accepted a candidate worse than the plain step",
        )

    def test_unknown_acceleration_type_raises(self):
        opt = torch.optim.SGD(self._fresh_model().parameters(), lr=1e-2)
        with self.assertRaises(ValueError):
            accelerate(opt, acceleration_type="not_a_real_kernel")

    def test_remove_acceleration_restores_step(self):
        opt = torch.optim.SGD(self._fresh_model().parameters(), lr=1e-2)
        original_step_func = opt.step.__func__
        accelerate(opt, acceleration_type="anderson")
        self.assertIsNot(opt.step.__func__, original_step_func)
        remove_acceleration(opt)
        # After removal the bound method should be the original optimizer step
        # and the acceleration attributes should be gone.
        self.assertIs(opt.step.__func__, original_step_func)
        self.assertFalse(hasattr(opt, "acc_type"))
        self.assertFalse(hasattr(opt, "acc_param_hist"))
        self.assertFalse(hasattr(opt, "avg_param_hist"))

    def test_identity_no_average_is_noop(self):
        # acceleration_type='identity' with average=False should leave step
        # unchanged.
        opt = torch.optim.SGD(self._fresh_model().parameters(), lr=1e-2)
        original_step_func = opt.step.__func__
        accelerate(opt, acceleration_type="identity", average=False)
        self.assertIs(opt.step.__func__, original_step_func)


if __name__ == "__main__":
    unittest.main()
