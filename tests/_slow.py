"""Common helpers for the legacy integration tests."""

import os
import unittest


def _truthy(val):
    return val is not None and val.lower() not in ("", "0", "false", "no")


SLOW_TESTS_ENABLED = _truthy(os.environ.get("RUN_SLOW_TESTS"))

slow = unittest.skipUnless(
    SLOW_TESTS_ENABLED,
    "slow integration test; set RUN_SLOW_TESTS=1 to enable",
)
