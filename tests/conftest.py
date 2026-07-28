"""Pytest/unittest path fixup so tests run from the repository root.

The legacy tests use ``sys.path.append('../utils')`` and
``sys.path.append('../model_zoo')`` which only works when launched from
inside ``tests/``. This conftest prepends the repo root, ``utils/`` and
``model_zoo/`` to ``sys.path`` regardless of CWD so ``unittest discover``
or ``pytest`` work from the repo root too.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "utils"), os.path.join(_REPO_ROOT, "model_zoo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
