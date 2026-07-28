"""Tests package marker.

Importing this module prepares ``sys.path`` so the legacy tests' bare
``from optimizers import ...`` and ``from TestFunctions_models import ...``
statements resolve regardless of the current working directory. This makes
``python -m unittest discover -s tests`` work from the repo root.
"""

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_HERE)

for _p in (_REPO_ROOT, _os.path.join(_REPO_ROOT, "utils"), _os.path.join(_REPO_ROOT, "model_zoo")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
