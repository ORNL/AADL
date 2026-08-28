"""Tests package marker.

Importing this module keeps the repository's legacy ``utils`` helpers
available while tests migrate to package-qualified model imports.
"""

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_HERE)

for _p in (_REPO_ROOT, _os.path.join(_REPO_ROOT, "utils")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
