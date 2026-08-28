"""Pytest/unittest path fixup so tests run from the repository root.

Some legacy test helpers still live in ``utils`` outside the installed
package. This conftest makes them available regardless of the current
working directory.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
