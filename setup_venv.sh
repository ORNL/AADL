#!/usr/bin/env bash
# setup_venv.sh — create a local .venv and install AADL with its dependencies.
#
# Usage:
#   ./setup_venv.sh                # use default python3
#   PYTHON=python3.11 ./setup_venv.sh
#   ./setup_venv.sh --recreate     # delete an existing .venv first

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "--recreate" ]] && [[ -d "$VENV_DIR" ]]; then
    echo ">>> Removing existing $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON' not found on PATH. Set PYTHON=<interpreter> and retry." >&2
    exit 1
fi

# Keep this check synchronized with python_requires in setup.py.
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then
    echo "ERROR: AADL requires Python >= 3.11, but '$PYTHON' is $("$PYTHON" --version 2>&1)." >&2
    echo "       Set PYTHON=<interpreter> to a newer Python, e.g. PYTHON=python3.11 ./setup_venv.sh" >&2
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo ">>> Creating virtual environment in $VENV_DIR using $($PYTHON --version)"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ">>> Upgrading pip / setuptools / wheel"
python -m pip install --upgrade pip setuptools wheel

echo ">>> Installing project requirements"
python -m pip install -r requirements.txt

echo ">>> Installing AADL in editable mode"
python -m pip install -e .

cat <<EOF

Done. Activate the environment with:

    source $VENV_DIR/bin/activate

Run the fast test suite (slow integration tests skipped) with:

    python -m unittest discover -s tests -t . -v

Run the full suite (slow tests included; takes minutes) with:

    RUN_SLOW_TESTS=1 python -m unittest discover -s tests -t . -v
EOF
