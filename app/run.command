#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT/runtime/venv"

export PYTHONPATH="$ROOT"

source "$VENV_DIR/bin/activate"

echo "Running eNPS_App..."
echo "Don't close this window for eNPS_App to work"

python "$ROOT/app.py"

deactivate
