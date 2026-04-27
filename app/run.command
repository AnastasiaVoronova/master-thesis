#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT/runtime/venv"

export PYTHONPATH="$ROOT"

source "$VENV_DIR/bin/activate"

echo "Running eNPS_App..."
echo "Don't close this window for eNPS_App to work"

(until curl -s http://localhost:8000 > /dev/null 2>&1; do sleep 1; done && open http://localhost:8000) &

uvicorn app:app --host 127.0.0.1 --port 8000

deactivate
