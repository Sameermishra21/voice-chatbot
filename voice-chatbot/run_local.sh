#!/usr/bin/env bash
# One-command local start (macOS / Linux). Usage:  bash run_local.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi
if [ ! -f backend/chatbot_model.keras ]; then
  ./.venv/bin/python backend/train_model.py
fi
echo "Open http://localhost:8000 in Chrome or Safari"
./.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000
