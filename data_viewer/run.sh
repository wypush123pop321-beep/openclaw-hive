#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
  # also ensure traj_pipeline deps
  .venv/bin/pip install -q openpyxl
fi

PORT=$(python3 -c "import json; c=json.load(open('config.json')); print(c.get('port', 8080))" 2>/dev/null || echo 8080)
echo "Trajectory Viewer running at http://0.0.0.0:${PORT}"
echo "Access from other machines: http://$(hostname -I | awk '{print $1}'):${PORT}"
.venv/bin/python server.py
