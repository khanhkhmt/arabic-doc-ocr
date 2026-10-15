#!/usr/bin/env bash
# Start the web interface on http://127.0.0.1:7862 (local only). Extra arguments go to `python -m arabic_doc_ocr web`,
# e.g.  bash run_web.sh --port 8000 --dots-device cuda:0 --htr-device cuda:1
# Remote server: forward the port over SSH (ssh -L 7862:127.0.0.1:7862 server) instead of opening it to the internet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "${VENV:-$ROOT/.venv}/bin/python" -m arabic_doc_ocr web "$@"
