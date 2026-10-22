#!/usr/bin/env bash
# OPTIONAL: install the LLM correction step (Qwen3.5-9B, ~18 GB download) — after bash install.sh.
#
#   bash install_llm.sh                 # venv ~/.cache/arabic_doc_ocr/llm_venv + model download
#   SKIP_MODELS=1 bash install_llm.sh   # venv only (the first --llm run downloads the model)
#
# Own venv, because the LLM needs transformers 5.x and the OCR models need 4.56.1. It reuses the system CUDA torch
# (--system-site-packages) and pins it: nothing replaces torch. Needs torch ≥ 2.6 (transformers 5.15.1).
# Runs only with ≥ 22 GB GPU memory in total (2 × T4 is enough, one T4 is not).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
HOME_DIR="${ARABIC_OCR_HOME:-$HOME/.cache/arabic_doc_ocr}"
VENV="${LLM_VENV:-$HOME_DIR/llm_venv}"
TF=5.15.1
fail() { echo "✘ $1"; exit 1; }
mkdir -p "$HOME_DIR"

"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
  || fail "the system Python ($PY) has no CUDA torch — the LLM step reuses it"
"$PY" -c 'import sys, torch; v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); sys.exit(0 if v >= (2, 6) else 1)' \
  || fail "torch $("$PY" -c 'import torch; print(torch.__version__)') is too old: transformers $TF (Qwen3.5) needs torch >= 2.6"

if command -v uv >/dev/null 2>&1; then UV=(uv)
else "$PY" -m pip install -q uv || fail "cannot install uv"; UV=("$PY" -m uv); fi

echo "== 1/2 LLM venv: $VENV (transformers $TF)"
[ -x "$VENV/bin/python" ] || "${UV[@]}" venv -q --seed "$VENV" -p "$PY" --system-site-packages
"$VENV/bin/python" -c "
from importlib.metadata import version, PackageNotFoundError
for n in ('torch', 'torchvision', 'torchaudio', 'triton'):
    try: print(f'{n}=={version(n)}')
    except PackageNotFoundError: pass" > "$VENV/torch_pin.txt"
# pip of the venv sees the system torch; the pin turns any attempt to replace it into an error
"$VENV/bin/python" -m pip install -q -c "$VENV/torch_pin.txt" "transformers==$TF" "accelerate>=1.0" \
  "huggingface_hub" "rapidfuzz>=3" "sentencepiece" "protobuf" || fail "pip install failed"
PYTHONPATH="$ROOT" "$VENV/bin/python" - <<'EOF' || fail "LLM venv check failed"
import os, sys
import torch, transformers, accelerate, rapidfuzz
assert not os.path.realpath(torch.__file__).startswith(os.path.realpath(sys.prefix) + os.sep), \
    f"torch is INSIDE the venv ({torch.__file__}) — not the system torch"
import arabic_doc_ocr.llm_fix  # noqa: F401
n = torch.cuda.device_count()
gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 2**30
print(f"   torch {torch.__version__} · transformers {transformers.__version__} · accelerate {accelerate.__version__}"
      f" · {n} GPU(s), {gb:.0f} GB" + ("" if gb >= 22 else "  ⚠ < 22 GB: the LLM step will not run on this machine"))
EOF

echo "== 2/2 model"
if [ -z "${SKIP_MODELS:-}" ]; then PYTHONPATH="$ROOT" "$VENV/bin/python" -m arabic_doc_ocr.llm_fix --download
else echo "   skipped"; fi
echo
echo "Done LLM. Web: tick \"LLM correction\".  CLI: .venv/bin/python -m arabic_doc_ocr page <file> --type printed --llm"
