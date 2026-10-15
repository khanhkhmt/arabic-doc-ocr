#!/usr/bin/env bash
# Install arabic-doc-ocr on a Linux machine with an NVIDIA GPU.
#
#   bash install.sh              # .venv (main) + Kraken venv + segmentation model + all models (~15 GB download)
#   SKIP_MODELS=1 bash install.sh   # do not download the big models now (first run downloads them)
#
# torch: if the system Python already has a CUDA torch (Colab, Kaggle, NVIDIA / PyTorch docker images) the venv reuses
# it (--system-site-packages) and pins it, so nothing replaces it. Otherwise torch is installed from PyPI (CUDA 12
# wheels; needs NVIDIA driver ≥ 525).  Kraken lives in its own venv with a CPU torch because it pins its own torch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"
HOME_DIR="${ARABIC_OCR_HOME:-$HOME/.cache/arabic_doc_ocr}"
KVENV="${KRAKEN_VENV:-$HOME_DIR/kraken_venv}"
SEG="${KRAKEN_SEG_MODEL:-$HOME_DIR/muharaf_seg_best.mlmodel}"
mkdir -p "$HOME_DIR"

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 needed"'
# uv (fast installer, can also fetch Python 3.12 for Kraken): already installed → use it; else pip; else its installer
if command -v uv >/dev/null 2>&1; then UV=(uv)
elif "$PY" -m pip install -q uv 2>/dev/null || "$PY" -m pip install -q --user uv 2>/dev/null; then UV=("$PY" -m uv)
else curl -LsSf https://astral.sh/uv/install.sh | sh; UV=("$HOME/.local/bin/uv"); fi

echo "== 1/4 main venv: $VENV"
if "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "   system torch with CUDA found → reused"
  [ -x "$VENV/bin/python" ] || "${UV[@]}" venv -q --seed "$VENV" -p "$PY" --system-site-packages
  "$VENV/bin/python" -c "
from importlib.metadata import version, PackageNotFoundError
for n in ('torch', 'torchvision', 'torchaudio', 'triton'):
    try: print(f'{n}=={version(n)}')
    except PackageNotFoundError: pass" > "$VENV/torch_pin.txt"
  # pip of the venv sees the system torch; the pin turns any attempt to replace it into an error
  "$VENV/bin/python" -m pip install -q -c "$VENV/torch_pin.txt" -r requirements.txt
  "$VENV/bin/python" -m pip install -q -c "$VENV/torch_pin.txt" --no-deps -e .
else
  echo "   no CUDA torch in $PY → installing torch from PyPI"
  [ -x "$VENV/bin/python" ] || "${UV[@]}" venv -q --seed "$VENV" -p "$PY"
  "${UV[@]}" pip install -q -p "$VENV/bin/python" "torch>=2.4,<2.9" "torchvision"
  "${UV[@]}" pip install -q -p "$VENV/bin/python" -r requirements.txt
  "${UV[@]}" pip install -q -p "$VENV/bin/python" --no-deps -e .
fi

echo "== 2/4 Kraken venv (CPU torch): $KVENV"
if [ ! -x "$KVENV/bin/kraken" ]; then
  "${UV[@]}" venv -q "$KVENV" -p 3.12 || "${UV[@]}" venv -q "$KVENV"
  "${UV[@]}" pip install -q -p "$KVENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu "kraken>=7.1,<8"
fi
"$KVENV/bin/kraken" --version | tail -1

echo "== 3/4 Muharaf line segmentation model (CC BY 4.0): $SEG"
[ -s "$SEG" ] || curl -fsSL -o "$SEG" "https://zenodo.org/records/14295555/files/muharaf_seg_best.mlmodel?download=1"
ls -la "$SEG"

echo "== 4/4 models"
if [ -z "${SKIP_MODELS:-}" ]; then "$VENV/bin/python" -m arabic_doc_ocr download; else echo "   skipped"; fi

"$VENV/bin/python" -m arabic_doc_ocr check || true
echo
echo "Done. Web:  bash run_web.sh        CLI:  $VENV/bin/python -m arabic_doc_ocr page <file> --type printed"
