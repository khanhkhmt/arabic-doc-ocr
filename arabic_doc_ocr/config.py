"""All tunable constants in one place. The values are the ones measured on the sample pages (see README) —
changing them changes the results."""

from __future__ import annotations

import os
from pathlib import Path

# --- where things live (override with environment variables) --------------------------------------------------------
HOME = Path(os.environ.get("ARABIC_OCR_HOME", Path.home() / ".cache" / "arabic_doc_ocr"))
KRAKEN_VENV = Path(os.environ.get("KRAKEN_VENV", HOME / "kraken_venv"))
KRAKEN_SEG_MODEL = Path(os.environ.get("KRAKEN_SEG_MODEL", HOME / "muharaf_seg_best.mlmodel"))
MUHARAF_SEG_URL = "https://zenodo.org/records/14295555/files/muharaf_seg_best.mlmodel?download=1"  # CC BY 4.0
# optional LLM correction (install_llm.sh): own venv, because the LLM needs transformers 5.x
LLM_VENV = Path(os.environ.get("LLM_VENV", HOME / "llm_venv"))

# --- models ---------------------------------------------------------------------------------------------------------
DOTS_MODEL = "dots-studio/dots.mocr"          # layout + printed text
HTR_BASE = "Misraj/Baseer__Nakba"             # handwritten line reader (Qwen2.5-VL-3B)
HTR_ADAPTERS = "khanh99/htr-baseer-nakba-ft"  # LoRA adapters ft3 (main reader) and ft4 (cross-check)
MAIN, CHECK = "ft3", "ft4"

# --- page preparation -----------------------------------------------------------------------------------------------
# small photos (long side < 1000 px) are enlarged to 1500 px (at most ×3) before anything else
UPSCALE_BELOW, UPSCALE_TO, UPSCALE_MAX = 1000, 1500, 3.0
# dots reads the page enlarged to a long side of 2400 px (at most ×3): printed text 77.6 % → 90.3 % on page 1.
# ×2 (3200 px) does not fit a 16 GB GPU. Out of memory → dots reads the page again without enlargement.
DOTS_LONG_SIDE, DOTS_MAX_FACTOR = 2400, 3.0
PDF_DPI = 200

# --- dots.mocr generation -------------------------------------------------------------------------------------------
DOTS_PROMPT = "prompt_layout_all_en"
DOTS_MAX_NEW_TOKENS = 24000
DOTS_TOKENS_PER_IMAGE_TOKEN = 8   # cap = 8 × image tokens + 512 (stops endless loops on small images)
DOTS_LOOP_MAX_PERIOD = 200        # stop when the output repeats a span of ≤ 200 tokens …
DOTS_LOOP_MIN_SPAN = 600          # … over ≥ 600 tokens

# --- line images ---------------------------------------------------------------------------------------------------
# Kraken line images are made taller by this % of the line thickness (half above, half below); the paper-colour mask
# is the line polygon shifted up/down, so curved / slanted lines keep their shape. 0 = the crop of the 2026-10-19
# results (results/two_gpus). Measured in ocr-bench (2026-10-23, ft3, Kraken lines, 66 Muharaf test pages, % correct
# characters): 0 % 63.6 · 15 % 65.6 · 30 % 66.7 · 50 % 67.2; sample page 2 (handwritten): 72.2 → 78.0.
LINE_PAD_PCT = 50

# --- line reader (authors' settings, github.com/misraj-ai/Nakba-pipeline) --------------------------------------------
HTR_MAX_NEW_TOKENS = 128
HTR_MIN_PIXELS, HTR_MAX_PIXELS = 28 * 28, 1280 * 28 * 28

# --- GPU memory (measured, see README "System requirements") -------------------------------------------------------
# one GPU with at least this much memory keeps dots.mocr and the line reader loaded together; a smaller single GPU
# swaps them: the idle model goes to CPU RAM if the machine has at least OFFLOAD_CPU_MIN_RAM_GB, else it is reloaded
ONE_GPU_BOTH_MODELS_GB = 22
OFFLOAD_CPU_MIN_RAM_GB = 24
# LLM correction: Qwen3.5-9B in fp16 (~18 GB) spread over all GPUs → needs this much GPU memory in total
LLM_MIN_GPU_GB = 22

# --- layout categories ----------------------------------------------------------------------------------------------
TEXT_CATS = {"Text", "Title", "Section-header", "List-item", "Caption", "Footnote", "Page-header", "Page-footer"}

# --- page types (which reader gives the main text) ------------------------------------------------------------------
HANDWRITTEN, PRINTED = "handwritten", "printed"
