#!/usr/bin/env bash
# Test on a GPU server (2 GPUs, ≥ 22 GB in total): line padding 50 % (config.LINE_PAD_PCT, measured in ocr-bench
# Đợt 1 #12) + the LLM correction, after bash install.sh and bash install_llm.sh.
#
#   bash scripts/check_pad50.sh 2>&1 | tee results/check_pad50.log
#
# 1. check + CPU tests  2. read the 4 sample pages with padding 50 % (two GPUs) → results/pad50/
# 3. score, compare with the 2026-10-19 results (results/two_gpus = padding 0 %); dots must be unchanged
# 4. LLM on the 2026-10-19 OCR → must give the measured output (results/llm_check.md)
# 5. LLM on the padding-50 % OCR → before / after scores (results/llm_pad50.md)
# 6. the demo path: page 2 with --llm (DOCX with the LLM changes in green) → results/pad50_llm_page2/
# 7. summary → results/pad50_summary.md
# Last line: "✔ XONG PAD50" or "✘ LỖI ở bước <n>: …". Writes only into results/ (scripts/push_results.sh pushes it).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
R="$ROOT/results"
PY="$ROOT/.venv/bin/python"
STEP=0
fail() { echo; echo "✘ LỖI ở bước $STEP: $1"; exit 1; }
step() { STEP=$((STEP + 1)); echo; echo "================ BƯỚC $STEP: $1 ($(date -u +%H:%M:%S) UTC)"; }
INPUTS=(examples/page_1/page.jpeg examples/page_2/page.jpeg examples/page_3/page.jpeg examples/page_4/page.jpeg)
TYPES="printed,handwritten,printed,printed"
ARGS=(--dots-device cuda:0 --htr-device cuda:1 --offload none)  # = results/two_gpus (the reference run)

step "kiểm tra máy, môi trường, test CPU"
[ -x "$PY" ] || fail "chưa cài: không có $PY (bash install.sh)"
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$N_GPU" -ge 2 ] || fail "cần 2 GPU (LLM cần ≥ 22 GB tổng; mốc đo trên 2 GPU) — thấy $N_GPU"
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000' | wc -l)
[ "$BUSY" -eq 0 ] || { nvidia-smi; fail "GPU đang bận (tiến trình khác đang dùng > 1 GB) — chờ việc kia xong"; }
"$PY" -m arabic_doc_ocr check > "$R/pad50_check.txt" 2>&1; tail -n 12 "$R/pad50_check.txt"
grep -qx "OK" "$R/pad50_check.txt" || fail "check không in OK (xem results/pad50_check.txt)"
grep -q "LLM correction (optional): venv" "$R/pad50_check.txt" || fail "chưa cài LLM (bash install_llm.sh)"
"$PY" -m pip install -q pytest > /dev/null 2>&1
"$PY" -m pytest -q tests > "$R/pad50_tests.txt" 2>&1 || { tail -30 "$R/pad50_tests.txt"; fail "pytest lỗi"; }
tail -n 1 "$R/pad50_tests.txt"

step "đọc 4 trang mẫu, nới dòng 50 %, 2 GPU (~25 phút)"
O="$R/pad50"; rm -rf "$O"; mkdir -p "$O"
"$PY" scripts/measure.py --out "$O/measure.json" -- "$PY" -m arabic_doc_ocr page "${INPUTS[@]}" --type "$TYPES" \
    --out "$O" --line-pad 50 "${ARGS[@]}" > "$O/log.txt" 2>&1 || { tail -30 "$O/log.txt"; fail "đọc 4 trang lỗi"; }
tail -n 4 "$O/log.txt"

step "chấm điểm, so với mốc 2026-10-19 (results/two_gpus, nới 0 %)"
"$PY" scripts/evaluate.py --results "$O" --ref "$R/two_gpus/eval.json" --json "$O/eval.json" > "$O/eval.md" \
  || fail "chấm điểm lỗi"
cat "$O/eval.md"
# dots đọc cả trang, không dùng ảnh dòng → phải giống hệt mốc; mọi trang phải ghi line_pad_pct = 50
"$PY" - "$O" "$R/two_gpus/eval.json" <<'EOF' || fail "pad50 khác điều kiện của mốc (xem dòng ✘ ở trên)"
import json, sys
from pathlib import Path
o, ref = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
now = json.loads((o / "eval.json").read_text())
ok = True
for page in sorted(ref):
    a, b = round(100 * now[page]["dots"]["chars"], 2), round(100 * ref[page]["dots"]["chars"], 2)
    pads = {p["info"].get("line_pad_pct") for p in json.loads((o / f"{page}_page.json").read_text())["pages"]}
    good = a == b and pads == {50}
    ok &= good
    print(f"{'✔' if good else '✘'} {page}: dots {a:.2f} (mốc {b:.2f}) · line_pad_pct {sorted(pads)}")
sys.exit(0 if ok else 1)
EOF

step "LLM trên OCR của lần đo 2026-10-19 → phải giống bản đo (~15 phút)"
"$PY" scripts/check_llm.py > "$R/pad50_llm_check.log" 2>&1; RC=$?
grep -E "LLM CHECK" "$R/pad50_llm_check.log" | tail -n 1
[ $RC -eq 0 ] || { tail -30 "$R/pad50_llm_check.log"; fail "LLM check lỗi / lệch bản đo (results/pad50_llm_check.log)"; }

step "LLM trên OCR nới 50 % (~15 phút)"
"$PY" scripts/check_llm.py --ocr results/pad50 --name llm_pad50 > "$R/pad50_llm.log" 2>&1 \
  || { tail -30 "$R/pad50_llm.log"; fail "LLM trên OCR nới 50 % lỗi (results/pad50_llm.log)"; }
grep -E "✔ LLM ON" "$R/pad50_llm.log"

step "chạy như demo: trang 2 (viết tay) với --llm (~10 phút)"
O2="$R/pad50_llm_page2"; rm -rf "$O2"; mkdir -p "$O2"
IN=$(mktemp -d); cp examples/page_2/page.jpeg "$IN/page_2.jpeg"  # tên file → tên kết quả page_2_* (evaluate.py)
"$PY" -m arabic_doc_ocr page "$IN/page_2.jpeg" --type handwritten --out "$O2" --llm "${ARGS[@]}" > "$O2/log.txt" 2>&1 \
  || { tail -30 "$O2/log.txt"; fail "page --llm lỗi"; }
rm -rf "$IN"
grep -E "LLM" "$O2/log.txt" | tail -n 2
ls "$O2"/page_2.docx "$O2"/page_2.json > /dev/null || fail "thiếu page_2.docx / page_2.json"
"$PY" - "$O2/page_2.json" <<'EOF' || fail "page --llm: LLM không chạy (xem results/pad50_llm_page2/log.txt)"
import json, sys
js = json.load(open(sys.argv[1], encoding="utf-8"))
p = js["pages"][0]
llm = p["info"].get("llm") or {}
n = sum(len(b.get("llm_changes") or []) for b in p["blocks"])
has_ocr = any("text_ocr" in b for b in p["blocks"])
print(f"{'✔' if llm.get('ok') and has_ocr else '✘'} page_2 --llm: ok={llm.get('ok')} · {n} chỗ sửa · "
      f"line_pad_pct={p['info'].get('line_pad_pct')} · JSON giữ chữ OCR: {has_ocr}")
sys.exit(0 if llm.get("ok") and has_ocr else 1)
EOF
"$PY" scripts/evaluate.py --results "$O2" > "$O2/eval.md" 2>/dev/null; grep page_2 "$O2/eval.md"

step "tóm tắt"
{
  echo "# Nới dòng 50 % + LLM — 4 trang mẫu"
  echo
  echo "$(date -u +%Y-%m-%d\ %H:%M) UTC · $(git log --oneline -1 | cut -c1-80) · $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $N_GPU"
  echo
  echo "## 1. OCR, nới dòng 50 % (results/pad50) — so với mốc 2026-10-19 (nới 0 %)"
  echo
  cat "$O/eval.md"
  echo
  echo "## 2. LLM trên OCR của mốc (phải giống bản đo)"
  echo
  sed -n '/^| page/,$p' "$R/llm_check.md"
  echo
  echo "## 3. LLM trên OCR nới 50 %"
  echo
  sed -n '/^| page/,$p' "$R/llm_pad50.md"
  echo
  echo "## 4. Demo: trang 2 với --llm (results/pad50_llm_page2)"
  echo
  grep -E "^\| page|^\|---|page_2" "$O2/eval.md"
} > "$R/pad50_summary.md"
cat "$R/pad50_summary.md"
echo
echo "✔ XONG PAD50 — results/pad50_summary.md (đẩy lên: bash scripts/push_results.sh)"
