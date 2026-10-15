#!/usr/bin/env bash
# Full acceptance run on a GPU server: install → tests → read the 4 sample pages in every GPU mode the machine allows
# (measuring GPU memory / RAM / time) → score → PDF, line and web checks → write docs/SYSTEM_REQUIREMENTS.md and
# docs/RESULTS.md from the measurements.
#
#   bash scripts/run_all.sh 2>&1 | tee run_all.log
#
# Last line is either "✔ XONG TẤT CẢ" (all done) or "✘ LỖI ở bước <n>" (error at step n).
# Re-running is safe: install and model downloads are skipped when already done; measurements are redone.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
R="$ROOT/results"
PY="$ROOT/.venv/bin/python"
STEP=0
fail() { echo; echo "✘ LỖI ở bước $STEP: $1"; exit 1; }
step() { STEP=$((STEP + 1)); echo; echo "================ BƯỚC $STEP: $1 ($(date -u +%H:%M:%S) UTC)"; }
mkdir -p "$R"

step "máy (GPU, RAM, CPU, ổ đĩa)"
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "os: $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}") · kernel $(uname -r)"
  echo "cpu: $(nproc) cores · $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "ram_gb: $(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
  echo "disk_free_gb: $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)"
  echo "python: $(python3 --version 2>&1)"
  echo "nvidia_driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader | sed 's/^/gpu: /'
} | tee "$R/hardware.txt"
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
[ "$N_GPU" -ge 1 ] || fail "không thấy GPU nào (nvidia-smi -L rỗng)"

step "cài đặt (install.sh)"
T0=$(date +%s)
bash install.sh > "$R/install.log" 2>&1 || { tail -30 "$R/install.log"; fail "install.sh lỗi (xem results/install.log)"; }
echo "{\"seconds\": $(( $(date +%s) - T0 ))}" > "$R/install.json"
tail -25 "$R/install.log"
grep -q "^OK$" "$R/install.log" || fail "lệnh check không in OK (xem results/install.log)"

step "test CPU (pytest)"
"$PY" -m pip install -q pytest > /dev/null 2>&1
"$PY" -m pytest -q tests > "$R/tests.txt" 2>&1 || { tail -30 "$R/tests.txt"; fail "pytest lỗi"; }
tail -2 "$R/tests.txt"

step "dung lượng ổ đĩa"
{
  echo "venv_gb: $(du -s --block-size=1M .venv | awk '{printf "%.2f", $1/1024}')"
  KV="${KRAKEN_VENV:-$HOME/.cache/arabic_doc_ocr/kraken_venv}"
  echo "kraken_venv_gb: $(du -s --block-size=1M "$KV" | awk '{printf "%.2f", $1/1024}')"
  "$PY" - <<'EOF'
from huggingface_hub import scan_cache_dir
from arabic_doc_ocr import config
have = {r.repo_id: r.size_on_disk for r in scan_cache_dir().repos}
for rid in (config.DOTS_MODEL, config.HTR_BASE, config.HTR_ADAPTERS):
    print(f"model_gb {rid}: {have.get(rid, 0) / 2**30:.2f}")
EOF
} | tee "$R/disk.txt"

INPUTS=(examples/page_1/page.jpeg examples/page_2/page.jpeg examples/page_3/page.jpeg examples/page_4/page.jpeg)
TYPES="printed,handwritten,printed,printed"
MODES=()
[ "$N_GPU" -ge 2 ] && MODES+=("two_gpus|--dots-device cuda:0 --htr-device cuda:1 --offload none")
MODES+=("one_gpu_unload|--dots-device cuda:0 --htr-device cuda:0 --offload unload")
if [ "$RAM_GB" -ge 20 ]; then MODES+=("one_gpu_cpu|--dots-device cuda:0 --htr-device cuda:0 --offload cpu")
else echo "{\"skipped\": \"RAM ${RAM_GB} GB < 20 GB\"}" > "$R/one_gpu_cpu.skipped.json"; fi
MODES+=("one_gpu_both|--dots-device cuda:0 --htr-device cuda:0 --offload none")
REF=""
for M in "${MODES[@]}"; do
  NAME="${M%%|*}"; ARGS="${M#*|}"
  step "đọc 4 trang mẫu, chế độ $NAME"
  O="$R/$NAME"; rm -rf "$O"; mkdir -p "$O"
  # shellcheck disable=SC2086
  "$PY" scripts/measure.py --out "$O/measure.json" -- "$PY" -m arabic_doc_ocr page "${INPUTS[@]}" \
      --type "$TYPES" --out "$O" $ARGS > "$O/log.txt" 2>&1
  RC=$?
  tail -8 "$O/log.txt"
  if [ $RC -ne 0 ]; then
    case "$NAME" in
      one_gpu_both) echo "   (chế độ thử: hai model cùng một GPU — lỗi = GPU không đủ chỗ, ghi vào báo cáo, đi tiếp)" ;;
      one_gpu_cpu) echo "   (lỗi ở chế độ cpu — ghi vào báo cáo, đi tiếp)" ;;
      *) tail -30 "$O/log.txt"; fail "chế độ $NAME lỗi (xem results/$NAME/log.txt)" ;;
    esac
    continue
  fi
  "$PY" scripts/evaluate.py --results "$O" --examples examples --json "$O/eval.json" | tee "$O/eval.md"
  [ -z "$REF" ] && REF="$NAME"
  # only the reference mode keeps the full outputs (DOCX, images, HTML); the others keep JSON / TXT
  rm -f "$O"/*_all.zip
  [ "$NAME" != "$REF" ] && rm -f "$O"/*.docx "$O"/*.jpg "$O"/*_review.html
done
[ -n "$REF" ] || fail "không chế độ nào chạy được"

step "PDF nhiều trang (chế độ $REF)"
"$PY" - <<'EOF' || fail "không tạo được PDF thử"
from PIL import Image
ims = [Image.open(f"examples/page_{i}/page.jpeg").convert("RGB") for i in (1, 4)]
ims[0].save("results/pdf_test.pdf", save_all=True, append_images=ims[1:], resolution=150)
EOF
REF_ARGS=""; for M in "${MODES[@]}"; do [ "${M%%|*}" = "$REF" ] && REF_ARGS="${M#*|}"; done
# shellcheck disable=SC2086
"$PY" -m arabic_doc_ocr page results/pdf_test.pdf --type printed --pages 1-2 --out "$R/pdf_test" $REF_ARGS \
  > "$R/pdf_test.log" 2>&1 || { tail -30 "$R/pdf_test.log"; fail "đọc PDF lỗi"; }
"$PY" -c "import json; d=json.load(open('results/pdf_test/pdf_test.json')); n=len(d['pages']); print('PDF pages read:', n); assert n == 2" \
  || fail "PDF: số trang đọc được khác 2"
rm -f "$R"/pdf_test/*_all.zip "$R"/pdf_test/*.jpg

step "đọc ảnh một dòng (lệnh line)"
REF="$REF" "$PY" - <<'EOF' || fail "không cắt được ảnh dòng"
import json, os
from PIL import Image
ref = f"results/{os.environ['REF']}/page_2_page.json"
d = json.load(open(ref))
im = Image.open("examples/page_2/page.jpeg").convert("RGB")
lines = [L for b in d["pages"][0]["blocks"] for L in b["lines"]][:2]
for i, L in enumerate(lines):
    im.crop(tuple(L["bbox"])).save(f"results/line_{i}.png")
EOF
# shellcheck disable=SC2086
"$PY" -m arabic_doc_ocr line results/line_0.png results/line_1.png --json "$R/line_test.json" \
  --htr-device cuda:0 --dots-device cuda:0 > "$R/line_test.log" 2>&1 || { tail -30 "$R/line_test.log"; fail "lệnh line lỗi"; }
cat "$R/line_test.log" | tail -6

step "web (mở, kiểm tra trả về trang, tắt)"
"$PY" -m arabic_doc_ocr web --port 7899 > "$R/web_test.log" 2>&1 &
WEB=$!
OK=""
for _ in $(seq 1 60); do
  sleep 2
  if curl -fs -o /dev/null http://127.0.0.1:7899/; then OK=1; break; fi
done
kill $WEB 2>/dev/null; wait $WEB 2>/dev/null
[ -n "$OK" ] || { tail -30 "$R/web_test.log"; fail "web không trả về trang trong 120 s"; }
echo "web OK (http 200)" | tee "$R/web_test.txt"

step "viết tài liệu từ số đo (docs/SYSTEM_REQUIREMENTS.md, docs/RESULTS.md)"
"$PY" scripts/make_report.py --results "$R" || fail "make_report.py lỗi"
head -40 docs/SYSTEM_REQUIREMENTS.md

echo
echo "✔ XONG TẤT CẢ — bước tiếp theo: bash scripts/push_results.sh"
