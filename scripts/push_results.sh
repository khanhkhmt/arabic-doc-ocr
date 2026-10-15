#!/usr/bin/env bash
# Commit and push ONLY results/ and docs/ (the measurements of scripts/run_all.sh and the documents generated
# from them) to the repository this folder was cloned from. Needs GITHUB_TOKEN in the environment (never printed).
#
#   bash scripts/push_results.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -n "${GITHUB_TOKEN:-}" ] || { echo "✘ LỖI: chưa có biến GITHUB_TOKEN"; exit 1; }
if grep -q "NOT MEASURED YET" docs/SYSTEM_REQUIREMENTS.md docs/RESULTS.md; then
  echo "✘ LỖI: docs chưa được sinh từ số đo — chạy scripts/run_all.sh tới dòng ✔ XONG TẤT CẢ trước"; exit 1
fi
# refuse if anything outside results/ docs/ changed (the run must not modify code)
OTHER=$(git status --porcelain --untracked-files=all | awk '{print $2}' | grep -vE '^(results/|docs/|run_all\.log$)' || true)
[ -z "$OTHER" ] || { echo "✘ LỖI: có file ngoài results/ docs/ bị đổi — không đẩy:"; echo "$OTHER"; exit 1; }
git add results docs
git -c user.name="khanhkhmt" -c user.email="nguyenquockhanh0909200513579@gmail.com" commit -q -m "results: $(grep -m1 '^gpu:' results/hardware.txt | cut -d, -f2 | xargs) · $(date -u +%Y-%m-%d)" || { echo "không có gì mới để đẩy"; exit 0; }
AUTH=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
BR=$(git rev-parse --abbrev-ref HEAD)
for i in 1 2 3; do
  if git -c http.extraHeader="Authorization: Basic $AUTH" push -q origin "HEAD:$BR"; then
    echo "✔ ĐÃ ĐẨY lên nhánh $BR: $(git log --oneline -1)"; exit 0
  fi
  git -c http.extraHeader="Authorization: Basic $AUTH" pull -q --rebase origin "$BR" || true
  sleep 5
done
echo "✘ LỖI: push thất bại 3 lần"; exit 1
