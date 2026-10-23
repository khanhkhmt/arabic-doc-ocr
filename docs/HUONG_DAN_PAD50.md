# HƯỚNG DẪN TEST NỚI DÒNG 50 % + LLM — dành cho agent trên server GPU (2 GPU)

Mục tiêu: trên nhánh **`llm`**, cài → chạy `scripts/check_pad50.sh` (4 trang mẫu với dòng nới 50 %, LLM sửa chữ,
chạy thử như demo) → đẩy `results/` lên GitHub → dán báo cáo. Mọi việc đã viết sẵn. Việc của bạn: **chép lệnh, chạy,
chờ, dán kết quả**. Không có gì phải tự nghĩ. Tổng ~2 giờ (cài ~45 phút, test ~1 giờ 15).

---

## 0. LUẬT CỨNG — đọc hết trước khi làm

1. **CHỈ ĐƯỢC** chạy đúng các lệnh trong file này, theo đúng thứ tự. Lệnh nào không có trong file này → **CẤM**.
2. **CẤM** sửa bất kỳ file nào trong repo (code, script, docs, config). **CẤM** tự gõ số liệu.
3. **CẤM** tự cài / gỡ / nâng cấp torch, CUDA, driver, transformers. Chỉ `install.sh` và `install_llm.sh` được cài.
4. **CẤM** in token ra màn hình hoặc ghi token vào file / `.git/config` / URL. Token chỉ nằm trong biến môi trường
   `GITHUB_TOKEN` và chỉ được dùng qua header như các lệnh dưới đây. Muốn biết có token chưa: `echo ${GITHUB_TOKEN:+CÓ}`.
5. **CẤM** `pkill`, `killall`, `kill -9`. Chỉ được tạo / tắt phiên tmux tên **`pad50`**. **CẤM** đụng các phiên tmux
   khác (ví dụ `dot1`, `hop_thu` của ocr-bench).
6. **CẤM** mở ngrok / tunnel / link chia sẻ (`--share`), **CẤM** đưa file lên Drive hay dịch vụ nào khác.
   Chỉ đẩy lên GitHub bằng `scripts/push_results.sh`.
7. **CẤM** tự đổi tham số, tự "sửa lỗi", tự chạy lại bước khác khi gặp lỗi.
8. Gặp **bất kỳ** lỗi nào (dòng có `✘ LỖI`, `Traceback`, lệnh trả mã khác 0, log không có dòng mới > 60 phút)
   → **DÁN NGUYÊN VĂN 30 dòng cuối log** rồi **DỪNG**, dùng mẫu tin nhắn ở mục 7.
9. Khi báo cáo: **dán nguyên văn** output. **CẤM** gõ lại, tóm tắt hay dịch chữ Ả Rập.
10. GPU phải RẢNH: nếu việc khác đang chạy trên GPU (ví dụ phiên tmux `dot1` của ocr-bench) → DỪNG ở bước 1, mẫu 7a.

---

## 1. Kiểm tra máy (2 phút)

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
df -BG --output=avail "$HOME" | tail -1
free -g | head -2
tmux ls 2>/dev/null || echo "(không có phiên tmux)"
for v in GITHUB_TOKEN HF_TOKEN; do
  if [ -z "${!v:-}" ] && tmux show-environment -g "$v" >/dev/null 2>&1; then export "$v=$(tmux show-environment -g "$v" | cut -d= -f2-)"; fi
done
echo "GITHUB_TOKEN: ${GITHUB_TOKEN:+CÓ}  HF_TOKEN: ${HF_TOKEN:+CÓ}"
```

(Mở shell mới thì chạy lại vòng `for v in …` ở trên trước các bước sau. `HF_TOKEN` không bắt buộc.)

Điều kiện để đi tiếp — **tất cả** phải đúng:
- có **2 dòng** GPU, cột `memory.used` mỗi GPU **< 1000 MiB**;
- ổ đĩa còn **≥ 60G** (model OCR ~15 GB + LLM ~18 GB);
- `tmux ls` **không** có phiên `dot1` (hoặc không có phiên tmux nào);
- dòng token có `GITHUB_TOKEN: CÓ`.

Sai điều kiện nào → DỪNG, mẫu 7a. Thiếu tmux → `apt-get install -y tmux` (một lần) rồi chạy lại bước 1.

## 2. Lấy code nhánh `llm` (1 phút)

```bash
W=$( [ -d /kaggle/working ] && echo /kaggle/working || { [ -d /content ] && echo /content || echo "$HOME"; } )
cd "$W"
AUTH=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
if [ -d arabic-doc-ocr/.git ]; then
  git -C arabic-doc-ocr -c http.extraHeader="Authorization: Basic $AUTH" fetch -q origin llm
  git -C arabic-doc-ocr checkout -q llm && git -C arabic-doc-ocr -c http.extraHeader="Authorization: Basic $AUTH" pull -q --ff-only origin llm
else
  git -c http.extraHeader="Authorization: Basic $AUTH" clone -q -b llm https://github.com/khanhkhmt/arabic-doc-ocr.git
fi
unset AUTH
echo "$W/arabic-doc-ocr" > "$HOME/.arabic_doc_ocr_dir"
cd "$W/arabic-doc-ocr" && git rev-parse --abbrev-ref HEAD && git log --oneline -1 && ls scripts/check_pad50.sh install_llm.sh
```

Phải in `llm`, một dòng commit và hai tên file. Từ đây mọi khối lệnh bắt đầu bằng `cd "$(cat ~/.arabic_doc_ocr_dir)"`.
Lỗi (`fatal`, `Authentication failed`, `No such file`, `Not possible to fast-forward`) → DỪNG, mẫu 7b.

## 3. Cài (~45 phút, chạy nền trong tmux)

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"
tmux new -d -s pad50 "bash install.sh > results/pad50_install.log 2>&1; bash install_llm.sh >> results/pad50_install.log 2>&1; echo '=== HẾT CÀI' >> results/pad50_install.log"
sleep 20; tail -3 results/pad50_install.log
```

Cứ **10 phút** kiểm tra một lần:

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"; tail -4 results/pad50_install.log
```

- Có dòng `=== HẾT CÀI` → chạy `grep -cx "OK" results/pad50_install.log; grep -c "^Done LLM" results/pad50_install.log`:
  - hai số đều **≥ 1** → sang bước 4;
  - có số `0` → `tail -30 results/pad50_install.log`, dán nguyên văn, DỪNG (mẫu 7c).
- Chưa có `=== HẾT CÀI` → chờ 10 phút. Tải model 10–30 phút không có dòng mới là BÌNH THƯỜNG.
- Cảnh báo pip kiểu "dependency resolver … incompatible" là BÌNH THƯỜNG.

## 4. Chạy test (~1 giờ 15, chạy nền trong tmux)

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"
tmux kill-session -t pad50 2>/dev/null
tmux new -d -s pad50 "bash scripts/check_pad50.sh 2>&1 | tee results/check_pad50.log"
sleep 30; tail -5 results/check_pad50.log
```

Cứ **10 phút** kiểm tra một lần:

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"; grep -E "BƯỚC|✔|✘|⚠" results/check_pad50.log | tail -6; tmux has-session -t pad50 2>/dev/null && echo "(đang chạy)" || echo "(phiên pad50 đã kết thúc)"
```

| thấy gì | làm gì |
|---|---|
| có `✔ XONG PAD50` | sang bước 5 |
| có `✘ LỖI ở bước …` | `tail -30 results/check_pad50.log`, dán nguyên văn, DỪNG (mẫu 7c) |
| `(phiên pad50 đã kết thúc)` mà không có hai dòng trên | `tail -30 results/check_pad50.log`, dán nguyên văn, DỪNG (mẫu 7c) |
| log không có dòng mới suốt 60 phút | `tail -30 results/check_pad50.log` và `nvidia-smi`, dán nguyên văn, DỪNG (mẫu 7c) |
| `(đang chạy)` | chờ 10 phút, kiểm tra lại |

BÌNH THƯỜNG, không phải lỗi: dòng `⚠ LLM CHECK: … blocks identical` (lệch nhỏ do GPU / thư viện — script vẫn đi
tiếp), số `ft3` / `ft4` / chữ chính trang 2 khác mốc (đó là điều đang đo), mỗi bước LLM 5–15 phút không có dòng mới.

## 5. Đẩy kết quả lên GitHub

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)" && bash scripts/push_results.sh
```

Phải in `✔ ĐÃ ĐẨY lên nhánh llm: …`. In `✘ LỖI` → dán nguyên văn, DỪNG (mẫu 7d). **CẤM** tự `git add` / `commit` /
`push` bằng tay.

## 6. Báo cáo (mẫu bắt buộc)

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"; git log --oneline -1; cat results/pad50_summary.md
```

```
✔ ĐÃ CHẠY XONG check_pad50 (nhánh llm)
Commit kết quả: <dòng git log>
--- results/pad50_summary.md ---
<dán nguyên văn>
```

Sau đó **DỪNG**. Không tắt server, không xoá thư mục, không chạy web.

## 7. Mẫu tin nhắn khi DỪNG

- **7a (máy không đủ điều kiện / GPU bận):** `✘ DỪNG ở bước 1 — <điều kiện sai>. Output: <dán nguyên văn output bước 1>`
- **7b (không lấy được code):** `✘ DỪNG ở bước 2 — <dán nguyên văn lỗi>`
- **7c (lỗi khi cài / chạy):** `✘ DỪNG ở bước <3 hoặc 4> — <dán nguyên văn 30 dòng cuối log>` (+ `nvidia-smi` nếu treo)
- **7d (không đẩy được):** `✘ DỪNG ở bước 5 — <dán nguyên văn output push_results.sh>`
