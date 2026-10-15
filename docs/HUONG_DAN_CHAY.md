# HƯỚNG DẪN CHẠY — dành cho agent (và người) trên server GPU

Mục tiêu: trên một server Linux có GPU NVIDIA, **cài → chạy thử toàn bộ hệ thống → đo VRAM / RAM / ổ đĩa / thời
gian → sinh tài liệu `docs/SYSTEM_REQUIREMENTS.md` và `docs/RESULTS.md` → đẩy kết quả lên GitHub**.
Mọi việc đã viết sẵn thành script. Việc của bạn: **chép lệnh, chạy, chờ, dán kết quả**. Không có gì phải tự nghĩ.

---

## 0. LUẬT CỨNG — đọc hết trước khi làm

1. **CHỈ ĐƯỢC** chạy đúng các lệnh trong file này, theo đúng thứ tự. Lệnh nào không có trong file này → **CẤM**.
2. **CẤM** sửa bất kỳ file nào trong repo (code, script, docs, config). Tài liệu do script tự sinh từ số đo.
   **CẤM** tự gõ số liệu hay viết thêm vào docs.
3. **CẤM** tự cài / gỡ / nâng cấp torch, CUDA, driver, transformers. Chỉ `install.sh` được cài thư viện.
4. **CẤM** in token ra màn hình hoặc ghi token vào file / `.git/config` / URL. Token chỉ nằm trong biến môi trường
   `GITHUB_TOKEN` và chỉ được dùng qua header như các lệnh dưới đây.
5. **CẤM** `pkill`, `killall`, `kill -9` tiến trình không do bạn tạo. Chỉ được dùng phiên tmux tên **`run`**.
6. **CẤM** mở ngrok / tunnel / link chia sẻ (`--share`), **CẤM** đưa file lên Drive hay dịch vụ nào khác.
   Chỉ đẩy lên GitHub bằng `scripts/push_results.sh`.
7. **CẤM** tự đổi tham số, tự "sửa lỗi", tự chạy lại bước khác khi gặp lỗi.
8. Gặp **bất kỳ** lỗi nào (dòng có `✘ LỖI`, `Traceback`, `Error`, lệnh trả mã khác 0, máy treo > 60 phút không
   có dòng log mới) → **DÁN NGUYÊN VĂN 30 dòng cuối log** rồi **DỪNG**, dùng mẫu tin nhắn ở mục 7.
9. Khi báo cáo: **dán nguyên văn** output. **CẤM** gõ lại, tóm tắt hay dịch chữ Ả Rập.

---

## 1. Kiểm tra máy (2 phút)

```bash
nvidia-smi -L
df -BG --output=avail "$HOME" | tail -1
free -g | head -2
[ -n "${GITHUB_TOKEN:-}" ] && echo "GITHUB_TOKEN: có" || echo "GITHUB_TOKEN: THIẾU"
command -v tmux >/dev/null && echo "tmux: có" || echo "tmux: THIẾU"
```

Điều kiện để đi tiếp — **tất cả** phải đúng:
- `nvidia-smi -L` in ra **ít nhất 1 dòng** `GPU 0: …`
- ổ đĩa còn **≥ 40G**
- `GITHUB_TOKEN: có`

Sai điều kiện nào → DỪNG, báo bằng mẫu 7a. Nếu chỉ thiếu tmux thì chạy `apt-get install -y tmux` (một lần), rồi
chạy lại lệnh kiểm tra.

## 2. Lấy code (1 phút)

Chép nguyên khối này (chọn thư mục làm việc tự động: `/kaggle/working`, `/content` hoặc `$HOME`):

```bash
W=$( [ -d /kaggle/working ] && echo /kaggle/working || { [ -d /content ] && echo /content || echo "$HOME"; } )
echo "Thư mục làm việc: $W"
cd "$W"
AUTH=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)
if [ -d arabic-doc-ocr/.git ]; then
  git -C arabic-doc-ocr -c http.extraHeader="Authorization: Basic $AUTH" pull -q --ff-only
else
  git -c http.extraHeader="Authorization: Basic $AUTH" clone -q https://github.com/khanhkhmt/arabic-doc-ocr.git
fi
unset AUTH
echo "$W/arabic-doc-ocr" > "$HOME/.arabic_doc_ocr_dir"
cd "$W/arabic-doc-ocr" && git log --oneline -1
```

Phải in ra một dòng commit. Từ đây mọi khối lệnh bắt đầu bằng `cd "$(cat ~/.arabic_doc_ocr_dir)"` (tự về đúng thư mục,
kể cả khi shell mới). Lỗi (`fatal`, `Authentication failed`, `not found`) → DỪNG, mẫu 7b.

## 3. Chạy toàn bộ (1–3 giờ, chạy nền trong tmux)

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"
tmux new -d -s run "bash scripts/run_all.sh 2>&1 | tee run_all.log"
sleep 20; tail -5 run_all.log
```

Nếu báo `duplicate session: run` → phiên cũ còn chạy: **KHÔNG** tạo phiên mới, sang bước 4 để theo dõi.

Script tự làm lần lượt (bạn không làm gì cả): ghi cấu hình máy → `install.sh` (venv, Kraken, tải ~15 GB model)
→ test CPU → đo ổ đĩa → đọc 4 trang mẫu ở từng chế độ GPU (2 GPU nếu có; 1 GPU nạp lại; 1 GPU qua RAM nếu RAM
≥ 20 GB; 1 GPU cả hai model — chế độ thử, **được phép** báo hết bộ nhớ) → chấm điểm → thử PDF, ảnh một dòng, web
→ sinh `docs/SYSTEM_REQUIREMENTS.md`, `docs/RESULTS.md`.

## 4. Theo dõi

Cứ **10 phút** chạy một lần (không chạy dày hơn):

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"; tail -5 run_all.log; tmux has-session -t run 2>/dev/null && echo "(đang chạy)" || echo "(phiên run đã kết thúc)"
```

Đọc kết quả:
| thấy gì | làm gì |
|---|---|
| dòng cuối có `✔ XONG TẤT CẢ` | sang bước 5 |
| có dòng `✘ LỖI ở bước …` | chạy `tail -30 run_all.log`, dán nguyên văn, DỪNG (mẫu 7c) |
| `(phiên run đã kết thúc)` mà không có hai dòng trên | chạy `tail -30 run_all.log`, dán nguyên văn, DỪNG (mẫu 7c) |
| log không thêm dòng mới suốt 60 phút | chạy `tail -30 run_all.log` và `nvidia-smi`, dán nguyên văn, DỪNG (mẫu 7c) |
| `(đang chạy)`, log có dòng mới | chờ 10 phút, chạy lại lệnh theo dõi |

Bình thường: bước cài đặt 10–30 phút không có dòng mới (log cài nằm trong `results/install.log`); mỗi chế độ đọc
4 trang mất 10–40 phút. Dòng `chế độ thử … lỗi = GPU không đủ chỗ` **không phải lỗi**, script tự đi tiếp.

## 5. Đẩy kết quả lên GitHub

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)" && bash scripts/push_results.sh
```

Phải in `✔ ĐÃ ĐẨY lên nhánh main: …`. Script chỉ đẩy `results/` và `docs/`; nó tự từ chối nếu có file khác bị đổi.
In `✘ LỖI` → dán nguyên văn, DỪNG (mẫu 7d). **CẤM** tự `git add` / `git commit` / `git push` bằng tay.

## 6. Báo cáo (mẫu bắt buộc)

```bash
cd "$(cat ~/.arabic_doc_ocr_dir)"; git log --oneline -1; cat docs/RESULTS.md; sed -n '/^## Measured/,/^## Disk/p;/^## Requirements/,/^## Maximum/p' docs/SYSTEM_REQUIREMENTS.md
```

Gửi đúng mẫu này, phần output **dán nguyên văn**:

```
✔ ĐÃ CHẠY XONG arabic-doc-ocr
Commit kết quả: <dòng git log>
--- output lệnh báo cáo ---
<dán nguyên văn>
```

Sau đó **DỪNG**, không làm gì thêm. Không tắt server, không xoá thư mục.

## 7. Mẫu tin nhắn khi DỪNG

- **7a (máy không đủ điều kiện):** `✘ DỪNG ở bước 1 — <điều kiện sai>. Output: <dán nguyên văn output bước 1>`
- **7b (không lấy được code):** `✘ DỪNG ở bước 2 — <dán nguyên văn lỗi>`
- **7c (lỗi khi chạy):** `✘ DỪNG ở bước 4 — <dán nguyên văn 30 dòng cuối run_all.log>` (+ `nvidia-smi` nếu treo)
- **7d (không đẩy được):** `✘ DỪNG ở bước 5 — <dán nguyên văn output push_results.sh>`

---

### Phụ lục cho người (không phải việc của agent)

- Chạy web sau khi cài: `bash run_web.sh` → `http://127.0.0.1:7862` (server từ xa: `ssh -L 7862:127.0.0.1:7862 <server>`).
- Đọc tài liệu bằng dòng lệnh: `.venv/bin/python -m arabic_doc_ocr page <file> --type printed|handwritten --out <thư mục>`.
- Chạy lại chỉ phần sinh tài liệu từ số đo có sẵn: `.venv/bin/python scripts/make_report.py --results results`.
