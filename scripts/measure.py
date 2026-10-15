"""Measure what a run really needs: GPU memory (nvidia-smi, includes the CUDA context), RAM of the process tree,
wall time — while running any arabic_doc_ocr command.

    python scripts/measure.py --out measure.json -- python -m arabic_doc_ocr page examples/*/page.jpeg --out ocr_out

Samples every 0.5 s. Prints peak GPU memory per GPU, peak RAM (RSS of the command and its children, e.g. Kraken),
wall time, and the per-stage peaks the pipeline records itself (ocr_out/run_info.json).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def gpu_used() -> dict[int, float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return {int(i): float(m) / 1024 for i, m in (ln.split(",") for ln in out.strip().splitlines())}
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="measure.json")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    base = gpu_used()
    peak_gpu, peak_rss, samples = dict(base), 0.0, []
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    p = psutil.Process(proc.pid)
    done = threading.Event()

    def sample():
        nonlocal peak_rss
        while not done.is_set():
            g = gpu_used()
            for i, v in g.items():
                peak_gpu[i] = max(peak_gpu.get(i, 0), v)
            try:
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
            except psutil.Error:
                rss = 0
            peak_rss = max(peak_rss, rss / 2**30)
            samples.append([round(time.time() - t0, 1), {i: round(v, 2) for i, v in g.items()}, round(rss / 2**30, 2)])
            time.sleep(0.5)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    rc = proc.wait()
    done.set()
    th.join()
    res = {"command": cmd, "returncode": rc, "seconds": round(time.time() - t0, 1),
           "gpu_used_before_gb": {i: round(v, 2) for i, v in base.items()},
           "gpu_peak_gb": {i: round(v, 2) for i, v in peak_gpu.items()}, "ram_peak_gb": round(peak_rss, 2),
           "samples": samples}
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"returncode {rc} · {res['seconds']} s · GPU peak (nvidia-smi) {res['gpu_peak_gb']} GB · "
          f"RAM peak {res['ram_peak_gb']} GB → {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
