"""Optional LLM correction step (after OCR), run in its own virtualenv — see llm_fix.py for the method.

    results = pipeline.process_document(pages, "printed")
    llm.correct(results, pipeline.models)      # main text corrected; the OCR text is kept in res.texts["ocr"]

The LLM (~18 GB in fp16) needs the GPUs to itself: dots.mocr and the line reader are freed first and reloaded on the
next document (~1 min). Needs ≥ LLM_MIN_GPU_GB of GPU memory in total (2 × T4 is enough, one T4 is not).
If the LLM step fails, the OCR text is kept and the failure is written in the page notes.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from . import config
from .pipeline import PageResult, free_gpu

ROOT = Path(__file__).resolve().parent.parent


def llm_python() -> Path | None:
    p = config.LLM_VENV / "bin" / "python"
    return p if p.exists() else None


def gpu_gb() -> float:
    import torch

    return sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 2**30


def why_not() -> str | None:
    """None if the LLM step can run, else the reason."""
    if llm_python() is None:
        return f"LLM venv not installed ({config.LLM_VENV}) — run: bash install_llm.sh"
    g = gpu_gb()
    if g < config.LLM_MIN_GPU_GB:
        return f"the LLM needs ≥ {config.LLM_MIN_GPU_GB} GB of GPU memory in total, this machine has {g:.0f} GB"
    return None


def units(results: list[PageResult]) -> list[dict]:
    """Blocks to correct: text blocks read by dots or ft3 (not Hebrew), non-empty main text."""
    out = []
    for pi, res in enumerate(results):
        for bi, (src, _) in sorted(res.sources.items()):
            if src not in ("dots", config.MAIN):
                continue
            main = res.texts["main"].get(bi, "")
            if main.strip():
                out.append({"key": f"{pi}:{bi}", "main": main, "check": res.texts["check"].get(bi, "")})
    return out


def correct(results: list[PageResult], models=None, progress=None) -> dict:
    """Correct the main text of every page in place. → info dict (also stored in each page's info["llm"])."""
    def say(p, msg):
        if progress:
            progress(p, msg)

    reason = why_not()
    if reason:
        return _note(results, {"ok": False, "error": reason})
    us = units(results)
    if not us:
        return _note(results, {"ok": True, "units": 0, "changed_words": 0})
    if models is not None:  # free the GPUs: the OCR models are reloaded lazily on the next document
        for name in ("_dots", "_htr"):
            m = getattr(models, name, None)
            if m is not None:
                m.close()
                setattr(models, name, None)
    free_gpu()
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="arabic_ocr_llm_") as d:
        fin, fout = Path(d) / "in.json", Path(d) / "out.json"
        fin.write_text(json.dumps({"units": us}, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        cmd = [str(llm_python()), "-m", "arabic_doc_ocr.llm_fix", "--in", str(fin), "--out", str(fout)]
        say(0.0, "LLM: loading Qwen3.5-9B (~1–2 min)")
        tail: list[str] = []
        with subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) as p:
            for line in p.stdout:
                line = line.rstrip()
                if line.startswith("@@unit "):
                    i, n = map(int, line.split()[1:3])
                    say(i / n, f"LLM: block {i}/{n}")
                elif line.startswith("@@loaded"):
                    say(0.0, f"LLM: loaded, correcting {len(us)} blocks")
                else:
                    tail = (tail + [line])[-30:]
        if p.returncode != 0 or not fout.exists():
            return _note(results, {"ok": False, "error": f"LLM process failed (exit {p.returncode})",
                                   "log_tail": tail})
        r = json.loads(fout.read_text(encoding="utf-8"))
    n_changed = apply(results, r)
    return _note(results, {"ok": True, "model": r["model"], "revision": r["revision"][:10], "method": r["method"],
                           "dtype": r["dtype"], "units": len(us), "changed_words": n_changed,
                           "load_seconds": r["load_seconds"], "seconds": round(time.time() - t0, 1)})


def apply(results: list[PageResult], r: dict) -> int:
    """Put the LLM output into the pages: main text replaced, the OCR text kept in texts["ocr"], accepted changes in
    res.llm_changes. → number of accepted changes."""
    n_changed = 0
    for pi, res in enumerate(results):
        res.texts["ocr"] = dict(res.texts["main"])
        res.llm_changes = {}
        for bi in list(res.texts["main"]):
            key = f"{pi}:{bi}"
            if key in r["texts"]:
                res.texts["main"][bi] = r["texts"][key]
                ch = [c for c in r["changes"].get(key, []) if not c.get("rejected")]
                res.llm_changes[bi] = ch
                n_changed += len(ch)
    return n_changed


def _note(results: list[PageResult], info: dict) -> dict:
    for res in results:
        res.info["llm"] = info
        if not info.get("ok"):
            res.info.setdefault("notes", []).append(f"LLM correction NOT applied: {info.get('error')}")
    return info
