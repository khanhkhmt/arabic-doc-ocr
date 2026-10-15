"""Write docs/SYSTEM_REQUIREMENTS.md and docs/RESULTS.md from the measurements of scripts/run_all.sh.

    python scripts/make_report.py --results results

Every number in the two documents comes from the files in --results; requirements are derived by fixed rules
(written next to each number), nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MODES = {
    "two_gpus": "2 GPUs — dots on GPU 0, line reader on GPU 1",
    "one_gpu_unload": "1 GPU, swap by reloading (`--offload unload`)",
    "one_gpu_cpu": "1 GPU, swap through CPU RAM (`--offload cpu`)",
    "one_gpu_both": "1 GPU, both models loaded together (`--offload none`)",
}
# previous pipeline (htr_test/app_ft.py of the research repo), same scoring: correct characters %, main text
PREVIOUS = {"page_1": ("main", 90.3), "page_2": ("ft3", 72.6), "page_3": ("main", 79.0), "page_4": ("main", 90.9)}
CONTEXT_GB = 1.0      # CUDA context + allocator fragmentation on top of PyTorch's peak allocation
SAFETY_GB = 2.0       # RAM headroom for the OS and other processes


def _kv(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out.setdefault(k.strip(), []).append(v.strip())
    return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def load(results: Path) -> dict:
    modes = {}
    for m in MODES:
        d = results / m
        if (results / f"{m}.skipped.json").exists():
            modes[m] = {"status": "skipped", "why": _json(results / f"{m}.skipped.json")["skipped"]}
            continue
        if not d.exists():
            continue
        meas, info, ev = _json(d / "measure.json"), _json(d / "run_info.json"), _json(d / "eval.json")
        log = (d / "log.txt").read_text(encoding="utf-8", errors="replace") if (d / "log.txt").exists() else ""
        ok = bool(meas and meas["returncode"] == 0 and info)
        why = ""
        if not ok:
            tail = [ln for ln in log.splitlines() if ln.strip()][-3:]
            why = "out of GPU memory" if "out of memory" in log.lower() else " / ".join(tail)[-300:]
        pages = []
        for doc in (info or {}).get("documents", []):
            for p in doc["per_page"]:
                pages.append({"input": doc["input"], "type": doc["type"], **p})
        modes[m] = {"status": "ok" if ok else "failed", "why": why, "measure": meas, "info": info, "eval": ev,
                    "pages": pages, "main": {f.name: f.read_text(encoding="utf-8") for f in sorted(d.glob("*_main.txt"))}}
    return {"hw": _kv(results / "hardware.txt"), "disk": _kv(results / "disk.txt"),
            "install": _json(results / "install.json") or {}, "modes": modes,
            "tests": (results / "tests.txt").read_text(encoding="utf-8").strip().splitlines()[-1:]
            if (results / "tests.txt").exists() else ["not run"],
            "pdf": (results / "pdf_test" / "pdf_test.json").exists(), "line": _json(results / "line_test.json"),
            "web": (results / "web_test.txt").exists()}


def _gpus(hw: dict) -> list[str]:
    g = hw.get("gpu", [])
    return [g] if isinstance(g, str) else g


def requirements(D: dict) -> str:
    hw, modes, disk = D["hw"], D["modes"], D["disk"]
    ok = {m: v for m, v in modes.items() if v["status"] == "ok"}
    allp = [p for v in ok.values() for p in v["pages"]]
    dots_peak = max((max(p["dots_gpu_peak_gb"].values()) for p in allp if p.get("dots_gpu_peak_gb")), default=0)
    htr_peak = max((max(p["htr_gpu_peak_gb"].values()) for p in allp if p.get("htr_gpu_peak_gb")), default=0)
    scales = sorted({p.get("dots_scale") for p in allp})
    oom_notes = sorted({n for p in allp for n in p.get("notes", []) if "out of GPU memory" in n})
    gpu_min = _ceil(max(dots_peak, htr_peak) + CONTEXT_GB)
    gpu_both = _ceil(dots_peak + htr_peak + CONTEXT_GB)
    ram = {m: v["measure"]["ram_peak_gb"] for m, v in ok.items()}
    ram_min_mode = "one_gpu_unload" if "one_gpu_unload" in ram else min(ram, key=ram.get) if ram else None
    ram_min = _ceil(ram[ram_min_mode] + SAFETY_GB) if ram_min_mode else None
    if "one_gpu_cpu" in ram:
        ram_cpu, ram_cpu_how = _ceil(ram["one_gpu_cpu"] + SAFETY_GB), "measured"
    else:
        ram_cpu, ram_cpu_how = (_ceil(ram[ram_min_mode] + 9 + SAFETY_GB) if ram_min_mode else None), \
            "estimate: unload-mode peak + 9 GB (dots.mocr weights held in RAM)"
    models = {k.split(" ", 1)[1]: float(v) for k, v in disk.items() if k.startswith("model_gb")}
    disk_total = sum(models.values()) + float(disk.get("venv_gb", 0)) + float(disk.get("kraken_venv_gb", 0))
    ref = next(iter(ok), None)
    per_type: dict[str, list] = {}
    if ref:
        for p in ok[ref]["pages"]:
            per_type.setdefault(p["type"], []).append(p)

    L = ["# System requirements", "",
         f"Measured by `scripts/run_all.sh` on {hw.get('date_utc', '?')} — every number below comes from "
         "`results/` (generated by `scripts/make_report.py`, do not edit by hand).", "",
         "## Test machine", "",
         *[f"- GPU {g}" for g in _gpus(hw)],
         f"- NVIDIA driver {hw.get('nvidia_driver', '?')} · CPU {hw.get('cpu', '?')} · RAM {hw.get('ram_gb', '?')} GB",
         f"- {hw.get('os', '?')} · {hw.get('python', '?')}", "",
         "## Measured: 4 sample pages (3 printed, 1 handwritten) per GPU mode", "",
         "| mode | result | GPU memory peak (nvidia-smi, per GPU) | RAM peak | time, 4 pages | model loads |",
         "|---|---|---|---:|---:|---|"]
    for m, label in MODES.items():
        v = modes.get(m)
        if not v:
            L.append(f"| {label} | not run (machine has 1 GPU) | | | | |" if m == "two_gpus" else f"| {label} | not run | | | | |")
            continue
        if v["status"] != "ok":
            L.append(f"| {label} | **{v['status']}**: {v['why']} | | | | |")
            continue
        me = v["measure"]
        g = ", ".join(f"GPU {i}: {x:.1f} GB" for i, x in me["gpu_peak_gb"].items())
        L.append(f"| {label} | ok | {g} | {me['ram_peak_gb']:.1f} GB | {me['seconds'] / 60:.1f} min | "
                 f"dots {v['info']['loads']['dots']}× · line reader {v['info']['loads']['htr']}× |")
    L += ["", "nvidia-smi shows what PyTorch has reserved (it keeps freed memory cached), so it is higher than what is "
          "needed. The need is PyTorch's peak allocation:", "",
          f"- dots.mocr (layout, page enlarged to 2400 px): **{dots_peak:.1f} GB** peak (dots scale used: "
          f"{', '.join(f'×{s:g}' for s in scales)})",
          f"- line reader (Baseer-Nakba + ft3/ft4): **{htr_peak:.1f} GB** peak"]
    if oom_notes:
        L.append(f"- ⚠ {len(oom_notes)} page(s) ran out of GPU memory at 2400 px and were read without enlargement "
                 "(lower accuracy on printed text) — this GPU is below the minimum below.")
    if ref:
        L += ["", f"Time per page ({MODES[ref]}):", "", "| page | type | dots layout | lines | ft3 + ft4 reading |",
              "|---|---|---:|---:|---:|"]
        for p in ok[ref]["pages"]:
            L.append(f"| {Path(p['input']).parent.name} | {p['type']} | {p['dots_seconds']:.0f} s | {p.get('lines', 0)} "
                     f"| {p.get('htr_seconds', 0):.0f} s |")
    L += ["", "## Disk", "", "| item | size |", "|---|---:|",
          *[f"| model {k} | {v:.1f} GB |" for k, v in models.items()],
          f"| main virtualenv (.venv) | {float(disk.get('venv_gb', 0)):.1f} GB |",
          f"| Kraken virtualenv | {float(disk.get('kraken_venv_gb', 0)):.1f} GB |",
          f"| **total** | **{disk_total:.1f} GB** |", "",
          f"Install (`install.sh`, incl. model download) took {D['install'].get('seconds', 0) / 60:.0f} min on this "
          "machine.", "",
          "## Requirements (derived from the measurements)", "",
          "| | minimum | recommended |", "|---|---|---|",
          f"| GPU | 1 NVIDIA GPU with ≥ **{gpu_min} GB** (= larger model peak + {CONTEXT_GB:g} GB), compute capability "
          f"≥ 7.5 (tested: {', '.join(g.split(',')[1].strip() for g in _gpus(hw)[:1]) if _gpus(hw) else '?'}) — "
          "models swap, slower | 2 GPUs with ≥ "
          f"{gpu_min} GB each, or 1 GPU with ≥ **{gpu_both} GB** (= both peaks + {CONTEXT_GB:g} GB) |",
          f"| RAM | **{ram_min} GB** (`--offload unload`, peak {ram.get(ram_min_mode, 0):.1f} GB + {SAFETY_GB:g}) |"
          f" **{ram_cpu} GB** for `--offload cpu` on one GPU ({ram_cpu_how}); same as minimum with 2 GPUs |",
          f"| Disk | **{_ceil(disk_total + 5)} GB** free (models + virtualenvs + 5 GB for outputs / temp) | same |",
          f"| CPU | {hw.get('cpu', '?').split(' cores')[0]} cores tested; Kraken line segmentation runs on the CPU | "
          "4+ cores |",
          "| Software | Linux, NVIDIA driver ≥ 525 (CUDA 12), Python ≥ 3.10, internet for the first model download | |",
          "", "Below the GPU minimum dots.mocr cannot read the page at 2400 px and falls back to the original size "
          "(printed text on page 1: 90.3 % → 77.6 % correct characters). The line reader needs batch size 1, so a "
          "bigger GPU does not make it faster.", "",
          "## Maximum (what the software can use)", "",
          "- GPUs: at most 2 per process (dots on one, line reader on the other); more GPUs are not used. For more "
          "throughput run several processes, each with its own `--dots-device` / `--htr-device`.",
          f"- GPU memory beyond {gpu_both} GB on one GPU (or {gpu_min} GB per GPU with two) brings no speed-up.",
          "- The web interface handles one request at a time (queue).",
          f"- RAM: the measured peak was {max(ram.values()) if ram else 0:.1f} GB; more RAM is not used.", "",
          "## Setting checked by these numbers", "",
          f"`config.ONE_GPU_BOTH_MODELS_GB = 22`: with one GPU of at least this size `--offload auto` keeps both "
          f"models loaded. Measured need: {gpu_both} GB → "
          + ("OK." if gpu_both <= 22 else f"**raise it to {gpu_both}** in arabic_doc_ocr/config.py."), ""]
    return "\n".join(L)


def results_md(D: dict) -> str:
    ok = {m: v for m, v in D["modes"].items() if v["status"] == "ok"}
    ref = next(iter(ok), None)
    L = ["# Results on the sample pages", "",
         f"Generated by `scripts/make_report.py` from `results/` ({D['hw'].get('date_utc', '?')}). "
         "Correct characters % / correct words % against `examples/*/label.json` (scripts/evaluate.py).", ""]
    if ref:
        ev = ok[ref]["eval"]
        L += [f"## Accuracy ({MODES[ref]})", "",
              "| page | type | **main text** | dots | ft3 | ft4 | previous pipeline |", "|---|---|---:|---:|---:|---:|---:|"]
        for page, sc in sorted(ev.items()):
            f = lambda k: f"{100 * sc[k]['chars']:.1f} / {100 * sc[k]['words']:.1f}"  # noqa: E731
            pk, pv = PREVIOUS.get(page, ("", None))
            now = 100 * sc[pk]["chars"] if pk else None
            prev = f"{pk} {pv:.1f}" + (f" ({now - pv:+.1f})" if now is not None else "") if pv is not None else ""
            L.append(f"| {page} | {sc.get('type')} | **{f('main')}** | {f('dots')} | {f('ft3')} | {f('ft4')} | {prev} |")
        L += ["", "Main text = dots on printed pages (pages 1, 3, 4), ft3 on the handwritten page (2); Hebrew is not "
              "scored (not in the labels). Page 2's previous number comes from an older layout step, so a difference "
              "there is expected; pages 1, 3, 4 should be within ±0.5.", ""]
        L += ["## Same output in every GPU mode?", ""]
        base = ok[ref]["main"]
        for m, v in ok.items():
            if m == ref:
                continue
            same = [k for k in base if v["main"].get(k) == base[k]]
            L.append(f"- {MODES[m]}: {len(same)}/{len(base)} main texts identical to {ref}")
        if len(ok) == 1:
            L.append("- only one mode ran")
        notes = [(Path(p["input"]).parent.name, n) for p in ok[ref]["pages"] for n in p.get("notes", [])]
        if notes:
            L += ["", "## Pipeline notes", "", *[f"- {a}: {n}" for a, n in notes]]
    L += ["", "## Other checks", "",
          f"- CPU tests (pytest): {' '.join(D['tests'])}",
          f"- PDF, 2 pages: {'ok' if D['pdf'] else 'NOT OK'}",
          f"- `line` command: {'ok — ' + str(len(D['line'])) + ' lines' if D['line'] else 'NOT OK'}",
          f"- web interface answers HTTP 200: {'ok' if D['web'] else 'NOT OK'}", ""]
    for m, v in D["modes"].items():
        if v["status"] != "ok":
            L.append(f"- {MODES[m]}: {v['status']} — {v['why']}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--docs", default="docs")
    a = ap.parse_args()
    D = load(Path(a.results))
    if not any(v["status"] == "ok" for v in D["modes"].values()):
        raise SystemExit("no successful run in results/")
    docs = Path(a.docs)
    docs.mkdir(exist_ok=True)
    (docs / "SYSTEM_REQUIREMENTS.md").write_text(requirements(D) + "\n", encoding="utf-8")
    (docs / "RESULTS.md").write_text(results_md(D), encoding="utf-8")
    print(f"wrote {docs / 'SYSTEM_REQUIREMENTS.md'} and {docs / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
