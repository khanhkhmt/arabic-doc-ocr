"""Command line.

    python -m arabic_doc_ocr check                                   # GPUs, library versions, Kraken, model cache
    python -m arabic_doc_ocr download                                # fetch all models into the cache (~15 GB)
    python -m arabic_doc_ocr page scan.pdf photo.jpg --type printed --out results/
    python -m arabic_doc_ocr line line1.png line2.png [--json lines.json]
    python -m arabic_doc_ocr web [--host 127.0.0.1] [--port 7862]

Devices: --dots-device / --htr-device (default: dots on cuda:0, line reader on cuda:1 if there is a second GPU).
--offload auto|none|cpu|unload: what to do when both models share one GPU (see README).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__, config


def _device_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dots-device", default=None, help="e.g. cuda:0 (default: first GPU)")
    ap.add_argument("--htr-device", default=None, help="e.g. cuda:1 (default: second GPU if any, else the first)")
    ap.add_argument("--offload", default="auto", choices=["auto", "none", "cpu", "unload"],
                    help="one shared GPU: keep both models (none), move the idle one to CPU RAM (cpu) or free and "
                         "reload it (unload); auto decides from GPU memory and RAM")


def _pipeline_kwargs(a) -> dict:
    return {"dots_device": a.dots_device, "htr_device": a.htr_device, "offload": a.offload}


def cmd_check(a) -> int:
    import importlib
    import platform
    import shutil

    from .lines import kraken_paths

    print(f"arabic_doc_ocr {__version__} · Python {platform.python_version()} · {platform.platform()}")
    for mod in ("torch", "transformers", "peft", "qwen_vl_utils", "gradio", "docx", "pypdfium2", "fitz", "psutil"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:14s} {getattr(m, '__version__', getattr(m, 'VersionBind', 'ok'))}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:14s} MISSING ({type(e).__name__}: {e})")
    ok = True
    try:
        import torch

        n = torch.cuda.device_count()
        print(f"  CUDA {torch.version.cuda} · {n} GPU(s)")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"    cuda:{i} {p.name} · {p.total_memory / 2**30:.1f} GiB · compute {p.major}.{p.minor}"
                  f" ({'bf16' if p.major >= 8 else 'fp16'})")
        if n == 0:
            ok = False
            print("  ✘ no GPU — dots.mocr needs a CUDA GPU")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  ✘ torch: {e}")
    try:
        import transformers

        if transformers.__version__ != "4.56.1":
            print(f"  ⚠ transformers {transformers.__version__}: tested with 4.56.1 only (other versions changed "
                  "dots.mocr / Qwen2.5-VL generation)")
    except ImportError:
        ok = False
    exe, model = kraken_paths()
    print(f"  Kraken: {exe or 'MISSING'} · segmentation model: {model or 'MISSING'}")
    if not exe or not model:
        print("  ⚠ without Kraken lines are found by horizontal projection (worse on handwriting) — run install.sh")
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"  RAM {vm.total / 2**30:.1f} GiB (free {vm.available / 2**30:.1f})")
    except ImportError:
        pass
    du = shutil.disk_usage(Path.home())
    print(f"  disk (home) free {du.free / 2**30:.1f} GiB")
    try:
        from huggingface_hub import scan_cache_dir

        have = {r.repo_id: r.size_on_disk for r in scan_cache_dir().repos}
        for rid in (config.DOTS_MODEL, config.HTR_BASE, config.HTR_ADAPTERS):
            print(f"  model {rid}: " + (f"cached ({have[rid] / 2**30:.1f} GiB)" if rid in have else "not downloaded"))
    except Exception:  # noqa: BLE001
        pass
    print("OK" if ok else "NOT READY")
    return 0 if ok else 1


def cmd_download(a) -> int:
    import urllib.request

    from huggingface_hub import snapshot_download

    for rid, pat in ((config.DOTS_MODEL, None), (config.HTR_BASE, None),
                     (config.HTR_ADAPTERS, [f"{config.MAIN}/*", f"{config.CHECK}/*"])):
        t = time.time()
        p = snapshot_download(rid, allow_patterns=pat)
        print(f"{rid} → {p} ({time.time() - t:.0f} s)")
    if not config.KRAKEN_SEG_MODEL.exists():
        config.KRAKEN_SEG_MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.MUHARAF_SEG_URL, config.KRAKEN_SEG_MODEL)
    print(f"Kraken segmentation model → {config.KRAKEN_SEG_MODEL}")
    return 0


def cmd_page(a) -> int:
    from .export import write_all
    from .pipeline import Pipeline, load_pages, parse_page_spec

    types = [t.strip() for t in a.type.split(",")]
    if len(types) == 1:
        types = types * len(a.inputs)
    bad = [t for t in types if t not in (config.HANDWRITTEN, config.PRINTED)]
    if bad or len(types) != len(a.inputs):
        sys.exit(f"--type: handwritten or printed, or one per input ({len(a.inputs)}); got {a.type!r}")
    pipe = Pipeline(**_pipeline_kwargs(a))
    out = Path(a.out)
    summary = []
    for f, page_type in zip(a.inputs, types):
        path = Path(f)
        nums = parse_page_spec(a.pages) if path.suffix.lower() == ".pdf" else None
        pages = load_pages(path, nums)
        nums = (nums or list(range(1, len(pages) + 1)))[:len(pages)]
        t0 = time.time()
        results = pipe.process_document(pages, page_type,
                                        (lambda p, msg: print(f"  [{100 * p:3.0f}%] {msg}", file=sys.stderr))
                                        if not a.quiet else None)
        name = path.stem if len(a.inputs) == len(set(Path(x).stem for x in a.inputs)) else \
            f"{path.parent.name}_{path.stem}"
        w = write_all(out, name, results, nums)
        dt = time.time() - t0
        summary.append({"input": str(path), "type": page_type, "pages": len(pages), "seconds": round(dt, 1),
                        "per_page": [r.info for r in results]})
        print(f"{path}: {len(pages)} page(s), {dt:.0f} s → {w['files'][0]}")
    (out / "run_info.json").write_text(json.dumps({"loads": pipe.models.loads, "offload": pipe.models.offload,
                                                   "dots_device": pipe.models.dots_device,
                                                   "htr_device": pipe.models.htr_device, "documents": summary},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def cmd_line(a) -> int:
    from PIL import Image

    from .pipeline import Pipeline

    pipe = Pipeline(**_pipeline_kwargs(a))
    ims = [Image.open(p).convert("RGB") for p in a.images]
    res = pipe.read_lines(ims)
    for p, r in zip(a.images, res):
        marked = " ".join(f"[?{w}]" if s else w for w, s, _ in r["words"])
        print(f"{p}\n  {config.MAIN}: {marked}\n  {config.CHECK}: {r['check']}")
    if a.json:
        Path(a.json).write_text(json.dumps([{"image": p, "ft3": r["text"], "ft4": r["check"],
                                             "agree": round(r["agree"], 3),
                                             "words_differ": [w for w, s, _ in r["words"] if s]}
                                            for p, r in zip(a.images, res)], ensure_ascii=False, indent=1),
                                encoding="utf-8")
    return 0


def cmd_web(a) -> int:
    from .web import serve

    serve(a.host, a.port, a.share, **_pipeline_kwargs(a))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arabic_doc_ocr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="check GPUs, libraries, Kraken and model cache")
    sub.add_parser("download", help="download all models into the cache")
    p = sub.add_parser("page", help="read documents (images / PDF)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--type", default=config.HANDWRITTEN,
                   help="text on the pages: handwritten (ft3 main text), printed (dots main text); one value for all inputs or a comma list, one per input")
    p.add_argument("--pages", default="", help="PDF pages, e.g. 1-3,5 (default: all)")
    p.add_argument("--out", default="ocr_out")
    p.add_argument("--quiet", action="store_true")
    _device_args(p)
    p = sub.add_parser("line", help="read single text-line images with ft3 + ft4")
    p.add_argument("images", nargs="+")
    p.add_argument("--json", default=None)
    _device_args(p)
    p = sub.add_parser("web", help="start the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--share", action="store_true", help="public Gradio link (do not use with client documents)")
    _device_args(p)
    a = ap.parse_args(argv)
    return {"check": cmd_check, "download": cmd_download, "page": cmd_page, "line": cmd_line, "web": cmd_web}[a.cmd](a)
