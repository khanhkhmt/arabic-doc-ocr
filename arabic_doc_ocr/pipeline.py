"""Page pipeline:

    page image ─► prepare (enlarge small photos) ─► dots.mocr layout on the page enlarged to 2400 px
               ─► Kraken line segmentation on the whole page, lines assigned to dots text blocks
               ─► ft3 and ft4 read every line ─► main text per block:
                    handwritten page → ft3 (ft4 = cross-check)   printed page → dots text (ft3 = cross-check)
                    mixed page → per block (experimental)        Hebrew block → dots text (line models read Arabic only)

    from arabic_doc_ocr.pipeline import Pipeline, load_pages
    p = Pipeline()                                   # GPUs picked automatically
    res = p.process_document(load_pages("scan.pdf"), page_type="printed")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from . import config
from .lines import assign_to_blocks, crop_poly, kraken_page, projection_lines
from .text import agreement, fix_arabic, is_hebrew

PAGE_TYPES = (config.HANDWRITTEN, config.PRINTED, config.MIXED)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


# --- input ----------------------------------------------------------------------------------------------------------

def load_pages(path: str | Path, pages: list[int] | None = None) -> list[Image.Image]:
    """PDF (rendered at 200 dpi) or image → page images. pages: 1-based page numbers (None = all)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            nums = pages or list(range(1, len(pdf) + 1))
            return [pdf[n - 1].render(scale=config.PDF_DPI / 72).to_pil().convert("RGB") for n in nums
                    if 1 <= n <= len(pdf)]
        finally:
            pdf.close()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"unsupported file type {ext} (PDF or image)")
    return [ImageOps.exif_transpose(Image.open(path)).convert("RGB")]


def parse_page_spec(spec: str | None) -> list[int] | None:
    """'1-3,5' → [1, 2, 3, 5]; empty → None (all pages)."""
    if not spec or not spec.strip():
        return None
    out = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def prepare(page: Image.Image) -> Image.Image:
    """Small photos (long side < 1000 px) are enlarged to 1500 px (at most ×3). All boxes refer to this image."""
    img = page.convert("RGB")
    long_side = max(img.size)
    if long_side >= config.UPSCALE_BELOW:
        return img
    s = min(config.UPSCALE_MAX, config.UPSCALE_TO / long_side)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


# --- models and GPU memory ------------------------------------------------------------------------------------------

def _is_oom(e: Exception) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def default_devices() -> tuple[str, str]:
    """dots on the first GPU, the line reader on the second if there is one."""
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA GPU found — dots.mocr needs a GPU (see README, system requirements)")
    return "cuda:0", ("cuda:1" if n > 1 else "cuda:0")


class Models:
    """Loads dots.mocr and the line reader lazily. When both share one GPU that cannot hold both
    (offload = "cpu" or "unload"), only the model in use stays on the GPU: the other is moved to CPU RAM ("cpu",
    a few seconds) or freed and reloaded from the disk cache ("unload", ~1 min, needs little RAM)."""

    def __init__(self, dots_device: str | None = None, htr_device: str | None = None, offload: str = "auto"):
        d0, d1 = default_devices() if not (dots_device and htr_device) else (dots_device, htr_device)
        self.dots_device, self.htr_device = dots_device or d0, htr_device or d1
        self.offload = self._auto_offload() if offload == "auto" else offload
        if self.offload not in ("none", "cpu", "unload"):
            raise ValueError("offload must be auto, none, cpu or unload")
        self._dots = self._htr = None
        self.loads = {"dots": 0, "htr": 0}

    def _auto_offload(self) -> str:
        import torch

        if self.dots_device != self.htr_device:
            return "none"
        gpu_gb = torch.cuda.get_device_properties(self.dots_device).total_memory / 2**30
        if gpu_gb >= config.ONE_GPU_BOTH_MODELS_GB:
            return "none"
        try:
            import psutil

            ram_gb = psutil.virtual_memory().total / 2**30
        except ImportError:
            ram_gb = 0
        return "cpu" if ram_gb >= config.OFFLOAD_CPU_MIN_RAM_GB else "unload"

    def _make_room(self, keep: str) -> None:
        if self.offload == "none":
            return
        other = "_htr" if keep == "dots" else "_dots"
        m = getattr(self, other)
        if m is None:
            return
        if self.offload == "cpu":
            m.to("cpu")
        else:
            m.close()
            setattr(self, other, None)
        free_gpu()

    def dots(self):
        from .dots_model import DotsLayout

        self._make_room("dots")
        if self._dots is None:
            self._dots = DotsLayout(self.dots_device).load()
            self.loads["dots"] += 1
        elif self.offload == "cpu":
            self._dots.to(self.dots_device)
        return self._dots

    def htr(self):
        from .htr_model import HTRReader

        self._make_room("htr")
        if self._htr is None:
            self._htr = HTRReader(self.htr_device).load()
            self.loads["htr"] += 1
        elif self.offload == "cpu":
            self._htr.to(self.htr_device)
        return self._htr


def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _GpuPeak:
    """Peak GPU memory allocated by PyTorch during a stage (GB per device)."""

    def __init__(self, devices):
        self.devices = sorted(set(d for d in devices if str(d).startswith("cuda")))
        if self.devices:
            import torch

            self.torch = torch

    def __enter__(self):
        for d in self.devices:
            self.torch.cuda.reset_peak_memory_stats(d)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self.t0, 1)
        self.peak_gb = {d: round(self.torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in self.devices}
        return False


# --- results --------------------------------------------------------------------------------------------------------

@dataclass
class Line:
    block: int
    box: tuple[int, int, int, int]
    image: Image.Image
    poly: list | None = None


@dataclass
class PageResult:
    image: Image.Image                  # prepared page; all boxes refer to it
    blocks: list[dict]                  # dots blocks: category, bbox, text (+ hebrew / fallback flags)
    lines: list[Line]
    readings: dict[str, list[str]] = field(default_factory=dict)   # {"ft3": [...], "ft4": [...]} per line
    sources: dict[int, tuple[str, float | None]] = field(default_factory=dict)  # block → (main reader, dots≈ft3)
    texts: dict[str, dict[int, str]] = field(default_factory=dict)  # main / check / ft3 / ft4 / dots → per block
    info: dict = field(default_factory=dict)

    def page_text(self, kind: str = "main") -> str:
        return "\n\n".join(t for _, t in sorted(self.texts.get(kind, {}).items()) if t.strip())


# --- pipeline -------------------------------------------------------------------------------------------------------

class Pipeline:
    def __init__(self, dots_device: str | None = None, htr_device: str | None = None, offload: str = "auto"):
        self.models = Models(dots_device, htr_device, offload)

    # stage 1: layout + lines
    def layout(self, page: Image.Image) -> PageResult:
        img = prepare(page)
        f = min(config.DOTS_MAX_FACTOR, config.DOTS_LONG_SIDE / max(img.size)) if max(img.size) < config.DOTS_LONG_SIDE \
            else 1.0
        notes = []
        dots = self.models.dots()

        def run(f):
            im = img if f == 1 else img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)
            return dots.parse(im)

        with _GpuPeak([self.models.dots_device]) as g:
            try:
                r = run(f)
            except Exception as e:  # noqa: BLE001
                if f == 1 or not _is_oom(e):
                    raise
                free_gpu()
                notes.append(f"dots ran out of GPU memory at ×{f:.2f} → read again without enlargement")
                f = 1.0
                r = run(f)
        blocks = []
        for b in r["blocks"]:
            bb = b.get("bbox")
            blk = {**b, "bbox": [v / f for v in bb] if bb and len(bb) == 4 else bb, "text": fix_arabic(b.get("text") or "")}
            if blk.get("category") in config.TEXT_CATS and is_hebrew(blk["text"]):
                blk["hebrew"] = True
            blocks.append(blk)
        if not any(b.get("category") in config.TEXT_CATS and len(b.get("bbox") or []) == 4 for b in blocks):
            why = ("dots found no text block" + (" (hit the token limit)" if r.get("hit_max_tokens") else
                                                 " (broken layout JSON)" if r.get("layout") == "filtered" else ""))
            notes.append(why + " → FALLBACK: the whole page is one block")
            blocks = [{"category": "Text", "bbox": [0, 0, img.width, img.height], "text": "", "fallback": True}]
        elif r.get("hit_max_tokens"):
            notes.append("dots hit the token limit — the last block may be incomplete")

        t1 = time.time()
        page_lines = kraken_page(img)
        boxes = {bi: tuple(int(round(v)) for v in b["bbox"]) for bi, b in enumerate(blocks)
                 if b.get("category") in config.TEXT_CATS and len(b.get("bbox") or []) == 4}
        by_block = assign_to_blocks(page_lines, boxes) if page_lines else {}
        lines, methods = [], set()
        for bi, b in enumerate(blocks):
            bb = b.get("bbox")
            if b.get("category") not in config.TEXT_CATS or not bb or len(bb) != 4:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in bb]
            if x1 - x0 < 8 or y1 - y0 < 8 or b.get("hebrew"):
                continue
            if by_block.get(bi):
                methods.add("kraken")
                lines += [Line(bi, L["box"], crop_poly(img, L), L.get("poly")) for L in by_block[bi]]
                continue
            methods.add("projection")
            for lx0, ly0, lx1, ly1 in projection_lines(img.crop((x0, y0, x1, y1))):
                box = (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1)
                lines.append(Line(bi, box, img.crop(box)))
        info = {"dots_scale": round(f, 3), "dots_seconds": g.seconds, "dots_gpu_peak_gb": g.peak_gb,
                "dots_tokens": r.get("new_tokens"), "dots_hit_max_tokens": r.get("hit_max_tokens"),
                "dots_stopped_loop": r.get("stopped_loop"), "dots_layout": r.get("layout"),
                "segmentation": sorted(methods), "segmentation_seconds": round(time.time() - t1, 1),
                "hebrew_blocks": sum(bool(b.get("hebrew")) for b in blocks), "notes": notes}
        return PageResult(img, blocks, lines, info=info)

    # stage 2: read lines
    def read(self, res: PageResult, progress=None) -> PageResult:
        htr = self.models.htr()
        crops = [L.image for L in res.lines]
        with _GpuPeak([self.models.htr_device]) as g:
            res.readings = {}
            for k, name in enumerate((config.MAIN, config.CHECK)):
                cb = (lambda i, n, k=k, name=name: progress((k * n + i) / (2 * max(1, n)), f"{name} line {i + 1}/{n}")) \
                    if progress else None
                res.readings[name] = htr.read(crops, name, cb)
        res.info.update(htr_seconds=g.seconds, htr_gpu_peak_gb=g.peak_gb, lines=len(crops))
        return res

    # stage 3: main text per block
    @staticmethod
    def assemble(res: PageResult, page_type: str) -> PageResult:
        if page_type not in PAGE_TYPES:
            raise ValueError(f"page_type must be one of {PAGE_TYPES}")
        ft3, ft4 = res.readings[config.MAIN], res.readings[config.CHECK]
        res.sources, res.texts = {}, {k: {} for k in ("main", "check", "ft3", "ft4", "dots")}
        for bi, b in enumerate(res.blocks):
            if b.get("category") not in config.TEXT_CATS:
                continue
            idx = [n for n, L in enumerate(res.lines) if L.block == bi]
            dots_text = b.get("text") or ""
            res.texts["ft3"][bi] = "\n".join(ft3[n] for n in idx)
            res.texts["ft4"][bi] = "\n".join(ft4[n] for n in idx)
            res.texts["dots"][bi] = dots_text
            if b.get("hebrew"):
                src, agree = "hebrew", None
            else:
                joined = " ".join(ft3[n] for n in idx)
                agree = agreement(dots_text, joined) if dots_text.strip() and joined.strip() else None
                if not dots_text.strip():
                    src = config.MAIN
                elif page_type == config.PRINTED or not joined.strip():
                    src = "dots"
                elif page_type == config.MIXED:
                    src = "dots" if agree is not None and agree >= config.MIXED_PRINTED_AGREEMENT else config.MAIN
                else:
                    src = config.MAIN
            res.sources[bi] = (src, agree)
            if src == "hebrew":
                res.texts["main"][bi] = res.texts["check"][bi] = dots_text  # kept as dots read it, not checked
            elif src == "dots":
                res.texts["main"][bi], res.texts["check"][bi] = dots_text, res.texts["ft3"][bi]
            else:
                res.texts["main"][bi], res.texts["check"][bi] = res.texts["ft3"][bi], res.texts["ft4"][bi]
        res.info["page_type"] = page_type
        return res

    def process(self, page: Image.Image, page_type: str = config.HANDWRITTEN, progress=None) -> PageResult:
        return self.process_document([page], page_type, progress)[0]

    def process_document(self, pages: list[Image.Image], page_type: str = config.HANDWRITTEN,
                         progress=None) -> list[PageResult]:
        """Layout of ALL pages first, then line reading of all pages — on a single GPU the models swap only once."""
        def say(p, msg):
            if progress:
                progress(p, msg)

        n = len(pages)
        results = []
        for i, page in enumerate(pages):
            say(0.4 * i / n, f"page {i + 1}/{n}: dots layout")
            results.append(self.layout(page))
        for i, res in enumerate(results):
            say(0.4 + 0.6 * i / n, f"page {i + 1}/{n}: reading {len(res.lines)} lines")
            self.read(res, (lambda p, msg, i=i: say(0.4 + 0.6 * (i + p) / n, f"page {i + 1}/{n}: {msg}"))
                      if progress else None)
            self.assemble(res, page_type)
        return results

    def read_lines(self, images: list[Image.Image]) -> list[dict]:
        return self.models.htr().read_check([im.convert("RGB") for im in images])
