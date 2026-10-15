"""Line segmentation: the line models read ONE text line at a time.

- Kraken (baseline segmentation, `blla`) with the Muharaf segmentation model, on the whole page: follows curved /
  slanted handwritten lines and ignores ruled-paper lines. Kraken runs as a separate program (its own virtualenv,
  see install.sh) because it pins its own torch version.
- Horizontal projection: fallback inside a layout block when Kraken is missing or found no line there
  (works for single-column text only).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import config

# --- Kraken ---------------------------------------------------------------------------------------------------------


def kraken_paths() -> tuple[str | None, str | None]:
    """(kraken executable, segmentation model) or None for what is missing."""
    exe = os.environ.get("KRAKEN_BIN") or str(config.KRAKEN_VENV / "bin" / "kraken")
    if not Path(exe).exists():
        exe = shutil.which("kraken")
    model = os.environ.get("KRAKEN_SEG_MODEL") or str(config.KRAKEN_SEG_MODEL)
    return exe, (model if Path(model).exists() else None)


def kraken_page(img: Image.Image) -> list[dict] | None:
    """→ [{"box": (x0, y0, x1, y1), "poly": [(x, y), …], "baseline": [(x, y), …]}] top to bottom;
    None when Kraken / the model is missing or Kraken failed."""
    exe, model = kraken_paths()
    if not exe or not model:
        return None
    with tempfile.TemporaryDirectory() as d:
        p, out = Path(d) / "page.png", Path(d) / "seg.json"
        img.convert("RGB").save(p)
        try:
            subprocess.run([exe, "-i", str(p), str(out), "segment", "-bl", "-i", model], check=True,
                           capture_output=True, timeout=600)
            seg = json.loads(out.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[lines] Kraken failed ({type(e).__name__}: {str(e)[:200]}) → projection", file=sys.stderr)
            return None
    lines = []
    for ln in seg.get("lines", []):
        poly = [(float(x), float(y)) for x, y in (ln.get("boundary") or [])]
        base = [(float(x), float(y)) for x, y in (ln.get("baseline") or [])]
        pts = poly or base
        if len(pts) < 2:
            continue
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        lines.append({"box": (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1), "poly": poly,
                      "baseline": base})
    lines.sort(key=lambda L: (L["box"][1] + L["box"][3]) / 2)
    return lines


def crop_poly(img: Image.Image, line: dict, pad: int = 4) -> Image.Image:
    """Crop a line along its Kraken polygon: outside the polygon is painted paper colour, so less ink from the lines
    above / below gets into the line image."""
    W, H = img.size
    x0, y0, x1, y1 = line["box"]
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crop = img.convert("RGB").crop((x0, y0, x1, y1))
    if len(line.get("poly") or []) < 3:
        return crop
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in line["poly"]], fill=255)
    bg = int(np.percentile(np.asarray(crop.convert("L")), 90))  # paper colour
    return Image.composite(crop, Image.new("RGB", crop.size, (bg, bg, bg)), mask)


def assign_to_blocks(lines: list[dict], boxes: dict[int, tuple[int, int, int, int]], margin: float = 0.02) -> dict:
    """Assign every line to the layout block containing the centre of its baseline (or box); blocks slightly enlarged;
    nested blocks → the smallest. → {block index: [line, …]}."""
    out: dict[int, list] = {}
    for L in lines:
        pts = L["baseline"] or [((L["box"][0] + L["box"][2]) / 2, (L["box"][1] + L["box"][3]) / 2)]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        best, best_area = None, None
        for bi, (x0, y0, x1, y1) in boxes.items():
            mx, my = margin * (x1 - x0), margin * (y1 - y0) + 4
            if x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best_area:
                    best, best_area = bi, area
        if best is not None:
            out.setdefault(best, []).append(L)
    return out


# --- horizontal projection ------------------------------------------------------------------------------------------


def _ink(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    bg, dark = np.percentile(g, 75), np.percentile(g, 0.5)
    if bg - dark < 30:
        return np.zeros(g.shape, bool)
    m = g < (bg + dark) / 2
    m[m.mean(axis=1) > 0.5, :] = False  # long horizontal rules (ruled paper, underlines) are not text
    return m


def _smooth(v: np.ndarray, k: float) -> np.ndarray:
    k = max(1, int(k))
    return np.convolve(v, np.ones(k) / k, mode="same")


def _bands(mask: np.ndarray) -> list[list[int]]:
    bands, start = [], None
    H = len(mask)
    for y in range(H + 1):
        if y < H and mask[y] and start is None:
            start = y
        elif (y == H or not mask[y]) and start is not None:
            bands.append([start, y])
            start = None
    return bands


def projection_rows(img: Image.Image, min_h: int = 6) -> list[tuple[int, int]]:
    """→ [(y0, y1)] text lines of a block image, top to bottom: ink profile per row, smoothed; unusually tall bands
    (touching lines) are cut at the lowest valley; very thin bands (dots, diacritics) merge into the nearest line."""
    m = _ink(img)
    H = m.shape[0]
    if not m.any():
        return []
    prof = m.sum(axis=1).astype(np.float32)
    rows = np.nonzero(prof > 0)[0]
    bands = _bands(prof > 0.05 * prof.max())
    if not bands:
        return [(int(rows.min()), int(rows.max()) + 1)]
    med = float(np.median([b - a for a, b in bands])) or 10.0
    sm = _smooth(prof, max(3, med / 4))
    bands = _bands(sm > 0.12 * sm.max())
    if not bands:
        return []
    med = float(np.median([b - a for a, b in bands]))
    out = []
    for a, b in bands:
        segs = [(a, b)]
        while True:
            a2, b2 = segs[-1]
            if b2 - a2 < 1.7 * med:
                break
            lo, hi = a2 + int(0.6 * med), b2 - int(0.6 * med)
            if hi <= lo:
                break
            cut = lo + int(np.argmin(sm[lo:hi]))
            segs[-1:] = [(a2, cut), (cut, b2)]
        out += segs
    merged: list[list[int]] = []
    for a, b in out:
        if b - a < max(min_h, 0.35 * med) and merged and a - merged[-1][1] < 0.6 * med:
            merged[-1][1] = max(merged[-1][1], b)
            continue
        merged.append([a, b])
    if len(merged) > 1 and merged[0][1] - merged[0][0] < 0.35 * med:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    pad = int(round(0.2 * med))
    return [(max(0, a - pad), min(H, b + pad)) for a, b in merged if b - a >= min_h]


def projection_lines(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """→ [(x0, y0, x1, y1)] lines inside a block image; horizontally only the inked part (+ small margin)."""
    W = img.size[0]
    rows = projection_rows(img)
    if not rows:
        return []
    m = _ink(img)
    out = []
    for y0, y1 in rows:
        cols = np.nonzero(m[y0:y1].any(axis=0))[0]
        if len(cols) == 0:
            continue
        pad = max(4, int(0.02 * W))
        out.append((max(0, int(cols.min()) - pad), y0, min(W, int(cols.max()) + 1 + pad), y1))
    return out
