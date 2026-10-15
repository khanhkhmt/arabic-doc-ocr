"""CPU tests with fake models: text helpers, line segmentation, block → reader choice, all output files.

    python -m pytest -q tests
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from arabic_doc_ocr import config
from arabic_doc_ocr.export import write_all
from arabic_doc_ocr.lines import assign_to_blocks, crop_poly, projection_lines
from arabic_doc_ocr.pipeline import Pipeline, load_pages, parse_page_spec, prepare
from arabic_doc_ocr.text import agreement, clean_repeated_substrings, compare_words, fix_arabic, is_hebrew


# --- text -----------------------------------------------------------------------------------------------------------

def test_fix_arabic():
    assert fix_arabic("کتاب یوم ۱۹۷۲") == "كتاب يوم ١٩٧٢"
    assert fix_arabic("و نحن") == "ونحن"
    assert fix_arabic("ذهب و عاد") == "ذهب وعاد"
    assert fix_arabic("نحو بيت") == "نحو بيت"  # و at the end of a word stays


def test_hebrew():
    assert is_hebrew("אישור חתימה 16/5/79")
    assert not is_hebrew("عقد بيع אב")
    assert not is_hebrew("")


def test_compare_words():
    w = compare_words("الفريق الاول باع الارض", "الفريق الأول باع الارص")
    assert [s for _, s, _ in w] == [False, False, False, True]
    assert w[3][2] == "الارص"
    assert agreement("a b c d", "a b c d") == 1.0
    assert agreement("", "x") == 1.0


def test_clean_repeated():
    s = "بداية " + "ab" * 150
    assert len(clean_repeated_substrings(s)) < len(s)
    assert clean_repeated_substrings("short") == "short"


# --- lines ----------------------------------------------------------------------------------------------------------

def _fake_page(n_lines=5, w=1200, h=900) -> Image.Image:
    im = Image.new("RGB", (w, h), (235, 225, 200))
    d = ImageDraw.Draw(im)
    for i in range(n_lines):
        y = 180 + i * 100
        for x in range(100, 1000, 40):
            d.rectangle([x, y, x + 20, y + 30], fill=(30, 30, 120))
    return im


def test_projection_lines():
    boxes = projection_lines(_fake_page(5))
    assert len(boxes) == 5
    ys = [b[1] for b in boxes]
    assert ys == sorted(ys)


def test_assign_and_crop():
    lines = [{"box": (10, 10, 200, 40), "poly": [(10, 10), (200, 10), (200, 40), (10, 40)],
              "baseline": [(10, 35), (200, 35)]},
             {"box": (10, 300, 200, 330), "poly": [], "baseline": []}]
    out = assign_to_blocks(lines, {0: (0, 0, 300, 100), 1: (0, 250, 300, 400), 2: (0, 0, 1000, 1000)})
    assert len(out[0]) == 1 and len(out[1]) == 1 and 2 not in out
    crop = crop_poly(_fake_page(), lines[0])
    assert crop.size == (198, 38)


def test_prepare_and_pages(tmp_path):
    small = Image.new("RGB", (550, 708), "white")
    assert prepare(small).size == (round(550 * 1500 / 708), 1500)
    big = Image.new("RGB", (1032, 1600), "white")
    assert prepare(big).size == (1032, 1600)
    p = tmp_path / "a.jpg"
    big.save(p)
    assert len(load_pages(p)) == 1
    assert parse_page_spec("1-3,5") == [1, 2, 3, 5] and parse_page_spec("") is None


# --- pipeline with fake models --------------------------------------------------------------------------------------

class FakeDots:
    def __init__(self, blocks):
        self.blocks, self.seen = blocks, []

    def parse(self, im):
        self.seen.append(im.size)
        f = im.size[0] / 1200  # boxes come back in the enlarged image's coordinates
        return {"blocks": [{**b, "bbox": [v * f for v in b["bbox"]]} for b in self.blocks], "new_tokens": 10,
                "hit_max_tokens": False, "stopped_loop": False, "layout": "ok"}


class FakeHTR:
    def read(self, crops, adapter=None, progress=None):
        return [("كلمة اولى ثانية" if adapter == "ft3" else "كلمة اولى ثالثة") for _ in crops]

    def read_check(self, ims):
        return [{"text": "a", "check": "a", "words": [("a", False, "")], "agree": 1.0} for _ in ims]


class FakeModels:
    dots_device = htr_device = "cpu"
    offload, loads = "none", {"dots": 1, "htr": 1}

    def __init__(self, blocks):
        self._d, self._h = FakeDots(blocks), FakeHTR()

    def dots(self):
        return self._d

    def htr(self):
        return self._h


BLOCKS = [{"category": "Title", "bbox": [50, 40, 1150, 140], "text": "عقد بيع"},
          {"category": "Text", "bbox": [50, 150, 1150, 680], "text": "كلمة اولى ثانية"},
          {"category": "Picture", "bbox": [800, 700, 1150, 890], "text": ""},
          {"category": "Text", "bbox": [50, 700, 750, 890], "text": "אישור חתימה מס׳ 12"}]


def _pipe(blocks=BLOCKS) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.models = FakeModels(blocks)
    return p


@pytest.mark.parametrize("page_type", [config.HANDWRITTEN, config.PRINTED, config.MIXED])
def test_pipeline_and_outputs(tmp_path, page_type, monkeypatch):
    monkeypatch.setattr("arabic_doc_ocr.pipeline.kraken_page", lambda img: None)  # projection only
    pipe = _pipe()
    res = pipe.process_document([_fake_page()], page_type)
    r = res[0]
    assert pipe.models._d.seen == [(2400, 1800)]  # dots sees the page enlarged to 2400 px
    assert r.blocks[1]["bbox"] == pytest.approx([50, 150, 1150, 680])  # boxes back on the prepared page
    assert r.blocks[3].get("hebrew") and r.sources[3][0] == "hebrew"
    assert 2 not in r.sources  # picture
    assert r.lines and all(L.block != 3 for L in r.lines)  # Hebrew block not given to the line readers
    src = r.sources[1][0]
    assert src == ("dots" if page_type in (config.PRINTED, config.MIXED) else "ft3")
    w = write_all(tmp_path, "doc", res)
    names = {p.name for p in w["files"]}
    assert {"doc_all.zip", "doc.docx", "doc_main.txt", "doc_ft3.txt", "doc_ft4.txt", "doc_review.html", "doc.json",
            "doc_p1_boxes.jpg"} <= names
    js = json.loads((tmp_path / "doc.json").read_text(encoding="utf-8"))
    assert js["pages"][0]["blocks"][3]["reader"] == "hebrew"
    assert "אישור" in (tmp_path / "doc_main.txt").read_text(encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "doc_all.zip") as z:
        assert len(z.namelist()) == len(w["files"]) - 1
    from docx import Document

    doc = Document(tmp_path / "doc.docx")
    assert any("كلمة" in p.text for p in doc.paragraphs)


def test_fallback_when_no_text_block(monkeypatch):
    monkeypatch.setattr("arabic_doc_ocr.pipeline.kraken_page", lambda img: None)
    pipe = _pipe([{"category": "Picture", "bbox": [0, 0, 1200, 900], "text": ""}])
    r = pipe.process(_fake_page(), config.HANDWRITTEN)
    assert r.blocks[0].get("fallback") and len(r.lines) == 5
    assert r.info["notes"] and "FALLBACK" in r.info["notes"][0]


def test_oom_retry(monkeypatch):
    monkeypatch.setattr("arabic_doc_ocr.pipeline.kraken_page", lambda img: None)
    pipe = _pipe()
    real = pipe.models._d.parse

    def parse(im):
        if im.size[0] > 1200:
            raise RuntimeError("CUDA out of memory")
        return real(im)

    pipe.models._d.parse = parse
    r = pipe.layout(_fake_page())
    assert r.info["dots_scale"] == 1.0 and "out of GPU memory" in r.info["notes"][0]


def test_vendored_dots_postprocess():
    """The vendored dots utils turn a layout JSON into blocks scaled to the original image."""
    from arabic_doc_ocr.dots_model import _dots_utils, layout_loop

    _, _, layout_utils = _dots_utils()
    origin = Image.new("RGB", (1000, 1400))
    resp = json.dumps([{"bbox": [10, 20, 500, 80], "category": "Text", "text": "نص"}])
    cells, filtered = layout_utils.post_process_output(resp, config.DOTS_PROMPT, origin, origin)
    assert not filtered and cells[0]["text"] == "نص"
    loop = "".join('{"bbox": [1, 2, 3, 4], "category": "Text", "text": "abcdefg"}, ' for _ in range(30))
    assert layout_loop(loop) and not layout_loop(loop[:200])


def test_projection_on_real_scan_shapes():
    """Ruled paper: long horizontal rules must not become lines."""
    im = _fake_page(3)
    d = ImageDraw.Draw(im)
    for y in (160, 260, 360, 460):
        d.line([0, y, 1199, y], fill=(30, 30, 120), width=2)
    assert len(projection_lines(im)) == 3
    assert np.asarray(im).shape == (900, 1200, 3)
