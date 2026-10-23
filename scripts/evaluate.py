"""Score OCR output against reference texts (character / word accuracy), the way the sample pages were scored.

    python scripts/evaluate.py --results ocr_out --examples examples

For every examples/<page>/label.json ({"type": …, "text": …}) the matching result <page>*.json in --results is
scored: main text, dots, ft3 and ft4. Accuracy = 100 % minus CER after normalisation: NFKC, no harakat / tatweel,
alef / yeh unified, Arabic-Indic digits → 0–9, punctuation removed, dates compared as sets of their parts, Hebrew
removed (the labels do not transcribe the Hebrew notary block).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

TEXT_CATS = {"Text", "Title", "Section-header", "List-item", "Caption", "Footnote", "Page-header", "Page-footer"}
_AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_BIDI = re.compile(r"[​-‏‪-‮⁦-⁩﻿؜]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "0123456789" * 2 + ".,")
_ALEF = re.compile("[آأإٱ]")  # آ أ إ ٱ -> ا
_HEBREW = re.compile(r"[֐-׿]+")
_PUNCT = re.compile(r"[.,،؛:;!?؟()\[\]{}«»\"'“”•\-–—ـ*/\\]+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _BIDI.sub("", s).replace("ـ", "")
    s = _AR_DIACRITICS.sub("", s)
    s = _ALEF.sub("ا", s).replace("ى", "ي").translate(_DIGITS)
    return " ".join(s.split())


def clean(s: str) -> str:
    s = re.sub(r"\[\?([^\]]*)\]", r"\1", s or "")
    s = re.sub(r"\[[^\]]*\]|\.\.\.|…", " ", s)
    s = re.sub(r"([0-9٠-٩]+)/([0-9٠-٩]+)/([0-9٠-٩]+)",
               lambda m: " ".join(sorted(m.groups(), key=lambda x: (len(x), x))), s)
    return " ".join(_PUNCT.sub(" ", normalize(_HEBREW.sub(" ", s))).split())


def score(ref: str, hyp: str) -> dict:
    r, h = clean(ref), clean(hyp)
    ce, we = Levenshtein.distance(r, h), Levenshtein.distance(r.split(), h.split())
    cr, ch = Counter(r.split()), Counter(h.split())
    common = sum((cr & ch).values())
    return {"chars": max(0.0, 1 - ce / max(1, len(r))), "words": max(0.0, 1 - we / max(1, len(r.split()))),
            "word_recall": common / max(1, sum(cr.values()))}


def readings(js: dict) -> dict[str, str]:
    """Whole-document text of each reader from the result JSON (blocks in dots reading order)."""
    out = {"main": [], "dots": [], "ft3": [], "ft4": []}
    for p in js["pages"]:
        for b in p["blocks"]:
            if b["category"] not in TEXT_CATS:
                continue
            out["main"].append(b.get("text") or "")
            out["dots"].append(b.get("dots") or "")
            out["ft3"].append("\n".join(L["ft3"] for L in b["lines"]))
            out["ft4"].append("\n".join(L["ft4"] for L in b["lines"]))
    return {k: "\n".join(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--examples", default="examples")
    ap.add_argument("--json", default=None, help="also write the scores to this file")
    ap.add_argument("--ref", default=None, help="eval.json of an earlier run (e.g. results/two_gpus/eval.json): "
                                                "also print the change per page")
    a = ap.parse_args()
    rows, out = [], {}
    for lab in sorted(Path(a.examples).glob("*/label.json")):
        page = lab.parent.name
        cand = [c for c in sorted(Path(a.results).glob(f"{page}*.json")) if not c.name.startswith("run_info")]
        if not cand:
            print(f"{page}: no result in {a.results}", file=sys.stderr)
            continue
        ref = json.loads(lab.read_text(encoding="utf-8"))
        rd = readings(json.loads(cand[0].read_text(encoding="utf-8")))
        sc = {k: score(ref["text"], v) for k, v in rd.items()}
        out[page] = {"type": ref.get("type"), **sc}
        rows.append(f"| {page} | {ref.get('type')} | " + " | ".join(
            f"{100 * sc[k]['chars']:.1f} / {100 * sc[k]['words']:.1f}" for k in rd) + " |")
    print("| page | type | main | dots | ft3 | ft4 |\n|---|---|---:|---:|---:|---:|")
    print("\n".join(rows))
    print("\n(correct characters % / correct words %)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    if a.ref:
        old = json.loads(Path(a.ref).read_text(encoding="utf-8"))
        print(f"\nChange vs {a.ref} (correct characters, points):\n")
        print("| page | main | dots | ft3 | ft4 |\n|---|---:|---:|---:|---:|")
        for page, sc in out.items():
            if page in old:
                print(f"| {page} | " + " | ".join(f"{100 * (sc[k]['chars'] - old[page][k]['chars']):+.2f}"
                                                for k in ("main", "dots", "ft3", "ft4")) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
