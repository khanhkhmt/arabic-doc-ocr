"""Check the LLM correction step on the 4 sample pages against the run measured on 2026-10-19 (ocr-bench llm_sua).

    .venv/bin/python scripts/check_llm.py            # after bash install_llm.sh; GPUs must be free (no web running)
    .venv/bin/python scripts/check_llm.py --ocr results/pad50 --name llm_pad50   # LLM on another OCR run (no
                                                     # measured output to compare with: only before / after scores)

Input: the OCR results stored in results/two_gpus/ (the same files the measured run corrected). The LLM corrects them
in its own venv (arabic_doc_ocr.llm_fix, exactly as the --llm option does), then:
  1. every corrected block is compared with tests/llm_expected.json (the measured output) → identical blocks;
  2. every page is scored before / after with scripts/evaluate.py against examples/*/label.json.
Writes results/llm_check.md + .json. Last line: "✔ LLM CHECK …" or "✘ LLM CHECK …".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arabic_doc_ocr import config  # noqa: E402
from arabic_doc_ocr.llm import llm_python  # noqa: E402


def _evaluate():
    spec = importlib.util.spec_from_file_location("adoc_evaluate", ROOT / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ocr", default="results/two_gpus", help="folder with page_N_page.json (OCR results)")
    ap.add_argument("--name", default="llm_check", help="writes results/<name>.md + .json")
    a = ap.parse_args()
    ocr_dir = (ROOT / a.ocr).resolve()
    measured = ocr_dir == (ROOT / "results" / "two_gpus").resolve()  # the OCR the measured run corrected
    exp = json.loads((ROOT / "tests" / "llm_expected.json").read_text(encoding="utf-8"))
    py = llm_python()
    if py is None:
        print(f"✘ LLM CHECK: LLM venv not installed ({config.LLM_VENV}) — bash install_llm.sh")
        return 1
    ocr, units = {}, []
    for page in sorted(exp["pages"]):
        js = json.loads((ocr_dir / f"{page}_page.json").read_text(encoding="utf-8"))
        ocr[page] = js
        for pi, p in enumerate(js["pages"]):
            for bi, b in enumerate(p["blocks"]):
                if b.get("category") not in config.TEXT_CATS or b.get("reader") not in ("dots", config.MAIN):
                    continue
                main = b.get("text") or ""
                if not main.strip():
                    continue
                other = config.MAIN if b["reader"] == "dots" else config.CHECK
                units.append({"key": f"{page}|{pi}:{bi}", "main": main,
                              "check": "\n".join(L[other] for L in b.get("lines", []))})
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="llm_check_") as d:
        fin, fout = Path(d) / "in.json", Path(d) / "out.json"
        fin.write_text(json.dumps({"units": units}, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        r = subprocess.run([str(py), "-m", "arabic_doc_ocr.llm_fix", "--in", str(fin), "--out", str(fout)],
                           env=env, cwd=str(ROOT))
        if r.returncode != 0 or not fout.exists():
            print(f"✘ LLM CHECK: the LLM process failed (exit {r.returncode}) — see the output above")
            return 1
        out = json.loads(fout.read_text(encoding="utf-8"))
    ev = _evaluate()
    rows, same_all, n_all, diffs = [], 0, 0, []
    for page in sorted(exp["pages"]):
        got = {k.split("|", 1)[1]: v for k, v in out["texts"].items() if k.startswith(page + "|")}
        want = exp["pages"][page]["texts"] if measured else {}
        same = sum(got.get(k) == v for k, v in want.items())
        same_all, n_all = same_all + same, n_all + len(want)
        diffs += [f"{page} block {k}: measured «{v[:80]}» · now «{got.get(k, '')[:80]}»"
                  for k, v in want.items() if got.get(k) != v][:3]
        label = json.loads((ROOT / "examples" / page / "label.json").read_text(encoding="utf-8"))["text"]

        def score(texts, js=ocr[page]):
            js = json.loads(json.dumps(js))
            for k, t in texts.items():
                pi, bi = map(int, k.split(":"))
                js["pages"][pi]["blocks"][bi]["text"] = t
            s = ev.score(label, ev.readings(js)["main"])
            return 100 * s["chars"], 100 * s["words"]

        b, n, m = score({}), score(got), (score(want) if measured else None)
        changed = sum(sum(1 for c in v if not c.get("rejected")) for k, v in out["changes"].items()
                      if k.startswith(page + "|"))
        rows.append((page, ocr[page]["pages"][0]["info"].get("page_type"), b, n, m, same, len(want), changed,
                     exp["pages"][page]["n_changes"] if measured else None))
    L = ["# LLM correction — check on the 4 sample pages" + ("" if measured else f" (OCR: {a.ocr})"), "",
         f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · {out['model']}@{out['revision'][:10]} · "
         f"{out['method']} · {out['dtype']} · load {out['load_seconds']:.0f} s · total {time.time() - t0:.0f} s", "",
         "Correct characters % / correct words % (scripts/evaluate.py, main text).", "",
         "| page | type | OCR only | + LLM now | + LLM measured 2026-10-19 | blocks identical to measured | "
         "words changed now / measured |", "|---|---|---:|---:|---:|---:|---:|"]
    for page, typ, b, n, m, same, tot, ch, ch0 in rows:
        if measured:
            L.append(f"| {page} | {typ} | {b[0]:.1f} / {b[1]:.1f} | {n[0]:.1f} / {n[1]:.1f} | {m[0]:.1f} / {m[1]:.1f} | "
                     f"{same}/{tot} | {ch} / {ch0} |")
        else:
            L.append(f"| {page} | {typ} | {b[0]:.1f} / {b[1]:.1f} | {n[0]:.1f} / {n[1]:.1f} | — | — | {ch} / — |")
    if diffs:
        L += ["", "First differences:", ""] + [f"- {x}" for x in diffs]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{a.name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / f"{a.name}.json").write_text(json.dumps({"ocr": a.ocr, "rows": rows, "identical": same_all,
                                                                 "blocks": n_all, "out": out},
                                                                ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    if not measured:
        avg = [sum(r[i][0] for r in rows) / len(rows) for i in (2, 3)]
        print(f"✔ LLM ON {a.ocr}: main text, mean of 4 pages {avg[0]:.2f} → {avg[1]:.2f} % correct characters "
              f"({avg[1] - avg[0]:+.2f})")
        return 0
    ok = same_all == n_all
    print(("✔" if ok else "⚠" if same_all >= 0.9 * n_all else "✘")
          + f" LLM CHECK: {same_all}/{n_all} blocks identical to the measured run"
          + ("" if ok else " (small differences can come from GPU / library versions — compare the scores above)"))
    return 0 if same_all >= 0.9 * n_all else 1


if __name__ == "__main__":
    sys.exit(main())
