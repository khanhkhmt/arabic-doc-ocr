"""Web interface (Gradio).

    python -m arabic_doc_ocr web [--host 127.0.0.1] [--port 7862] [--dots-device cuda:0] [--htr-device cuda:1]

Tab "Document": image or PDF → dots.mocr layout → Kraken lines → ft3 + ft4 read every line → (optional) LLM
correction → review table, DOCX / TXT / HTML / JSON / page images to download.  Tab "One line": one line image →
ft3 / ft4.
One request at a time (the GPU models are shared).
"""

from __future__ import annotations

import html
import json
import tempfile
import time
from pathlib import Path

import gradio as gr

from . import config
from .export import CSS, draw_boxes, marked_html, write_all
from .pipeline import Pipeline, load_pages, parse_page_spec
from .text import agreement, compare_words

TYPES = {
    "Handwritten — ft3 reads, ft4 cross-checks": config.HANDWRITTEN,
    "Printed — dots reads, ft3 cross-checks": config.PRINTED,
}
STATE: dict = {"pipeline": None, "args": {}}


def pipeline() -> Pipeline:
    if STATE["pipeline"] is None:
        STATE["pipeline"] = Pipeline(**STATE["args"])
    return STATE["pipeline"]


def run_document(file, pages_spec, type_label, use_llm=False, progress=gr.Progress()):
    if not file:
        return None, "<p>No file.</p>", "", None
    path = Path(file if isinstance(file, str) else file.name)
    nums = parse_page_spec(pages_spec) if path.suffix.lower() == ".pdf" else None
    pages = load_pages(path, nums)
    if not pages:
        return None, "<p>No page in range.</p>", "", None
    nums = nums or list(range(1, len(pages) + 1))
    nums = nums[:len(pages)]
    t0 = time.time()
    results = pipeline().process_document(pages, TYPES[type_label], lambda p, msg: progress(p, desc=msg))
    li = ""
    if use_llm:
        from . import llm

        t1 = time.time()
        info = llm.correct(results, pipeline().models, lambda p, msg: progress(p, desc=msg))
        li = (f" · LLM: {info['changed_words']} word change(s), {time.time() - t1:.0f} s" if info.get("ok")
              else f" · LLM NOT applied: {html.escape(str(info.get('error')))}")
    out = Path(tempfile.mkdtemp(prefix="arabic_ocr_"))
    w = write_all(out, path.stem, results, nums)
    head = f"<p class=meta>{len(pages)} page(s) · {time.time() - t0:.0f} s{li}</p>"
    return (draw_boxes(results[0]), f"<div class=htr>{head}</div>{w['html']}", w["main"],
            [str(p) for p in w["files"]])


def run_line(image):
    if image is None:
        return "<p>No image.</p>", None
    t = time.time()
    r = pipeline().read_lines([image])[0]
    a, b = r["text"], r["check"]
    tag = "✓ both readings agree" if agreement(a, b) >= 0.999 else "yellow = ft4 reads differently"
    f = Path(tempfile.mkdtemp(prefix="arabic_ocr_")) / "line.json"
    f.write_text(json.dumps({"ft3": a, "ft4": b, "words_differ": [w for w, s, _ in compare_words(a, b) if s]},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"<div class=htr><p class=meta>{time.time() - t:.1f} s · {tag}</p><table>"
            f"<tr><th>ft3</th><td class=ar>{marked_html(a, b)}</td></tr>"
            f"<tr><th>ft4</th><td class=ar>{html.escape(b)}</td></tr></table></div>"), [str(f)]


def build():
    from . import llm

    gr6 = int(gr.__version__.split(".")[0]) >= 6
    llm_reason = llm.why_not()
    with gr.Blocks(title="Arabic document OCR", **({} if gr6 else {"css": CSS})) as demo:
        gr.Markdown("## Arabic document OCR — printed and handwritten\n"
                    "dots.mocr finds the blocks and reads printed text → Kraken splits lines → **ft3** reads every "
                    "line, **ft4** reads it again. Where two readings differ the word is **highlighted yellow** — "
                    "that is where a person must check. The result is a **draft**, not a final text.")
        with gr.Tab("Document"):
            with gr.Row():
                with gr.Column(scale=1):
                    f = gr.File(label="Image or PDF", file_types=["image", ".pdf"], type="filepath")
                    pages = gr.Textbox(label="PDF pages (e.g. 1-3,5; empty = all)", value="")
                    kind = gr.Radio(list(TYPES), value=next(iter(TYPES)), label="Text on the page")
                    use_llm = gr.Checkbox(
                        value=False, interactive=llm_reason is None,
                        label="LLM correction (Qwen3.5-9B) — slower: +1–2 min to load, ~1 min per page",
                        info=(llm_reason or "fixes OCR letter mistakes; changed words are GREEN in the DOCX and the "
                              "review table. Measured on 4 sample pages: +1…3 % correct characters, ~1 wrong change "
                              "for 7 good ones — check green words too."))
                    go = gr.Button("Read", variant="primary")
                with gr.Column(scale=2):
                    vis = gr.Image(label="Blocks (blue; purple = Hebrew) · lines: green = ft3/ft4 agree, orange = differ",
                                   type="pil")
            gr.Markdown("### Downloads\n`_all.zip` everything · `.docx` text with highlights + comments · "
                        "`_main.txt` main text (`_ft3.txt`, `_ft4.txt` each reader) · `_review.html` line-by-line "
                        "table · `.json` boxes + all readings · `_boxes.jpg` page with boxes")
            files = gr.Files(label="Download")
            with gr.Accordion("Main text", open=False):
                txt = gr.Textbox(lines=18, rtl=True)
            out = gr.HTML()
            go.click(run_document, [f, pages, kind, use_llm], [vis, out, txt, files])
        with gr.Tab("One line"):
            li = gr.Image(label="Image of ONE text line", type="pil")
            lb = gr.Button("Read", variant="primary")
            lo = gr.HTML()
            ldl = gr.Files(label="Download (JSON)")
            lb.click(run_line, [li], [lo, ldl])
    return demo, gr6


def serve(host: str = "127.0.0.1", port: int = 7862, share: bool = False, **pipeline_args) -> None:
    STATE["args"] = pipeline_args
    demo, gr6 = build()
    demo.queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, share=share,
                                                   **({"css": CSS} if gr6 else {}))
