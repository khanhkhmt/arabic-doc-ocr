"""LLM correction of the OCR main text (optional step) — runs in its OWN virtualenv (install_llm.sh).

The LLM (Qwen3.5-9B) needs transformers 5.x, the OCR models need transformers 4.56.1 → this module runs as a separate
process:

    <llm venv>/bin/python -m arabic_doc_ocr.llm_fix --in units.json --out fixed.json
    <llm venv>/bin/python -m arabic_doc_ocr.llm_fix --download

Method "chon_sua" — the one measured on the 4 sample pages (ocr-bench htr_test/llm_sua.py, 2026-10-19: printed
+1.1…+3.0 correct characters / +8…+13 correct words, handwritten +3.0 / +5.3; 77 + 12 words fixed, 10 + 2 broken).
The code, constants, prompt and model revision below are the measured ones — changing them changes the results.

  chon  where the two readings of a block differ (printed: dots ↔ ft3; handwritten: ft3 ↔ ft4; ≤ 3 words per side,
        no digits, no Latin / Hebrew) the LLM scores the sentence with reading A and with reading B → keeps the more
        probable one. No new words.
  sua   then the LLM rewrites the text (the second reading is given as a hint). ONLY 1 word ↔ 1 word replacements
        are accepted, ≤ 2 letters changed and ≤ 40 % of the word, no digits, Arabic letters only. Everything else
        (added / dropped / merged / split words, numbers, rewritten sentences) is rejected: the OCR text stays.

This module imports only the standard library at the top (it must also import in the main venv for tests).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

from .text import _key

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # the revision measured on 2026-10-19
METHOD = "chon_sua"

# --- acceptance rules (fixed) -----------------------------------------------------------------------------------------
MAX_SPAN_WORDS = 3            # chon: ≤ 3 words per side
CTX_LEFT, CTX_RIGHT = 40, 12  # chon: context words left / right
MAX_EDIT_CHARS = 2            # sua: ≤ 2 letters changed …
MAX_EDIT_FRAC = 0.4           # … and ≤ 40 % of the original word
CHUNK_WORDS = 60              # sua: ~60 words per LLM call (whole lines)

SYSTEM = (
    "You are an expert proofreader of Arabic legal documents from the 1970s (sale contracts, powers of attorney, "
    "official letters from Palestine, Jordan and Egypt). The text you receive was produced by OCR and contains "
    "letter-level mistakes: letters with similar shapes are confused (for example ع/ح/غ/خ/ج, ة/ه, ب/ت/ث/ن/ي, ق/ف, "
    "د/ذ/ر/ز, س/ش/ص/ض) and some letters are dropped.\n"
    "Rules:\n"
    "1. Correct only words that are OCR mistakes. Change as few letters as possible.\n"
    "2. Never change numbers, digits or dates.\n"
    "3. Do not change personal names or place names unless the mistake is obvious.\n"
    "4. Do not add, remove, merge, split or reorder words. Keep the line breaks.\n"
    "5. Do not add diacritics (harakat).\n"
    "6. Output only the corrected text, with no explanation."
)

_AR_WORD = re.compile(r"[ء-ي]+")
_AR_WORD_HARAKAT = re.compile(r"[ء-يً-ْ]+")
_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")
_FOREIGN = re.compile(r"[A-Za-z֐-׿]")
_EDGE = re.compile(r"^([^ء-ي]*)(.*?)([^ء-ي]*)$", re.S)


# --- word alignment ---------------------------------------------------------------------------------------------------

def _spans(main: str, check: str) -> tuple[list[str], list[tuple[int, int, list[str]]]]:
    """Where main ↔ check differ (replacement of ≤ 3 words per side, no digits, no foreign letters) →
    (main words, [(i1, i2, check words)])."""
    a, b = main.split(), check.split()
    sm = difflib.SequenceMatcher(None, [_key(w) for w in a], [_key(w) for w in b], autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or not (1 <= i2 - i1 <= MAX_SPAN_WORDS and 1 <= j2 - j1 <= MAX_SPAN_WORDS):
            continue
        both = " ".join(a[i1:i2] + b[j1:j2])
        if _DIGIT.search(both) or _FOREIGN.search(both):
            continue
        out.append((i1, i2, b[j1:j2]))
    return a, out


def _rebuild(text: str, repl: dict[int, str]) -> str:
    """Replace word i (as split()) by repl[i], keeping the spaces / line breaks of the original."""
    parts, n = re.split(r"(\s+)", text), 0
    for k, p in enumerate(parts):
        if p and not p.isspace():
            if n in repl:
                parts[k] = repl[n]
            n += 1
    return "".join(parts)


def _accept(old: str, new: str) -> str | None:
    """Acceptance rule for one 1 word ↔ 1 word change of "sua" → the new word (keeping the punctuation around the
    original word) or None."""
    from rapidfuzz.distance import Levenshtein

    if _DIGIT.search(old) or _DIGIT.search(new):
        return None
    po, co, so = _EDGE.match(old).groups()
    _, cn, _ = _EDGE.match(new).groups()
    if len(co) < 2 or not _AR_WORD_HARAKAT.fullmatch(co) or not _AR_WORD.fullmatch(cn or "-"):
        return None
    d = Levenshtein.distance(co, cn)
    if d == 0 or d > MAX_EDIT_CHARS or d > MAX_EDIT_FRAC * len(co):
        return None
    return po + cn + so


def _chunks(words: list[str], text: str) -> list[tuple[int, int]]:
    """Word ranges [s, e), whole lines grouped until ≥ CHUNK_WORDS words."""
    out, s, n = [], 0, 0
    for line in text.split("\n"):
        n += len(line.split())
        if n - s >= CHUNK_WORDS:
            out.append((s, n))
            s = n
    if s < len(words):
        out.append((s, len(words)))
    return out


def _map_check(a: list[str], b: list[str]) -> list[int]:
    """Index of a main word → matching position in the check reading (to cut the hint for each chunk)."""
    m = [0] * (len(a) + 1)
    sm = difflib.SequenceMatcher(None, [_key(w) for w in a], [_key(w) for w in b], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(i1, i2):
            m[k] = j1 + (k - i1 if tag == "equal" else min(k - i1, j2 - j1))
    m[len(a)] = len(b)
    return m


# --- the LLM ----------------------------------------------------------------------------------------------------------

class LLM:
    def __init__(self, repo: str = MODEL, revision: str = REVISION):
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer

        self.torch, self.repo, self.revision = torch, repo, revision
        path = snapshot_download(repo, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"])
        cfg = AutoConfig.from_pretrained(path)
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU in the LLM venv")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        mm = {i: f"{int(torch.cuda.get_device_properties(i).total_memory / 2**30 - 1.5)}GiB"
              for i in range(torch.cuda.device_count())}
        kw = {"device_map": "auto", "max_memory": mm}
        self.dtype = str(dtype).replace("torch.", "")
        tv = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
        kw["dtype" if tv >= (4, 56) else "torch_dtype"] = dtype
        if getattr(cfg, "vision_config", None) is not None:
            from transformers import AutoModelForImageTextToText as Cls
        else:
            from transformers import AutoModelForCausalLM as Cls
        self.model, info = Cls.from_pretrained(path, output_loading_info=True, **kw)
        missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
        if missing:
            raise RuntimeError(f"missing weights when loading ({len(missing)}, e.g. {missing[:3]}) — output would be garbage")
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.proc = None
        if not getattr(self.tok, "chat_template", None):
            from transformers import AutoProcessor

            self.proc = AutoProcessor.from_pretrained(path)
        self.dev = self.model.get_input_embeddings().weight.device

    def logprob(self, text: str) -> float:
        torch = self.torch
        # fixed prefix: the first token (not scored) is the same for reading A and reading B
        ids = self.tok("نص: " + text, return_tensors="pt").input_ids.to(self.dev)
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None]).sum().item()
        if lp != lp:  # NaN
            raise RuntimeError("logprob = NaN (fp16 overflow?)")
        return lp

    def chat(self, user: str) -> str:
        torch = self.torch
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if self.proc is None:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in msgs]
            prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
        n_text = len(self.tok(user).input_ids)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=min(1024, n_text + 64), pad_token_id=self.tok.eos_token_id)
        s = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
        return s.replace("<<<", "").replace(">>>", "").replace("```", "").strip()


# --- the two steps ----------------------------------------------------------------------------------------------------

def fix_chon(llm, main: str, check: str) -> tuple[str, list[dict]]:
    a, spans = _spans(main, check)
    cur, repl, changes = list(a), {}, []
    for i1, i2, alt in spans:
        left = " ".join(cur[max(0, i1 - CTX_LEFT):i1])
        right = " ".join(cur[i2:i2 + CTX_RIGHT])
        ctx = lambda mid: " ".join(x for x in (left, mid, right) if x)  # noqa: E731
        lp_a, lp_b = llm.logprob(ctx(" ".join(a[i1:i2]))), llm.logprob(ctx(" ".join(alt)))
        if lp_b > lp_a:
            # the whole span: its first word gets the new span, the other words become empty
            repl[i1] = " ".join(alt)
            for k in range(i1 + 1, i2):
                repl[k] = ""
            cur[i1:i2] = [" ".join(alt)] + [""] * (i2 - i1 - 1)
            changes.append({"old": " ".join(a[i1:i2]), "new": " ".join(alt), "lp_old": round(lp_a, 2),
                            "lp_new": round(lp_b, 2)})
    text = _rebuild(main, repl)
    return re.sub(r"[ \t]{2,}", " ", text), changes


def fix_sua(llm, main: str, check: str) -> tuple[str, list[dict]]:
    a, b = main.split(), check.split()
    m = _map_check(a, b) if b else None
    repl, changes = {}, []
    for s, e in _chunks(a, main):
        piece = " ".join(a[s:e])
        hint = " ".join(b[m[s]:m[e]]) if m else ""
        user = f"OCR text:\n<<<\n{piece}\n>>>\n"
        if hint.strip():
            user += ("\nA second OCR system read the same passage as follows (it also makes mistakes; use it only "
                     f"as a hint):\n<<<\n{hint}\n>>>\n")
        user += "\nCorrected text:"
        nw = llm.chat(user).split()
        sm = difflib.SequenceMatcher(None, a[s:e], nw, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "replace" or i2 - i1 != j2 - j1:
                continue
            for k in range(i2 - i1):
                old, new = a[s + i1 + k], nw[j1 + k]
                ok = _accept(old, new)
                if ok:
                    repl[s + i1 + k] = ok
                    changes.append({"old": old, "new": ok})
                else:
                    changes.append({"old": old, "new": new, "rejected": True})
    return _rebuild(main, repl), changes


def fix_unit(llm, main: str, check: str) -> tuple[str, list[dict]]:
    """chon, then sua. → (corrected text, changes; rejected proposals are kept with "rejected": true)."""
    t, c1 = fix_chon(llm, main, check)
    t, c2 = fix_sua(llm, t, check)
    return t, [{**c, "step": "chon"} for c in c1] + [{**c, "step": "sua"} for c in c2]


# --- worker -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m arabic_doc_ocr.llm_fix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", help='JSON {"units": [{"key", "main", "check"}]}')
    ap.add_argument("--out", help='JSON {"texts": {key: text}, "changes": {key: [...]}, …}')
    ap.add_argument("--download", action="store_true", help="only download the model into the cache")
    a = ap.parse_args(argv)
    if a.download:
        from huggingface_hub import snapshot_download

        t = time.time()
        p = snapshot_download(MODEL, revision=REVISION,
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"])
        print(f"{MODEL}@{REVISION[:10]} → {p} ({time.time() - t:.0f} s)")
        return 0
    if not a.inp or not a.out:
        ap.error("--in and --out are needed (or --download)")
    units = json.loads(Path(a.inp).read_text(encoding="utf-8"))["units"]
    t0 = time.time()
    llm = LLM()
    load_s = round(time.time() - t0, 1)
    print(f"@@loaded {load_s}", flush=True)
    texts, changes = {}, {}
    for i, u in enumerate(units):
        texts[u["key"]], changes[u["key"]] = fix_unit(llm, u["main"], u["check"])
        print(f"@@unit {i + 1} {len(units)}", flush=True)
    Path(a.out).write_text(json.dumps({"model": MODEL, "revision": REVISION, "method": METHOD, "dtype": llm.dtype,
                                       "load_seconds": load_s, "seconds": round(time.time() - t0, 1),
                                       "texts": texts, "changes": changes}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
