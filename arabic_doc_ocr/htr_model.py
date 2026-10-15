"""Handwritten line reader: Misraj/Baseer__Nakba (Qwen2.5-VL-3B) + LoRA adapters ft3 / ft4 sharing one base model.

Settings exactly as the authors' Baseer_Nakba.py (github.com/misraj-ai/Nakba-pipeline): prompt "Extract the text from
the above document." with the default system prompt, min_pixels 28·28, max_pixels 1280·28·28, greedy decoding,
repetition_penalty 1.1, at most 128 new tokens, repeated-tail cleanup. Lines are read ONE BY ONE: batching makes
Qwen2.5-VL stop early on transformers 4.56 (72 % vs 91 % correct characters on 32 test lines).
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

from PIL import Image

from . import config
from .dots_model import pick_dtype
from .text import clean_repeated_substrings, compare_words, fix_arabic

TEMPLATE = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{p}<|im_end|>\n"
            "<|im_start|>assistant\n")
PROMPT = "Extract the text from the above document."
_EOS, _PAD = [151645, 151643], 151643


class HTRReader:
    """read(lines, adapter="ft3") → texts. adapter="base" = Baseer-Nakba without adapter."""

    def __init__(self, device: str = "cuda:0", adapters=(config.MAIN, config.CHECK), base: str = config.HTR_BASE,
                 repo: str = config.HTR_ADAPTERS):
        self.device, self.adapters, self.base, self.repo = device, list(adapters), base, repo
        self.model = self.processor = None
        self._lock = threading.Lock()

    @staticmethod
    def _fix_config(cfg):
        """The base repo stores the image-token ids only in text_config; transformers 4.56 looks for them at the top
        level ('vision_start_token_id' AttributeError). Copy the values (unchanged)."""
        tc = getattr(cfg, "text_config", None)
        for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
                  "vision_token_id"):
            v = getattr(tc, k, None) if tc is not None else None
            if v is not None and getattr(cfg, k, None) is None:
                setattr(cfg, k, v)
        return cfg

    def load(self) -> "HTRReader":
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        try:  # an old torchao (e.g. on Colab) makes peft raise ImportError on plain layers — torchao is never used
            import peft.import_utils as iu
            import peft.tuners.lora.torchao as lt
            iu.is_torchao_available = lt.is_torchao_available = lambda: False
        except Exception:  # noqa: BLE001
            pass
        self.torch = torch
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kw = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": pick_dtype(torch, self.device),
              "device_map": {"": self.device}}
        cfg = self._fix_config(AutoConfig.from_pretrained(self.base))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.base, config=cfg, **kw).eval()
        if self.adapters:
            root = Path(snapshot_download(self.repo, allow_patterns=[f"{a}/*" for a in self.adapters]))
            for i, a in enumerate(self.adapters):
                if not (root / a / "adapter_config.json").exists():
                    raise FileNotFoundError(f"adapter {a} not found in {self.repo}")
                if i == 0:
                    model = PeftModel.from_pretrained(model, str(root / a), adapter_name=a).eval()
                else:
                    model.load_adapter(str(root / a), adapter_name=a)
            model.set_adapter(self.adapters[0])
        self.model = model
        try:
            self.processor = AutoProcessor.from_pretrained(self.base, min_pixels=config.HTR_MIN_PIXELS,
                                                           max_pixels=config.HTR_MAX_PIXELS)
        except TypeError:  # newer processors take a size dict
            self.processor = AutoProcessor.from_pretrained(
                self.base, size={"shortest_edge": config.HTR_MIN_PIXELS, "longest_edge": config.HTR_MAX_PIXELS})
        self.processor.tokenizer.padding_side = "left"
        return self

    def to(self, device: str) -> None:
        self.model.to(device)

    def _select(self, adapter: str | None):
        if adapter == "base":
            return self.model.disable_adapter() if self.adapters else contextlib.nullcontext()
        if self.adapters:
            a = adapter or self.adapters[0]
            if a not in self.adapters:
                raise ValueError(f"adapter {a} not loaded ({self.adapters})")
            self.model.set_adapter(a)
        return contextlib.nullcontext()

    def read(self, lines: list[Image.Image], adapter: str | None = None, progress=None) -> list[str]:
        if self.model is None:
            self.load()
        torch, out = self.torch, []
        prompt = TEMPLATE.format(p=PROMPT)
        for i, im in enumerate(lines):
            if progress:
                progress(i, len(lines))
            inputs = self.processor(text=[prompt], images=[im.convert("RGB")], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self._lock, torch.no_grad(), self._select(adapter):
                ids = self.model.generate(**inputs, max_new_tokens=config.HTR_MAX_NEW_TOKENS, do_sample=False,
                                          repetition_penalty=1.1, eos_token_id=_EOS, pad_token_id=_PAD)
            n_in = inputs["input_ids"].shape[1]
            text = self.processor.decode(ids[0][n_in:], skip_special_tokens=True).strip()
            out.append(fix_arabic(clean_repeated_substrings(text)))
        return out

    def read_check(self, lines: list[Image.Image], main: str = config.MAIN, check: str = config.CHECK) -> list[dict]:
        """Two readings per line → [{"text", "check", "words": [(word, suspicious, alt)], "agree"}]."""
        a, b = self.read(lines, main), self.read(lines, check)
        out = []
        for x, y in zip(a, b):
            w = compare_words(x, y)
            out.append({"text": x, "check": y, "words": w,
                        "agree": 1.0 if not w else sum(not s for _, s, _ in w) / len(w)})
        return out

    def close(self) -> None:
        self.model = None
