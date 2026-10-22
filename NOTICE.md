# Third-party components and licenses

| component | used for | license |
|---|---|---|
| [dots-studio/dots.mocr](https://huggingface.co/dots-studio/dots.mocr) (model) | layout + printed text | dots.ocr LICENSE AGREEMENT (`arabic_doc_ocr/_vendor/dots_ocr/dots.ocr LICENSE AGREEMENT`): commercial use allowed; §3.3 restricts processing sensitive personal data without consent / lawful basis |
| dots.ocr code, `dots_ocr/utils` (vendored unchanged, commit 36d7248) | pre/post-processing | MIT (`arabic_doc_ocr/_vendor/dots_ocr/LICENSE`) |
| [Misraj/Baseer__Nakba](https://huggingface.co/Misraj/Baseer__Nakba) (model) | handwritten lines | **CC BY-NC-SA 4.0 — non-commercial** |
| [khanh99/htr-baseer-nakba-ft](https://huggingface.co/khanh99/htr-baseer-nakba-ft) ft3 / ft4 (LoRA) | handwritten lines | **CC BY-NC-SA 4.0** (trained on Muharaf, CC BY-NC-SA 4.0; Omar Al-Saleh, CC BY 4.0; AraMS; KHATT) |
| [Kraken](https://kraken.re) | line segmentation | Apache 2.0 |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (model, optional `--llm` step) | OCR correction | Apache 2.0 |
| [Muharaf segmentation model](https://zenodo.org/records/14295555) | line segmentation | CC BY 4.0 |
| transformers, peft, gradio, python-docx, pypdfium2, PyMuPDF | libraries | Apache 2.0 / MIT / BSD; **PyMuPDF is AGPL-3.0** |

Inference settings of Baseer-Nakba follow the authors' code (github.com/misraj-ai/Nakba-pipeline).
The sample pages in `examples/` are published with the document owner's permission.
