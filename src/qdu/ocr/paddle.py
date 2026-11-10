"""PaddleOCR 3.x (PP-OCRv5) engine with the same on-disk cache as Tesseract.

Install the optional extra first: ``uv sync --extra paddle``. Two model sizes are offered:
``mobile`` (fast, CPU-friendly) and ``server`` (more accurate, slower). Document orientation,
unwarping and text-line orientation are off: benchmark pages are upright scans and renders.

On Windows the oneDNN CPU backend fails on PP-OCRv5's graph ("ConvertPirAttribute2Runtime-
Attribute not support"), so oneDNN is disabled there; this makes CPU inference much slower
(~24 s per page on a laptop) but correct. Elsewhere (Linux, Colab) oneDNN stays on unless
``QDU_PADDLE_MKLDNN=0``. The engine name records which, since the setting changes the measured
OCR time.

The first prediction pays one-off initialisation, so the engine runs once on a blank page
when it is built; otherwise the first page of every session would report an inflated time.
"""

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from qdu.ocr.base import OcrLine, OcrResult, cache_file, read_cache, write_cache
from qdu.provenance import cpu_info

if TYPE_CHECKING:
    from PIL.Image import Image

PADDLE_MODELS = ("mobile", "server")


class PaddleOcr:
    def __init__(self, model: str = "mobile", cache_dir: Path | None = None) -> None:
        if model not in PADDLE_MODELS:
            raise ValueError(f"unknown PP-OCRv5 model {model!r}; choose from {PADDLE_MODELS}")
        try:
            import paddleocr
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("PaddleOCR is not installed: run `uv sync --extra paddle`") from exc

        prefix = f"PP-OCRv5_{model}"
        mkldnn = sys.platform != "win32" and os.environ.get("QDU_PADDLE_MKLDNN", "1") != "0"
        self._ocr = PaddleOCR(
            text_detection_model_name=f"{prefix}_det",
            text_recognition_model_name=f"{prefix}_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=mkldnn,
        )
        self.model = model
        self.name = f"paddleocr-{paddleocr.__version__}-ppocrv5-{model}" + (
            "" if mkldnn else "-nomkldnn"
        )
        self.hardware = cpu_info()
        self.cache_dir = cache_dir / f"paddle-{model}" if cache_dir else None

        import numpy as np

        self._ocr.predict(np.full((64, 256, 3), 255, dtype=np.uint8))  # warm-up, not timed

    def _cache_path(self, key: str) -> Path | None:
        return cache_file(self.cache_dir, key) if self.cache_dir else None

    def extract(self, image: "Image", key: str) -> OcrResult:
        path = self._cache_path(key)
        if (cached := read_cache(path)) is not None:
            return cached

        import numpy as np

        start = time.perf_counter()
        pages = self._ocr.predict(np.array(image.convert("RGB")))
        latency = time.perf_counter() - start
        lines = tuple(
            OcrLine(text, round(float(score), 4), tuple(int(v) for v in box))
            for page in pages
            for text, score, box in zip(
                page["rec_texts"], page["rec_scores"], page["rec_boxes"], strict=True
            )
        )
        result = OcrResult(
            "\n".join(line.text for line in lines),
            self.name,
            latency,
            hardware=self.hardware,
            lines=lines,
            page_size=image.size,
        )
        write_cache(path, result)
        return result
