"""PaddleOCR 3.x (PP-OCRv5) engine with the same on-disk cache as Tesseract.

Install the optional extra first: ``uv sync --extra paddle``. Two model sizes are offered:
``mobile`` (fast, CPU-friendly) and ``server`` (more accurate, slower). Document orientation,
unwarping and text-line orientation are off: benchmark pages are upright scans and renders.

On Windows the oneDNN CPU backend fails on PP-OCRv5's graph ("ConvertPirAttribute2Runtime-
Attribute not support"), so oneDNN is disabled; this makes CPU inference slower but correct.
"""

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from qdu.ocr.base import OcrResult, cache_file

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
        self._ocr = PaddleOCR(
            text_detection_model_name=f"{prefix}_det",
            text_recognition_model_name=f"{prefix}_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        self.model = model
        self.name = f"paddleocr-{paddleocr.__version__}-ppocrv5-{model}"
        self.cache_dir = cache_dir / f"paddle-{model}" if cache_dir else None

    def _cache_path(self, key: str) -> Path | None:
        return cache_file(self.cache_dir, key) if self.cache_dir else None

    def extract(self, image: "Image", key: str) -> OcrResult:
        path = self._cache_path(key)
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return OcrResult(data["text"], data["engine"], data["latency_s"], cached=True)

        import numpy as np

        start = time.perf_counter()
        pages = self._ocr.predict(np.array(image.convert("RGB")))
        text = "\n".join(line for page in pages for line in page["rec_texts"])
        result = OcrResult(text, self.name, time.perf_counter() - start)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"text": text, "engine": self.name, "latency_s": result.latency_s}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return result
