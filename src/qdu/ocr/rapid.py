"""PP-OCRv5 through RapidOCR (ONNX Runtime), on the CPU or the GPU.

Install the optional extra first: ``uv sync --extra rapid``; for the GPU also install
``onnxruntime-gpu`` in place of ``onnxruntime``. RapidOCR runs the same PP-OCRv5 detection and
recognition models as PaddleOCR, converted to ONNX, so a GPU is usable without the
paddlepaddle-gpu build (whose package index was unreachable in Nov 2026).

Settings follow :mod:`qdu.ocr.paddle` where they affect the output: no angle classifier,
pages longer than 4000 pixels downscaled for detection, and no recognition-score threshold
(PaddleOCR's default keeps every line). On the GPU cuDNN's algorithm search is "HEURISTIC":
on a 223-line page (L4) it recognised in 1.18 s against 2.3 s with "DEFAULT", and it matches
RapidOCR's "EXHAUSTIVE" (1.15 s) without re-running a search for every new page shape.
Recognition batches of 6 lines (RapidOCR's default) beat 64 (2.5 s) on the GPU.

Results carry their own engine name (``-gpu`` on the GPU), cache directory and hardware
record, as for PaddleOCR. The engine runs once on a page of text (`warmup_page`) when built, so
one-off initialisation is not charged to the first page.
"""

import time
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from qdu.ocr.base import OcrLine, OcrResult, cache_file, read_cache, warmup_page, write_cache
from qdu.provenance import cpu_info, gpu_info

if TYPE_CHECKING:
    from PIL.Image import Image

RAPID_MODELS = ("mobile", "server")


class RapidOcr:
    def __init__(
        self, model: str = "mobile", cache_dir: Path | None = None, gpu: bool = False
    ) -> None:
        if model not in RAPID_MODELS:
            raise ValueError(f"unknown PP-OCRv5 model {model!r}; choose from {RAPID_MODELS}")
        try:
            from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("RapidOCR is not installed: run `uv sync --extra rapid`") from exc

        model_type = ModelType.MOBILE if model == "mobile" else ModelType.SERVER
        params = {
            "Global.use_cls": False,
            "Global.max_side_len": 4000,
            "Global.text_score": 0.0,
            "Global.log_level": "warning",
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": model_type,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": model_type,
            "EngineConfig.onnxruntime.use_cuda": gpu,
        }
        if gpu:
            params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "HEURISTIC"
            # Grow GPU memory by what is requested, not to the next power of two: next to vLLM
            # the doubling ran out of memory on a large page (Colab L4, page 315 of full.csv).
            params["EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy"] = (
                "kSameAsRequested"
            )
        self._ocr = RapidOCR(params=params)
        self.model = model
        suffix = "-gpu" if gpu else ""
        self.name = f"rapidocr-{metadata.version('rapidocr')}-ppocrv5-{model}{suffix}"
        gpus = gpu_info() if gpu else None
        self.hardware = gpus[0]["name"] if gpus else cpu_info()
        self.cache_dir = cache_dir / f"rapid-{model}{suffix}" if cache_dir else None

        self._ocr(warmup_page())  # warm-up, not timed

    def _cache_path(self, key: str) -> Path | None:
        return cache_file(self.cache_dir, key) if self.cache_dir else None

    def extract(self, image: "Image", key: str) -> OcrResult:
        path = self._cache_path(key)
        if (cached := read_cache(path, image.size)) is not None:
            return cached

        import numpy as np

        start = time.perf_counter()
        out = self._ocr(np.array(image.convert("RGB")))
        latency = time.perf_counter() - start
        texts = out.txts or ()
        scores = out.scores or ()
        boxes = out.boxes if out.boxes is not None else ()
        lines = tuple(
            OcrLine(
                text,
                round(float(score), 4),
                (
                    int(min(p[0] for p in box)),
                    int(min(p[1] for p in box)),
                    int(max(p[0] for p in box)),
                    int(max(p[1] for p in box)),
                ),
            )
            for text, score, box in zip(texts, scores, boxes, strict=True)
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
