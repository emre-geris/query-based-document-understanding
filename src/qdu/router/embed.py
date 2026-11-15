"""OCR-free embedding signals S5 (question) and S6 (page image) for the router (Phase 7 C).

- **S5** question embedding: ``sentence-transformers/all-MiniLM-L6-v2`` (384-d, normalised).
- **S6** page embedding: ``sentence-transformers/clip-ViT-B-32`` image vector (512-d,
  normalised), plus ``qsim``: the cosine between the CLIP text embedding of the question and the
  page vector (does the question "look like" the page?).

Neither needs OCR or layout, so both can serve stage 1 of the two-stage router. The vectors are
computed once by ``qdu router embed`` (needs the optional ``embed`` extra and the page images)
and committed as two deterministic gzips next to the run, like the OCR and layout:

- ``clip_pages.jsonl.gz``: one row per page, ``{doc_id, model, hardware, latency_s, vector}``;
- ``question_embeddings.jsonl.gz``: one row per question, ``{question_id, model, latency_s,
  vector, clip_model, qsim, qsim_s}``.

Vectors are stored as base64 float16 (a few MB per file). Each embedding is timed alone (no
batching) after a warm-up, the way a deployed router would compute it for one question, and the
time is charged to the router: S5's and the question's CLIP text time per question
(``question_s``), the page's CLIP image time per page (``page_s``, shared by the page's
questions in scenario B). Loading the image is not timed, as for the other signals.

The extractors only read these files, so routers train on a laptop without
sentence-transformers. ``sentence_transformers`` is imported only by `load_encoder`.
"""

import base64
import gzip
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from qdu.gz import open_gzip_text
from qdu.router.features import RouterInput

if TYPE_CHECKING:
    from PIL.Image import Image

    from qdu.data import QARecord

QUESTION_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
DEFAULT_PAGE_EMBEDDINGS = Path("results/full-qwen3.5-4b/clip_pages.jsonl.gz")
DEFAULT_QUESTION_EMBEDDINGS = Path("results/full-qwen3.5-4b/question_embeddings.jsonl.gz")


class Encoder(Protocol):
    """The part of ``SentenceTransformer`` used here."""

    def encode(self, items: Sequence[Any], **kwargs: Any) -> np.ndarray: ...


def load_encoder(model: str, device: str = "cpu") -> Encoder:
    """A sentence-transformers model (``uv sync --extra embed``)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(
            "embedding signals need the 'embed' extra: uv sync --extra embed"
        ) from exc
    return SentenceTransformer(model, device=device)


def device_hardware(device: str) -> str:
    """What the embeddings were timed on (recorded per page, like the OCR's hardware)."""
    if device.startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(torch.device(device))
    from qdu.provenance import cpu_info

    return cpu_info()


def encode_vector(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="<f2").tobytes()).decode("ascii")


def decode_vector(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f2").astype(np.float32)


def _one(encoder: Encoder, item: Any) -> tuple[np.ndarray, float]:
    """Embed one item (normalised); returns the vector and the seconds it took."""
    start = time.perf_counter()
    v = encoder.encode([item], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(v, dtype=np.float32), time.perf_counter() - start


@dataclass(frozen=True)
class PageEmbedding:
    doc_id: str
    model: str
    hardware: str
    latency_s: float
    vector: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class QuestionEmbedding:
    question_id: str
    model: str
    latency_s: float
    vector: np.ndarray = field(repr=False, compare=False)
    clip_model: str
    qsim: float
    """Cosine between the question's CLIP text embedding and its page's CLIP image vector."""
    qsim_s: float
    """Time of the question's CLIP text embedding."""


def _read_lines(path: Path) -> Iterable[dict]:
    """Rows of a ``.jsonl`` (work file) or ``.jsonl.gz`` (committed file)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        yield from (json.loads(line) for line in f if line.strip())


def load_page_embeddings(path: Path = DEFAULT_PAGE_EMBEDDINGS) -> dict[str, PageEmbedding]:
    return {
        d["doc_id"]: PageEmbedding(
            d["doc_id"], d["model"], d["hardware"], d["latency_s"], decode_vector(d["vector"])
        )
        for d in _read_lines(path)
    }


def load_question_embeddings(
    path: Path = DEFAULT_QUESTION_EMBEDDINGS,
) -> dict[str, QuestionEmbedding]:
    return {
        d["question_id"]: QuestionEmbedding(
            d["question_id"],
            d["model"],
            d["latency_s"],
            decode_vector(d["vector"]),
            d["clip_model"],
            d["qsim"],
            d["qsim_s"],
        )
        for d in _read_lines(path)
    }


def _page_json(p: PageEmbedding) -> dict[str, Any]:
    return {
        "doc_id": p.doc_id,
        "model": p.model,
        "hardware": p.hardware,
        "latency_s": round(p.latency_s, 6),
        "vector": encode_vector(p.vector),
    }


def _question_json(q: QuestionEmbedding) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "model": q.model,
        "latency_s": round(q.latency_s, 6),
        "vector": encode_vector(q.vector),
        "clip_model": q.clip_model,
        "qsim": round(q.qsim, 6),
        "qsim_s": round(q.qsim_s, 6),
    }


def embed_pages(
    records: Sequence["QARecord"],
    get_image: Callable[["QARecord"], "Image"],
    encoder: Encoder,
    out: Path,
    *,
    model: str = PAGE_MODEL,
    hardware: str = "",
    on_page: Callable[[PageEmbedding], None] | None = None,
) -> int:
    """Append the CLIP vector of every page of ``records`` not yet in ``out`` (a ``.jsonl``);
    returns the number of pages embedded in this call (resumable, like ``qdu layout detect``)."""
    done = set(load_page_embeddings(out)) if out.exists() else set()
    todo = {r.doc_id: r for r in records if r.doc_id not in done}
    if not todo:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    _one(encoder, get_image(next(iter(todo.values()))).convert("RGB"))  # warm-up, not timed
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for doc_id, record in todo.items():
            image = get_image(record).convert("RGB")  # loading is not timed
            vector, seconds = _one(encoder, image)
            page = PageEmbedding(doc_id, model, hardware, seconds, vector)
            f.write(json.dumps(_page_json(page)) + "\n")
            f.flush()
            if on_page:
                on_page(page)
    return len(todo)


def embed_questions(
    records: Sequence["QARecord"],
    pages: Mapping[str, PageEmbedding],
    question_encoder: Encoder,
    clip_encoder: Encoder,
    out: Path,
    *,
    model: str = QUESTION_MODEL,
    clip_model: str = PAGE_MODEL,
) -> int:
    """Append S5 and the CLIP question-page similarity of every question not yet in ``out``;
    every question's page must already be in ``pages``."""
    done = set(load_question_embeddings(out)) if out.exists() else set()
    todo = [r for r in records if r.question_id not in done]
    if not todo:
        return 0
    missing = sorted({r.doc_id for r in todo} - set(pages))
    if missing:
        raise ValueError(f"no page embedding for {len(missing)} pages, e.g. {missing[:3]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    _one(question_encoder, todo[0].question)  # warm-up
    _one(clip_encoder, todo[0].question)
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for r in todo:
            vector, seconds = _one(question_encoder, r.question)
            text, text_s = _one(clip_encoder, r.question)
            qsim = float(text @ pages[r.doc_id].vector)
            q = QuestionEmbedding(r.question_id, model, seconds, vector, clip_model, qsim, text_s)
            f.write(json.dumps(_question_json(q)) + "\n")
            f.flush()
    return len(todo)


def compress_pages(src: Path, dst: Path) -> int:
    """Write ``src`` (.jsonl) as gzipped ``dst`` sorted by page id (byte-identical on rewrite)."""
    pages = sorted(load_page_embeddings(src).values(), key=lambda p: p.doc_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_gzip_text(dst) as f:
        for p in pages:
            f.write(json.dumps(_page_json(p)) + "\n")
    return len(pages)


def compress_questions(src: Path, dst: Path) -> int:
    questions = sorted(load_question_embeddings(src).values(), key=lambda q: q.question_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open_gzip_text(dst) as f:
        for q in questions:
            f.write(json.dumps(_question_json(q)) + "\n")
    return len(questions)


@dataclass
class QuestionEmbeddingSignals:
    """S5: the question's MiniLM embedding (no OCR, no layout). Charged its embedding time."""

    path: Path = DEFAULT_QUESTION_EMBEDDINGS
    name: str = "s5"
    needs_ocr: bool = False
    needs_layout: bool = False
    _table: dict[str, QuestionEmbedding] | None = field(default=None, repr=False)

    def _get(self, item: RouterInput) -> QuestionEmbedding:
        if self._table is None:
            self._table = load_question_embeddings(self.path)
        try:
            return self._table[item.question_id]
        except KeyError:
            raise KeyError(
                f"{item.question_id}: no question embedding in {self.path} (run qdu router embed)"
            ) from None

    def __call__(self, item: RouterInput) -> dict[str, float]:
        return {f"e{i}": float(v) for i, v in enumerate(self._get(item).vector)}

    def cost(self, item: RouterInput) -> tuple[float, float]:
        """(seconds per question, seconds per page) measured when the embedding was made."""
        return self._get(item).latency_s, 0.0


@dataclass
class PageEmbeddingSignals:
    """S6: the page's CLIP image vector and the question-page CLIP similarity (no OCR, no
    layout). Charged the page's CLIP image time per page and the question's CLIP text time."""

    path: Path = DEFAULT_PAGE_EMBEDDINGS
    question_path: Path = DEFAULT_QUESTION_EMBEDDINGS
    name: str = "s6"
    needs_ocr: bool = False
    needs_layout: bool = False
    _pages: dict[str, PageEmbedding] | None = field(default=None, repr=False)
    _questions: dict[str, QuestionEmbedding] | None = field(default=None, repr=False)

    def _get(self, item: RouterInput) -> tuple[PageEmbedding, QuestionEmbedding]:
        if self._pages is None:
            self._pages = load_page_embeddings(self.path)
            self._questions = load_question_embeddings(self.question_path)
        assert self._questions is not None
        if item.doc_id not in self._pages or item.question_id not in self._questions:
            raise KeyError(
                f"{item.question_id}: no page or question embedding in {self.path} / "
                f"{self.question_path} (run qdu router embed)"
            )
        return self._pages[item.doc_id], self._questions[item.question_id]

    def __call__(self, item: RouterInput) -> dict[str, float]:
        page, question = self._get(item)
        return {**{f"e{i}": float(v) for i, v in enumerate(page.vector)}, "qsim": question.qsim}

    def cost(self, item: RouterInput) -> tuple[float, float]:
        page, question = self._get(item)
        return question.qsim_s, page.latency_s
