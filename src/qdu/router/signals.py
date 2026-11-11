"""Router signals S1-S4 (CLAUDE.md, Phase 3), as feature extractors.

- **S1** question (`QuestionSignals`, no OCR, no layout): length, wh-word, cue words for numbers,
  graphics, layout and comparison. A TF-IDF of the question is not an extractor: its vocabulary
  must be fitted on the training split only, so it belongs to the model (Phase 4).
- **S2** OCR text (`OcrSignals`): how much text the page has and how confidently it was read.
- **S3** page (`PageSignals`, layout only): page size and the area shares of text, table,
  figure and formula regions (PP-DocLayout, `qdu.layout`).
- **S4** evidence location (`EvidenceSignals`, OCR + layout): which OCR lines contain the
  question's terms, how confidently they were read, and whether they sit in text, a table or a
  figure. The idea: a question whose terms are found in confidently read text is answerable
  from OCR (cheap text path); one whose terms are missing or sit in figures needs the image.

Each extractor reads only the question, the run's OCR and the page layout (`RouterInput`).
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from qdu.layout import PageLayout, Region
from qdu.router.features import RouterInput

# PP-DocLayout labels grouped into the region kinds the signals use.
REGION_KIND = {
    **dict.fromkeys(
        (
            "text", "paragraph_title", "doc_title", "abstract", "content", "reference",
            "footnote", "header", "footer", "aside_text", "number", "algorithm",
            "reference_content", "vision_footnote",
        ),
        "text",
    ),
    **dict.fromkeys(("table", "table_title"), "table"),
    **dict.fromkeys(
        (
            "image", "chart", "figure_title", "chart_title", "header_image", "footer_image",
            "seal",
        ),
        "figure",
    ),
    **dict.fromkeys(("formula", "formula_number"), "formula"),
}  # fmt: skip
KINDS = ("text", "table", "figure", "formula", "other")

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = frozenset(
    (
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and",
        "or", "is", "are", "was", "were", "be", "been", "what", "which", "who", "whom",
        "whose", "when", "where", "why", "how", "many", "much", "does", "do", "did", "this",
        "that", "these", "those", "it", "its", "as", "per", "given", "shown", "mentioned",
        "according", "name", "number", "value", "total", "there", "their", "than", "into",
    )
)  # fmt: skip
_WH = ("what", "which", "who", "when", "where", "how many", "how much", "how", "why")
_CUES = {
    "numeric": r"how many|how much|total|sum|difference|average|mean|percent|percentage|%|ratio"
    r"|number of|amount|value|rate|increase|decrease|change",
    "visual": r"colou?r|chart|graph|bar|line|pie|figure|image|logo|icon|map|shown|picture|symbol",
    "layout": r"table|column|row|left|right|top|bottom|above|below|first|last|section|page"
    r"|heading|title|header|footer",
    "compare": r"more|less|higher|lower|highest|lowest|largest|smallest|most|least|compared"
    r"|greater|fewer|better|worse|maximum|minimum",
    "year": r"(?:19|20)\d\d",
}
_CUE = {k: re.compile(rf"\b(?:{v})\b") for k, v in _CUES.items()}


def kind(label: str) -> str:
    return REGION_KIND.get(label, "other")


def terms(text: str) -> set[str]:
    """Content words and numbers of ``text``, lowercased (for matching question and OCR)."""
    return {
        w.replace(",", "")
        for w in _WORD.findall(text.lower())
        if w not in _STOP and (len(w) >= 3 or w.isdigit())
    }


@dataclass(frozen=True)
class QuestionSignals:
    """S1: the question alone (no OCR, no layout)."""

    name: str = "s1"
    needs_ocr: bool = False
    needs_layout: bool = False

    def __call__(self, item: RouterInput) -> dict[str, float]:
        text = item.question.lower().strip()
        wh = next((w for w in _WH if text.startswith(w) or f" {w} " in text), "other")
        words = text.split()
        return {
            "n_words": len(words),
            "n_terms": len(terms(text)),
            "has_digit": any(c.isdigit() for c in text),
            "has_quote": any(q in text for q in "\"'\u201c\u201d\u2018\u2019"),
            **{f"cue_{k}": bool(p.search(text)) for k, p in _CUE.items()},
            **{f"wh_{w.replace(' ', '_')}": wh == w for w in (*_WH, "other")},
        }


@dataclass(frozen=True)
class OcrSignals:
    """S2: the page's OCR text: amount, density and recognition confidence."""

    name: str = "s2"
    needs_ocr: bool = True
    needs_layout: bool = False

    def __call__(self, item: RouterInput) -> dict[str, float]:
        ocr = item.ocr
        lines = ocr.lines or ()
        scores = [line.score for line in lines]
        words = ocr.text.split()
        numeric = sum(any(c.isdigit() for c in w) for w in words)
        width, height = ocr.page_size or (1, 1)
        heights = [line.box[3] - line.box[1] for line in lines]
        return {
            "n_lines": len(lines),
            "log_chars": math.log1p(len(ocr.text)),
            "log_words": math.log1p(len(words)),
            "words_per_mpx": len(words) / (width * height / 1e6),
            "numeric_share": numeric / len(words) if words else 0.0,
            "mean_conf": sum(scores) / len(scores) if scores else 0.0,
            "min_conf": min(scores, default=0.0),
            "low_conf_share": sum(s < 0.8 for s in scores) / len(scores) if scores else 0.0,
            "very_low_conf_share": sum(s < 0.5 for s in scores) / len(scores) if scores else 0.0,
            "median_line_height": sorted(heights)[len(heights) // 2] / height if heights else 0.0,
            "empty": not words,
        }


def _shares(regions: Iterable[Region], area: float) -> dict[str, float]:
    covered = dict.fromkeys(KINDS, 0.0)
    for r in regions:
        covered[kind(r.label)] += r.area
    return {k: min(1.0, v / area) for k, v in covered.items()}


@dataclass(frozen=True)
class PageSignals:
    """S3: page size and what its regions are made of (layout, no OCR)."""

    name: str = "s3"
    needs_ocr: bool = False
    needs_layout: bool = True

    def __call__(self, item: RouterInput) -> dict[str, float]:
        layout = item.layout
        width, height = layout.page_size
        area = float(width * height)
        counts = dict.fromkeys(KINDS, 0)
        for r in layout.regions:
            counts[kind(r.label)] += 1
        largest = max((r.area for r in layout.regions), default=0.0) / area
        return {
            "log_mpx": math.log(area / 1e6),
            "aspect": height / width,
            "n_regions": len(layout.regions),
            **{f"n_{k}": v for k, v in counts.items()},
            **{f"share_{k}": v for k, v in _shares(layout.regions, area).items()},
            "largest_region_share": min(1.0, largest),
        }


def _region_kind_at(layout: PageLayout, box: tuple[int, int, int, int]) -> str:
    """Kind of the smallest region containing the centre of ``box`` ("none" if outside all)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    inside = [
        r for r in layout.regions if r.box[0] <= cx <= r.box[2] and r.box[1] <= cy <= r.box[3]
    ]
    return kind(min(inside, key=lambda r: r.area).label) if inside else "none"


@dataclass(frozen=True)
class EvidenceSignals:
    """S4: where the question's terms appear in the OCR, how well read, in which region kind."""

    name: str = "s4"
    needs_ocr: bool = True
    needs_layout: bool = True

    def __call__(self, item: RouterInput) -> dict[str, float]:
        q = terms(item.question)
        lines = item.ocr.lines or ()
        line_terms = [terms(line.text) for line in lines]
        found = set().union(*line_terms) & q if lines else set()
        hits = [(line, len(t & q)) for line, t in zip(lines, line_terms, strict=True) if t & q]
        best = max((n for _, n in hits), default=0)
        where = dict.fromkeys((*KINDS, "none"), 0.0)
        for line, n in hits:
            where[_region_kind_at(item.layout, line.box)] += n
        total = sum(where.values())
        scores = [line.score for line, _ in hits]
        numbers = {t for t in q if any(c.isdigit() for c in t)}
        return {
            "match_share": len(found) / len(q) if q else 0.0,
            "n_matched_lines": len(hits),
            "best_line_share": best / len(q) if q else 0.0,
            "matched_mean_conf": sum(scores) / len(scores) if scores else 0.0,
            "matched_min_conf": min(scores, default=0.0),
            "numbers_found": len(numbers & found) / len(numbers) if numbers else -1.0,
            **{f"in_{k}": v / total if total else 0.0 for k, v in where.items()},
        }
