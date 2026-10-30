"""Prompt templates and answer post-processing shared by all pipelines.

DocVQA answers are short extractive spans, so every prompt asks for the shortest answer
using the document's own words. Keeping the instruction identical across pipelines means
differences in score come from the input modality, not from the wording.
"""

import re

ANSWER_INSTRUCTION = (
    "Answer with the exact words from the document whenever possible, as briefly as "
    "possible (a word, number or short phrase). Do not explain."
)

TEXT_PROMPT = """You are answering a question about a document page.
The page text below was extracted with OCR and may contain recognition errors.

OCR text:
<<<
{ocr_text}
>>>

Question: {question}
{instruction}"""

VISION_PROMPT = """You are answering a question about the attached document page image.

Question: {question}
{instruction}"""

HYBRID_PROMPT = """You are answering a question about the attached document page image.
The OCR text of the same page is given below. It may contain recognition errors or miss
content; when it disagrees with the image, trust the image.

OCR text:
<<<
{ocr_text}
>>>

Question: {question}
{instruction}"""

_PREFIX = re.compile(r"^\s*(final answer|answer)\s*[:\-]\s*", re.IGNORECASE)
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"  # ASCII and typographic quotes


def text_prompt(question: str, ocr_text: str) -> str:
    return TEXT_PROMPT.format(
        ocr_text=ocr_text.strip(), question=question, instruction=ANSWER_INSTRUCTION
    )


def vision_prompt(question: str) -> str:
    return VISION_PROMPT.format(question=question, instruction=ANSWER_INSTRUCTION)


def hybrid_prompt(question: str, ocr_text: str) -> str:
    return HYBRID_PROMPT.format(
        ocr_text=ocr_text.strip(), question=question, instruction=ANSWER_INSTRUCTION
    )


# A space (or no-break / thin space) between a digit and a following group of exactly three
# digits is a thousands separator: "140 858" -> "140858", "1 413.68" -> "1413.68".
_THOUSANDS_SPACE = re.compile(r"(?<=\d)[ \u00a0\u2009\u202f](?=\d{3}\b)")


def clean_answer(raw: str) -> str:
    """Normalise a raw model output to the answer string that is scored.

    First non-empty line, without an 'Answer:' prefix, wrapping quotes or a final period, and
    with space thousands separators removed. The last rule matters for the hybrid path: given
    OCR text such as "140 858" (as printed on Statista charts) the model copies that spacing,
    which ANLS and relaxed accuracy would otherwise score as wrong.
    """
    line = next((ln for ln in raw.strip().splitlines() if ln.strip()), "")
    line = _PREFIX.sub("", line).strip().strip(_QUOTES).strip()
    line = _THOUSANDS_SPACE.sub("", line)
    return line.removesuffix(".").rstrip()
