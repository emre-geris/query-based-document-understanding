"""One user's demo session: the open page, the answers so far and the presentation state.

Independent of Gradio (`qdu.demo.app` only wires buttons to these methods), so the behaviour
is tested without a browser. The heavy parts (router, models, OCR) live in the shared
`DemoService`; a session holds only its own page and answers.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qdu.demo import render
from qdu.demo.examples import Example
from qdu.demo.service import DemoAnswer, DemoService
from qdu.pipelines import PipelineOutput

if TYPE_CHECKING:
    from PIL.Image import Image

MODE_OF = {"Otomatik": "auto", "Metin": "text", "Görüntü": "vision", "Hibrit": "hybrid"}
DATASET_NAME = {
    "docvqa": "DocVQA",
    "infographicvqa": "InfographicVQA",
    "chartqa": "ChartQA",
    "tatdqa": "TAT-DQA",
}


@dataclass
class Page:
    page_id: str
    name: str
    info: str
    image: "Image"
    examples: list[Example] = field(default_factory=list)
    """For an example page: its showcase questions (gold answers known)."""

    @property
    def is_example(self) -> bool:
        return bool(self.examples)


def page_label(page: Sequence[Example]) -> str:
    first = page[0]
    return f"{DATASET_NAME.get(first.dataset, first.dataset)} · {first.doc_id}"


class DemoSession:
    def __init__(
        self,
        service: DemoService,
        showcase: Sequence[Sequence[Example]],
        load_image: Callable[[Example], "Image"],
        recorded: Mapping[str, Mapping[str, PipelineOutput]] | None = None,
    ) -> None:
        self.service = service
        self.showcase = [list(p) for p in showcase]
        self.load_image = load_image
        self.recorded = recorded or {}
        self.page: Page | None = None
        self.answers: list[DemoAnswer] = []
        self.selected: int | None = None
        self.scenario = "C"
        self.present = False
        self.step = 0

    # pages ---------------------------------------------------------------------------------

    def example_choices(self) -> list[str]:
        return [page_label(p) for p in self.showcase]

    def _open(self, page: Page) -> None:
        self.service.forget(page.page_id)  # a reopened page starts fresh (scenario B count)
        self.page = page
        self.answers, self.selected = [], None

    def open_example(self, index: int) -> None:
        examples = self.showcase[index]
        image = self.load_image(examples[0])
        w, h = image.size
        self._open(
            Page(
                examples[0].doc_id,
                page_label(examples),
                f"{w}x{h} piksel · val kümesinden · metin okuma ve sayfa düzeni bu sayfa için "
                "bir kez yapılır",
                image,
                list(examples),
            )
        )

    def open_upload(self, image: "Image", name: str = "yüklenen sayfa") -> None:
        image = image.convert("RGB")
        digest = hashlib.sha1(image.tobytes()).hexdigest()[:12]
        w, h = image.size
        self._open(
            Page(
                f"upload/{digest}",
                name,
                f"{w}x{h} piksel · doğru cevap bilinmiyor, puan gösterilmez",
                image,
            )
        )

    def question_choices(self) -> list[str]:
        return [e.record.question for e in self.page.examples] if self.page else []

    # asking --------------------------------------------------------------------------------

    def ask(self, question: str, mode: str = "Otomatik", compare: bool = True) -> DemoAnswer:
        if self.page is None:
            raise ValueError("önce bir sayfa açın ya da yükleyin")
        question = question.strip()
        if not question:
            raise ValueError("soru boş")
        example = next((e for e in self.page.examples if e.record.question == question), None)
        kwargs = {}
        if example is not None:  # gold answers, and the run's recorded answers in replay mode
            kwargs = {
                "answers": example.record.answers,
                "dataset": example.dataset,
                "recorded": self.recorded.get(example.question_id),
            }
        answer = self.service.ask(
            question,
            self.page.image,
            self.page.page_id,
            MODE_OF.get(mode, mode),
            compare,
            **kwargs,
        )
        self.answers.append(answer)
        self.selected = len(self.answers) - 1
        return answer

    def select(self, index: int) -> None:
        if 0 <= index < len(self.answers):
            self.selected = index

    def set_scenario(self, scenario: str) -> None:
        self.scenario = "B" if scenario.startswith("B") else "C"

    # presentation --------------------------------------------------------------------------

    def start_presentation(self, index: int) -> None:
        """Open a showcase page and answer its first question (router, all paths compared)."""
        self.open_example(index)
        self.present, self.step = True, 0
        self.ask(self.page.examples[0].record.question, "Otomatik", compare=True)

    def next_step(self) -> None:
        if not self.present or self.page is None:
            return
        if self.step + 1 < len(self.page.examples):
            self.step += 1
            self.ask(self.page.examples[self.step].record.question, "Otomatik", compare=True)

    def previous_step(self) -> None:
        if self.present and self.step > 0:
            self.step -= 1
            self.selected = self.step

    def stop_presentation(self) -> None:
        self.present = False

    @property
    def at_last_step(self) -> bool:
        return self.page is not None and self.step + 1 >= len(self.page.examples)

    # views ---------------------------------------------------------------------------------

    def header_html(self) -> str:
        if self.page is None:
            return render.page_header_html("Sayfa seçilmedi", "Bir örnek açın ya da yükleyin.",
                                           False, False)  # fmt: skip
        recorded = self.page.is_example and bool(self.recorded)
        return render.page_header_html(
            self.page.name, self.page.info, self.page.is_example, recorded
        )

    def chat_html(self) -> str:
        return render.chat_html(self.shown(), self.selected, self.present)

    def panel_html(self) -> str:
        a = self.answers[self.selected] if self.selected is not None else None
        return render.panel_html(a, self.scenario, self.present)

    def story(self) -> tuple[str, str, int, int] | None:
        """The presentation step's title, text, number and count; None outside a presentation."""
        if not self.present or not self.answers or self.page is None:
            return None
        n, of = self.step + 1, len(self.page.examples)
        title, text = render.step_story(self.answers[self.step], n, of)
        if self.at_last_step:
            text += (
                " Bu sayfa yönlendiricinin nerede kazandırdığını ve nerede yanıldığını gösteriyor; "
                "genel başarı val ve test tablolarındadır."
            )
        return title, text, n, of

    def story_html(self) -> str:
        story = self.story()
        return render.story_html(*story) if story else ""

    def shown(self) -> list[DemoAnswer]:
        """The answers on screen: in a presentation, only those of the steps reached."""
        return self.answers[: self.step + 1] if self.present else self.answers

    def reset(self) -> None:
        """A new chat: no page, no answers."""
        self.page, self.answers, self.selected = None, [], None
        self.present, self.step = False, 0

    def answer_choices(self) -> list[str]:
        return [f"{i + 1}. cevap" for i in range(len(self.shown()))]
