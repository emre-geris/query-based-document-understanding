"""Build models from short specs.

- ``fake``: deterministic stand-in for tests and wiring checks.
- ``openai:<model>``: any OpenAI-compatible endpoint, e.g. vLLM serving
  ``openai:Qwen/Qwen3.5-4B`` (see :mod:`qdu.models.openai_compat`).

Every provider implements :class:`qdu.models.base.LanguageModel` and reports token usage.
"""

from qdu.models.base import LanguageModel
from qdu.models.fake import FakeModel

SUPPORTED = ("fake", "openai:<model>")


def create_model(spec: str) -> LanguageModel:
    if spec == "fake":
        return FakeModel()
    if spec.startswith("openai:") and len(spec) > len("openai:"):
        from qdu.models.openai_compat import OpenAICompatModel

        return OpenAICompatModel(spec.removeprefix("openai:"))
    raise ValueError(
        f"unknown model spec {spec!r}; supported: {', '.join(SUPPORTED)}. "
        "Add a provider in src/qdu/models/ and register it here."
    )
