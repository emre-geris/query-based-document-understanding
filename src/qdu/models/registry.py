"""Build models from short specs such as ``"fake"``.

Real providers are added here once the team settles on model access. Each provider module
should implement :class:`qdu.models.base.LanguageModel` and report token usage.
"""

from qdu.models.base import LanguageModel
from qdu.models.fake import FakeModel

SUPPORTED = ("fake",)


def create_model(spec: str) -> LanguageModel:
    if spec == "fake":
        return FakeModel()
    raise ValueError(
        f"unknown model spec {spec!r}; supported: {', '.join(SUPPORTED)}. "
        "Add a provider in src/qdu/models/ and register it here."
    )
