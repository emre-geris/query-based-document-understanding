"""Model clients behind a provider-independent interface."""

from qdu.models.base import Generation, LanguageModel
from qdu.models.fake import FakeModel
from qdu.models.registry import create_model

__all__ = ["FakeModel", "Generation", "LanguageModel", "create_model"]
