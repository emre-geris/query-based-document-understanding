from types import SimpleNamespace

from PIL import Image

from qdu.models import create_model
from qdu.models.openai_compat import OpenAICompatModel


class RecordingClient:
    """Stands in for openai.OpenAI: records the request, returns a canned completion."""

    def __init__(self, text: str = "1975") -> None:
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.text = text

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            usage=SimpleNamespace(prompt_tokens=812, completion_tokens=3),
        )


def test_text_request_is_greedy_short_and_thinking_disabled() -> None:
    client = RecordingClient()
    model = OpenAICompatModel("Qwen/Qwen3.5-4B", client=client)
    out = model.generate("What year?")
    request = client.requests[0]
    assert request["model"] == "Qwen/Qwen3.5-4B"
    assert request["messages"] == [{"role": "user", "content": "What year?"}]
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 64
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert (out.text, out.input_tokens, out.output_tokens) == ("1975", 812, 3)
    assert out.latency_s >= 0


def test_image_request_sends_png_data_url_before_the_text() -> None:
    client = RecordingClient()
    OpenAICompatModel("m", client=client).generate("Read it", [Image.new("RGB", (4, 4))])
    content = client.requests[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "Read it"}


def test_registry_builds_openai_models() -> None:
    model = create_model("openai:Qwen/Qwen3.5-4B")
    assert model.name == "Qwen/Qwen3.5-4B"
    assert model.supports_images
