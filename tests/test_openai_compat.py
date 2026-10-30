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


def test_large_pages_are_downscaled_keeping_aspect_ratio() -> None:
    from qdu.models.openai_compat import limit_pixels

    tall = Image.new("RGB", (1000, 8000))  # 8 MP infographic
    small = limit_pixels(tall, 2_000_000)
    assert small.size[0] * small.size[1] <= 2_000_000
    assert abs(small.size[1] / small.size[0] - 8.0) < 0.05
    page = Image.new("RGB", (800, 1000))
    assert limit_pixels(page, 2_000_000) is page  # small pages are untouched


def test_image_cap_comes_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("QDU_MAX_IMAGE_PIXELS", "123456")
    assert OpenAICompatModel("m", client=RecordingClient()).max_image_pixels == 123456


def test_describe_records_settings_even_without_a_server(monkeypatch) -> None:
    monkeypatch.delenv("QDU_MAX_IMAGE_PIXELS", raising=False)
    monkeypatch.setenv("QDU_SERVER_ARGS", "--no-enable-prefix-caching")
    model = OpenAICompatModel("m", base_url="http://127.0.0.1:9/v1", client=RecordingClient())
    info = model.describe()
    assert info["thinking_disabled"] is True
    assert info["max_image_pixels"] == 16_777_216
    assert info["server_args"] == "--no-enable-prefix-caching"
    assert info["server_version"] is None  # unreachable server is recorded, not fatal
