import json
from pathlib import Path

import pytest
from PIL import Image

from qdu.cli import build_parser, main


def test_answer_with_vision_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = tmp_path / "page.png"
    Image.new("RGB", (40, 20), "white").save(image)
    code = main(
        ["answer", "--image", str(image), "--question", "Total?", "--pipeline", "vision", "--json"]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["pipeline"] == "vision"
    assert out["model"] == "fake"


def test_benchmark_commands_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(["benchmark", "run", "--text-model", "a", "--vision-model", "b"])
    assert args.pipelines == "text,vision,hybrid"
    assert args.manifest == Path("benchmarks/pilot_300.csv")
    args = parser.parse_args(["benchmark", "report", "runs/x", "--tolerance", "0.05"])
    assert args.tolerance == 0.05


def test_unknown_pipeline_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["answer", "--image", "x.png", "--question", "q", "--pipeline", "audio"]
        )
