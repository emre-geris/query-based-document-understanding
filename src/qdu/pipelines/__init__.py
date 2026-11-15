"""The three answering pipelines a router chooses between."""

from qdu.pipelines.base import Pipeline, PipelineInput, PipelineName, PipelineOutput
from qdu.pipelines.factory import ALL_PIPELINES, PIPELINE_NAMES, build_pipeline
from qdu.pipelines.hybrid import HybridPipeline
from qdu.pipelines.text import TextPipeline
from qdu.pipelines.vision import VisionPipeline

__all__ = [
    "ALL_PIPELINES",
    "PIPELINE_NAMES",
    "HybridPipeline",
    "Pipeline",
    "PipelineInput",
    "PipelineName",
    "PipelineOutput",
    "TextPipeline",
    "VisionPipeline",
    "build_pipeline",
]
