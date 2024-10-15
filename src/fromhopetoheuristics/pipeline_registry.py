"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from fromhopetoheuristics.pipelines.generation.pipeline import (
    create_pipeline as create_generation_pipeline,
)
from fromhopetoheuristics.pipelines.science.pipeline import (
    create_pipeline as create_science_pipeline,
)
from fromhopetoheuristics.pipelines.visualization.pipeline import (
    create_pipeline as create_visualization_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    pipelines["__default__"] = (
        create_generation_pipeline()
        + create_science_pipeline()
        # + create_visualization_pipeline() # FIXME
    )
    return pipelines
