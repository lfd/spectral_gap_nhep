"""Project pipelines."""

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from fromhopetoheuristics.pipelines.qubo.pipeline import (
    create_pipeline as create_qubo_pipeline,
)
from fromhopetoheuristics.pipelines.adiabatic_maxcut.pipeline import (
    create_pipeline as create_adiabatic_maxcut_pipeline,
)
from fromhopetoheuristics.pipelines.qaoa_maxcut.pipeline import (
    create_pipeline as create_qaoa_maxcut_pipeline,
)
from fromhopetoheuristics.pipelines.adiabatic_trackrec.pipeline import (
    create_pipeline as create_adiabatic_trackrec_pipeline,
)
from fromhopetoheuristics.pipelines.qaoa_trackrec.pipeline import (
    create_pipeline as create_qaoa_trackrec_pipeline,
)
from fromhopetoheuristics.pipelines.generation.pipeline import (
    create_pipeline as create_generation_pipeline,
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

    pipelines["qaoa_maxcut"] = create_qaoa_maxcut_pipeline()
    pipelines["adiabatic_maxcut"] = create_adiabatic_maxcut_pipeline()
    pipelines["maxcut"] = (
        create_qaoa_maxcut_pipeline() + create_adiabatic_maxcut_pipeline()
    )

    pipelines["qubo"] = create_generation_pipeline() + create_qubo_pipeline()
    pipelines["qaoa_trackrec"] = (
        create_generation_pipeline()
        + create_qubo_pipeline()
        + create_qaoa_trackrec_pipeline()
    )
    pipelines["adiabatic_trackrec"] = (
        create_generation_pipeline()
        + create_qubo_pipeline()
        + create_adiabatic_trackrec_pipeline()
    )
    pipelines["trackrec"] = (
        create_generation_pipeline()
        + create_qubo_pipeline()
        + create_qaoa_trackrec_pipeline()
        + create_adiabatic_trackrec_pipeline()
        # + create_visualization_pipeline() # FIXME
    )
    pipelines["__default__"] = pipelines["trackrec"]
    return pipelines
