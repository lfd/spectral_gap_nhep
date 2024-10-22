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
from fromhopetoheuristics.pipelines.anneal_schedule.pipeline import (
    create_maxcut_pipeline as create_maxcut_anneal_schedule_pipeline,
)
from fromhopetoheuristics.pipelines.anneal_schedule.pipeline import (
    create_trackrecon_pipeline as create_trackrecon_anneal_schedule_pipeline,
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

    pipelines["qaoa_maxcut"] = (
        create_qaoa_maxcut_pipeline() + create_maxcut_anneal_schedule_pipeline()
    )
    pipelines["adiabatic_maxcut"] = create_adiabatic_maxcut_pipeline()

    pipelines["maxcut"] = pipelines["qaoa_maxcut"] + pipelines["adiabatic_maxcut"]

    pipelines["qubo"] = create_generation_pipeline() + create_qubo_pipeline()
    pipelines["qaoa_trackrec"] = pipelines["qubo"] + create_qaoa_trackrec_pipeline()
    pipelines["adiabatic_trackrec"] = (
        pipelines["qubo"] + create_adiabatic_trackrec_pipeline()
    )
    pipelines["anneal_schedule"] = (
        create_maxcut_anneal_schedule_pipeline()
        + create_trackrecon_anneal_schedule_pipeline()
    )

    pipelines["trackrec"] = (
        pipelines["qaoa_trackrec"]
        + pipelines["adiabatic_trackrec"]
        + pipelines["anneal_schedule"]
        + create_visualization_pipeline()
    )
    pipelines["__default__"] = pipelines["trackrec"] + pipelines["maxcut"]
    return pipelines
