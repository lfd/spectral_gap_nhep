from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_anneal_schedule


def create_maxcut_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_anneal_schedule,
                {
                    "results": "qaoa_maxcut_results",
                },
                {"results": "anneal_schedule_maxcut_results"},
            ),
        ]
    )


def create_trackrecon_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_anneal_schedule,
                {
                    "results": "qaoa_track_reconstruction_results",
                },
                {"results": "anneal_schedule_track_reconstruction_results"},
            ),
        ]
    )
