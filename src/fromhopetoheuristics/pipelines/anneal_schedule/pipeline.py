from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_maxcut_anneal_schedule, create_trackrecon_anneal_schedule


def create_maxcut_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_maxcut_anneal_schedule,
                {
                    "results": "qaoa_maxcut_results",
                    "maxcut_max_qubits": "params:maxcut_max_qubits",
                },
                {"results": "anneal_schedule_maxcut_results"},
            ),
        ]
    )


def create_trackrecon_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_trackrecon_anneal_schedule,
                {
                    "results": "qaoa_track_reconstruction_results",
                    "num_angle_parts": "params:num_angle_parts",
                },
                {"results": "anneal_schedule_track_reconstruction_results"},
            ),
        ]
    )
