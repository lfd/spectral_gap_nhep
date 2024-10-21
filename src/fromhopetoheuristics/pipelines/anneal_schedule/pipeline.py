from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_maxcut_anneal_schedule, create_trackrecon_anneal_schedule


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_maxcut_anneal_schedule,
                {
                    "q": "params:q",
                    "max_p": "params:max_p",
                    "maxcut_max_qubits": "params:maxcut_max_qubits",
                },
                {},
            ),
            node(
                create_trackrecon_anneal_schedule,
                {
                    "q": "params:q",
                    "max_p": "params:max_p",
                    "num_angle_parts": "params:num_angle_parts",
                },
                {},
            ),
        ]
    )
