from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_anneal_schedule,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_anneal_schedule,
                {
                    "qaoa_result_file": "params:qaoa_result_file",
                    "q": "params:q",
                    "max_p": "params:max_p",
                    "num_angle_parts": "params:num_angle_parts",
                    "maxcut_max_qubits": "params:maxcut_max_qubits",
                },
                {},
            ),
        ]
    )
