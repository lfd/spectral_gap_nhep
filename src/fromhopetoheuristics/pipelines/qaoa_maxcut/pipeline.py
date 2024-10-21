from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_maxcut_qaoa,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                run_maxcut_qaoa,
                {
                    "result_path_prefix": "params:output_path",
                    "seed": "params:seed",
                    "max_p": "params:max_p",
                    "q": "params:q",
                    "maxcut_max_qubits": "params:maxcut_max_qubits",
                    "optimiser": "params:optimiser",
                },
                {"qaoa_solution_path": "params:qaoa_result_file"},
            )
        ]
    )
