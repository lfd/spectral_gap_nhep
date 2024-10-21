from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_maxcut_annealing,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                run_maxcut_annealing,
                {
                    "seed": "params:seed",
                    "num_anneal_fractions": "params:num_anneal_fractions",
                    "maxcut_max_qubits": "params:maxcut_max_qubits",
                },
                {"results": "adiabatic_maxcut_results"},
            )
        ]
    )
