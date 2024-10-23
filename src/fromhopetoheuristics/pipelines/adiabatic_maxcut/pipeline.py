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
                    "maxcut_n_qubits": "params:maxcut_n_qubits",
                    "maxcut_graph_density": "params:maxcut_graph_density",
                },
                {"results": "adiabatic_maxcut_results"},
            )
        ]
    )
