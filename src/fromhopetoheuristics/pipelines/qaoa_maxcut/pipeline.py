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
                    "seed": "params:seed",
                    "max_p": "params:max_p",
                    "q": "params:q",
                    "maxcut_n_qubits": "params:maxcut_n_qubits",
                    "maxcut_graph_density": "params:maxcut_graph_density",
                    "optimiser": "params:optimiser",
                    "tolerance": "params:tolerance",
                    "maxiter": "params:maxiter",
                    "apply_bounds": "params:apply_bounds",
                    "initialisation": "params:initialisation",
                    "options": "params:options",
                },
                {"results": "qaoa_maxcut_results"},
            )
        ]
    )
