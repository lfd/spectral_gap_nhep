from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_track_reconstruction_annealing,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                run_track_reconstruction_annealing,
                {
                    "qubos": "qubos",
                    "num_anneal_fractions": "params:num_anneal_fractions",
                },
                {
                    "results": "adiabatic_track_reconstruction_results",
                },
            ),
        ]
    )
