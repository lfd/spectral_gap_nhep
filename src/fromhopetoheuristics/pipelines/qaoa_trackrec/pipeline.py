from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_track_reconstruction_qaoa,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                run_track_reconstruction_qaoa,
                {
                    "qubos": "qubos",
                    "event_path": "event_path",
                    "seed": "params:seed",
                    "max_p": "params:max_p",
                    "q": "params:q",
                },
                {},
            ),
        ]
    )
