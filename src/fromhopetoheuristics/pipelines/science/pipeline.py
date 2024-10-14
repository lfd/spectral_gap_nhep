from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_qubo, solve_qubo


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                build_qubo,
                {
                    "event_path": "event_path",
                    "output_path": "params:output_path",
                    "prefix": "params:prefix",
                },
                {"qubo_path": "qubo_path"},
            ),
            node(
                solve_qubo,
                {
                    "event_path": "event_path",
                    "qubo_path": "qubo_path",
                    "output_path": "params:output_path",
                    "prefix": "params:prefix",
                    "seed": "params:seed",
                },
                {
                    "response": "response",
                },
            ),
        ]
    )
