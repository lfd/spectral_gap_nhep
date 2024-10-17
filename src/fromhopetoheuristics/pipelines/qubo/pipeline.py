from kedro.pipeline import Pipeline, node, pipeline
from kedro.config import OmegaConfigLoader  # noqa: E402

from .nodes import (
    build_qubos,
    load_qubos,
    solve_qubos,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                build_qubos,
                {
                    "data_wrapper": "data_wrapper",
                    "event_path": "event_path",
                    "num_angle_parts": "params:num_angle_parts",
                },
                {"qubo_paths": "qubo_paths"},
            ),
            node(
                load_qubos,
                {"qubo_paths": "qubo_paths"},
                {"qubos": "qubos"},
            ),
            node(
                solve_qubos,
                {
                    "data_wrapper": "data_wrapper",
                    "qubo_paths": "qubo_paths",
                    "event_path": "event_path",
                    "seed": "params:seed",
                },
                {
                    "responses": "responses",
                },
            ),
        ]
    )
