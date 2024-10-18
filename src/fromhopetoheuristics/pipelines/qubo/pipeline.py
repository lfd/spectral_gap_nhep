from kedro.pipeline import Pipeline, node, pipeline
from kedro.config import OmegaConfigLoader  # noqa: E402

from .nodes import (
    build_qubos,
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
                {"qubos": "qubos"},
            ),
            node(
                solve_qubos,
                {
                    "qubos": "qubos",
                    "seed": "params:seed",
                },
                {
                    "responses": "responses",
                },
            ),
        ]
    )
