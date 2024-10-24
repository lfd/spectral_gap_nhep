from kedro.pipeline import Pipeline, node, pipeline
from kedro.config import OmegaConfigLoader  # noqa: E402

from .nodes import (
    build_qubos,
    solve_qubos,
    create_metadata,
    create_qallse_datawrapper,
)


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_metadata,
                {
                    "seed": "params:seed",
                    "f": "params:data_fraction",
                    "event_hits": "event_hits",
                    "event_particles": "event_particles",
                    "event_truth": "event_truth",
                },
                {
                    "hits": "hits",
                    "truth": "truth",
                    "particles": "particles",
                    "doublets": "doublets",
                    "metadata": "metadata",
                },
            ),
            node(
                create_qallse_datawrapper,
                {
                    "hits": "hits",
                    "truth": "truth",
                },
                {
                    "data_wrapper": "data_wrapper",
                },
            ),
            node(
                build_qubos,
                {
                    "data_wrapper": "data_wrapper",
                    "doublets": "doublets",
                    "num_angle_parts": "params:num_angle_parts",
                    "geometric_index": "params:geometric_index",
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
