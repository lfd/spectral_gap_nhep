from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_qubos,
    load_qubos,
    solve_qubos,
    run_maxcut_annealing,
    run_maxcut_qaoa,
    run_track_reconstruction_annealing,
    run_track_reconstruction_qaoa,
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
                    "qubos": "qubos",
                    "result_path_prefix": "params:output_path",
                    "seed": "params:seed",
                },
                {
                    "responses": "responses",
                },
            ),
            node(
                run_maxcut_annealing,
                {
                    "seed": "params:seed",
                },
                {},
            ),
            node(
                run_maxcut_qaoa,
                {
                    "seed": "params:seed",
                },
                {},
            ),
            node(
                run_track_reconstruction_annealing,
                {
                    "qubos": "qubos",
                    "event_path": "event_path",
                    "seed": "params:seed",
                },
                {},
            ),
            node(
                run_track_reconstruction_qaoa,
                {
                    "qubos": "qubos",
                    "event_path": "event_path",
                    "seed": "params:seed",
                    "max_p": "params:max_p",
                },
                {},
            ),
        ]
    )
