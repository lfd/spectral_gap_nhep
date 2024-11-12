from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_track_reconstruction_qaoa


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                run_track_reconstruction_qaoa,
                {
                    "qubos": "qubos",
                    "seed": "params:seed",
                    "max_p": "params:max_p",
                    "q": "params:q",
                    "optimiser": "params:optimiser",
                    "tolerance": "params:tolerance",
                    "maxiter": "params:maxiter",
                    "geometric_index": "params:geometric_index",
                    "apply_bounds": "params:apply_bounds",
                    "initialisation": "params:initialisation",
                    "options": "params:options",
                    "hyperhyper_trial_id": "params:hyperhyper_trial_id",
                },
                {"results": "qaoa_track_reconstruction_results"},
            ),
        ]
    )
