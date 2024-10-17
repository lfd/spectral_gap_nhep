from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_metadata, create_qallse_datawrapper


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_metadata,
                {
                    "result_path_prefix": "params:output_path",
                    "trackml_input_path": "params:trackml_input_path",
                    "num_angle_parts": "params:num_angle_parts",
                    "seed": "params:seed",
                    "f": "params:data_fraction",
                },
                {
                    "metadata": "metadata",
                    "event_path": "event_path",
                },
            ),
            node(
                create_qallse_datawrapper,
                {
                    "event_path": "event_path",
                },
                {
                    "data_wrapper": "data_wrapper",
                },
            ),
        ]
    )
