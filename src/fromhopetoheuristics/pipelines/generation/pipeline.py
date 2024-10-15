from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_metadata


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_metadata,
                {
                    "output_path": "params:output_path",
                    "prefix": "params:prefix",
                    "seed": "params:seed",
                },
                {
                    "metadata": "metadata",
                    "event_path": "event_path",
                },
            )
        ]
    )
