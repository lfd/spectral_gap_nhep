from kedro.pipeline import Pipeline, node, pipeline

from .nodes import visualize


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                visualize,
                {
                    "response": "response",
                    "event_path": "event_path",
                    "output_path": "params:output_path",
                    "prefix": "params:prefix",
                },
                {
                    "plot_doublets": "plot_doublets",
                    "plot_triplets": "plot_triplets",
                },
            )
        ]
    )
