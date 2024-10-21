from kedro.pipeline import Pipeline, node, pipeline

from .nodes import visualize


def create_pipeline() -> Pipeline:

    return pipeline(
        [
            node(
                visualize,
                {
                    "responses": "responses",
                    "data_wrapper": "data_wrapper",
                },
                {
                    "figures": "figures",
                },
            )
        ]
    )
