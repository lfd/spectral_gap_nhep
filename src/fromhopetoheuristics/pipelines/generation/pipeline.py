from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_metadata, create_qallse_datawrapper


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            node(
                create_metadata,
                {
                    "seed": "params:seed",
                    "num_angle_parts": "params:num_angle_parts",
                    "f": "params:data_fraction",
                    "event-cells": "event-cells",
                    "event-hits": "event-hits",
                    "event-particles": "event-particles",
                    "event-truth": "event-truth",
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
        ]
    )
