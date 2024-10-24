from kedro.pipeline import Pipeline, node, pipeline

from .nodes import visualize, export_parameters


def create_trackrecon_pipeline() -> Pipeline:

    return pipeline(
        [
            node(
                visualize,
                {
                    "responses": "responses",
                    "data_wrapper": "data_wrapper",
                },
                {
                    "figures": "trackrecon_figures",
                    # FIXME: find suitable catalog entry
                },
            ),
            node(
                export_parameters,
                {
                    "data_fraction": "params:data_fraction",
                    "num_angle_parts": "params:num_angle_parts",
                    "seed": "params:seed",
                    "max_p": "params:max_p",
                    "q": "params:q",
                    "optimiser": "params:optimiser",
                    "num_anneal_fractions": "params:num_anneal_fractions",
                    "geometric_index": "params:geometric_index",
                },
                {"parameters": "trackrecon_parameters"},
            ),
        ]
    )


def create_maxcut_pipeline() -> Pipeline:

    return pipeline(
        [
            node(
                export_parameters,
                {
                    "seed": "params:seed",
                    "max_p": "params:max_p",
                    "q": "params:q",
                    "optimiser": "params:optimiser",
                    "maxcut_n_qubits": "params:maxcut_n_qubits",
                    "maxcut_graph_density": "params:maxcut_graph_density",
                    "num_anneal_fractions": "params:num_anneal_fractions",
                },
                {"parameters": "maxcut_parameters"},
            ),
        ]
    )
