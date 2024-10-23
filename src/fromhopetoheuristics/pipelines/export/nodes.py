from qallse.cli.func import (
    diff_rows,
    process_response,
)
from qallse.plotting import iplot_results
import pandas as pd

import logging

log = logging.getLogger(__name__)


def visualize(responses, data_wrapper):
    dims = list("xy")
    figs = {}
    for i, response in enumerate(responses):
        final_doublets, _ = process_response(response)
        _, missings, _ = diff_rows(final_doublets, data_wrapper.get_real_doublets())

        figs[i] = iplot_results(
            data_wrapper, final_doublets, missings, dims=dims, return_fig=True
        )

    return {"figures": figs}


def export_parameters(**kwargs):
    params = pd.DataFrame.from_dict(kwargs)

    return {"parameters": params}
