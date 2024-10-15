import os

from qallse.cli.func import (
    diff_rows,
    process_response,
)
from qallse.data_wrapper import DataWrapper
from qallse.plotting import iplot_results

import logging

log = logging.getLogger(__name__)


def visualize(responses, data_wrapper, event_path):

    dims = list("xy")
    figs = []
    for i, response in enumerate(responses):
        output_path = os.path.join(event_path, "plots")
        prefix = f"plot{i:02d}"
        final_doublets, final_tracks = process_response(response)
        _, missings, _ = diff_rows(final_doublets, data_wrapper.get_real_doublets())
        dout = os.path.join(output_path, prefix + "plot-doublets.html")
        tout = os.path.join(output_path, prefix + "plot-triplets.html")
        iplot_results(data_wrapper, final_doublets, missings, dims=dims, filename=dout)
        figs.append((dout, tout))

    return {"figures": figs}
