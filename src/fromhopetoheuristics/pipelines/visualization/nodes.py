import os

from qallse.cli.func import (
    diff_rows,
    process_response,
)
from qallse.data_wrapper import DataWrapper
from qallse.plotting import iplot_results

import logging

log = logging.getLogger(__name__)


def visualize(response, event_path, output_path, prefix):
    dw = DataWrapper.from_path(event_path)

    dims = list("xy")

    final_doublets, final_tracks = process_response(response)
    _, missings, _ = diff_rows(final_doublets, dw.get_real_doublets())
    dout = os.path.join(output_path, prefix + "plot-doublets.html")
    tout = os.path.join(output_path, prefix + "plot-triplets.html")
    iplot_results(dw, final_doublets, missings, dims=dims, filename=dout)

    return {"plot_doublets": dout, "plot_triplets": tout}
