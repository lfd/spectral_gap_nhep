from qallse.cli.func import (
    diff_rows,
    process_response,
)
from qallse.plotting import iplot_results
import pandas as pd

from typing import Dict
import logging

log = logging.getLogger(__name__)


def visualize(responses: Dict, data_wrapper) -> dict:
    """
    Visualize the results from responses.

    Parameters
    ----------
    responses : Dict
        A dict of response objects to process.
    data_wrapper :
        An object that provides access to real doublets data.

    Returns
    -------
    dict
        A dictionary containing figure objects for each response.
    """
    dims = list("xy")
    figs = {}
    for i, response in responses.items():
        final_doublets, _ = process_response(response)
        _, missings, _ = diff_rows(final_doublets, data_wrapper.get_real_doublets())

        figs[i] = iplot_results(
            data_wrapper, final_doublets, missings, dims=dims, return_fig=True
        )

    return {"figures": figs}


def export_parameters(**kwargs: Dict) -> Dict:
    """
    Export the given parameters.

    Parameters
    ----------
    kwargs : Dict
        A dictionary of parameters to export.

    Returns
    -------
    Dict
        A dictionary containing the exported parameters.
    """
    log.info(f"Exporting parameters: {kwargs}")

    return {"parameters": kwargs}
