from qallse.cli.func import (
    diff_rows,
    process_response,
)
from qallse.plotting import iplot_results
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from fromhopetoheuristics.utils.plot_utils import LFD_COLOURS

from typing import Dict
import logging

log = logging.getLogger(__name__)


def visualize(
    responses: Dict,
    data_wrapper,
    qaoa_results: pd.DataFrame,
    adiabatic_results: pd.DataFrame,
):
    """
    Calls all plot functions and combines them to one data structure

    :param responses Dict: simulated annealing responses
    :param data_wrapper DataWrapper: Qallse DataWrapper for loading
        track-reconstruction data
    :param qaoa_results pd.DataFrame: QAOA Results
    :param adiabatic_results pd.DataFrame: Results from computing the adiabatic
        evolution
    """
    figs = visualize_responses(responses, data_wrapper)
    figs.update(plot_QAOA_solution_quality(qaoa_results))
    figs.update(plot_spectral_gap(adiabatic_results))
    return {"figures": figs}


def plot_QAOA_solution_quality(qaoa_results: pd.DataFrame):
    """
    Creates a plotly graph, which plots the energy of QAOA vs the Optimal
    Energy for each Trotter step p.

    :param qaoa_results pd.DataFrame: QAOA Results
    """
    plot = px.line(
        qaoa_results,
        x="p",
        y=["min_energy", "qaoa_energy"],
        color_discrete_sequence=LFD_COLOURS,
        width=600,
        height=400,
        labels={"value": "Energy"},
    )
    names = {"min_energy": "Optimal Energy", "qaoa_energy": "QAOA Energy"}
    plot.for_each_trace(lambda t: t.update(name=names[t.name]))
    plot.update_layout(legend_title_text="")
    return {"qaoa_solution_quality": plot}


def plot_spectral_gap(adiabatic_results: pd.DataFrame):
    """
    Creates a plotly graph, which plots the curse of the energy for the ground
    state and the first excited state, as well as the energy difference between
    these

    :param adiabatic_results pd.DataFrame: Results from computing the adiabatic
        evolution
    """
    plot = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        x_title="Anneal Fraction",
        y_title="Energy",
    )
    energy_plot = px.line(
        adiabatic_results,
        x="fraction",
        y=["gs", "fes", "gap"],
        color_discrete_sequence=LFD_COLOURS,
    )
    names = {
        "gs": "Ground State",
        "fes": "First Excited State",
        "gap": "Spectral Gap",
    }
    energy_plot.for_each_trace(lambda t: t.update(name=names[t.name]))
    energy_plot.update_layout(legend_title_text="")

    plot.add_trace(energy_plot.data[0], row=1, col=1)
    plot.add_trace(energy_plot.data[1], row=1, col=1)
    plot.add_trace(energy_plot.data[2], row=1, col=2)
    plot.update_layout(width=1000, height=400)
    return {"spectral_gap": plot}


def visualize_responses(responses: Dict, data_wrapper) -> dict:
    """
    Visualize the results from simulated annealing responses.

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

        figs[f"vis_{i}"] = iplot_results(
            data_wrapper, final_doublets, missings, dims=dims, return_fig=True
        )

    return figs


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
