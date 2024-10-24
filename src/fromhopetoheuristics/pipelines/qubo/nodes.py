from typing import Dict, Tuple
import numpy as np
import pandas as pd
import random
from datetime import datetime

from qallse.cli.func import solve_neal
from qallse.data_wrapper import DataWrapper
from trackml.dataset import load_event, load_dataset
from fromhopetoheuristics.utils.model import QallseSplit, build_model


from typing import List, Dict, Any, Tuple, Optional

import logging

log = logging.getLogger(__name__)

# TODO: currently we're loading data using the the external trackml-library
# However, we should switch to the kedro internal dataloader and catalog
BARREL_VOLUME_IDS = [8, 13, 17]


def build_qubos(
    data_wrapper: DataWrapper,
    doublets: List,
    num_angle_parts: int,
    geometric_index: int = -1,
) -> Dict[str, QallseSplit]:
    """
    Creates partial QUBO from TrackML data using Qallse. The data is split into
    several parts by the angle in the XY-plane of the detector, from which the
    QUBO is built.

    :param data_wrapper: Qallse data wrapper
    :type data_wrapper: DataWrapper
    :param doublets: List of doublets to be used in the QUBO
    :type doublets: List[ExtendedDoublet]
    :param num_angle_parts: Number of angle segments in the detector, equals
        the number of resulting QUBOs
    :param geometric_index: The angle part, for which to build the QUBO, if -1
        build all
    :type geometric_index: int
    :return: The QUBOs in dictionary form, where the keys are the angle part
        indices
    :rtype: Dict[str, QallseSplit]
    """
    qubos = {}
    log.info(f"Generating {num_angle_parts} QUBOs")

    # FIXME: only generate for one index
    if geometric_index == -1:
        angle_parts = range(num_angle_parts)
    else:
        angle_parts = [geometric_index]

    for i in angle_parts:
        extra_config = {
            "geometric_index": i,
            "xy_angle_parts ": num_angle_parts,
        }
        model = QallseSplit(data_wrapper, **extra_config)
        build_model(doublets=doublets, model=model, add_missing=False)

        qubos[i] = model
        log.info(f"Generated QUBO {i+1}/{num_angle_parts}")
    return {"qubos": qubos}


def solve_qubos(
    qubos: Dict[str, QallseSplit],
    seed: int,
) -> Dict[str, Any]:
    """
    Solve the provided QUBOs using a specified seed.

    Parameters
    ----------
    qubos : Dict[str, QallseSplit]
        A dictionary of QUBOs to be solved, where the key is the identifier.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the responses for each solved QUBO.
    """
    responses = {}
    log.info(f"Solving {len(qubos)} QUBOs")

    for i, qubo in qubos.items():
        response = solve_neal(qubo, seed=seed)
        # print_stats(data_wrapper, response, qubo) # FIXME: solve no track found case

        log.info(f"Solved QUBO {i+1}/{len(qubos)}")
        responses[i] = response

    return {"responses": responses}  # FIXME find suitable catalog entry


def create_dataset(
    hits: pd.DataFrame,
    particles: pd.DataFrame,
    truth: pd.DataFrame,
    density: float = 0.1,
    min_hits_per_track: int = 5,
    high_pt_cut: float = 1.0,
    double_hits_ok: bool = False,
    random_seed: Optional[int] = None,
    phi_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict, str]:
    """
    Create a dataset for Qallse from the given hits, particles, and truth.

    Parameters
    ----------
    hits : pd.DataFrame
        The hits.
    particles : pd.DataFrame
        The particles.
    truth : pd.DataFrame
        The truth.
    density : float, optional
        The density of the tracks to sample. Defaults to 0.1.
    min_hits_per_track : int, optional
        The minimum number of hits per track. Defaults to 5.
    high_pt_cut : float, optional
        The high Pt cut. Defaults to 1.0.
    double_hits_ok : bool, optional
        If True, double hits are allowed. Defaults to False.
    random_seed : Optional[int], optional
        The random seed. Defaults to None.
    phi_bounds : Optional[Tuple[float, float]], optional
        The phi bounds. Defaults to None.

    Returns
    -------
    Dict
        The metadata.
    str
        The path to the dataset.
    """

    input_params = locals()

    # initialise random
    if random_seed is None:
        random_seed = random.randint(0, 1 << 30)
    random.seed(random_seed)

    # ---------- prepare data

    # load the data

    # add indexes
    particles.set_index("particle_id", drop=False, inplace=True)
    truth.set_index("hit_id", drop=False, inplace=True)
    hits.set_index("hit_id", drop=False, inplace=True)

    # create a merged dataset with hits and truth
    df = hits.join(truth, rsuffix="_", how="inner")

    log.debug(f"Loaded {len(df)} hits.")

    # ---------- filter hits
    # keep only hits in the barrel region
    df = df[hits.volume_id.isin(BARREL_VOLUME_IDS)]
    log.debug(f"Filtered hits from barrel. Remaining hits: {len(df)}.")

    if phi_bounds is not None:
        df["phi"] = np.arctan2(df.y, df.x)
        df = df[(df.phi >= phi_bounds[0]) & (df.phi <= phi_bounds[1])]
        log.debug(f"Filtered using phi bounds {phi_bounds}. Remaining hits: {len(df)}.")

    # store the noise for later, then remove them from the main dataframe
    # do this before filtering double hits, as noise will be thrown away as duplicates
    noise_df = df.loc[df.particle_id == 0]
    df = df[df.particle_id != 0]

    if not double_hits_ok:
        df.drop_duplicates(
            ["particle_id", "volume_id", "layer_id"], keep="first", inplace=True
        )
        log.debug(f"Dropped double hits. Remaining hits: {len(df) + len(noise_df)}.")

    # ---------- sample tracks

    num_tracks = int(df.particle_id.nunique() * density)
    sampled_particle_ids = random.sample(df.particle_id.unique().tolist(), num_tracks)
    df = df[df.particle_id.isin(sampled_particle_ids)]

    # ---------- sample noise

    num_noise = int(len(noise_df) * density)
    sampled_noise = random.sample(noise_df.hit_id.values.tolist(), num_noise)
    noise_df = noise_df.loc[sampled_noise]

    # ---------- recreate hits, particle and truth

    new_hit_ids = df.hit_id.values.tolist() + noise_df.hit_id.values.tolist()
    new_hits = hits.loc[new_hit_ids]
    new_truth = truth.loc[new_hit_ids]
    new_particles = particles.loc[sampled_particle_ids]

    # ---------- fix truth

    if high_pt_cut > 0:
        # set low pt weights to 0
        hpt_mask = np.sqrt(truth.tpx**2 + truth.tpy**2) >= high_pt_cut
        new_truth.loc[~hpt_mask, "weight"] = 0
        log.debug(f"High Pt hits: {sum(hpt_mask)}/{len(new_truth)}")

    if min_hits_per_track > 0:
        short_tracks = new_truth.groupby("particle_id").filter(
            lambda g: len(g) < min_hits_per_track
        )
        new_truth.loc[short_tracks.index, "weight"] = 0

    new_truth.weight = new_truth.weight / new_truth.weight.sum()

    # ---------- write metadata

    metadata = dict(
        num_hits=new_hits.shape[0],
        num_tracks=num_tracks,
        num_important_tracks=new_truth[new_truth.weight != 0].particle_id.nunique(),
        num_noise=num_noise,
        random_seed=random_seed,
        time=datetime.now().isoformat(),
    )
    for k, v in metadata.items():
        log.debug(f"  {k}={v}")

    metadata["params"] = input_params

    # ------------ gen doublets

    from qallse.seeding import generate_doublets

    doublets_df = generate_doublets(hits=new_hits)

    return new_hits, new_truth, new_particles, doublets_df, metadata


def load_event_data(event: str) -> dict[str, pd.DataFrame]:
    """
    Load the hits, cells, particles, and truth information for a given event.

    Args:
        event (str): The event id

    Returns:
        dict[str, pd.DataFrame]: A dictionary containing the hits, cells, particles, and truth data.
    """
    hits, cells, particles, truth = load_event(f"dataset/{event}")

    return {
        "hits": hits,
        "cells": cells,
        "particles": particles,
        "truth": truth,
    }


def load_dataset_data() -> Dict[str, pd.DataFrame]:
    """
    Load the hits, cells, particles, and truth information for the entire dataset.

    Returns:
        dict[str, pd.DataFrame]: A dictionary containing the dataset data.
    """
    dataset = pd.DataFrame()

    for event_id, hits, cells, particles, truth in load_dataset("dataset"):
        dataset = dataset.append(
            {
                "event_id": event_id,
                "hits": hits,
                "cells": cells,
                "particles": particles,
                "truth": truth,
            },
            ignore_index=True,
        )

    return {"dataset": dataset}


def create_metadata(
    seed: int,
    f: float,
    event_hits: pd.DataFrame,
    event_particles: pd.DataFrame,
    event_truth: pd.DataFrame,
) -> Tuple[Dict[str, Any], str]:
    """
    Creates/filters and stores TrackML event data for a given fraction of data.
    Uses Qallse to filter the data.

    Parameters
    ----------
    seed : int
        seed for randomised data filtering, defaults to 12345
    f : float
        fraction of the data, which is to be used, defaults to 0.1
    event_cells : pd.DataFrame
        cells of the event
    event_hits : pd.DataFrame
        hits of the event
    event_particles : pd.DataFrame
        particles of the event
    event_truth : pd.DataFrame
        truth information of the event

    Returns
    -------
    tuple
        A tuple containing:
        - a dictionary with the filtered data
        - the path, where the filtered data is stored
    """
    hits, truth, particles, doublets, metadata = create_dataset(
        hits=event_hits,
        truth=event_truth,
        particles=event_particles,
        density=f,
        random_seed=seed,
    )

    return {
        "hits": hits,
        "truth": truth,
        "particles": particles,
        "doublets": doublets,
        "metadata": metadata,
    }


def create_qallse_datawrapper(
    hits: pd.DataFrame, truth: pd.DataFrame
) -> Dict[str, DataWrapper]:
    """
    Creates a qallse data wrapper form a given event path

    Parameters
    ----------
    hits : pd.DataFrame
        hits of the event
    truth : pd.DataFrame
        truth information of the event

    Returns
    -------
    dict
        A dictionary containing the qallse data wrapper
    """
    dw = DataWrapper(hits=hits, truth=truth)
    return {"data_wrapper": dw}
