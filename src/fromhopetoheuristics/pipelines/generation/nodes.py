from trackml.dataset import load_event, load_dataset
import pandas as pd
from typing import Tuple, Dict
import os
import re
import random
import numpy as np
from datetime import datetime

from qallse.data_wrapper import DataWrapper

# from qallse.dsmaker import create_dataset

import logging

log = logging.getLogger(__name__)

# TODO: currently we're loading data using the the external trackml-library
# However, we should switch to the kedro internal dataloader and catalog
BARREL_VOLUME_IDS = [8, 13, 17]


def create_dataset(
    hits,
    particles,
    truth,
    density=0.1,
    min_hits_per_track=5,
    high_pt_cut=1.0,
    double_hits_ok=False,
    random_seed=None,
    phi_bounds=None,
) -> Tuple[Dict, str]:
    # capture all parameters, so we can dump them to a file later

    input_params = locals()

    # initialise random
    if random_seed is None:
        random_seed = random.randint(0, 1 << 30)
    random.seed(random_seed)

    # event_id = re.search("(event[0-9]+)", input_path)[0]

    # compute the prefix
    # if prefix is None:
    #     prefix = f"ez-{density}"
    #     if high_pt_cut > 0:
    #         prefix += f"_hpt-{high_pt_cut}"
    #     else:
    #         prefix += "_baby"
    #     if double_hits_ok:
    #         prefix += "_dbl"

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

    # ---------- write data

    # write the dataset to disk
    # output_path = os.path.join(output_path, prefix)
    # os.makedirs(output_path, exist_ok=True)
    # output_path = os.path.join(output_path, event_id)

    # new_hits.to_csv(output_path + "-hits.csv", index=False)
    # new_truth.to_csv(output_path + "-truth.csv", index=False)
    # new_particles.to_csv(output_path + "-particles.csv", index=False)

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

    # with open(output_path + "-meta.json", "w") as f:
    #     json.dump(metadata, f, indent=4)

    # ------------ gen doublets

    from qallse.seeding import generate_doublets

    doublets_df = generate_doublets(hits=new_hits)
    # with open(output_path + "-doublets.csv", "w") as f:
    #     doublets_df.to_csv(f, index=False)
    #     log.info(
    #         f"Doublets (len={len(doublets_df)}) generated in f{output_path}."
    # )

    return new_hits, new_truth, new_particles, doublets_df, metadata


def load_event_data(event) -> dict:
    hits, cells, particles, truth = load_event(f"dataset/{event}")

    return {
        "hits": hits,
        "cells": cells,
        "particles": particles,
        "truth": truth,
    }


def load_dataset_data() -> dict:
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
    num_angle_parts: str,
    f: float,
    event_cells,
    event_hits,
    event_particles,
    event_truth,
) -> Tuple[dict, str]:
    """
    Creates/filters and stores TrackML event data for a given fraction of data.
    Uses Qallse to filter the data.

    :param result_path_prefix: Where to store the data
    :type result_path_prefix: str
    :param seed: seed for randomised data filtering, defaults to 12345
    :type seed: int, optional
    :param trackml_input_path: Path of the TrackML event, which is to be used
    :type trackml_input_path: str
    :param f: fraction of the data, which is to be used, defaults to 0.1
    :type f: float, optional

    :return: Qallse event metadata
    :rtype: dict
    :return: Path, where the filtered data is stored
    :rtype: str
    """
    prefix = f"data_frac{int(f*100)}_seed{seed}_num_parts{num_angle_parts}"

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


def create_qallse_datawrapper(hits, truth) -> DataWrapper:
    """
    Creates a qallse data wrapper form a given event path

    :return: Qallse data wrapper
    :rtype: DataWrapper
    """
    dw = DataWrapper(hits=hits, truth=truth)
    return {"data_wrapper": dw}
