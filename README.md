# From Hope To Heuristics
## Realistic Runtime Estimates for Quantum Optimisation in NHEP

## :rocket: Getting Started

When cloning, make sure to get the submodule:
```
git clone --recurse-submodules git@github.com:majafranz/spectral_gap_nhep.git
```

If you have poetry installed, run `poetry install`.
With pip, make sure to include the dependencies in the submodule `trackml-library` (pandas and numpy).

To get the data, head over to the [Kaggle TrackML Particle Tracking Challenge](https://www.kaggle.com/c/trackml-particle-identification/data) and download e.g. the `train_sample.zip` file which is a reduced version of the overall dataset.
Extract the data into a `dataset` folder, such that the structure is as follows:
```bash
\dataset\event*-hits.csv
\dataset\event*-particles.csv
\dataset\event*-truth.csv
```
Head over to the [TrackML Library Repo](https://github.com/stroblme/trackml-library) for more details.

## Approach

For the QUBO formulation we build on the [HEPQPR.Qallse](https://github.com/derlin/hepqpr-qallse) project.

TODO: add description

To work with smaller sized QUBOs, we only focus on hit-triplets present in a specified angle, similar to the approach presented by [Schwägerl et al.](https://arxiv.org/pdf/2303.13249).

## Minimum working example

no spectral gap, yet, just:
- generating partial dataset from event
- building the QUBO
- optimising with simulated annealing

```
export PYTHONPATH=$PATHONPATH:src/
python src/qallse_wrapper/main.py
```
