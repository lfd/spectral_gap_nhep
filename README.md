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
\data\01_raw\event*-hits.csv
\data\01_raw\event*-particles.csv
\data\01_raw\event*-truth.csv
```
Head over to the [TrackML Library Repo](https://github.com/stroblme/trackml-library) for more details.

