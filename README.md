# From Hope To Heuristics
**Realistic Runtime Estimates for Quantum Optimisation in NHEP**

## Approach

For the QUBO formulation we build on the [HEPQPR.Qallse](https://github.com/derlin/hepqpr-qallse) project.

<!-- TODO: add description -->

To work with smaller sized QUBOs, we only focus on hit-triplets present in a specified angle, similar to the approach presented by [Schwägerl et al.](https://arxiv.org/pdf/2303.13249).

## :rocket: Getting Started

### Setup

When cloning, make sure to get the submodule:
```
git clone --recurse-submodules git@github.com:majafranz/spectral_gap_nhep.git
```

If you have poetry installed, run `poetry install`.
With pip, make sure to include the dependencies in the submodule `trackml-library` (pandas and numpy) and `hepqpr-qallse` (pandas, numpy, plotly).


<!-- To get the data, head over to the [Kaggle TrackML Particle Tracking Challenge](https://www.kaggle.com/c/trackml-particle-identification/data) and download e.g. the `train_sample.zip` file which is a reduced version of the overall dataset.
Extract the data into a `dataset` folder, such that the structure is as follows:
```bash
\data\01_raw\event*-hits.csv
\data\01_raw\event*-particles.csv
\data\01_raw\event*-truth.csv
```
Head over to the [TrackML Library Repo](https://github.com/stroblme/trackml-library) for more details. -->

### Quickstart

After installing all the dependencies, simply execute
```
kedro run
```

This will run the default pipeline which consists of all individual pipelines described in the following section.

### Pipelines

#### QAOA Maxcut

Solving the Maxcut problem using Quadratic Approximate Optimization Algorithm (QAOA).
It also calculates the annealing schedules for this problem.

#### Adiabatic Maxcut

Solving the Maxcut problem using adiabatic quantum computing, aka. annealing.

#### Qubo Formulation

This pipeline loads event data and prepares a qubo for the following two track reconstruction pipelines.

#### QAOA Track Reconstruction

Solving the track reconstruction problem using QAOA.
It also calculates the annealing schedules for this problem.

#### Adiabatic Track Reconstruction

Solving the track reconstruction problem using quantum annealing.

#### Visualization

Visualization of all results.

## Project Structure

The following list gives a brief explanation of the most important locations in our repository:
- `conf/base/parameters.yml`: Parameters used in the experiments
- `conf/base/catalog.yml`: Datasets and models description
- `data/**`: Contains all initial, intermediate and final data(sets)
- `src/fromhopetoheuristics`: Main code divided into pipelines and utilities (shared among several pipelines)
- `tests`: All the tests for verification

Besides that, we make use of two submodules:
- `hepqpr-qallse`: Currently, all the data loading and QUBO formulation is done using this submodule
- `trackml-library`: Not in use currently

## 🚧 Contributing

Contributions are highly welcome! Take a look at our [Contribution Guidelines](https://github.com/cirKITers/qml-essentials/blob/main/CONTRIBUTING.md).