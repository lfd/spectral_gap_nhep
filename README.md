# From Hope To Heuristics
**Realistic Runtime Estimates for Quantum Optimisation in NHEP**

## :book: Project Description

This is the repository for our contribution to [CHEP24](https://indico.cern.ch/event/1338689/contributions/6010081/) consisting of two key aspects:
Firstly, we estimate runtimes and scalability for common NHEP problems addressed via QUBO formulations by identifying minimum energy solutions of intermediate Hamiltonian operators encountered during the annealing process. 
Secondly, we investigate how the classical parameter space in the QAOA, together with approximation techniques such as a Fourier-analysis based heuristic, proposed by Zhou et al. (2018), can help to achieve (future) quantum advantage, considering a trade-off between computational complexity and solution quality.
Those approaches are evaluated on two benchmark problems: the Maxcut problem and the track reconstruction problem.

## Approach

For the QUBO formulation of the track reconstruction problem, we build on the [HEPQPR.Qallse](https://github.com/derlin/hepqpr-qallse) project.
To work with smaller sized QUBOs, we only focus on hit-triplets present in a specified angle, similar to the approach presented by [Schwägerl et al.](https://arxiv.org/pdf/2303.13249).

## :rocket: Getting Started

### Setup

When cloning, make sure to get the submodule:
```
git clone --recurse-submodules git@github.com:lfd/spectral_gap_nhep.git
```
This will clone [our fork of hepqr-qallse](https://github.com/lfd/hepqpr-qallse) recursively.

If you have poetry installed, run `poetry install`.
With pip, make sure to include the dependencies in the submodule `hepqpr-qallse` (pandas, numpy, plotly) and `trackml` (pandas, numpy).


### Quickstart

After installing all the dependencies, simply execute
```
kedro run
```

This will run the default pipeline which consists of all individual pipelines described in the following section.
You can get an interactive overview of the pipeline in your browser by running
```
kedro viz
```

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


## Project Structure

The following list gives a brief explanation of the most important locations in our repository:
- `conf/base/parameters.yml`: Parameters used in the experiments
- `conf/base/catalog.yml`: Datasets and models description
- `data/**`: Contains all initial, intermediate and final data(sets)
- `src/fromhopetoheuristics`: Main code divided into pipelines and utilities (shared among several pipelines)
- `tests`: All the tests for verification

Besides that, we make use of two submodules:
- `hepqpr-qallse`: Currently, all the data loading and QUBO formulation is done using this submodule

## Hyperparameter Optimization

This project uses Optuna for hyperparameter optimization.
There is a dedicated kedro pipeline that takes care of the hyperparameter optimization and submission of jobs to a SLURM cluster.
```
kedro run --pipeline hyperparameter_study
```

If you don't have a SLURM cluster available, head to `pipelines/hyperparameter_study/nodes.py` switch the subprocess command such that it spawns a single kedro job instead of a submission to the cluster.

Supposing everything goes well, you can take a look at the experiments by running
```
optuna-dashboard sqlite:///studies/fhth.db
```
supposing that the path to the sqlite database where Optuna stores its results is `studies/fhth.db`.


## Reproduction

The numerical results in our study can be reproduced using the `reproduction.sh` script.
The script executes all runs sequentially. Feel free to change the script for parallelisation, depending on your system size.

### Numerical data
All results are stored in the `data/` folder. The subfolders `04_adiabatic`, `05_qaoa` and `06_schedules` can contain results in CSV format, with the corresponding run configuration in `00_parameters`, stored as JSON.

### Proceedings results
We copied the results obtained by us to `analysis/proceedings_results/`.

### Data visualisation
The data for the proceedings article in `analysis/proceedings_results/` can be plotted using R with GGplot.
The following R libraries are required:
- tidyverse
- ggh4x
- stringr
- tikzDevice
- patchwork
- rjson

If you have R installed on your system, the libraries can be installed via:
```
R -e "install.packages('<library>', dependencies=TRUE, repos='http://cran.rstudio.com/')"
```

### To tikz or not to tikz
For the plots in the article, we used the tikzDevice export, which can be used by setting the variable`tikz` in the plot script `analysis/plot.r` to `TRUE`.
If you are fine with plain PDF, keep it as it is.

### Plotting
Once everything is set up, we can run the following to obtain the plots:
```
cd analysis
Rscript plot.r
```
Output plots can then be found in either `analysis/img-tikz`, or `analysis/img-pdf`.


## 🚧 Contributing

Contributions are highly welcome! Take a look at our [Contribution Guidelines](https://github.com/lfd/spectral_gap_nhep/blob/main/CONTRIBUTING.md).

---

![overview](doc/kedro-pipeline.svg)
