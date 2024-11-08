from typing import Dict, List
import subprocess

from fromhopetoheuristics.utils.hyperparam_optimizer import Hyperparam_Optimizer

import logging

log = logging.getLogger(__name__)


def create_hyperparam_optimizer(
    n_trials: str,
    timeout: int,
    enabled_hyperparameters: List,
    optimization_metric: List,
    path: str,
    sampler: str,
    sampler_seed: int,
    pruner_strategy: str,
    pruner_startup_trials: int,
    pruner_warmup_steps: int,
    pruner_interval_steps: int,
    pruner_min_trials: int,
    selective_optimization: bool,
    resume_study: bool,
    n_jobs: int,
    run_id: str,
    hyperparameters: Dict,
) -> Hyperparam_Optimizer:
    if run_id is None:
        name = mlflow.active_run().info.run_id
    else:
        name = run_id

    hyperparam_optimizer = Hyperparam_Optimizer(
        name=name,
        sampler=sampler,
        seed=sampler_seed,
        n_trials=n_trials,
        timeout=timeout,
        enabled_hyperparameters=enabled_hyperparameters,
        optimization_metric=optimization_metric,
        path=path,
        n_jobs=n_jobs,
        selective_optimization=selective_optimization,
        resume_study=resume_study,
        pruner=pruner_strategy,
        pruner_startup_trials=pruner_startup_trials,
        pruner_warmup_steps=pruner_warmup_steps,
        pruner_interval_steps=pruner_interval_steps,
        pruner_min_trials=pruner_min_trials,
    )

    hyperparam_optimizer.set_variable_parameters(hyperparameters)
    hyperparam_optimizer.set_fixed_parameters({})

    def objective(trial, parameters, report_callback, early_stop_callback):
        subprocess.run(
            [
                "kedro",
                "run",
                "--pipeline",
                "trackrec",
                f"--params={','.join([f'{k}={v}' for k, v in parameters.items()])}",
            ]
        )
        raise NotImplementedError("Objective method must be set!")

    hyperparam_optimizer.objective = objective

    return {"hyperparam_optimizer": hyperparam_optimizer}


def run_optuna(
    hyperparam_optimizer: Hyperparam_Optimizer,
):
    hyperparam_optimizer.minimize(idx=0)

    # try:
    #     hyperparam_optimizer.log_study(
    #         selected_parallel_params=optuna_selected_parallel_params,
    #         selected_slice_params=optuna_selected_slice_params,
    #     )
    # except Exception as e:
    #     log.exception("Error while logging study")

    return {}
