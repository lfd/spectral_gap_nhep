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

    def objective(
        trial, parameters, report_callback=None, early_stop_callback=None
    ) -> float:
        """This function is the optimization target that is called by Optuna
        for each trial. It runs the experiment with the given parameters
        and reports the result to Optuna.
        Note that the `report_callback` and early_stop_callback is optional. It's just important
        that this objective returns a float value that corresponds to our
        optimization metric.
        Parameters are those hyperparameters that are left after filtering with
        `enabled_hyperparameters`.

        Args:
            trial: The Optuna trial object.
            parameters: The hyperparameters for this trial.
            report_callback: The callback function to report the result to Optuna. This is only
                necessary for statistics and if we have pruning enabled.
            early_stop_callback: The callback function to stop the trial early
                if the result is not promising, based on the pruner chosen. This function
                returns "True" if the trial should be pruned and else "False".

        Returns:
            The result of the experiment.
        """
        # TODO: eventually this will be a slurm script that we can provide an id
        # use the `trial._trial_id` for that purpose
        subprocess.run(
            [
                "kedro",
                "run",
                "--pipeline",
                "trackrec",
                f"--params={','.join([f'{k}={v}' for k, v in parameters.items()])}",
            ]
        )

        # TODO: here we should find a way to retrieve our optimization value from this subprocess
        # We could either do so by checking the files (as in `qnd_hyper`)or, ideally get the value straight from the output of the subprocess.
        # Because I'm not sure how well it goes with the output files if we run the experiments in parallel...
        # Note that we can trigger this hyperparameter experiment as often as we want,
        # optuna will take care of syncing those experiments based on the common database and the experiment name

        return 0.0

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
