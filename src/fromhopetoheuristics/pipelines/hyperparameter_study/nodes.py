from typing import Dict, List, Optional, Callable
import subprocess
import pandas as pd
import os
import optuna

from fromhopetoheuristics.utils.hyperparam_optimizer import Hyperparam_Optimizer

import logging

log = logging.getLogger(__name__)


def create_hyperparam_optimizer(
    n_trials: int,
    timeout: int,
    enabled_hyperparameters: List[str],
    optimization_metric: List[str],
    path: str,
    sampler: str,
    sampler_seed: int,
    pruner_strategy: str,
    pruner_startup_trials: int,
    pruner_warmup_steps: int,
    pruner_interval_steps: int,
    pruner_min_trials: int,
    resume_study: bool,
    n_jobs: int,
    run_id: str,
    hyperparameters: Dict[str, List[float]],
) -> Dict[str, Hyperparam_Optimizer]:

    hyperparam_optimizer = Hyperparam_Optimizer(
        name=run_id,
        sampler=sampler,
        seed=sampler_seed,
        n_trials=n_trials,
        timeout=timeout,
        enabled_hyperparameters=enabled_hyperparameters,
        optimization_metric=optimization_metric,
        path=path,
        n_jobs=n_jobs,
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
        trial: optuna.trial.Trial,
        parameters: Dict[str, float],
        report_callback: Optional[Callable[[Dict[str, float], int], None]] = None,
        early_stop_callback: Optional[Callable[[], bool]] = None,
    ) -> float:
        """This function is the optimization target that is called by Optuna
        for each trial. It runs the experiment with the given parameters
        and reports the result to Optuna.
        Note that the `report_callback` and early_stop_callback is optional.
        It's just important that this objective returns a float value
        that corresponds to our optimization metric.
        Parameters are those hyperparameters that are left after filtering with
        `enabled_hyperparameters`.

        Args:
            trial: The Optuna trial object.
            parameters: The hyperparameters for this trial.
            report_callback: The callback function to report the result to Optuna.
                This is only necessary for statistics and if we have pruning enabled.
            early_stop_callback: The callback function to stop the trial early
                if the result is not promising, based on the pruner chosen.
                This function
                returns "True" if the trial should be pruned and else "False".

        Returns:
            The result of the experiment.
        """
        # TODO: eventually this will be a slurm script that we can provide an id
        # use the `trial._trial_id` for that purpose

        log.info(f"Running trial {trial._trial_id} with parameters {parameters}.")

        parameters["hyperhyper_trial_id"] = trial._trial_id
        kedro_params = ",".join([f"{k}={v}" for k, v in parameters.items()])
        # subprocess.run(
        #     [
        #         "kedro",
        #         "run",
        #         "--pipeline qaoa_trackrec",
        #         f"--params={kedro_params}",
        #     ]
        # )
        subprocess.run(
            [
                "sbatch",  # slurm
                "--job-name",
                f"sgnhep{trial._trial_id}",  # identify the job in the queue
                "--wait",  # ensure that we wait until the job is finished
                "./slurm.sh",  # our submission script
                "--pipeline qaoa_trackrec",  # evaluation pipeline
                f"--params={kedro_params}",  # kedro parameters
            ]
        )

        def get_objective_for_trial(trial_id: int) -> float:
            tmp_file_name = f".hyperhyper{trial_id}.json"
            results = pd.read_json(tmp_file_name)
            os.remove(tmp_file_name)

            last5 = results.loc[results["p"] > results.shape[0] - 5]
            return {
                "qaoa_energy": last5["qaoa_energy"].mean(),
                "min_energy": last5["min_energy"].mean(),
            }

        objective_vals = get_objective_for_trial(trial._trial_id)
        log.info(
            f"Trial {trial._trial_id} received objective value(s) {objective_vals}."
        )

        return objective_vals

    hyperparam_optimizer.objective = objective

    return {"hyperparam_optimizer": hyperparam_optimizer}


def run_optuna(hyperparam_optimizer: Hyperparam_Optimizer) -> None:
    """
    Run the hyperparameter optimization study.

    Args:
        hyperparam_optimizer: The hyperparameter optimizer to run.

    Returns:
        None
    """
    hyperparam_optimizer.minimize(idx=0)

    return {}
