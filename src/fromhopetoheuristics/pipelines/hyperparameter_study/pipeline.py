from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_hyperparam_optimizer, run_optuna


def create_pipeline() -> Pipeline:

    return pipeline(
        [
            node(
                create_hyperparam_optimizer,
                inputs={
                    "n_trials": "params:optuna.n_trials",
                    "timeout": "params:optuna.timeout",
                    "enabled_hyperparameters": "params:optuna.enabled_hyperparameters",
                    "optimization_metric": "params:optuna.optimization_metric",
                    "path": "params:optuna.path",
                    "sampler": "params:optuna.sampler",
                    "sampler_seed": "params:optuna.sampler_seed",
                    "pruner_strategy": "params:optuna.pruner_strategy",
                    "pruner_startup_trials": "params:optuna.pruner_startup_trials",
                    "pruner_warmup_steps": "params:optuna.pruner_warmup_steps",
                    "pruner_interval_steps": "params:optuna.pruner_interval_steps",
                    "pruner_min_trials": "params:optuna.pruner_min_trials",
                    "selective_optimization": "params:optuna.selective_optimization",
                    "resume_study": "params:optuna.resume_study",
                    "n_jobs": "params:optuna.n_jobs",
                    "run_id": "params:optuna.run_id",
                    "hyperparameters": "params:optuna.hyperparameters",
                },
                outputs={
                    "hyperparam_optimizer": "hyperparam_optimizer",
                },
            ),
            node(
                run_optuna,
                inputs={
                    "hyperparam_optimizer": "hyperparam_optimizer",
                },
                outputs={},
            ),
        ]
    )
