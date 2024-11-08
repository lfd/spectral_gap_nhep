import optuna as o
from typing import List, Dict, Optional, Any

import plotly.graph_objects as go

from fromhopetoheuristics.utils.design import design


class Hyperparam_Optimizer:
    def __init__(
        self,
        name: str,
        sampler: str,
        seed: int,
        path: str,
        n_trials: int,
        timeout: int,
        enabled_hyperparameters: List[str],
        optimization_metric: List[str],
        n_jobs: int,
        resume_study: bool,
        pruner: Optional[str],
        pruner_startup_trials: int,
        pruner_warmup_steps: int,
        pruner_interval_steps: int,
        pruner_min_trials: int,
    ) -> None:
        """
        Initialize the Hyperparam_Optimizer class.

        Args:
            name (str): The name of the study to be stored in the database.
            sampler (str): The sampler to be used for the study.
            seed (int): The seed for the random sampler.
            path (str): The path to the database file.
            n_trials (int): The number of trials to run for the study.
            timeout (int): The timeout for the study in seconds.
            enabled_hyperparameters (List[str]): The hyperparameters to be optimized.
            optimization_metric (List[str]): The optimization metric to be used.
            n_jobs (int): The number of jobs to run in parallel.
            resume_study (bool): Whether to resume the study from the database.
            pruner (Optional[str]): The pruner to be used for the study.
            pruner_startup_trials (int): The number of trials before pruning starts.
            pruner_warmup_steps (int): The number of trials to be used for warmup.
            pruner_interval_steps (int): The interval between pruning conditions.
            pruner_min_trials (int): The minimum number of trials to be used for pruning.
        """

        if pruner is None:
            self.pruner = None
        elif pruner == "MedianPruner":
            self.pruner = o.pruners.MedianPruner(
                n_startup_trials=pruner_startup_trials,
                n_warmup_steps=pruner_warmup_steps,
                interval_steps=pruner_interval_steps,
                n_min_trials=pruner_min_trials,
            )
        elif pruner == "PercentilePruner":
            self.pruner = o.pruners.PercentilePruner(
                percentile=10.0,
                n_warmup_steps=pruner_warmup_steps,
                n_startup_trials=pruner_startup_trials,
            )
        else:
            raise ValueError(f"Pruner {pruner} is not supported")

        if sampler == "TPESampler":
            sampler = o.samplers.TPESampler(
                seed=seed, multivariate=True, constant_liar=True
            )
        elif sampler == "RandomSampler":
            sampler = o.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"Sampler {sampler} is not supported")

        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.timeout = timeout

        self.enabled_hyperparameters = enabled_hyperparameters

        self.studies = []
        self.n_jobs = n_jobs

        direction = "minimize"  # TODO: replace if required

        for n_it in range(self.n_jobs):
            resume_study = resume_study or (self.n_jobs > 1 and n_it != 0)

            self.studies.append(
                o.create_study(
                    pruner=self.pruner,
                    sampler=sampler,
                    direction=direction,
                    load_if_exists=resume_study,
                    study_name=name,
                    storage=f"sqlite:///{path}",
                )
            )

            self.studies[-1].set_metric_names([self.optimization_metric])

    @staticmethod
    def objective():
        raise NotImplementedError("Objective method must be set!")

    def select_hyperparams(self, pair):
        key, _ = pair
        return (
            len(self.enabled_hyperparameters) == 0
            or key in self.enabled_hyperparameters
        )

    def set_variable_parameters(self, parameters: Dict):
        self.variable_parameters = dict(
            filter(self.select_hyperparams, parameters.items())
        )

    def set_fixed_parameters(self, parameters: Dict):
        self.fixed_parameters = dict(
            filter(self.select_hyperparams, parameters.items())
        )

    def update_variable_parameters(
        self, trial: o.trial.Trial, parameters: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Updates the variable hyperparameters based on the trial suggestions.
        There is a special postfix keyword for the hyperparameters: "_choice"
        This keyword allows nested hyperparameters that depend on the
        categorical decision.
        So e.g. one could have the following structure:

        optimizer_choice
            A:
                A.1
                A.2
            B:
                B.1
                B.2

        where you cannot combine A.1 and B.1, in a single study because A, B
        might be optimizers and their sub-parameters are options.
        This code then detects "optimizer_choice" and only selects either
        A with its sub-parameters or B with its sub-parameters depending
        on a categorical suggestion of the current trial instance.

        Besides that, Hyperparametes can have the following "types":
        - continous values: [START, END, ['log'|'linear']]
        - discrete values: [A, B, C, ...]
        The former is used for either int or float suggestion based on the type
        of the hyperparameter. The latter is used for categorical suggestion.
        Note that continous values allow specific log or linear scaling.

        Args:
            trial: The Optuna trial object.
            parameters: A dictionary of hyperparameters to be optimized.
            prefix: A string to prepend to each parameter name for uniqueness.

        Returns:
            A dictionary with updated hyperparameters.
        """
        updated_variable_parameters = dict()
        choice = None  # indicate that this level is not a choice

        for parameter, value in parameters.items():
            if "_choice" in parameter:
                param_name = parameter.replace("_choice", "")
                assert type(value) == dict
                # if we have the choice; go and ask the trial what to do
                choice = trial.suggest_categorical(param_name, value.keys())
            else:
                param_name = parameter

            # now, check if the hyperparameter is nested, i.e. there is another level below
            if isinstance(value, Dict):
                # this is a nested parameter, check if we had the choice
                if choice is not None:
                    # we want to skip through this choice (i.e. not iterate at this level)
                    # therefore create a new dict
                    updated_variable_parameters[param_name] = {}
                    # and assign the result as a new dict to the only key in this dict (this is just to preserve the structure given by the config)
                    # note that we preselect value[choice] and modify the prefix such that it includes the choice (to not get duplicates later when running trial.suggest(..))
                    updated_variable_parameters[param_name][choice] = (
                        self.update_variable_parameters(
                            trial,
                            value[choice],
                            prefix=f"{prefix}{param_name}_{choice}_",
                        )
                    )
                else:
                    # the easy case; just go one level deeper and pass as prefix the current prefix (in case we are multilevel) as well as the current parameter name
                    updated_variable_parameters[param_name] = (
                        self.update_variable_parameters(
                            trial, value, prefix=f"{prefix}{param_name}_"
                        )
                    )

                # as this is just a "virtual" level, there is no reason the check the following
                continue
            # ok, so this is not nested, and therefore can only be a list (i.e. a _range hyperparameter)
            elif not isinstance(value, List):
                raise RuntimeError(
                    "Provides parameter is not a dictionary or a list. Cannot infer hyperparameters."
                )

            # ----
            # here ends the recursive call from previous section

            # if we have three values (-> no bool) and they are not categorical (str) and the last one is a str (linear/log)
            if (
                len(value) == 3
                and not isinstance(value[0], str)
                and not isinstance(value[1], str)
                and isinstance(value[2], str)
            ):
                low = value[0]
                high = value[1]

                # if the third parameter specifies the scale
                if isinstance(value[2], str):
                    if value[2] == "log":
                        log = True
                        step = None
                    else:
                        log = False
                        step = value[0]
                else:
                    log = False
                    step = value[2]

                #
                if isinstance(low, float) and isinstance(high, float):
                    updated_variable_parameters[param_name] = trial.suggest_float(
                        prefix + param_name, value[0], value[1], step=step, log=log
                    )
                elif isinstance(low, int) and isinstance(high, int):
                    updated_variable_parameters[param_name] = trial.suggest_int(
                        prefix + param_name, value[0], value[1], step=1, log=log
                    )
                else:
                    raise ValueError(
                        f"Unexpected type of range for trial suggestion for parameter {param_name}. Expected one of 'float' or 'int', got [{type(low)}, {type(high)}]."
                    )

            else:
                # if we don't find the scheme, go for categorical
                updated_variable_parameters[param_name] = trial.suggest_categorical(
                    prefix + param_name, value
                )

        return updated_variable_parameters

    def minimize(self, idx: int):
        result = self.studies[idx].optimize(
            self.run_trial, n_trials=self.n_trials, n_jobs=self.n_jobs
        )
        self.studies[idx].tell(result)

    def run_trial(self, trial):
        updated_variable_parameters = self.update_variable_parameters(
            trial, self.variable_parameters
        )
        parameters = self.fixed_parameters | updated_variable_parameters

        report_callback = lambda metrics, step: trial.report(
            metrics[self.optimization_metric], step=step
        )

        early_stop_callback = trial.should_prune if self.pruner else lambda: False

        metric = self.objective(
            trial=trial,
            parameters=parameters,
            report_callback=report_callback,
            early_stop_callback=early_stop_callback,
        )

        return metric[self.optimization_metric]

    # def log_study(self, selected_parallel_params, selected_slice_params):
    #     study = self.studies[0]
    #     # for study in self.studies:
    #     plt = o.visualization.plot_optimization_history(study)

    #     plt.update_layout(
    #         yaxis=dict(
    #             showgrid=design.showgrid,
    #         ),
    #         xaxis=dict(
    #             showgrid=design.showgrid,
    #         ),
    #         title=dict(
    #             text=(
    #                 f"Optimization History for {study.study_name}"
    #                 if design.print_figure_title
    #                 else ""
    #             ),
    #             font=dict(
    #                 size=design.title_font_size,
    #             ),
    #         ),
    #         font=dict(
    #             size=design.legend_font_size,
    #         ),
    #         template="simple_white",
    #     )
    #     mlflow.log_figure(plt, "optuna_optimization_history_{study_name}.html")

    #     plt = o.visualization.plot_intermediate_values(study)

    #     plt.update_layout(
    #         yaxis=dict(
    #             showgrid=design.showgrid,
    #         ),
    #         xaxis=dict(
    #             showgrid=design.showgrid,
    #         ),
    #         title=dict(
    #             text=(
    #                 f"Intermediate Values for {study.study_name}"
    #                 if design.print_figure_title
    #                 else ""
    #             ),
    #             font=dict(
    #                 size=design.title_font_size,
    #             ),
    #         ),
    #         hovermode="x",
    #         font=dict(
    #             size=design.legend_font_size,
    #         ),
    #         template="simple_white",
    #     )
    #     mlflow.log_figure(plt, "optuna_intermediate_values_{study_name}.html")

    #     # TODO: the following is highly customizable and maybe should get more attention in the future
    #     plt = o.visualization.plot_parallel_coordinate(
    #         study, params=selected_parallel_params, target_name=self.optimization_metric
    #     )

    #     plt.update_layout(
    #         title=dict(
    #             text=(
    #                 f"Parallel Coordinates for {study.study_name}"
    #                 if design.print_figure_title
    #                 else ""
    #             ),
    #             font=dict(
    #                 size=design.title_font_size,
    #             ),
    #         ),
    #         font=dict(
    #             size=design.legend_font_size,
    #         ),
    #         margin=go.layout.Margin(
    #             b=100  # increase bottom margin so that we can plot long param names correctly
    #         ),
    #         template="simple_white",
    #     )
    #     mlflow.log_figure(plt, "optuna_parallel_coordinate_{study_name}.html")

    #     plt = o.visualization.plot_slice(
    #         study, params=selected_slice_params, target_name=self.optimization_metric
    #     )

    #     plt.update_layout(
    #         title=dict(
    #             text=(
    #                 f"Slice for {study.study_name}" if design.print_figure_title else ""
    #             ),
    #             font=dict(
    #                 size=design.title_font_size,
    #             ),
    #         ),
    #         font=dict(
    #             size=design.legend_font_size,
    #         ),
    #         template="simple_white",
    #     )
    #     mlflow.log_figure(plt, "optuna_slice_{study_name}.html")
