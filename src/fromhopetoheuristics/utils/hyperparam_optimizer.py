import optuna as o
from typing import List, Dict

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
        enabled_hyperparameters,
        optimization_metric: List,
        n_jobs: int,
        selective_optimization: bool,
        resume_study: bool,
        pool_process,
        pruner,
        pruner_startup_trials,
        pruner_warmup_steps,
        pruner_interval_steps,
        pruner_min_trials,
    ):
        # storage = self.initialize_storage(host, port, path, password)

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
        self.selective_optimization = selective_optimization
        self.pool_process = pool_process

        self.enabled_hyperparameters = enabled_hyperparameters

        self.studies = []
        self.n_jobs = n_jobs

        n_studies = self.n_jobs if self.pool_process else 1

        direction = "minimize"  # TODO: replace if required

        for n_it in range(n_studies):
            resume_study = resume_study or (n_studies > 1 and n_it != 0)

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

    def update_variable_parameters(self, trial, parameters, prefix=""):
        raise NotImplementedError("need to be revised")
        updated_variable_parameters = dict()
        choice = None  # indicate that this level is not a choice

        for parameter, value in parameters.items():
            # the following section removes any reserved strings from the parameter name and draws a choice if possible
            if "_range_quant" in parameter:
                if not self.toggle_classical_quant and self.selective_optimization:
                    continue  # continue if its a quantum parameter and we are classical
                param_name = parameter.replace("_range_quant", "")
            elif "_range" in parameter:
                if self.toggle_classical_quant and self.selective_optimization:
                    continue  # continue if its a classical parameter and we are quantum
                param_name = parameter.replace("_range", "")
            elif "_choice" in parameter:
                param_name = parameter.replace("_choice", "")
                assert type(value) == dict
                # if we have the choice; go and ask the trial what to do
                choice = trial.suggest_categorical(param_name, value.keys())
            else:
                # there is nothing fancy going on, copy the parameter name as is
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
                updated_variable_parameters[param_name] = trial.suggest_categorical(
                    prefix + param_name, value
                )

        return updated_variable_parameters

    def minimize(self, idx: int):
        self.studies[idx].optimize(
            self.run_trial, n_trials=self.n_trials, n_jobs=self.n_jobs
        )

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
