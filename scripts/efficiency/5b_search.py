"""Step 5b · Hyperparameter search with a budget.

uv run --group efficiency python scripts/efficiency/5b_search.py

Four ways to spend the same budget (BUDGET_S seconds of wall time, at most 27 trials) on
three LightGBM parameters. A trial is scored on 3 time folds of 30 days inside the last
training year, early stopping on each fold. The best parameters of each search are then
refitted on both training years and scored once on the test year, next to two untuned
models scored the same way.
"""

import time

import numpy as np
import optuna
from _common import DAILY, fit_lightgbm, history_dataset, mae, show, split
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lucerne_temp.common.model import Forecaster
from lucerne_temp.training.folds import time_folds

BUDGET_S, MAX_TRIALS = 60, 27
GRID = {
    "learning_rate": [0.03, 0.1, 0.3],
    "num_leaves": [7, 31, 127],
    "min_child_samples": [5, 50, 500],
}


def space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500, log=True),
    }


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train, test = split(history_dataset())
    last_year = train[train.index >= train.index.max() - np.timedelta64(365, "D")]
    folds = time_folds(last_year, n_splits=3, test_days=30)
    ((fit_idx, val_idx),) = time_folds(train, n_splits=1, test_days=90)

    def cv(params: dict, trial: optuna.Trial | None = None) -> float:
        scores = []
        for k, (a, b) in enumerate(folds):
            fold_fit, fold_val = last_year.iloc[a], last_year.iloc[b]
            model = fit_lightgbm(fold_fit, fold_val, DAILY, **params)
            scores.append(mae(fold_val["target"], model.predict(fold_val)))
            if trial is not None:
                trial.report(float(np.mean(scores)), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores))

    def refit_test(params: dict) -> float:
        model = fit_lightgbm(train.iloc[fit_idx], train.iloc[val_idx], DAILY, **params)
        return round(mae(test["target"], model.predict(test)), 3)

    searches = {
        "grid (3x3x3)": (optuna.samplers.GridSampler(GRID, seed=0), optuna.pruners.NopPruner()),
        "random": (optuna.samplers.RandomSampler(seed=0), optuna.pruners.NopPruner()),
        "random + median pruner": (
            optuna.samplers.RandomSampler(seed=0),
            optuna.pruners.MedianPruner(n_startup_trials=5),
        ),
        "TPE": (optuna.samplers.TPESampler(seed=0), optuna.pruners.NopPruner()),
    }
    rows = []
    for name, (sampler, pruner) in searches.items():
        study = optuna.create_study(sampler=sampler, pruner=pruner)
        t0 = time.perf_counter()
        study.optimize(lambda t: cv(space(t), t), n_trials=MAX_TRIALS, timeout=BUDGET_S)
        states = [t.state for t in study.trials]
        rows.append(
            {
                "search": name,
                "wall s": round(time.perf_counter() - t0),
                "trials": len(states),
                "pruned": states.count(optuna.trial.TrialState.PRUNED),
                "best CV MAE": round(study.best_value, 3),
                "test MAE": refit_test(study.best_params),
                "best params": {k: round(v, 3) for k, v in study.best_params.items()},
            }
        )
    rows.append(
        {
            "search": "untuned LightGBM, defaults + early stopping",
            "best CV MAE": round(cv({"learning_rate": 0.1}), 3),
            "test MAE": refit_test({"learning_rate": 0.1}),
        }
    )
    ridge = lambda: Forecaster(make_pipeline(StandardScaler(), Ridge()), DAILY)  # noqa: E731
    ridge_cv = [
        mae(
            last_year.iloc[b]["target"],
            ridge().fit(last_year.iloc[a], last_year.iloc[a]["target"]).predict(last_year.iloc[b]),
        )
        for a, b in folds
    ]
    ridge_test = mae(test["target"], ridge().fit(train, train["target"]).predict(test))
    rows.append(
        {
            "search": "untuned Ridge",
            "best CV MAE": round(float(np.mean(ridge_cv)), 3),
            "test MAE": round(ridge_test, 3),
        }
    )
    show(
        rows, f"Budget {BUDGET_S} s or {MAX_TRIALS} trials per search; CV = 3 time folds x 30 days"
    )


if __name__ == "__main__":
    main()
