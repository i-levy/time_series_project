"""
Hyperparameter tuning for LightGBM quantile regression flood forecasting.

Tunes one set of hyperparameters per forecast horizon (1, 3, 7 days),
shared across the three quantile models (2.5th, 50th, 97.5th percentiles).

The objective function averages MAE across all three quantiles and all
CV folds, giving a single scalar for Optuna to minimize.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler
import json
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Configuration ─────────────────────────────────────────────────────────────

HORIZONS   = [1, 3, 7]
QUANTILES  = [0.025, 0.50, 0.975]
N_SPLITS   = 5      # Walk-forward CV folds
N_TRIALS   = 200    # Optuna trials per horizon — increase for more thorough search
RANDOM_SEED = 59

# Output path for best parameters
OUTPUT_PATH = "best_params.json"


# ── Data ──────────────────────────────────────────────────────────────────────
# Replace these with your actual dataframes/variables

# df_features : your full feature-engineered dataframe
# targets     : dict mapping horizon -> target column name
#               e.g. {1: 'gage_height_t1', 3: 'gage_height_t3', 7: 'gage_height_t7'}
# feature_cols: list of feature column names (excludes date, target cols, raw gage height)

# Example (replace with your actual variables):
# from your_notebook import df_features, targets, feature_cols


# ── Objective function ────────────────────────────────────────────────────────

def make_objective(horizon, X, y, n_splits, quantiles, seed):
    """
    Returns an Optuna objective function for a given forecast horizon.
    Minimizes mean MAE averaged across quantile models and CV folds.
    """
    def objective(trial):
        params = {
            "objective":        "quantile",
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbose":          -1,
            "n_jobs":           -1,
            "random_state":     seed,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_maes = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            quantile_maes = []
            for q in quantiles:
                model = LGBMRegressor(alpha=q, **params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(False),
                    ],
                )
                preds = model.predict(X_val)
                quantile_maes.append(mean_absolute_error(y_val, preds))

            fold_maes.append(np.mean(quantile_maes))

        return np.mean(fold_maes)

    return objective


# ── Coverage evaluation helper ────────────────────────────────────────────────

def evaluate_coverage(params, horizon, X, y, n_splits, quantiles):
    """
    Evaluate interval coverage and width for a given set of hyperparameters.
    Useful for sanity-checking tuned parameters before final training.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        preds = {}
        for q in quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(False),
                ],
            )
            preds[q] = model.predict(X_val)

        mae      = mean_absolute_error(y_val, preds[0.50])
        coverage = np.mean(
            (y_val.values >= preds[quantiles[0]]) &
            (y_val.values <= preds[quantiles[-1]])
        )
        width = np.mean(preds[quantiles[-1]] - preds[quantiles[0]])

        fold_metrics.append({
            "fold":           fold + 1,
            "mae":            mae,
            "coverage":       coverage,
            "interval_width": width,
        })
        print(f"  Horizon {horizon}d | Fold {fold+1} | "
              f"MAE: {mae:.3f} ft | "
              f"Coverage: {coverage:.1%} | "
              f"Interval Width: {width:.3f} ft")

    results = pd.DataFrame(fold_metrics)
    print(f"\n  Horizon {horizon}d summary — "
          f"MAE: {results['mae'].mean():.3f} ± {results['mae'].std():.3f} ft | "
          f"Coverage: {results['coverage'].mean():.1%} | "
          f"Interval Width: {results['interval_width'].mean():.3f} ft\n")

    return results


# ── Main tuning loop ──────────────────────────────────────────────────────────

def tune_all_horizons(df_features, targets, feature_cols):
    """
    Run Optuna tuning for each forecast horizon and return best parameters.

    Args:
        df_features  : feature-engineered dataframe
        targets      : dict mapping horizon -> target column name
        feature_cols : list of feature column names

    Returns:
        best_params  : dict mapping horizon -> best hyperparameter dict
    """
    X = df_features[feature_cols]
    best_params = {}

    for horizon in HORIZONS:
        print(f"{'='*60}")
        print(f"Tuning horizon: {horizon} day(s) ahead")
        print(f"{'='*60}")

        y = df_features[targets[horizon]]

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=RANDOM_SEED),
        )

        study.enqueue_trial({
            "n_estimators":      1000,
            "learning_rate":     0.01,
            "max_depth":         4,
            "num_leaves":        31,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "min_child_samples": 20,
            "reg_alpha":         1e-8,
            "reg_lambda":        1e-8,
        })
        
        objective = make_objective(
            horizon=horizon,
            X=X,
            y=y,
            n_splits=N_SPLITS,
            quantiles=QUANTILES,
            seed=RANDOM_SEED,
        )


        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

        print(f"\nBest MAE for horizon {horizon}d: {study.best_value:.4f} ft")
        print(f"Best params: {study.best_params}\n")

        best_params[horizon] = study.best_params

        # Sanity check: re-evaluate with tuned params including coverage
        print(f"Evaluating coverage with tuned params for horizon {horizon}d...")
        evaluate_coverage(
            params=study.best_params,
            horizon=horizon,
            X=X,
            y=y,
            n_splits=N_SPLITS,
            quantiles=QUANTILES,
        )

    # Save best params to JSON for use in final model training
    serializable = {str(k): v for k, v in best_params.items()}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Best parameters saved to {OUTPUT_PATH}")

    return best_params


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Replace with your actual variables
    best_params = tune_all_horizons(df_features, targets, feature_cols)
