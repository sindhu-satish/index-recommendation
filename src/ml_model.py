"""
ml_model.py
-----------
Trains an XGBoost regression model to predict the benefit score (HypoPG cost delta)
for a given (query, candidate index) pair.

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → training_dataset → ml_model

Usage:
    python ml_model.py --train
    python ml_model.py --train --no-grid-search
    python ml_model.py --recommend --top-k 10
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from candidate_generator import generate_candidates
from db_utils import get_connection
from feature_extractor import build_feature_rows
from training_dataset import (
    add_example_ids,
    feature_matrix,
    infer_numeric_feature_columns,
)
from workload_parser import parse_workload

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)

TRAINING_DIR = os.path.join(_REPO_ROOT, "data", "training")
MODEL_PATH = os.path.join(_REPO_ROOT, "data", "model.json")

PARAM_GRID = {
    "learning_rate":    [0.05, 0.1],
    "max_depth":        [3, 5, 7],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "n_estimators":     [50, 100, 300, 500],
}

FIXED_PARAMS = {
    "early_stopping_rounds": 15,
    "eval_metric":           "rmse",
    "objective":             "reg:squarederror",
    "random_state":          42,
    "n_jobs":                -1,
    "verbosity":             0,
}

DEFAULT_MIN_COST_IMPACT = 50000.0


def feature_cols_path_for_model(model_path: str) -> str:
    """Save feature-column order next to the model file."""
    base, _ = os.path.splitext(model_path)
    return f"{base}_feature_cols.txt"


def load_splits(training_dir: str):
    """Load train / val / test CSVs produced by training_dataset.py."""
    train = pd.read_csv(os.path.join(training_dir, "train.csv"))
    val = pd.read_csv(os.path.join(training_dir, "val.csv"))
    test = pd.read_csv(os.path.join(training_dir, "test.csv"))
    return train, val, test


def add_is_marginal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatibility safety net.

    New training_dataset.py already exports numeric is_marginal.
    If it is missing, derive it from label_source when available.
    """
    df = df.copy()
    if "is_marginal" in df.columns:
        return df
    if "label_source" in df.columns:
        df["is_marginal"] = (df["label_source"] == "marginal").astype(float)
    else:
        df["is_marginal"] = 0.0
    return df


def check_pipeline_alignment(df: pd.DataFrame, label_col: str):
    """Print all features and label stats to confirm the pipeline is wired correctly."""
    feature_cols = infer_numeric_feature_columns(df)

    print("\n" + "=" * 50)
    print("PIPELINE ALIGNMENT CHECK")
    print("=" * 50)

    print(f"\nTotal rows: {len(df)}  |  Total features: {len(feature_cols)}")

    print("\n--- FEATURES ---")
    for col in feature_cols:
        print(f"  {col}")

    print(f"\n--- LABEL: {label_col} ---")
    if label_col in df.columns:
        s = df[label_col]
        print(
            f"  count={len(s)}  min={s.min():.4f}  max={s.max():.4f}  "
            f"mean={s.mean():.4f}  median={s.median():.4f}  nulls={s.isna().sum()}"
        )
    else:
        print(f"  '{label_col}' not found in dataframe!")

    print("=" * 50 + "\n")


def grid_search_cv(
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_column: str,
    param_grid: dict = PARAM_GRID,
    fixed_params: dict = FIXED_PARAMS,
) -> tuple[dict, float, list[str]]:
    """
    Grid search over explicit hyperparameter combinations, including n_estimators,
    evaluated on the train/val split.

    Since n_estimators is now part of the grid, we use the existing validation set
    directly instead of xgb.cv with early stopping choosing tree count implicitly.
    """
    feature_cols = infer_numeric_feature_columns(train)
    X_train, y_train = feature_matrix(train, feature_cols, label_column)
    X_val, y_val = feature_matrix(val, feature_cols, label_column)

    combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    total = len(combinations)

    print(f"\nGrid search: {total} combinations on train/val split")
    print(f"  early_stopping_rounds: {fixed_params['early_stopping_rounds']}")
    print(f"  n_estimators included in search: {param_grid['n_estimators']}\n")

    best_rmse = float("inf")
    best_params = {}

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))

        t0 = time.time()
        model = xgb.XGBRegressor(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            n_estimators=params["n_estimators"],
            objective=fixed_params["objective"],
            random_state=fixed_params["random_state"],
            n_jobs=fixed_params["n_jobs"],
            verbosity=fixed_params["verbosity"],
            eval_metric=fixed_params["eval_metric"],
            early_stopping_rounds=fixed_params["early_stopping_rounds"],
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        elapsed = time.time() - t0

        actual_trees = getattr(model, "best_iteration", None)
        if actual_trees is not None:
            actual_trees += 1
        else:
            actual_trees = params["n_estimators"]

        flag = " <- best" if rmse < best_rmse else ""
        print(
            f"  [{i:2d}/{total}] "
            f"lr={params['learning_rate']} "
            f"depth={params['max_depth']} "
            f"sub={params['subsample']} "
            f"col={params['colsample_bytree']} "
            f"n_estimators={params['n_estimators']} "
            f"-> val RMSE {rmse:.4f} "
            f"(actual={actual_trees} trees, {elapsed:.1f}s){flag}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {**params}

    print(f"\nBest params:  {best_params}")
    print(f"Best val RMSE: {best_rmse:.4f}")

    return best_params, best_rmse, feature_cols


def train_with_best_params(
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_column: str,
    best_params: dict,
    feature_cols: list[str],
) -> xgb.XGBRegressor:

    trainval = pd.concat([train, val], ignore_index=True)
    X_all, y_all = feature_matrix(trainval, feature_cols, label_column)

    model = xgb.XGBRegressor(
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        n_estimators=best_params["n_estimators"],
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(X_all, y_all, verbose=False)
    return model


def train_default(
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_column: str,
) -> tuple[xgb.XGBRegressor, list[str]]:

    feature_cols = infer_numeric_feature_columns(train)
    X_train, y_train = feature_matrix(train, feature_cols, label_column)
    X_val, y_val = feature_matrix(val, feature_cols, label_column)

    model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.7,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=15,
        eval_metric="rmse",
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    return model, feature_cols


def evaluate(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_column: str = "label",
    split_name: str = "test",
) -> dict:
    X, y = feature_matrix(df, feature_cols, label_column)
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    r2 = r2_score(y, preds)

    print(f"\n{split_name.upper()} METRICS  (labels in log space)")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def print_feature_importance(
    model: xgb.XGBRegressor,
    feature_cols: list[str],
    top_n: int = 15,
) -> None:
    importance = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\nTOP {top_n} FEATURES BY GAIN")
    for fname, score in ranked:
        try:
            idx = int(fname[1:])
            real_name = feature_cols[idx]
        except (ValueError, IndexError):
            real_name = fname
        print(f"  {score:10.1f}  {real_name}")


def build_recommendation_features(
    repo_root: str,
    min_cost_impact: float = DEFAULT_MIN_COST_IMPACT,
) -> pd.DataFrame:
    """
    Rebuild fresh feature rows from the current workload/candidate pipeline.
    Recommendation should score current candidates, not reuse old train/test rows.
    """
    queries_dir = os.path.join(repo_root, "queries")
    workload = parse_workload(queries_dir)
    candidates = generate_candidates(workload, min_cost_impact=min_cost_impact)

    conn = get_connection()
    try:
        rows = build_feature_rows(
            conn,
            candidates,
            workload,
            queries_dir=queries_dir,
            schema="public",
        )
    finally:
        conn.close()

    rows = add_example_ids(rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def feature_matrix_inference(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Build inference matrix without requiring a label column.
    Missing features are added as 0.0 so old/new schemas fail gracefully.
    """
    work = df.copy()
    for col in feature_columns:
        if col not in work.columns:
            work[col] = 0.0
    return work.loc[:, feature_columns].astype(np.float64).fillna(0.0)


def recommend(
    model: xgb.XGBRegressor,
    feature_cols: list[str],
    features_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Predict benefit for every (query, candidate) row.

    The model predicts labels in signed-log space, because training_dataset.py
    applies a signed log1p transform:
        raw benefit >= 0  ->  log1p(raw)
        raw benefit <  0  -> -log1p(|raw|)

    To recover total workload benefit for a candidate, we must:
        1. predict per-query benefit in log space
        2. invert each prediction back to raw-space
        3. sum raw per-query benefits across queries

    Summing log-space predictions directly is incorrect.
    """
    X = feature_matrix_inference(features_df, feature_cols)
    features_df = features_df.copy()
    features_df["predicted_log_benefit"] = model.predict(X)

    def inverse_signed_log1p(x: float) -> float:
        x = float(x)
        return np.expm1(x) if x >= 0 else -np.expm1(abs(x))

    features_df["predicted_raw_benefit"] = features_df["predicted_log_benefit"].apply(
        inverse_signed_log1p
    )

    ranked = (
        features_df
        .groupby(
            ["candidate_table", "candidate_cols", "candidate_type"],
            as_index=False
        )[["predicted_raw_benefit", "predicted_log_benefit"]]
        .sum()
        .sort_values("predicted_raw_benefit", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    ranked.index += 1
    return ranked


def format_recommendations(ranked: pd.DataFrame) -> None:
    print("\nRECOMMENDED INDEXES")
    print("=" * 72)
    for rank, row in ranked.iterrows():
        print(f"\n#{rank}")
        print(f"  Predicted total raw benefit: {row['predicted_raw_benefit']:,.0f}")
        print(f"  Sum of per-query log scores: {row['predicted_log_benefit']:.4f}")
        print(f"  CREATE INDEX ON {row['candidate_table']} ({row['candidate_cols']});")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the index recommendation model.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--recommend", action="store_true")
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip grid search and use default hyperparameters.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--training-dir", default=TRAINING_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--repo-root", default=_REPO_ROOT)
    parser.add_argument("--min-cost-impact", type=float, default=DEFAULT_MIN_COST_IMPACT)
    args = parser.parse_args()

    cols_path = feature_cols_path_for_model(args.model_path)

    if args.train:
        print("Loading splits...")
        train, val, test = load_splits(args.training_dir)
        train, val, test = add_is_marginal(train), add_is_marginal(val), add_is_marginal(test)
        print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

        check_pipeline_alignment(train, args.label_column)

        if args.no_grid_search:
            print("\nSkipping grid search — using default hyperparameters.")
            model, feature_cols = train_default(train, val, args.label_column)
            evaluate(model, test, feature_cols, args.label_column, split_name="test")
        else:
            best_params, _, feature_cols = grid_search_cv(train, val, args.label_column)

            print(
                f"\nTraining final model on train+val "
                f"({len(train)+len(val)} rows, {best_params['n_estimators']} trees)..."
            )
            model = train_with_best_params(
                train, val, args.label_column, best_params, feature_cols
            )
            evaluate(model, test, feature_cols, args.label_column, split_name="test")

        print_feature_importance(model, feature_cols)

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save_model(args.model_path)
        with open(cols_path, "w") as f:
            f.write("\n".join(feature_cols))

        print(f"\nModel saved to {args.model_path}")
        print(f"Feature columns saved to {cols_path}")

    elif args.recommend:
        if not os.path.exists(args.model_path):
            print(f"No model found at {args.model_path}. Run with --train first.")
            sys.exit(1)

        if not os.path.exists(cols_path):
            print(f"No feature-column file found at {cols_path}. Run with --train first.")
            sys.exit(1)

        model = xgb.XGBRegressor()
        model.load_model(args.model_path)

        with open(cols_path) as f:
            feature_cols = [line.strip() for line in f if line.strip()]

        print("Building fresh recommendation features from current workload...")
        features_df = build_recommendation_features(
            repo_root=os.path.abspath(args.repo_root),
            min_cost_impact=args.min_cost_impact,
        )

        if features_df.empty:
            print("No recommendation feature rows were generated.")
            sys.exit(1)

        features_df = add_is_marginal(features_df)

        ranked = recommend(model, feature_cols, features_df, top_k=args.top_k)
        format_recommendations(ranked)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()