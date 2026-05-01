"""
training_dataset.py
-------------------
Assemble ML-ready tables from `feature_extractor` output, merge labels
produced by `hypopg_labeler`, and export train / validation / test splits.

Changes from original:
    - get_connection imported from db_utils instead of feature_extractor
      (fixes broken import since get_connection was moved out of feature_extractor).
    - load_labels_csv updated for Option B per-query labels: when query_name is
      present in the labels CSV, example_id is constructed from the three columns
      (query_name|candidate_table|candidate_cols) so merge_features_and_labels
      joins on it exactly — one label per feature row, no broadcasting.
      Option A (per-candidate broadcasting) still works as a legacy compatibility
      path when query_name is absent.
    - min_frequency replaced with min_cost_impact throughout.
    - apply_log_transform updated to signed log1p: compresses the huge cost scale
      so the model isn't dominated by q20's billion-scale costs. Uses signed log
      (positive: log1p(x), negative: -log1p(|x|)) so indexes that actively hurt
      performance are distinguishable from neutral ones. Previously clip(lower=0)
      destroyed the anti-pattern signal by treating harmful and neutral indexes
      identically.
    - is_marginal conversion moved here from ml_model.py: label_source
      ('marginal' | 'individual') is converted to a numeric is_marginal float
      column inside merge_features_and_labels so the exported CSVs are
      self-contained. ml_model.py's add_is_marginal() remains as a safety net
      for older CSVs that predate this change.

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → training_dataset → ml_model
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from candidate_generator import generate_candidates
from db_utils import get_connection
from feature_extractor import build_feature_rows
from workload_parser import parse_workload

ID_METADATA_COLUMNS: Tuple[str, ...] = (
    "example_id",
    "query_name",
    "candidate_table",
    "candidate_cols",
    "candidate_type",
    # label_source is string metadata; the numeric version is is_marginal
    "label_source",
)


def example_id(query_name: str, candidate_table: str, candidate_cols: str) -> str:
    """Stable row id for merging features with hypopg_labeler output."""
    return f"{query_name}|{candidate_table}|{candidate_cols}"


def add_example_ids(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["example_id"] = example_id(
            str(r["query_name"]),
            str(r["candidate_table"]),
            str(r["candidate_cols"]),
        )
        out.append(r)
    return out


def build_feature_dataframe(
    conn,
    candidates: List[dict],
    workload: List[dict],
    queries_dir: str,
    schema: str = "public",
) -> pd.DataFrame:
    rows = build_feature_rows(
        conn,
        candidates,
        workload,
        queries_dir=queries_dir,
        schema=schema,
    )
    rows = add_example_ids(rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_labels_csv(path: str) -> pd.DataFrame:
    """
    Load labels from hypopg_labeler CSV.

    Handles two formats:

    Option B (per-query labels — preferred):
        query_name, candidate_table, candidate_cols, label, label_source
        q3,  lineitem, l_partkey,l_suppkey, 12000, marginal
        q12, lineitem, l_partkey,l_suppkey, 1098000000, marginal

        When query_name is present, example_id is constructed from the three
        columns so merge_features_and_labels can join on it exactly — one label
        per feature row, no broadcasting.

    Option A (per-candidate labels — legacy compatibility path):
        candidate_table, candidate_cols, label, label_source
        lineitem, l_partkey,l_suppkey, 1100426205, marginal

        When query_name is absent, the label is broadcast to all query rows
        for that candidate in merge_features_and_labels.
    """
    df = pd.read_csv(path)

    if "example_id" not in df.columns:
        if all(c in df.columns for c in ("query_name", "candidate_table", "candidate_cols")):
            # Option B — construct example_id so the exact join path is used
            df = df.copy()
            df["example_id"] = (
                df["query_name"].astype(str) + "|" +
                df["candidate_table"].astype(str) + "|" +
                df["candidate_cols"].astype(str)
            )
        else:
            # Option A — legacy broadcasting path, just needs candidate_table + candidate_cols
            missing = {
                c for c in ("candidate_table", "candidate_cols")
                if c not in df.columns
            }
            if missing:
                raise ValueError(
                    f"Labels CSV must have 'example_id', or all three of "
                    f"query_name/candidate_table/candidate_cols (Option B), "
                    f"or at least candidate_table/candidate_cols (Option A legacy broadcasting). "
                    f"Missing: {sorted(missing)}"
                )
    return df


def merge_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Merge labels onto features.

    Preferred path:
        If labels has 'example_id', join on that so each (query, candidate)
        feature row receives exactly its own per-query label.

    Legacy compatibility path:
        If labels lacks 'example_id', join on (candidate_table, candidate_cols)
        and broadcast the label to every query row that shares that candidate.
        This supports older per-candidate label files.

    After merging, label_source ('marginal' | 'individual') is converted to a
    numeric is_marginal column (1.0 / 0.0) so it is picked up by
    infer_numeric_feature_columns and included in the exported CSVs as a proper
    feature. label_source itself is retained as metadata for debugging.
    """
    if features.empty:
        return features.copy()

    label_key_cols = {"example_id", "candidate_table", "candidate_cols", "query_name"}

    if label_columns is None:
        label_columns = [c for c in labels.columns if c not in label_key_cols]
    else:
        for c in label_columns:
            if c not in labels.columns:
                raise ValueError(f"Label column {c!r} not in labels dataframe")

    if "example_id" in labels.columns:
        merge_df = labels[["example_id"] + list(label_columns)].drop_duplicates(subset=["example_id"])
        merged = features.merge(merge_df, on="example_id", how="left")
    else:
        join_keys = ["candidate_table", "candidate_cols"]
        merge_df = labels[join_keys + list(label_columns)].drop_duplicates(subset=join_keys)
        merged = features.merge(merge_df, on=join_keys, how="left")

    # Convert label_source -> is_marginal so exported CSVs are self-contained.
    if "label_source" in merged.columns:
        merged["is_marginal"] = (merged["label_source"] == "marginal").astype(float)
    else:
        merged["is_marginal"] = 0.0

    return merged


def apply_log_transform(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    Signed log1p-transform the label column to compress the large cost scale
    while preserving the sign of negative benefits.

    Benefit scores span many orders of magnitude — q20 alone has costs in
    the billions while small-table queries sit in the thousands. Without this,
    the model learns almost entirely from high-cost outliers and all
    recommendations end up being lineitem indexes.

    Signed log transform:
        positive benefit  ->  log1p(x)
        zero benefit      ->  0.0
        negative benefit  ->  -log1p(|x|)

    NaN values are preserved as NaN.
    """
    def _signed_log1p(x: Any) -> float:
        if pd.isna(x):
            return np.nan
        x = float(x)
        return np.log1p(x) if x >= 0 else -np.log1p(abs(x))

    df = df.copy()
    df[label_column] = df[label_column].apply(_signed_log1p)
    return df


def infer_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """All numeric columns except ids and label columns."""
    skip = set(ID_METADATA_COLUMNS)
    cols: List[str] = []
    for c in df.columns:
        if c in skip:
            continue
        if c.startswith("label_") or c == "label":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def train_val_test_split_dataframe(
    df: pd.DataFrame,
    label_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.dropna(subset=[label_column]).copy()
    if work.empty:
        raise ValueError("No rows with non-null labels; cannot split.")

    strat = None
    if stratify:
        y = work[label_column]
        if y.nunique() >= 2 and y.value_counts().min() >= 2:
            strat = y

    idx = np.arange(len(work))
    i_trainval, i_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=strat
    )
    trainval = work.iloc[i_trainval]
    test = work.iloc[i_test]

    val_ratio = val_size / max(1e-9, (1.0 - test_size))
    if len(trainval) < 2:
        train, val = trainval, trainval.iloc[0:0]
    else:
        y_tv = trainval[label_column]
        strat_tv = None
        if stratify and y_tv.nunique() >= 2 and y_tv.value_counts().min() >= 2:
            strat_tv = y_tv
        i_train, i_val = train_test_split(
            np.arange(len(trainval)),
            test_size=val_ratio,
            random_state=random_state,
            stratify=strat_tv,
        )
        train = trainval.iloc[i_train]
        val = trainval.iloc[i_val]

    return train, val, test


def feature_matrix(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_columns is None:
        feature_columns = infer_numeric_feature_columns(df)
    X = df.loc[:, list(feature_columns)].astype(np.float64).fillna(0.0)
    y = df[label_column]
    return X, y


def write_labels_template(features: pd.DataFrame, path: str) -> None:
    cols = ["example_id", "query_name", "candidate_table", "candidate_cols", "label"]
    have = [c for c in cols if c in features.columns]
    tpl = features[have].drop_duplicates(subset=["example_id"]).copy()
    if "label" not in tpl.columns:
        tpl["label"] = np.nan
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tpl.to_csv(path, index=False)


def export_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str,
    fmt: str = "csv",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "csv":
        train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    elif fmt == "parquet":
        try:
            train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
            val.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
            test.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
        except ImportError as e:
            raise ImportError("Parquet export requires pyarrow.") from e
    else:
        raise ValueError("fmt must be 'csv' or 'parquet'")


def build_training_splits(
    conn,
    workload: List[dict],
    candidates: List[dict],
    queries_dir: str,
    labels_path: Optional[str] = None,
    label_column: str = "label",
    label_columns: Optional[Sequence[str]] = None,
    placeholder_label: Optional[float] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = False,
    schema: str = "public",
    log_transform: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_feature_dataframe(
        conn, candidates, workload, queries_dir=queries_dir, schema=schema
    )
    if features.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    if labels_path:
        labels = load_labels_csv(labels_path)
        features = merge_features_and_labels(features, labels, label_columns=label_columns)
        if label_column not in features.columns:
            raise ValueError(f"After merge, expected label column {label_column!r} is missing")
    elif placeholder_label is not None:
        features = features.copy()
        features[label_column] = float(placeholder_label)

    if label_column not in features.columns or features[label_column].isna().all():
        return features, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if log_transform and label_column in features.columns:
        features = apply_log_transform(features, label_column)

    train, val, test = train_val_test_split_dataframe(
        features,
        label_column=label_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
    return features, train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training dataset from feature_extractor + labels CSV."
    )
    parser.add_argument("--repo-root", default=os.path.dirname(_SRC_DIR))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--labels", default=None, help="CSV from hypopg_labeler.")
    parser.add_argument("--label-template", default=None)
    parser.add_argument(
        "--min-cost-impact",
        type=float,
        default=50000.0,
        help="Candidate generator min_cost_impact threshold.",
    )
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--placeholder-label", type=float, default=None)
    parser.add_argument("--fmt", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument(
        "--no-log-transform",
        action="store_true",
        help="Disable signed log1p label transform.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    queries_dir = os.path.join(repo_root, "queries")
    out_dir = args.out_dir or os.path.join(repo_root, "data", "training")

    workload = parse_workload(queries_dir)
    candidates = generate_candidates(workload, min_cost_impact=args.min_cost_impact)

    conn = get_connection()
    try:
        if args.label_template:
            features = build_feature_dataframe(conn, candidates, workload, queries_dir=queries_dir)
            write_labels_template(features, args.label_template)
            print(f"Wrote label template ({len(features)} rows) to {args.label_template}")
            return

        features, train, val, test = build_training_splits(
            conn,
            workload,
            candidates,
            queries_dir=queries_dir,
            labels_path=args.labels,
            label_column=args.label_column,
            placeholder_label=args.placeholder_label,
            stratify=args.stratify,
            log_transform=not args.no_log_transform,
        )
    finally:
        conn.close()

    if features.empty:
        print("No feature rows; check DB, workload, and candidates.")
        return

    if train.empty and args.labels is None and args.placeholder_label is None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "features_unlabeled.csv")
        features.to_csv(path, index=False)
        print(f"Wrote {len(features)} unlabeled rows to {path}.")
        return

    if train.empty:
        print("No labeled rows to split; check labels file and label column.")
        return

    export_splits(train, val, test, out_dir, fmt=args.fmt)
    print(
        f"Wrote train/val/test to {out_dir} ({args.fmt}): "
        f"{len(train)} / {len(val)} / {len(test)} rows."
    )


if __name__ == "__main__":
    main()