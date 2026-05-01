"""
feature_extractor.py
--------------------
Pulls PostgreSQL catalog statistics (pg_stats) and optimizer estimates from
EXPLAIN (FORMAT JSON) for workload queries. Produces fixed-width numeric
summaries suitable for tree-based models.

Changes from original:
    - get_connection, normalize_query_for_postgres, explain_query_json,
      _walk_plan, summarize_explain_json all moved to db_utils.py.
      Imported from there to eliminate circular dependency with workload_parser.
    - Added estimate_write_penalty: approximates index maintenance cost from
      table size and index width. Important for workload-driven design —
      benefit must outweigh write overhead.
    - Added clustered_candidate feature to build_feature_rows so the model
      knows when a candidate is competing for the single clustered index slot.
    - Added workload_clause_features: extracts per-column GROUP BY and ORDER BY
      cost impact from the updated workload_parser output (in_group_by /
      in_order_by flags). Exposes 6 new features in build_feature_rows:
        workload_max_group_by_freq, workload_sum_group_by_freq,
        workload_max_order_by_freq, workload_sum_order_by_freq,
        first_col_in_group_by, first_col_in_order_by.
      This allows the model to learn sort-elimination opportunities: a B+ tree
      index on a GROUP BY or ORDER BY column lets the optimizer skip an
      expensive in-memory Sort node entirely. The first_col_* features are
      particularly important because sort elimination only applies when the
      leading index column matches the sort key.
    - Added is_composite feature: binary 1.0/0.0 encoding of candidate_type
      ('composite' vs 'single'). candidate_type was previously in
      ID_METADATA_COLUMNS and skipped by infer_numeric_feature_columns entirely.
      Single vs composite indexes have fundamentally different cost/benefit
      profiles — composite indexes carry higher write penalty and maintenance
      cost but can serve multiple predicates and enable index-only scans across
      more columns. The model was previously blind to this distinction.
    - Fixed conn.rollback() no-op in explain_workload: autocommit=True makes
      rollback a no-op. Removed it since EXPLAIN failures leave no state to
      clean up.
    - Fixed n_distinct sign in fetch_pg_stats_row: PostgreSQL stores negative
      n_distinct as a fraction of table rows (e.g. -1.0 = all rows unique,
      -0.5 = 50% distinct). The model was seeing -1.0 as low cardinality when
      it actually means highest cardinality. Fixed with abs() so the sign is
      always positive and the model reads cardinality correctly.
    - Fixed write_penalty_proxy negative values in estimate_write_penalty:
      pg_class.reltuples returns -1 when ANALYZE has not been run yet. Clamped
      with max(0.0, reltuples) so the penalty is never negative.
    - Made workload-derived features cost-based instead of count-based:
      workload_column_frequencies and workload_clause_features now sum query_cost
      so feature extraction aligns with the cost-aware workload_parser and
      candidate_generator design.

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → ml_model
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from db_utils import get_connection, normalize_query_for_postgres, explain_query_json
from workload_parser import load_queries, parse_workload

QUERIES_DIR = "queries"


def _parse_pg_array(value: Any) -> Optional[List[Any]]:
    """Parse PostgreSQL array text or pass through list from psycopg2."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if value == "{}":
            return []
        inner = value.strip()
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1]
        if not inner:
            return []
        parts = re.split(r",(?![^(]*\))", inner)
        return [p.strip().strip('"') for p in parts if p.strip()]
    return None


def _histogram_summary(bounds: Any) -> Dict[str, float]:
    """Turn histogram_bounds into a small numeric fingerprint."""
    arr = _parse_pg_array(bounds)
    if not arr:
        return {"hist_n_buckets": 0.0, "hist_span": 0.0}
    n = len(arr)
    try:
        first = float(arr[0])
        last = float(arr[-1])
        span = abs(last - first)
    except (TypeError, ValueError):
        span = float(n)
    return {
        "hist_n_buckets": float(max(0, n - 1)),
        "hist_span": float(span),
    }


def _mcv_top_freq(freqs: Any) -> float:
    arr = _parse_pg_array(freqs)
    if not arr:
        return 0.0
    best = 0.0
    for x in arr:
        try:
            best = max(best, float(x))
        except (TypeError, ValueError):
            continue
    return best


def fetch_pg_stats_row(
    conn, table: str, column: str, schema: str = "public"
) -> Optional[Dict[str, Any]]:
    """
    One row from pg_stats for (schema, table, column).
    Returns None if missing (e.g. system column or no stats yet).

    n_distinct sign handling: PostgreSQL stores negative n_distinct as a
    fraction of table rows rather than an absolute count. For example:
        -1.0  → every value is unique (highest cardinality)
        -0.5  → ~50% of rows have distinct values
    The model would misread -1.0 as low cardinality (opposite of reality).
    abs() normalizes the sign so the model always sees a non-negative value.
    Note: this loses the fraction-vs-absolute distinction, but for a tree-based
    model that splits on thresholds, the corrected sign is more important than
    preserving the exact semantic.
    """
    q = """
        SELECT
            null_frac,
            avg_width,
            n_distinct,
            correlation,
            most_common_freqs,
            histogram_bounds
        FROM pg_stats
        WHERE schemaname = %s AND tablename = %s AND attname = %s
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table, column))
        row = cur.fetchone()
    if not row:
        return None
    null_frac, avg_width, n_distinct, correlation, mcf, hb = row
    hist = _histogram_summary(hb)
    corr = correlation if correlation is not None else 0.0

    nd = abs(float(n_distinct)) if n_distinct is not None else 0.0

    return {
        "null_frac": float(null_frac or 0.0),
        "avg_width": float(avg_width or 0.0),
        "n_distinct": nd,
        "correlation": float(corr),
        "mcv_top_freq": _mcv_top_freq(mcf),
        **hist,
    }


def fetch_pg_stats_for_columns(
    conn,
    table_columns: Iterable[Tuple[str, str]],
    schema: str = "public",
) -> Dict[str, Dict[str, float]]:
    """
    Batch-fetch pg_stats for unique (table, column) pairs.
    Keys are 'table.column' (same convention as candidate_generator frequencies).
    """
    seen: Set[Tuple[str, str]] = set()
    out: Dict[str, Dict[str, float]] = {}
    for table, col in table_columns:
        key_t = (table, col)
        if key_t in seen:
            continue
        seen.add(key_t)
        row = fetch_pg_stats_row(conn, table, col, schema)
        if row:
            out[f"{table}.{col}"] = row
        else:
            out[f"{table}.{col}"] = {
                "null_frac": 0.0,
                "avg_width": 0.0,
                "n_distinct": 0.0,
                "correlation": 0.0,
                "mcv_top_freq": 0.0,
                "hist_n_buckets": 0.0,
                "hist_span": 0.0,
            }
    return out


def aggregate_column_stats(
    stats_by_col: Mapping[str, Mapping[str, float]], columns: Sequence[str], table: str
) -> Dict[str, float]:
    """
    Combine per-column pg_stats for a multi-column candidate into one fixed vector
    using simple summaries (mean / min / max).
    """
    keys = [f"{table}.{c}" for c in columns]
    rows = [stats_by_col[k] for k in keys if k in stats_by_col]
    if not rows:
        return {
            "colstats_mean_null_frac": 0.0,
            "colstats_mean_corr": 0.0,
            "colstats_min_ndistinct": 0.0,
            "colstats_max_mcv_top": 0.0,
            "colstats_mean_hist_buckets": 0.0,
        }

    def mean(name: str) -> float:
        return sum(float(r[name]) for r in rows) / len(rows)

    return {
        "colstats_mean_null_frac": mean("null_frac"),
        "colstats_mean_corr": mean("correlation"),
        "colstats_min_ndistinct": min(float(r["n_distinct"]) for r in rows),
        "colstats_max_mcv_top": max(float(r["mcv_top_freq"]) for r in rows),
        "colstats_mean_hist_buckets": mean("hist_n_buckets"),
    }


def estimate_write_penalty(
    conn, table: str, columns: Sequence[str], schema: str = "public"
) -> Dict[str, float]:
    """
    Approximates the maintenance overhead of an index.
    A wider index on a table with more rows incurs a higher penalty during
    INSERT/UPDATE operations.

    reltuples is clamped to 0 with max(0.0, reltuples) because PostgreSQL sets
    reltuples = -1 when ANALYZE has not been run on the table yet. Without the
    clamp, the penalty would be negative, which is semantically wrong and
    misleads the model into thinking a large index has negative write cost.
    """
    try:
        q = """
            SELECT c.reltuples, c.relpages
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
        """
        with conn.cursor() as cur:
            cur.execute(q, (schema, table))
            row = cur.fetchone()
            if not row:
                return {"write_penalty_proxy": 0.0}
            reltuples, relpages = row

            # Roughly estimate the width of the index keys (4 bytes per column as a naive baseline)
            index_width_bytes = len(columns) * 4.0

            penalty = (max(0.0, float(reltuples)) / 1000.0) * index_width_bytes
            return {"write_penalty_proxy": float(penalty)}
    except Exception:
        return {"write_penalty_proxy": 0.0}


def list_indexed_column_sets(conn, table: str, schema: str = "public") -> List[Tuple[str, ...]]:
    """Return list of column tuples for each non-primary index on the table (ordered)."""
    q = """
        SELECT ARRAY(
            SELECT a.attname
            FROM unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ord)
            JOIN pg_attribute a
              ON a.attrelid = ix.indrelid AND a.attnum = k.attnum AND NOT a.attisdropped
            ORDER BY k.ord
        ) AS cols
        FROM pg_class t
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_index ix ON t.oid = ix.indrelid
        WHERE n.nspname = %s AND t.relkind = 'r' AND t.relname = %s
          AND NOT ix.indisprimary
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table))
        rows = cur.fetchall()
    out: List[Tuple[str, ...]] = []
    for (cols,) in rows:
        if cols:
            out.append(tuple(cols))
    return out


def existing_index_overlap_features(
    conn, table: str, columns: Sequence[str], schema: str = "public"
) -> Dict[str, float]:
    """Heuristic: does an index already cover this column set as a prefix?"""
    want = tuple(columns)
    idx_sets = list_indexed_column_sets(conn, table, schema)
    exact = 0.0
    prefix = 0.0
    for icols in idx_sets:
        if icols == want:
            exact = 1.0
        if len(icols) >= len(want) and icols[: len(want)] == want:
            prefix = 1.0
    return {
        "idx_already_exact": exact,
        "idx_already_prefix": max(exact, prefix),
        "n_existing_indexes_on_table": float(len(idx_sets)),
    }


def explain_workload(
    conn, queries: Optional[Mapping[str, str]] = None, queries_dir: str = QUERIES_DIR
) -> Dict[str, Dict[str, float]]:
    """Summarized EXPLAIN output per query name."""
    if queries is None:
        queries = load_queries(queries_dir)
    out: Dict[str, Dict[str, float]] = {}
    for name in sorted(queries.keys()):
        sql = queries[name]
        try:
            out[name] = explain_query_json(conn, sql)
        except Exception as ex:
            print(f"[warn] EXPLAIN failed for {name}: {ex}")
            out[name] = {
                "plan_total_cost": 0.0,
                "plan_startup_cost": 0.0,
                "plan_rows": 0.0,
                "n_seq_scan": 0.0,
                "n_index_scan": 0.0,
                "explain_error": 1.0,
                "explain_error_msg_len": float(len(str(ex))),
            }
        else:
            out[name]["explain_error"] = 0.0
            out[name]["explain_error_msg_len"] = 0.0
    return out


def workload_column_frequencies(workload: List[dict]) -> Dict[str, float]:
    """
    Sum total query_cost per table.column.

    This is cost-based, not count-based, so columns referenced by more expensive
    queries receive larger feature values. workload_parser already deduplicates
    (table, column) within a query, so each query contributes its cost at most
    once per column.
    """
    freq: Dict[str, float] = {}
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        freq[key] = freq.get(key, 0.0) + float(item.get("query_cost", 1.0))
    return freq


def workload_clause_features(workload: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Per-column cost impact of how often that column participates in GROUP BY
    or ORDER BY clauses, using the in_group_by / in_order_by flags added by
    the updated workload_parser.

    These values are cost-weighted rather than count-weighted: a column used in
    the ORDER BY of one very expensive query should matter more than one used in
    many cheap queries.

    Returns dict mapping 'table.column' ->
        {'group_by_cost': float, 'order_by_cost': float}.

    Safe to call on old-format workloads that lack the clause flags — defaults to 0.
    """
    result: Dict[str, Dict[str, float]] = {}
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        if key not in result:
            result[key] = {"group_by_cost": 0.0, "order_by_cost": 0.0}
        qcost = float(item.get("query_cost", 1.0))
        if item.get("in_group_by"):
            result[key]["group_by_cost"] += qcost
        if item.get("in_order_by"):
            result[key]["order_by_cost"] += qcost
    return result


def queries_touching_table(workload: List[dict], table: str) -> Set[str]:
    return {item["query"] for item in workload if item["table"] == table}


def build_feature_rows(
    conn,
    candidates: List[dict],
    workload: List[dict],
    queries: Optional[Mapping[str, str]] = None,
    queries_dir: str = QUERIES_DIR,
    schema: str = "public",
) -> List[Dict[str, Any]]:
    """
    Join candidate index metadata with pg_stats, existing-index flags, write
    penalty estimates, workload cost-impact features, clause cost features,
    and per-query EXPLAIN summaries for queries that touch the candidate's table.

    Each row represents one (query_name, candidate) pair.

    Clause features added (6 total):
        workload_max_group_by_freq  — max GROUP BY cost impact across candidate columns
        workload_sum_group_by_freq  — sum of GROUP BY cost impact
        workload_max_order_by_freq  — max ORDER BY cost impact across candidate columns
        workload_sum_order_by_freq  — sum of ORDER BY cost impact
        first_col_in_group_by       — 1.0 if the leading column appears in GROUP BY
        first_col_in_order_by       — 1.0 if the leading column appears in ORDER BY

    The first_col_* features matter most: sort elimination only applies when
    the leading index column matches the sort key. A composite index on
    (l_shipdate, l_quantity) eliminates a sort on l_shipdate; one on
    (l_quantity, l_shipdate) does not.

    Note:
        The feature names retain *_freq for compatibility with downstream code,
        but they now represent cost-weighted workload impact rather than raw counts.
    """
    if queries is None:
        queries = load_queries(queries_dir)

    pairs: Set[Tuple[str, str]] = set()
    for c in candidates:
        for col in c["columns"]:
            pairs.add((c["table"], col))
    stats_by_col = fetch_pg_stats_for_columns(conn, pairs, schema)

    explain_by_q = explain_workload(conn, queries=queries)
    freqs = workload_column_frequencies(workload)
    clause_feats = workload_clause_features(workload)

    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        table = cand["table"]
        cols = cand["columns"]

        colstats = aggregate_column_stats(stats_by_col, cols, table)
        idxmeta = existing_index_overlap_features(conn, table, cols, schema)
        write_penalty = estimate_write_penalty(conn, table, cols, schema)
        touch = queries_touching_table(workload, table)

        # Per-column GROUP BY / ORDER BY cost impacts
        group_by_costs = [
            clause_feats.get(f"{table}.{c}", {}).get("group_by_cost", 0.0)
            for c in cols
        ]
        order_by_costs = [
            clause_feats.get(f"{table}.{c}", {}).get("order_by_cost", 0.0)
            for c in cols
        ]

        base = {
            "candidate_table": table,
            "candidate_cols": ",".join(cols),
            "candidate_type": cand.get("type", "unknown"),
            "n_index_columns": float(len(cols)),
            "is_composite": 1.0 if cand.get("type") == "composite" else 0.0,
            "clustered_candidate": float(cand.get("clustered_candidate", False)),
            **colstats,
            **idxmeta,
            **write_penalty,
        }

        # Cost-based workload impact of these columns
        base["workload_max_col_freq"] = max(
            (freqs.get(f"{table}.{c}", 0.0) for c in cols), default=0.0
        )
        base["workload_sum_col_freq"] = sum(freqs.get(f"{table}.{c}", 0.0) for c in cols)

        # Cost-based clause features: sort-elimination signal from GROUP BY / ORDER BY
        base["workload_max_group_by_freq"] = max(group_by_costs) if group_by_costs else 0.0
        base["workload_sum_group_by_freq"] = sum(group_by_costs)
        base["workload_max_order_by_freq"] = max(order_by_costs) if order_by_costs else 0.0
        base["workload_sum_order_by_freq"] = sum(order_by_costs)

        # Leading-column sort-elimination eligibility signal
        base["first_col_in_group_by"] = 1.0 if group_by_costs and group_by_costs[0] > 0 else 0.0
        base["first_col_in_order_by"] = 1.0 if order_by_costs and order_by_costs[0] > 0 else 0.0

        for qname in sorted(touch):
            if qname not in explain_by_q:
                continue
            exp = explain_by_q[qname]
            row = {
                "query_name": qname,
                **base,
                **{f"q_{k}": v for k, v in exp.items()},
            }
            rows.append(row)
    return rows


if __name__ == "__main__":
    from pprint import pprint
    from candidate_generator import generate_candidates

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _queries_dir = os.path.join(_repo_root, "queries")
    workload = parse_workload(_queries_dir)
    candidates = generate_candidates(workload, min_cost_impact=50000.0)
    conn = get_connection()
    try:
        rows = build_feature_rows(
            conn,
            candidates[:5],
            workload,
            queries_dir=_queries_dir,
        )
        print(f"Sample feature rows (first of {len(rows)}): ")
        if rows:
            pprint(rows[0])
        else:
            print("(no rows — check DB and workload)")
    finally:
        conn.close()