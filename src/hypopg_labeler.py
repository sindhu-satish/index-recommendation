"""
hypopg_labeler.py
-----------------
Uses HypoPG to simulate candidate indexes without building them.
Produces per-query labels so training_dataset.py can join on example_id
directly, eliminating the label broadcasting mismatch.

Output CSV format (one row per query/candidate pair):
    query_name, candidate_table, candidate_cols, label, label_source

Changes from original:
    - get_connection moved to db_utils.
    - normalize_query_for_postgres and explain_query_json imported from db_utils.
    - Fixed conn.rollback() no-ops: autocommit=True makes rollback a no-op.
      Replaced with hypopg_reset() + early return on context failures.
    - label_candidate now returns a per-query benefit dict {query_name: float}
      instead of a single summed float. The model can now learn which queries
      each index helps most rather than predicting a single workload total.
    - label_candidate_with_context returns the same per-query dict format.
      Total marginal benefit (sum of dict values) is used for greedy selection.
    - label_all_candidates removed — dead code, never called.
    - __main__ writes one row per (query_name, candidate) pair instead of one
      row per candidate. This matches what training_dataset.py expects when
      joining on example_id.
    - label_source field ('marginal' or 'individual') preserved on all rows.

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → ml_model
"""

import os
import pandas as pd

from db_utils import get_connection, explain_query_json
from feature_extractor import queries_touching_table
from workload_parser import load_queries, parse_workload
from candidate_generator import generate_candidates

QUERIES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'queries')

GREEDY_POOL_SIZE = 60
MIN_COST_IMPACT = 50000.0


def get_explain_cost(conn, sql: str) -> float:
    """Run EXPLAIN (FORMAT JSON) and return the total plan cost."""
    try:
        # explain_query_json() should own normalization behavior centrally
        return explain_query_json(conn, sql)['plan_total_cost']
    except Exception as e:
        print(f"Error getting explain cost: {e}")
        return 0.0


def compute_baseline_costs(conn, queries: dict) -> dict:
    """
    Cost for every query with no hypothetical indexes active.
    These are the reference costs used to compute per-query benefit deltas.
    """
    return {
        query_name: get_explain_cost(conn, sql)
        for query_name, sql in queries.items()
    }


def _build_index_sql(candidate: dict) -> str:
    """Build CREATE INDEX statement for HypoPG from candidate metadata."""
    table = candidate['table']
    columns = candidate['columns']
    return f"CREATE INDEX ON {table} ({', '.join(columns)})"


def _hypopg_create_index(cursor, index_sql: str) -> None:
    """
    Create a hypothetical index through HypoPG.

    hypopg_create_index takes a SQL string argument, so we pass it as a bound
    parameter instead of interpolating it directly into the outer SELECT.
    """
    cursor.execute("SELECT * FROM hypopg_create_index(%s)", (index_sql,))


def _hypopg_reset(cursor) -> None:
    """Reset all hypothetical indexes."""
    cursor.execute("SELECT hypopg_reset()")


def label_candidate(
    conn, candidate: dict, queries: dict, baseline_costs: dict, workload: list
) -> dict:
    """
    Evaluate a single candidate in isolation.
    Returns per-query benefits: {query_name: baseline_cost - cost_with_index}.
    Positive = index helped that query. Negative = index hurt that query.
    Used in Pass 1 to rank candidates before greedy selection.
    """
    cursor = conn.cursor()
    table = candidate['table']
    touching = queries_touching_table(workload, table)
    index_sql = _build_index_sql(candidate)

    try:
        _hypopg_create_index(cursor, index_sql)
    except Exception as e:
        print(f"Error creating hypothetical index: {index_sql} — {e}")
        return {q: 0.0 for q in touching}

    per_query = {}
    try:
        for query_name, sql in queries.items():
            if query_name in touching:
                cost = get_explain_cost(conn, sql)
                per_query[query_name] = baseline_costs[query_name] - cost
    finally:
        try:
            _hypopg_reset(cursor)
        except Exception as e:
            print(f"Error resetting HypoPG — {e}")

    return per_query


def label_candidate_with_context(
    conn, candidate: dict, existing_indexes: list,
    queries: dict, baseline_costs: dict, workload: list
) -> dict:
    """
    Evaluate a candidate given already-selected context indexes.
    Returns per-query marginal benefits: improvement over having context alone.
    Total marginal benefit = sum(dict.values()) — used for greedy selection.
    """
    cursor = conn.cursor()
    table = candidate['table']
    touching = queries_touching_table(workload, table)

    try:
        # Step 1: create context indexes
        for existing in existing_indexes:
            existing_sql = _build_index_sql(existing)
            try:
                _hypopg_create_index(cursor, existing_sql)
            except Exception as e:
                print(f"Error creating context index: {existing_sql} — {e}")
                _hypopg_reset(cursor)
                return {q: 0.0 for q in touching}

        # Step 2: measure cost with context only, per query
        context_costs = {}
        for query_name, sql in queries.items():
            if query_name in touching:
                context_costs[query_name] = get_explain_cost(conn, sql)

        # Step 3: add candidate on top of context
        index_sql = _build_index_sql(candidate)
        try:
            _hypopg_create_index(cursor, index_sql)
        except Exception as e:
            print(f"Error creating hypothetical index: {index_sql} — {e}")
            _hypopg_reset(cursor)
            return {q: 0.0 for q in touching}

        # Step 4: measure cost with context + candidate, per query
        per_query = {}
        for query_name, sql in queries.items():
            if query_name in touching:
                cost_plus = get_explain_cost(conn, sql)
                per_query[query_name] = context_costs[query_name] - cost_plus

        return per_query

    finally:
        try:
            _hypopg_reset(cursor)
        except Exception as e:
            print(f"Error resetting HypoPG — {e}")


def label_all_candidates_greedy(
    conn, candidates: list, queries: dict, workload: list,
    pool_size: int = GREEDY_POOL_SIZE
) -> list:
    """
    Two-pass greedy labeling that produces per-query benefit labels.

    Pass 1: Score all candidates independently. Each entry stores
            per_query_benefits dict and total_benefit (sum) for ranking.
    Pass 2: Greedy marginal selection on top pool_size candidates.
            At each round, pick the candidate with highest total marginal
            benefit given already-selected indexes. Store per-query breakdown.

    Remaining candidates outside pool_size keep their Pass 1 per-query scores.
    """
    baseline_costs = compute_baseline_costs(conn, queries)

    print(f"Pass 1: Scoring all {len(candidates)} candidates independently...")
    all_labeled = []
    for i, candidate in enumerate(candidates):
        per_query = label_candidate(conn, candidate, queries, baseline_costs, workload)
        total = sum(per_query.values())
        all_labeled.append({
            **candidate,
            'per_query_benefits': per_query,
            'total_benefit': total,
            'label_source': 'individual',
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(candidates)} done...")

    all_labeled.sort(key=lambda x: x['total_benefit'], reverse=True)
    pool = all_labeled[:pool_size]
    rest = all_labeled[pool_size:]

    print(f"\nPass 2: Greedy marginal evaluation on top {min(pool_size, len(pool))} candidates...")
    selected = []
    remaining = list(pool)
    greedy_labeled = []

    while remaining:
        best = None
        best_total = -float('inf')
        best_per_query = {}

        for candidate in remaining:
            per_query = label_candidate_with_context(
                conn, candidate, selected, queries, baseline_costs, workload
            )
            total_marginal = sum(per_query.values())
            if total_marginal > best_total:
                best_total = total_marginal
                best = candidate
                best_per_query = per_query

        entry = {
            **best,
            'per_query_benefits': best_per_query,
            'total_benefit': best_total,
            'label_source': 'marginal',
            'selection_round': len(selected) + 1,
        }
        greedy_labeled.append(entry)

        # Keep only the plain candidate fields in the selected context list
        selected.append({
            'table': best['table'],
            'columns': best['columns'],
            'type': best.get('type', 'unknown'),
            'clustered_candidate': best.get('clustered_candidate', False),
        })

        remaining.remove(best)
        print(
            f"  Round {len(selected):2d}: {best['table']:12} {best['columns']} "
            f"— total marginal benefit={best_total:,.0f}"
        )

    return greedy_labeled + rest


if __name__ == '__main__':
    queries = load_queries(QUERIES_DIR)
    workload = parse_workload(QUERIES_DIR)
    candidates = generate_candidates(workload, min_cost_impact=MIN_COST_IMPACT)
    conn = get_connection()

    try:
        labeled = label_all_candidates_greedy(conn, candidates, queries, workload)

        # Explode per-query benefits into one row per (query_name, candidate)
        rows = []
        for entry in labeled:
            candidate_table = entry['table']
            candidate_cols = ','.join(entry['columns'])
            label_source = entry['label_source']
            for query_name, benefit in entry['per_query_benefits'].items():
                rows.append({
                    'query_name': query_name,
                    'candidate_table': candidate_table,
                    'candidate_cols': candidate_cols,
                    'label': benefit,
                    'label_source': label_source,
                })

        df = pd.DataFrame(rows)

        output_path = os.path.join(os.path.dirname(QUERIES_DIR), 'data', 'labels.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df[['query_name', 'candidate_table', 'candidate_cols', 'label', 'label_source']].to_csv(
            output_path, index=False
        )

        n_marginal = (df['label_source'] == 'marginal').sum()
        n_individual = (df['label_source'] == 'individual').sum()
        print(f"\nWrote {len(df)} labeled rows to {output_path}")
        print(f"  {n_marginal} marginal  |  {n_individual} individual")
        print(
            f"  {df['candidate_table'].nunique()} unique tables  "
            f"|  {df[['candidate_table','candidate_cols']].drop_duplicates().shape[0]} unique candidates  "
            f"|  {df['query_name'].nunique()} unique queries"
        )

        print("\nTop 10 candidates by total workload benefit:")
        totals = (
            df.groupby(['candidate_table', 'candidate_cols'])['label']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        for (table, cols), total in totals.items():
            print(f"  {total:15.0f}  {table:12}  {cols}")

    finally:
        conn.close()