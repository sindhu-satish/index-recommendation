"""
candidate_generator.py
----------------------
Generates candidate indexes from TPC-H workload by analyzing column usage patterns.

Changes from original:
    - count_column_frequency now sums query_cost instead of counting appearances.
      A column that appears in one expensive query outranks one that appears in
      many cheap queries on small tables — directly aligns with workload-driven design.
    - generate_candidates uses min_cost_impact threshold instead of min_frequency
      so columns from expensive single-occurrence queries are not pruned.
    - Each candidate gets a clustered_candidate flag: True when the dominant
      predicate type is range. Only one clustered index can exist per relation,
      and clustered indexes are most beneficial for range scans.
    - Dominant predicate type is now also cost-weighted, so expensive range/equality
      usage has more influence than many cheap occurrences.

Input:
    - Workload list from workload_parser, keys: table, column, clause,
      predicate_type, query, query_cost, in_where, in_group_by, in_order_by
Output:
    - List of candidate index dicts, keys: table, columns (list), type,
      clustered_candidate (bool)
"""

import collections
from workload_parser import parse_workload

QUERIES_DIR = 'queries/'


def count_column_frequency(workload: list) -> dict:
    """
    Calculate total cost-impact of each column across the workload.

    Sums query_cost instead of counting appearances (+1).
    workload_parser deduplicates on (table, column) per query so each
    query's cost is already counted at most once per column — no extra
    deduplication needed here.

    Args:
        workload: list of dicts with keys: table, column, clause,
                  predicate_type, query, query_cost
    Returns:
        dict mapping "table.column" → total cost impact (float)
    """
    frequencies = collections.defaultdict(float)
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        frequencies[key] += item.get('query_cost', 1.0)
    return frequencies


def get_column_predicate_types(workload: list) -> dict:
    """
    Determine the dominant predicate type (equality or range) for each column.

    Used for:
        1. Ordering columns in composite indexes (equality first, range last).
        2. Flagging clustered index candidates (range columns benefit most
           from clustering since only one clustered index exists per relation).

    This is cost-weighted to stay consistent with the rest of the pipeline:
    a predicate type used in an expensive query should matter more than the
    same predicate type used repeatedly in cheap queries.

    Args:
        workload: list of dicts with keys: table, column, clause,
                  predicate_type, query_cost
    Returns:
        dict mapping "table.column" → "equality" or "range"
    """
    counts = collections.defaultdict(lambda: {'equality': 0.0, 'range': 0.0})
    for item in workload:
        if item['clause'] == 'WHERE' and 'predicate_type' in item:
            key = f"{item['table']}.{item['column']}"
            counts[key][item['predicate_type']] += item.get('query_cost', 1.0)

    result = {}
    for key, type_counts in counts.items():
        if type_counts['range'] > type_counts['equality']:
            result[key] = 'range'
        else:
            result[key] = 'equality'  # default to equality on tie
    return result


def generate_candidates(workload: list, min_cost_impact: float = 50000.0) -> list:
    """
    Generate candidate indexes based on query cost impact.

    Steps:
        1. Sum cost impact per column and filter below threshold.
           Columns from expensive queries clear the threshold even with
           single appearances — avoids discarding high-value candidates.
        2. Build single-column candidates, flagging range columns as
           clustered_candidate=True.
        3. Pair high-impact columns from the same table into composite
           candidates, ordered equality-first then range, assigning
           clustering flags heuristically based on the presence of range
           predicates.

    Note:
        Composite candidates are generated heuristically from high-impact
        columns on the same table. This is not strict per-query co-occurrence.

    Args:
        workload: list of dicts with keys: table, column, clause,
                  predicate_type, query, query_cost
        min_cost_impact: minimum total optimizer cost a column must be
                         associated with to be considered. Scale-dependent —
                         tune per dataset size.
    Returns:
        list of candidate dicts with keys: table, columns, type,
        clustered_candidate
    """
    frequencies = count_column_frequency(workload)
    predicate_types = get_column_predicate_types(workload)
    candidates = []

    # 1. Filter columns by cost threshold
    frequent_columns = {k: v for k, v in frequencies.items() if v >= min_cost_impact}

    # 2. Generate Single-Column Candidates
    for key in frequent_columns:
        table, column = key.split('.', 1)
        pred_type = predicate_types.get(key, 'equality')
        candidates.append({
            'table': table,
            'columns': [column],
            'type': 'single',
            # Heuristic flag only: range predicate columns are the strongest
            # clustered-index candidates because clustering helps range scans
            # far more than point lookups.
            'clustered_candidate': pred_type == 'range'
        })

    # Group high-impact columns by table for composite candidate generation
    table_to_columns = collections.defaultdict(set)
    for item in workload:
        key = f"{item['table']}.{item['column']}"
        if key in frequent_columns:
            table_to_columns[item['table']].add(item['column'])

    # 3. Build composite candidates, applying structural heuristics
    for table, columns in table_to_columns.items():
        columns = list(columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col_a = columns[i]
                col_b = columns[j]

                type_a = predicate_types.get(f"{table}.{col_a}", 'equality')
                type_b = predicate_types.get(f"{table}.{col_b}", 'equality')

                # Edge Case 1: (Equality, Range)
                if (type_a == 'equality' and type_b == 'range') or (type_a == 'range' and type_b == 'equality'):
                    # Force equality first for optimal B+ tree traversal
                    ordered_cols = [col_a, col_b] if type_a == 'equality' else [col_b, col_a]
                    candidates.append({
                        'table': table,
                        'columns': ordered_cols,
                        'type': 'composite',
                        # Heuristic: any composite with a range suffix is worth
                        # considering as a clustered option, but final selection
                        # should happen downstream at the table level.
                        'clustered_candidate': True
                    })

                # Edge Case 2: (Range, Range)
                elif type_a == 'range' and type_b == 'range':
                    # Generate both permutations (we don't know selectivity yet)
                    candidates.append({
                        'table': table,
                        'columns': [col_a, col_b],
                        'type': 'composite',
                        'clustered_candidate': True
                    })
                    candidates.append({
                        'table': table,
                        'columns': [col_b, col_a],
                        'type': 'composite',
                        'clustered_candidate': True
                    })

                # Edge Case 3: (Equality, Equality)
                elif type_a == 'equality' and type_b == 'equality':
                    # Generate both permutations (we don't know which equality is more selective)
                    candidates.append({
                        'table': table,
                        'columns': [col_a, col_b],
                        'type': 'composite',
                        'clustered_candidate': False
                    })
                    candidates.append({
                        'table': table,
                        'columns': [col_b, col_a],
                        'type': 'composite',
                        'clustered_candidate': False
                    })

    return candidates


if __name__ == '__main__':
    workload = parse_workload(QUERIES_DIR)

    frequencies = count_column_frequency(workload)
    print("Top 10 columns by Total Query Cost Impact:")
    for key, cost in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cost:12.2f} {key}")

    candidates = generate_candidates(workload)

    total_candidates = len(candidates)
    single_count = sum(1 for c in candidates if c['type'] == 'single')
    composite_count = total_candidates - single_count
    clustered_count = sum(1 for c in candidates if c.get('clustered_candidate'))

    print(f"\n{'='*40}")
    print(f"TOTAL CANDIDATES GENERATED: {total_candidates}")
    print(f"  Single-column indexes:    {single_count}")
    print(f"  Composite indexes:        {composite_count}")
    print(f"  Clustered candidates:     {clustered_count}")
    print(f"{'='*40}")