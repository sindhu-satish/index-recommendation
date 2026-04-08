import psycopg2
from dotenv import load_dotenv
import os
from feature_extractor import normalize_query_for_postgres, explain_query_json, queries_touching_table
from workload_parser import load_queries, parse_workload
from candidate_generator import generate_candidates
def get_connection():
    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'tpch'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'password')
    )
    # Prevents failed EXPLAINs from breaking the whole loop
    conn.autocommit = True 
    return conn

def get_explain_cost(conn, sql: str) -> float:
    try:
        explain_result = explain_query_json(conn, sql)
        return explain_result['plan_total_cost']
    except Exception as e:
        print(f"Error getting explain cost for SQL: {sql}")
        print(e)
        conn.rollback()
        return 0.0

def compute_baseline_costs(conn, queries: dict) -> dict:
    baseline_costs = {}
    for query_name, sql in queries.items():
        normalized = normalize_query_for_postgres(sql)
        cost = get_explain_cost(conn, normalized)
        baseline_costs[query_name] = cost
    return baseline_costs

def label_candidate(conn, candidate: dict, queries: dict, baseline_costs: dict, workload: list) -> float:
    """
    For a single candidate index:
    1. Build the CREATE INDEX statement from the candidate dict.
    2. Create the hypothetical index using hypopg.
    3. Run EXPLAIN (FORMAT JSON) for each query in the workload with the hypothetical index.
    4. Compute benefit = sum(baseline_cost) - sum(cost_with_index)
    5. Reset HypoPG to remove the hypothetical index.
    6. Return the benefit score
    """
    cursor = conn.cursor()

    # Build the CREATE INDEX statement from the candidate dict
    # e.g. {'table': 'lineitem', columns': ['l_suppkey', 'l_orderkey']} → "CREATE INDEX ON lineitem (l_suppkey, l_orderkey)"
    table = candidate['table']
    columns = candidate['columns']
    index_sql = f"CREATE INDEX ON {table} ({', '.join(columns)})"

    # find which queries are actually touch this candidate's table
    # No point on running EXPLAIN on querires that dont use table

    touching = queries_touching_table(workload, table)

    # Create the hypothetical index using hypopg
    try:
        cursor.execute(f"SELECT * FROM hypopg_create_index('{index_sql}')")
    except Exception as e:
        print(f"Error creating hypothetical index: {index_sql} — {e}")
        conn.rollback()
        return 0.0

    # Run EXPLAIN (FORMAT JSON) for each query in the workload that touches the candidate's table
    # sum up the costs with the hypothetical index
    cost_with_index = 0.0 
    for query_name, sql in queries.items():
        if query_name in touching:
            normalized = normalize_query_for_postgres(sql)

            cost_with_index += get_explain_cost(conn, normalized)

    # benefit = how much cheaper the workload is with this index
    benefit = sum(
        cost for query_name, cost in baseline_costs.items()
        if query_name in touching
    ) - cost_with_index
    # reset HypoPG — removes all hypothetical indexes for the next candidate

    try:
        cursor.execute("SELECT hypopg_reset()")
    except Exception as e:
        print(f"Error resetting HypoPG — {e}")
        conn.rollback()

    return benefit

def label_all_candidates(conn, candidates: list, queries: dict, workload: list) -> list:
    """
    Labels all candidates with benefit score
    Returns the candidate lists with a 'benefit' field added to each
    """

    # Compute baseline costs before the loop 
    # no hypoethetical indexes, just the raw cost of the workload
    baseline_costs = compute_baseline_costs(conn, queries)

    # loop through candidates
    labeled = []
    for candidate in candidates:
        benefit = label_candidate(conn, candidate, queries, baseline_costs, workload)
        labeled.append({**candidate, 'benefit': benefit})

    return labeled

if __name__ == '__main__':
    QUERIES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'queries')


    queries = load_queries(QUERIES_DIR)
    workload = parse_workload(QUERIES_DIR)
    candidates = generate_candidates(workload)
    conn = get_connection()

    try:
        labeled = label_all_candidates(conn, candidates, queries, workload)
        for c in sorted(labeled, key=lambda x: x['benefit'], reverse=True)[:25]: # print top 10 candidates
            print(f"{c['benefit']:12.2f} {c['table']} {c['columns']}")
    finally:
        conn.close()
