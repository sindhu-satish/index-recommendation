
"""
Parses TPC-H SQL query files and extracts column references 
from WHERE, ORDER BY, and GROUP BY clauses.

Input: 
    - Directory of .sql files (TPC-H queries)
Output:
    - List of dicts, each representing a column reference with:
        {
            "table": "lineitem",
            "column": "l_shipdate",
            "clause": "WHERE",
            "predicate_type": "range",  # only for WHERE
            "query": "q1"  # which query it came from
        }
"""
import sqlparse
import os

# Directory where TPC-H queries are stored
QUERIES_DIR = 'queries'
# Maps TPC-H column prefixes to thier table names
# TPC-H usesconsistent anmin convention where each column starts with a prefix that indicates its table (e.g., l_ for lineitem, o_ for orders)
PREFIX_TO_TABLE = {
    'l': 'lineitem',
    'o': 'orders',
    'c': 'customer',
    's': 'supplier',
    'n': 'nation',
    'r': 'region',
    'p': 'part',
    'ps': 'partsupp'
}
def get_operator(comparison) -> str:
    """Extract the operator from a sqlparse comparison token.
        sqlparse does not expose a direct .operator attribute, so we have to look for the token with Comparison ttype and grab the one with
        ttype Comparision(e.g. =, <, >, IN, etc). If we can't find one, default to '='.
    
    Args:
        comparison: a sqlparse.sql.Comparison token object
    Returns:
        The operator as a string (e.g., '=', '<', 'IN', etc)
        Defaults to '=' if no operator token is found (which is a safe fallback for most cases)
    
    """
    for token in comparison.tokens:
        if token.ttype in (sqlparse.tokens.Comparison, sqlparse.tokens.Token.Comparison):
            return token.value.strip()
    return '='
def get_predicate_type(operator: str) -> str:
    """
    Determine the predicate type based on the operator(equality vs range).
    """
    operator = operator.strip()
    if operator in ['=', '!=', '<>', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE']:
        return 'equality'
    elif operator in ['<', '<=', '>', '>=', 'BETWEEN', 'NOT BETWEEN']:
        return 'range'
    else:
        return 'equality'  # default to equality for unknown operators


def load_queries(queries_dir: str) -> dict:
    """
    Load all .sql files from the queries directory.
    Returns a dict like {"q1": "SELECT ...", "q2": "SELECT ..."}

    Args:
        queries_dir: path to the directory containing .sql files
    Returns:
        A dictionary mapping query names (without .sql extension) to their SQL string content.
    """
    queries = {}
    for filename in os.listdir(queries_dir):
        if filename.endswith('.sql'):
            # strip the .sql extension to use as key
            query_name = os.path.splitext(filename)[0]
            with open(os.path.join(queries_dir, filename), 'r') as f:
                queries[query_name] = f.read()
    return queries

def extract_columns(sql: str) -> list:
    """
    Parse a single SQL query and extract column names used in
    WHERE, ORDER BY, and GROUP BY clauses.

    Handles three main cases:
    1. Direct comparisons in WHERE: col = val, col > val.
    2. Parenthesized conditions in WHERE: (col = val OR col = val).
    3. Subqueries: SELECT ... FROM (SELECT ... WHERE col = val).

    Only columns with a recognized prefix (e.g., l_, o_, c_) are included, and the prefix is used to determine the table name.
    Duplicate references (same table/column/clause) are removed while preserving order.

    Args:
    sql: The SQL query string to parse.
    Returns:
    List of dicts, each containing:
        - table: the table name (derived from column prefix)
        - column: the column name
        - clause: which clause it was found in (WHERE, ORDER BY, GROUP BY)
        - predicate_type: for WHERE clause, whether it's an 'equality' or 'range' predicate
    """
    results = []

    # parse the SQL into tokens
    parsed = sqlparse.parse(sql)[0]

    for token in parsed.tokens:
        # skip whitespace and newlines
        if token.is_whitespace:
            continue

        # -- WHERE clause --
        # look for Comparison tokens directly in the WHERE clause, and also handle parenthesized conditions (e.g., (col = val OR col = val))
        if isinstance(token, sqlparse.sql.Where):
            for item in token.tokens:
                # direct comparison (e.g., col = val)
                if isinstance(item, sqlparse.sql.Comparison):
                    column_name = str(item.left).strip()
                    prefix = column_name.split('_')[0]
                    if prefix in PREFIX_TO_TABLE:
                        results.append({
                            'table': PREFIX_TO_TABLE[prefix],
                            'column': column_name,
                            'clause': 'WHERE',
                            'predicate_type': get_predicate_type(get_operator(item))
                        })
                # handle parenthesized conditions (e.g., (col = val OR col = val))
                elif isinstance(item, sqlparse.sql.Parenthesis):
                    for subitem in item.tokens:
                        if isinstance(subitem, sqlparse.sql.Comparison):
                            column_name = str(subitem.left).strip()
                            prefix = column_name.split('_')[0]
                            if prefix in PREFIX_TO_TABLE:
                                results.append({
                                    'table': PREFIX_TO_TABLE[prefix],
                                    'column': column_name,
                                    'clause': 'WHERE',
                                    'predicate_type': get_predicate_type(get_operator(subitem))

                                })


        # -- ORDER BY clause —-
        # ORDER By is a plain keyword followed by an IdentifierList of columns, so we look for the ORDER BY keyword and then grab the following IdentifierList
        if token.ttype is sqlparse.tokens.Keyword and token.normalized == 'ORDER BY':
            idx = parsed.tokens.index(token)
            for j in range(idx + 1, len(parsed.tokens)):
                if not parsed.tokens[j].is_whitespace:
                    next_token = parsed.tokens[j]
                    if isinstance(next_token, sqlparse.sql.IdentifierList):
                        for identifier in next_token.get_identifiers():
                            column_name = str(identifier).strip()
                            column_name = column_name.split()[0]  # ← removes 'desc'/'asc'
                            prefix = column_name.split('_')[0]
                            if prefix in PREFIX_TO_TABLE:
                                results.append({
                                    'table': PREFIX_TO_TABLE[prefix],
                                    'column': column_name,
                                    'clause': 'ORDER BY'
                                })
                    break
        # -- GROUP BY clause —-
        # Same pattern as ORDER BY, look for the GROUP BY keyword and then grab the following IdentifierList of columns
        if token.ttype is sqlparse.tokens.Keyword and token.normalized == 'GROUP BY':
            idx = parsed.tokens.index(token)
            for j in range(idx + 1, len(parsed.tokens)):
                if not parsed.tokens[j].is_whitespace:
                    next_token = parsed.tokens[j]
                    if isinstance(next_token, sqlparse.sql.IdentifierList):
                        for identifier in next_token.get_identifiers():
                            column_name = str(identifier).strip()
                            column_name = column_name.split()[0]  # ← removes 'desc'/'asc'
                            prefix = column_name.split('_')[0]
                            if prefix in PREFIX_TO_TABLE:
                                results.append({
                                    'table': PREFIX_TO_TABLE[prefix],
                                    'column': column_name,
                                    'clause': 'GROUP BY'
                                })
                    break

    # --- Subqueries handling ---
    # Some queries wrap thier main logic in a subquery, so we need to look for any parenthesized subqueries and recursively extract columns from them as well. 
    # We check if the parenthesis contains a SELECT statement to identify it as a subquery.
    for token in parsed.tokens:
        if isinstance(token, sqlparse.sql.Identifier):
            for subtoken in token.tokens:
                if isinstance(subtoken, sqlparse.sql.Parenthesis):
                    # Check if it's subquery (contains SELECT)
                    subquery_str = str(subtoken)[1:-1]  # remove parentheses
                    if 'select' in subquery_str.lower():
                        results.extend(extract_columns(subquery_str))

    # --- Deduplicate results ---
    # Same column can appear multiple times in a query (e.g., in both WHERE and ORDER BY), but we only want to count it once per clause. 
    # We use a set to track seen (table, column, clause) combinations and filter out duplicates while preserving order.
    seen = set()
    unique_results = []
    for r in results:
        key = (r['table'], r['column'], r['clause'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    

    return unique_results

    

def parse_workload(queries_dir: str) -> list:
    """
    Main function — loads all queries and runs extract_columns on each.
    Calls load_queries to get all SQL strings, then runs extract_columns
    on each query. Adds a 'query' field to each column reference dict to track which query it came from.

    Args:
    queries_dir: path to the directory containing .sql files
    Returns:
    A flat list of all column references across all queries, each with:
        - table: the table name
        - column: the column name
        - clause: which clause it was found in (WHERE, ORDER BY, GROUP BY)
        - predicate_type: for WHERE clause, whether it's an 'equality' or 'range' predicate
    """
    queries = load_queries(queries_dir)
    all_results = []
    for query_name, sql in sorted(queries.items()):
        columns = extract_columns(sql)
        for col in columns:
            col['query'] = query_name  # track which query it came from
        all_results.extend(columns)
    return all_results


if __name__ == '__main__':
    results = parse_workload(QUERIES_DIR)
    print(f"\nTotal column references found: {len(results)}")
    for r in results:
        print(r)
