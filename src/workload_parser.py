"""
workload_parser.py
------------------
Parses TPC-H SQL query files and extracts column references
used in WHERE, GROUP BY, and ORDER BY clauses.

Input:
    - Directory of .sql query files (queries/)

Output:
    - List of dicts, each representing a column reference:
      {
          "table": "lineitem",
          "column": "l_shipdate",
          "clause": "WHERE",
          "predicate_type": "range",
          "query": "q1"
      }

Pipeline position:
    workload_parser → candidate_generator → feature_extractor → hypopg_labeler → ml_model
"""

import sqlparse
import os

QUERIES_DIR = 'queries'

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
    """
    Extract the comparison operator from a sqlparse Comparison token.
    Defaults to '=' if no operator token is found.
    """
    for token in comparison.tokens:
        if token.ttype in (sqlparse.tokens.Comparison, sqlparse.tokens.Token.Comparison):
            return token.value.strip()
    return '='


def get_predicate_type(operator: str) -> str:
    """
    Classify a SQL comparison operator as equality or range.
    """
    operator = operator.strip().upper()
    if operator in ['=', '!=', '<>', 'IN', 'NOT IN', 'LIKE', 'NOT LIKE']:
        return 'equality'
    elif operator in ['<', '<=', '>', '>=', 'BETWEEN', 'NOT BETWEEN']:
        return 'range'
    else:
        return 'equality'


def strip_alias(column_name: str) -> str:
    """
    Strip table alias from a column name.
    e.g. 'l1.l_suppkey' → 'l_suppkey', 'n1.n_name' → 'n_name'
    """
    if '.' in column_name:
        return column_name.split('.')[-1]
    return column_name


def load_queries(queries_dir: str) -> dict:
    """
    Load all .sql files from a directory into a dictionary.
    Returns dict mapping query name to SQL string.
    """
    queries = {}
    for filename in os.listdir(queries_dir):
        if filename.endswith('.sql'):
            query_name = os.path.splitext(filename)[0]
            with open(os.path.join(queries_dir, filename), 'r') as f:
                queries[query_name] = f.read()
    return queries


def extract_columns_from_token(token, clause, predicate_type='n/a'):
    """
    Recursively flattens an AST token and extracts any TPC-H columns.
    Bypasses all nesting issues (functions, math ops, double parentheses).
    """
    cols = []
    
    # 1. Define the fake aliases we want to ignore
    IGNORE_ALIASES = {'l_year', 'o_year', 'c_count'}

    # .flatten() yields all leaf nodes, destroying nested structures
    for leaf in token.flatten():
        val = str(leaf).strip()
        
        # Look for TPC-H column signatures: contains '_' and isn't a literal string
        if '_' in val and not val.startswith("'") and not val.startswith('"'):
            # Strip table aliases (e.g., "l1.l_suppkey" -> "l_suppkey")
            col_name = val.split('.')[-1].lower()
            
            # 2. Skip the column if it is in our ignore list
            if col_name in IGNORE_ALIASES:
                continue
                
            prefix = col_name.split('_')[0]
            
            if prefix in PREFIX_TO_TABLE:
                cols.append({
                    'table': PREFIX_TO_TABLE[prefix],
                    'column': col_name,
                    'clause': clause,
                    'predicate_type': predicate_type
                })
    return cols


def extract_columns(sql: str) -> list:
    """
    Parses a SQL query using a recursive walker to catch ALL nested columns
    in WHERE, ON, HAVING, GROUP BY, and ORDER BY clauses.
    """
    results = []
    parsed = sqlparse.parse(sql)[0]
    
    # State tracking so we know what clause we are currently walking inside
    current_clause = 'n/a'

    def walk(token_list):
        nonlocal current_clause
        idx = 0
        tokens = token_list.tokens
        
        while idx < len(tokens):
            token = tokens[idx]

            if token.is_whitespace:
                idx += 1
                continue

            # 1. Update Context based on Keywords
            if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML):
                val = token.value.upper()
                if val in ('WHERE', 'ON', 'HAVING'):
                    # Treat ON and HAVING as WHERE for index recommendation purposes
                    current_clause = 'WHERE'
                elif val == 'GROUP BY':
                    current_clause = 'GROUP BY'
                elif val == 'ORDER BY':
                    current_clause = 'ORDER BY'
                elif val in ('SELECT', 'FROM'):
                    current_clause = 'n/a' # Stop extracting until we hit WHERE/ON/GROUP BY

            # 2. Extract from Filtering Clauses (WHERE, ON, HAVING)
            if current_clause == 'WHERE':
                if isinstance(token, sqlparse.sql.Comparison):
                    try:
                        op = get_operator(token)
                        ptype = get_predicate_type(op)
                    except:
                        ptype = 'equality' # Fallback for complex comparisons
                    
                    # Extract left side (handles functions like substring automatically)
                    results.extend(extract_columns_from_token(token.left, current_clause, ptype))
                    
                    # Extract right side (if it's a subquery, skip and let the recursion handle it)
                    if not isinstance(token.right, sqlparse.sql.Parenthesis) or 'select' not in str(token.right).lower():
                        results.extend(extract_columns_from_token(token.right, current_clause, ptype))

                # Handle IN and BETWEEN 
                elif isinstance(token, (sqlparse.sql.Identifier, sqlparse.sql.Function)):
                    peek_idx = idx + 1
                    while peek_idx < len(tokens) and tokens[peek_idx].is_whitespace:
                        peek_idx += 1
                    
                    if peek_idx < len(tokens):
                        next_tok = tokens[peek_idx]
                        if next_tok.ttype is sqlparse.tokens.Keyword:
                            kw = next_tok.value.upper()
                            if kw == 'IN':
                                results.extend(extract_columns_from_token(token, current_clause, 'equality'))
                            elif kw == 'BETWEEN':
                                results.extend(extract_columns_from_token(token, current_clause, 'range'))

            # 3. Extract from Grouping/Sorting Clauses
            elif current_clause in ('GROUP BY', 'ORDER BY'):
                if isinstance(token, (sqlparse.sql.Identifier, sqlparse.sql.IdentifierList, sqlparse.sql.Function)):
                    results.extend(extract_columns_from_token(token, current_clause, 'n/a'))

            # 4. RECURSION: Dive into containers (Parentheses, Where blocks, nested Identifiers)
            if hasattr(token, 'tokens'):
                # Save the current state
                prev_clause = current_clause
                
                # If diving into a subquery, turn off extraction until we hit its WHERE/GROUP BY
                if isinstance(token, sqlparse.sql.Parenthesis) and 'select' in str(token).lower():
                    current_clause = 'n/a'
                
                walk(token) # Dive in
                
                # Restore the state when bubbling back out
                current_clause = prev_clause

            idx += 1

    # Start the recursive walk
    walk(parsed)

    # Deduplicate results
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
    Main entry point — load all queries and extract column references.
    """
    queries = load_queries(queries_dir)
    all_results = []
    for query_name, sql in sorted(queries.items()):
        columns = extract_columns(sql)
        for col in columns:
            col['query'] = query_name
        all_results.extend(columns)
    return all_results


if __name__ == '__main__':
    queries = load_queries(QUERIES_DIR)

    with open('query_analysis.txt', 'w') as f:
        total = 0
        for name, sql in sorted(queries.items()):
            result = extract_columns(sql)
            f.write(f"{'='*60}\n")
            f.write(f"QUERY: {name}\n")
            f.write(f"{'='*60}\n")
            f.write(sql)
            f.write(f"\n{'─'*60}\n")
            f.write(f"EXTRACTED COLUMNS ({len(result)}):\n")
            for r in result:
                predicate = r.get('predicate_type', 'n/a')
                f.write(f"  {r['clause']:10} {predicate:10} {r['table']}.{r['column']}\n")
            f.write('\n')
            total += len(result)
        f.write(f"{'='*60}\n")
        f.write(f"TOTAL COLUMN REFERENCES: {total}\n")

    print("Written to query_analysis.txt")