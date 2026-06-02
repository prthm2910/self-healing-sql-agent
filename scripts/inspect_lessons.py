"""
Inspect lessons stored in the pgvector PostgresStore for duplicates.

Run: python scripts/inspect_lessons.py
"""

import os
from dotenv import load_dotenv
import psycopg
from psycopg_pool import ConnectionPool
from urllib.parse import urlparse, urlunparse

load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL not found in environment")

# Point to neondb (where LangGraph store tables live)
parsed = urlparse(db_url)
neondb_url = urlunparse(parsed._replace(path="/neondb"))

def list_store_entries():
    """Raw SQL query to read all lessons from the LangGraph store tables."""
    pool = ConnectionPool(
        conninfo=neondb_url,
        max_size=1,
        min_size=1,
        kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row}
    )

    with pool.connection() as conn:
        # The LangGraph PostgresStore uses a table called "store" (or "store_vectors")
        # Let's first discover the table structure
        tables = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name LIKE '%store%'
        """).fetchall()
        print(f"=== Store-related tables found: {[t['table_name'] for t in tables]} ===\n")

        if not tables:
            print("No store tables found. Looking for any tables with 'lesson' or 'vector'...")
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """).fetchall()
            print(f"All public tables: {[t['table_name'] for t in tables]}")
            return

        # Inspect the first store table
        store_table = tables[0]["table_name"]
        cols = conn.execute(f"""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_name = '{store_table}' ORDER BY ordinal_position
        """).fetchall()
        print(f"=== Columns in '{store_table}': ===")
        for c in cols:
            print(f"  - {c['column_name']} ({c['data_type']})")
        print()

        # Count total entries
        count = conn.execute(f"SELECT COUNT(*) as cnt FROM {store_table}").fetchone()
        print(f"Total entries in '{store_table}': {count['cnt']}\n")

        # Fetch all rows
        rows = conn.execute(f"SELECT * FROM {store_table} ORDER BY created_at DESC").fetchall()

        # Print lessons grouped by namespace
        from collections import defaultdict
        by_namespace = defaultdict(list)
        for row in rows:
            # Build namespace key from prefix columns
            ns_raw = row.get("prefix", [])
            ns_key = tuple(ns_raw) if isinstance(ns_raw, list) else (str(ns_raw),)
            ns_str = ".".join(str(x) for x in ns_key)
            value = row.get("value", {})
            by_namespace[ns_str].append({
                "key": row.get("key"),
                "value": value,
                "created_at": row.get("created_at"),
                "namespace": ns_key,
            })

        for ns, entries in by_namespace.items():
            print(f"\n{'='*60}")
            print(f"Namespace: {ns} ({len(entries)} entries)")
            print(f"{'='*60}")
            for i, entry in enumerate(entries, 1):
                val = entry["value"]
                title = val.get("title", "<no title>") if isinstance(val, dict) else str(val)
                instruction = val.get("instruction", "")[:120] if isinstance(val, dict) else ""
                tags = val.get("tags", []) if isinstance(val, dict) else []
                print(f"  [{i}] key={entry['key'][:12]}...")
                print(f"      title: {title}")
                print(f"      tags: {tags}")
                print(f"      instruction: {instruction}...")
                print()

        # Check for duplicate titles
        print(f"\n{'='*60}")
        print("DUPLICATE TITLE CHECK")
        print(f"{'='*60}")
        all_titles = []
        for entries in by_namespace.values():
            for entry in entries:
                val = entry["value"]
                if isinstance(val, dict):
                    all_titles.append((val.get("title", ""), entry["key"][:12]))

        from collections import Counter
        title_counts = Counter(t[0] for t in all_titles)
        duplicates = {title: count for title, count in title_counts.items() if count > 1}

        if duplicates:
            print(f"Found {len(duplicates)} duplicate title(s):\n")
            for title, count in duplicates.items():
                print(f"  TITLE: '{title}' (appears {count} times)")
                matching = [key for t, key in all_titles if t == title]
                for key in matching:
                    print(f"    - key: {key}")
        else:
            print("No duplicate titles found.")

        # Also check for near-duplicate instructions (same first 80 chars)
        print(f"\n{'='*60}")
        print("NEAR-DUPLICATE INSTRUCTION CHECK (first 80 chars)")
        print(f"{'='*60}")
        all_instructions = []
        for entries in by_namespace.values():
            for entry in entries:
                val = entry["value"]
                if isinstance(val, dict):
                    instr = val.get("instruction", "")[:80]
                    title = val.get("title", "")
                    all_instructions.append((instr, title, entry["key"][:12]))

        instr_counts = Counter(i[0] for i in all_instructions)
        dup_instructions = {instr: count for instr, count in instr_counts.items() if count > 1 and instr}

        if dup_instructions:
            print(f"Found {len(dup_instructions)} near-duplicate instruction(s):\n")
            for instr, count in dup_instructions.items():
                matching = [(t, k) for i, t, k in all_instructions if i == instr]
                print(f"  INSTRUCTION (prefix): '{instr}...'")
                print(f"    Appears {count} times:")
                for t, k in matching:
                    print(f"    - title: '{t}', key: {k}")
                print()
        else:
            print("No near-duplicate instructions found.")

    pool.close()

if __name__ == "__main__":
    list_store_entries()
