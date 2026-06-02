"""Quick check: what's actually in the lessons namespaces now?"""

import os
from dotenv import load_dotenv
import psycopg
from psycopg_pool import ConnectionPool
from urllib.parse import urlparse, urlunparse

load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL not found in environment")

parsed = urlparse(db_url)
neondb_url = urlunparse(parsed._replace(path="/neondb"))

pool = ConnectionPool(
    conninfo=neondb_url,
    max_size=1,
    min_size=1,
    kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row}
)

with pool.connection() as conn:
    # Check what prefix format is actually stored
    print("=== All distinct prefixes in store ===")
    rows = conn.execute(
        "SELECT DISTINCT prefix, COUNT(*) as cnt FROM store GROUP BY prefix"
    ).fetchall()
    for r in rows:
        print(f"  prefix={repr(r['prefix'])}  count={r['cnt']}")

    print("\n=== Lesson entries (pinned) ===")
    rows = conn.execute(
        "SELECT key, value->>'title' as title FROM store WHERE prefix = %s",
        ('{global,lessons,pinned}',)
    ).fetchall()
    for r in rows:
        print(f"  key={r['key'][:16]}...  title={r['title']}")

    print(f"\n=== Lesson entries (dynamic) ===")
    rows = conn.execute(
        "SELECT key, value->>'title' as title FROM store WHERE prefix = %s",
        ('{global,lessons,dynamic}',)
    ).fetchall()
    for r in rows:
        print(f"  key={r['key'][:16]}...  title={r['title']}")

pool.close()
