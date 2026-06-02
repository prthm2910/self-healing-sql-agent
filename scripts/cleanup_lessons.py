"""
Clean up duplicate and noisy lessons from the LangGraph pgvector store.

Run: python scripts/cleanup_lessons.py
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

parsed = urlparse(db_url)
neondb_url = urlunparse(parsed._replace(path="/neondb"))


def cleanup_lessons():
    pool = ConnectionPool(
        conninfo=neondb_url,
        max_size=1,
        min_size=1,
        kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row}
    )

    deleted = []
    kept = []

    with pool.connection() as conn:
        # 1. Delete test "Vegetarian Filter" (key starts with 'test_lesson_')
        print("1. Deleting test 'Vegetarian Filter' duplicate...")
        conn.execute(
            "DELETE FROM store WHERE prefix = 'global.lessons.dynamic' AND key LIKE 'test_lesson_%'"
        )
        deleted.append("test_lesson_... (test 'Vegetarian Filter')")

        # 2. Delete botched "Boolean vs Integer" lessons
        #    These have instructions that don't match their titles at all.
        print("2. Deleting botched 'Boolean vs Integer' lessons...")
        for key_prefix in ["e8b2ca0d-709", "d9f0a307-9c2"]:
            conn.execute(
                "DELETE FROM store WHERE prefix = 'global.lessons.pinned' AND key LIKE %s",
                (f"{key_prefix}%",)
            )
            deleted.append(f"{key_prefix}... (botched Boolean lesson)")

        # 3. Delete duplicate "column existence" lesson
        #    Keep '1cb95c60-5d0' (title: "Avoiding SQL Joins with Non-Existent Columns")
        #    Delete '8d94895e-f6c' (title: "Avoiding SQL Errors with Nested Joins") -- same instruction
        print("3. Deleting duplicate 'column existence' lesson...")
        conn.execute(
            "DELETE FROM store WHERE prefix = 'global.lessons.pinned' AND key LIKE %s",
            ("8d94895e-f6c%",)
        )
        deleted.append("8d94895e-f6c... (duplicate 'column existence')")
        kept.append("1cb95c60-5d0... ('Avoiding SQL Joins with Non-Existent Columns')")

        # 4. Delete generic "Ambiguous Column References" duplicate
        #    Keep '44782af2-80e' (dynamic, more specific title with "in JOINs")
        #    Delete '5f3d2b79-a33' (dynamic, generic title, same instruction)
        print("4. Deleting generic 'Ambiguous Column References' duplicate...")
        conn.execute(
            "DELETE FROM store WHERE prefix = 'global.lessons.dynamic' AND key LIKE %s",
            ("5f3d2b79-a33%",)
        )
        deleted.append("5f3d2b79-a33... (generic 'Ambiguous Column References')")
        kept.append("44782af2-80e... ('Avoid Ambiguous Column References in JOINs')")

    # Verify remaining counts
    pool2 = ConnectionPool(
        conninfo=neondb_url,
        max_size=1,
        min_size=1,
        kwargs={"autocommit": True, "row_factory": psycopg.rows.dict_row}
    )
    with pool2.connection() as conn:
        for ns_label in ["pinned", "dynamic"]:
            prefix = f"global.lessons.{ns_label}"
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM store WHERE prefix = %s",
                (prefix,)
            ).fetchone()
            print(f"\n   Remaining in global.lessons.{ns_label}: {count['cnt']}")

            rows = conn.execute(
                "SELECT key, value->>'title' as title FROM store WHERE prefix = %s ORDER BY value->>'title'",
                (prefix,)
            ).fetchall()
            for r in rows:
                print(f"     - {r['title'][:70]}")

    pool.close()
    pool2.close()

    print(f"\n{'='*60}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"\nDeleted {len(deleted)} entries:")
    for d in deleted:
        print(f"  - {d}")
    print(f"\nKept (deduplicated):")
    for k in kept:
        print(f"  + {k}")


if __name__ == "__main__":
    cleanup_lessons()
