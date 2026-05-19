import pytest
from src.services.sql_engine import SQLTranspiler

def test_merge_snippets_simple():
    snippets = {
        "t1": "SELECT film_id, title FROM film WHERE title = 'ACADEMY DINOSAUR'",
        "t2": "SELECT film_id, actor_id FROM film_actor"
    }
    join_plan = {
        "base_task": "t1",
        "steps": [
            {"left": "t1", "right": "t2", "on": "film_id", "join_type": "inner"}
        ],
        "final_select": "t1.title, t2.actor_id"
    }
    
    final_sql = SQLTranspiler.merge_snippets(snippets, join_plan)
    
    # Assertions - Check for presence of components (dialect agnostic or normalized)
    assert "t1 AS (" in final_sql
    assert "t2 AS (" in final_sql
    assert "SELECT" in final_sql
    assert "t1.title" in final_sql
    assert "t2.actor_id" in final_sql
    assert "JOIN t2" in final_sql
    assert "ON t1.film_id = t2.film_id" in final_sql or "ON t1.film_id = t2.film_id" in final_sql.replace("\n", "").replace("  ", " ")

def test_merge_snippets_complex():
    snippets = {
        "t1": "SELECT film_id FROM film_category fc JOIN category c ON fc.category_id = c.category_id WHERE c.name = 'Action'",
        "t2": "SELECT actor_id, film_id FROM film_actor",
        "t3": "SELECT actor_id, first_name, last_name FROM actor"
    }
    join_plan = {
        "base_task": "t1",
        "steps": [
            {"left": "t1", "right": "t2", "on": "film_id", "join_type": "inner"},
            {"left": "t2", "right": "t3", "on": "actor_id", "join_type": "inner"}
        ],
        "final_select": "t3.first_name, t3.last_name"
    }
    
    final_sql = SQLTranspiler.merge_snippets(snippets, join_plan)
    
    assert "t1 AS (" in final_sql
    assert "t2 AS (" in final_sql
    assert "t3 AS (" in final_sql
    assert "JOIN t2" in final_sql
    assert "JOIN t3" in final_sql
    # Check join condition with normalization
    normalized_sql = final_sql.replace("\n", " ").replace("  ", " ")
    assert "ON t1.film_id = t2.film_id" in normalized_sql
    assert "ON t2.actor_id = t3.actor_id" in normalized_sql
