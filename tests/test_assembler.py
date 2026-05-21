from src.services.sql_assembler import sql_assembler

def test_merge_snippets_simple():
    snippets = {
        "t1": "SELECT film_id, title FROM film WHERE title = 'ACADEMY DINOSAUR'",
        "t2": "SELECT film_id, actor_id FROM film_actor"
    }
    join_plan = {
        "base_task": "t1",
        "steps": [
            {"left": "t1", "right": "t2", "on": "t1.t1_film_id = t2.t2_film_id", "join_type": "inner"}
        ],
        "final_select": "t1_title, t2_actor_id"
    }
    
    final_sql = sql_assembler.assemble(snippets, join_plan)
    
    # Assertions - Check for presence of components
    assert "t1 AS (" in final_sql
    assert "t2 AS (" in final_sql
    assert "SELECT" in final_sql
    assert "t1_title" in final_sql
    assert "t2_actor_id" in final_sql
    assert "JOIN t2" in final_sql

def test_merge_snippets_complex():
    snippets = {
        "t1": "SELECT film_id FROM film_category fc JOIN category c ON fc.category_id = c.category_id WHERE c.name = 'Action'",
        "t2": "SELECT actor_id, film_id FROM film_actor",
        "t3": "SELECT actor_id, first_name, last_name FROM actor"
    }
    join_plan = {
        "base_task": "t1",
        "steps": [
            {"left": "t1", "right": "t2", "on": "t1.t1_film_id = t2.t2_film_id", "join_type": "inner"},
            {"left": "t2", "right": "t3", "on": "t2.t2_actor_id = t3.t3_actor_id", "join_type": "inner"}
        ],
        "final_select": "t3_first_name, t3_last_name"
    }
    
    final_sql = sql_assembler.assemble(snippets, join_plan)
    
    assert "t1 AS (" in final_sql
    assert "t2 AS (" in final_sql
    assert "t3 AS (" in final_sql
    assert "JOIN t2" in final_sql
    assert "JOIN t3" in final_sql
    # Check join condition with normalization
    normalized_sql = final_sql.replace("\n", " ").replace("  ", " ")
    assert "t1.t1_film_id = t2.t2_film_id" in normalized_sql
    assert "t2.t2_actor_id = t3.t3_actor_id" in normalized_sql
