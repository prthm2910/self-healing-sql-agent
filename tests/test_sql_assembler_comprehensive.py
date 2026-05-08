import time
from src.services.sql_assembler import sql_assembler
from src.utils.logger import logger

def run_test(name, islands, join_plan, final_select):
    print(f"\n{'='*20} TEST: {name} {'='*20}")
    try:
        start_time = time.time()
        
        # 1. Assembly (Injection is now handled INTERNALLY)
        final_sql = sql_assembler.assemble(islands, join_plan, final_select)
        
        duration = (time.time() - start_time) * 1000
        print(f"STATUS: SUCCESS ({duration:.2f}ms)")
        print("GENERATED SQL:")
        print(final_sql)
        return True, duration
    except Exception as e:
        print(f"STATUS: FAILED")
        print(f"ERROR: {e}")
        return False, 0

def comprehensive_suite():
    results = []

    # --- 1. SIMPLE: 2 Islands, 1 Join ---
    islands_1 = {
        "actor_info": "SELECT first_name, last_name FROM actor",
        "film_counts": "SELECT actor_id, COUNT(*) as total FROM film_actor GROUP BY actor_id"
    }
    join_plan_1 = [{"source": "actor_info", "target": "film_counts", "on": "actor_info.actor_info_actor_id = film_counts.film_counts_actor_id"}]
    # Note: final_select must use the aliased names: {island_id}_{original_name}
    final_select_1 = ["actor_info_first_name", "actor_info_last_name", "film_counts_total"]
    results.append(("SIMPLE", *run_test("SIMPLE: Basic 2-Island Join", islands_1, join_plan_1, final_select_1)))

    # --- 2. MEDIUM: 3 Islands, Column Collisions ---
    islands_2 = {
        "cust": "SELECT first_name, last_name, address_id FROM customer",
        "addr": "SELECT address_id, city_id, address FROM address",
        "city": "SELECT city_id, city FROM city"
    }
    # Both 'customer' and 'city' have a 'city' like column? No, but let's assume 'name' collisions.
    # In Pagila, address and city both have 'last_update'.
    islands_2_collision = {
        "cust": "SELECT first_name, last_name, address_id FROM customer",
        "addr": "SELECT address_id, city_id, last_update FROM address",
        "city": "SELECT city_id, city, last_update FROM city"
    }
    join_plan_2 = [
        {"source": "cust", "target": "addr", "on": "cust.cust_address_id = addr.addr_address_id"},
        {"source": "addr", "target": "city", "on": "addr.addr_city_id = city.city_city_id"}
    ]
    final_select_2 = ["cust_first_name", "cust_last_name", "addr_last_update", "city_last_update"]
    results.append(("MEDIUM", *run_test("MEDIUM: 3-Island Collision Handling", islands_2_collision, join_plan_2, final_select_2)))

    # --- 3. HARD: Aggregates + Sub-filters + Complex Aliasing ---
    islands_3 = {
        "sales": "SELECT staff_id, SUM(amount) as revenue FROM payment WHERE payment_date > '2024-01-01' GROUP BY staff_id",
        "staff_info": "SELECT staff_id, first_name, last_name, store_id FROM staff",
        "store_location": "SELECT store_id, address_id FROM store"
    }
    join_plan_3 = [
        {"source": "sales", "target": "staff_info", "on": "sales.sales_staff_id = staff_info.staff_info_staff_id"},
        {"source": "staff_info", "target": "store_location", "on": "staff_info.staff_info_store_id = store_location.store_location_store_id"}
    ]
    final_select_3 = ["staff_info_first_name", "sales_revenue", "store_location_store_id"]
    results.append(("HARD", *run_test("HARD: Multi-Join with Filters/Aggs", islands_3, join_plan_3, final_select_3)))

    # --- 4. ABSOLUTE HARD NUT: Circular Logic + Multi-Path Bridge ---
    # User: "Find actors who worked with staff member 'Mike' in 'Action' films"
    islands_4 = {
        "mike_rentals": "SELECT inventory_id FROM rental r JOIN staff s ON r.staff_id = s.staff_id WHERE s.first_name = 'Mike'",
        "action_inv": "SELECT inventory_id, film_id FROM inventory i JOIN film_category fc ON i.film_id = fc.film_id JOIN category c ON fc.category_id = c.category_id WHERE c.name = 'Action'",
        "film_actors": "SELECT film_id, actor_id FROM film_actor",
        "actor_names": "SELECT actor_id, first_name, last_name FROM actor"
    }
    join_plan_4 = [
        {"source": "mike_rentals", "target": "action_inv", "on": "mike_rentals.mike_rentals_inventory_id = action_inv.action_inv_inventory_id"},
        {"source": "action_inv", "target": "film_actors", "on": "action_inv.action_inv_film_id = film_actors.film_actors_film_id"},
        {"source": "film_actors", "target": "actor_names", "on": "film_actors.film_actors_actor_id = actor_names.actor_names_actor_id"}
    ]
    final_select_4 = ["actor_names_first_name", "actor_names_last_name", "COUNT(*) as appearances"]
    results.append(("HARD NUT", *run_test("HARD NUT: 4-Island Deep Chain", islands_4, join_plan_4, final_select_4)))

    # --- FINAL REPORT ---
    print("\n" + "#"*50)
    print("### SQL ASSEMBLER COMPREHENSIVE REPORT ###")
    print("#"*50)
    print(f"{'CATEGORY':<15} | {'STATUS':<10} | {'LATENCY':<10}")
    print("-" * 40)
    for cat, status, lat in results:
        stat_str = "✅ PASS" if status else "❌ FAIL"
        print(f"{cat:<15} | {stat_str:<10} | {lat:.2f}ms")
    print("#"*50)

if __name__ == "__main__":
    comprehensive_suite()
