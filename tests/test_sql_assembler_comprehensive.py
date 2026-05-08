import unittest
import time
from src.services.sql_assembler import sql_assembler

class TestSQLAssembler(unittest.TestCase):
    """
    Comprehensive, assertion-based Test Suite for SQLAssembler.
    Covers Simple to 'Hard Nut' scenarios with architectural hardening.
    """

    def test_01_simple_2_island_join(self):
        """SIMPLE: Basic 2-Island Join with internal key injection."""
        islands = {
            "actor_info": "SELECT first_name, last_name FROM actor",
            "film_counts": "SELECT actor_id, COUNT(*) as total FROM film_actor GROUP BY actor_id"
        }
        join_plan = [{"source": "actor_info", "target": "film_counts", "on": "actor_info.actor_info_actor_id = film_counts.film_counts_actor_id"}]
        final_select = ["actor_info_first_name", "actor_info_last_name", "film_counts_total"]
        
        final_sql = sql_assembler.assemble(islands, join_plan, final_select)
        
        self.assertIn("actor_id AS actor_info_actor_id", final_sql)
        self.assertIn("INNER JOIN film_counts", final_sql)

    def test_02_medium_collision_handling(self):
        """MEDIUM: 3-Island Join with column name collisions (auto-aliasing)."""
        islands = {
            "cust": "SELECT first_name, last_name, address_id FROM customer",
            "addr": "SELECT address_id, city_id, last_update FROM address",
            "city": "SELECT city_id, city, last_update FROM city"
        }
        join_plan = [
            {"source": "cust", "target": "addr", "on": "cust.cust_address_id = addr.addr_address_id"},
            {"source": "addr", "target": "city", "on": "addr.addr_city_id = city.city_city_id"}
        ]
        final_select = ["cust_first_name", "addr_last_update", "city_last_update"]
        
        final_sql = sql_assembler.assemble(islands, join_plan, final_select)
        
        # Verify both last_update columns are unique
        self.assertIn("last_update AS addr_last_update", final_sql)
        self.assertIn("last_update AS city_last_update", final_sql)

    def test_03_group_by_injection_safety(self):
        """CRITICAL: Ensure injected keys are added to GROUP BY to prevent Postgres errors."""
        islands = {
            "sales": "SELECT SUM(amount) as revenue FROM payment GROUP BY staff_id"
        }
        join_plan = [{"source": "sales", "target": "other", "on": "sales.sales_payment_id = other.id"}]
        
        final_sql = sql_assembler.assemble(islands, join_plan, ["sales_revenue"])
        
        # Verify 'payment_id' was injected into both SELECT and GROUP BY
        self.assertIn("payment_id AS sales_payment_id", final_sql)
        self.assertIn("GROUP BY", final_sql)
        self.assertIn("payment_id", final_sql.split("GROUP BY")[1])

    def test_04_union_support_aliasing(self):
        """HIGH: Ensure UNION queries are correctly aliased recursively."""
        islands = {
            "people": "SELECT first_name FROM actor UNION SELECT first_name FROM staff"
        }
        join_plan = []
        final_select = ["people_first_name"]
        
        final_sql = sql_assembler.assemble(islands, join_plan, final_select)
        
        # Check for aliased output in the CTE
        self.assertIn("first_name AS people_first_name", final_sql)

    def test_05_hard_nut_4_island_chain(self):
        """HARD NUT: Verify the full deep chain with clean, non-duplicated aliases."""
        islands = {
            "mike_rentals": "SELECT inventory_id FROM rental r JOIN staff s ON r.staff_id = s.staff_id WHERE s.first_name = 'Mike'",
            "action_inv": "SELECT inventory_id, film_id FROM inventory i JOIN film_category fc ON i.film_id = fc.film_id JOIN category c ON fc.category_id = c.category_id WHERE c.name = 'Action'",
            "film_actors": "SELECT film_id, actor_id FROM film_actor",
            "actor_names": "SELECT actor_id, first_name, last_name FROM actor"
        }
        join_plan = [
            {"source": "mike_rentals", "target": "action_inv", "on": "mike_rentals.mike_rentals_inventory_id = action_inv.action_inv_inventory_id"},
            {"source": "action_inv", "target": "film_actors", "on": "action_inv.action_inv_film_id = film_actors.film_actors_film_id"},
            {"source": "film_actors", "target": "actor_names", "on": "film_actors.film_actors_actor_id = actor_names.actor_names_actor_id"}
        ]
        final_select = ["actor_names_first_name", "actor_names_last_name"]
        
        final_sql = sql_assembler.assemble(islands, join_plan, final_select)
        
        # Core structure checks
        self.assertIn("WITH mike_rentals AS", final_sql)
        self.assertIn("actor_names_last_name", final_sql)
        # Ensure prefix stripping worked (no island_id prefix repetition)
        self.assertIn("inventory_id AS mike_rentals_inventory_id", final_sql)
        self.assertNotIn("mike_rentals_inventory_id AS mike_rentals_mike_rentals_inventory_id", final_sql)

if __name__ == "__main__":
    unittest.main()
