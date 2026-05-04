from src.services.sql_engine import sql_engine

def find_max_staff():
    query = """
    SELECT co.country, COUNT(s.staff_id) AS staff_count 
    FROM staff s 
    JOIN address a ON s.address_id = a.address_id 
    JOIN city ci ON a.city_id = ci.city_id 
    JOIN country co ON ci.country_id = co.country_id 
    GROUP BY co.country 
    ORDER BY staff_count DESC 
    LIMIT 1;
    """
    result = sql_engine.execute_query(query)
    if result["status"] == "success" and result["data"]:
        data = result["data"][0]
        print(f"Country: {data['country']}")
        print(f"Count: {data['staff_count']}")
    else:
        print(f"Error or no data: {result.get('error_message', 'No data found')}")

if __name__ == "__main__":
    find_max_staff()
