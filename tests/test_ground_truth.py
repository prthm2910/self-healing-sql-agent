from src.services.sql_engine import sql_engine

def check_q2():
    query = """
    SELECT TO_CHAR(rental_date, 'YYYY-MM') as month, COUNT(*) as rental_count 
    FROM rental 
    WHERE rental_date >= '2022-01-01' AND rental_date <= '2022-12-31' 
    GROUP BY month 
    ORDER BY month;
    """
    result = sql_engine.execute_query(query)
    print("Q2 (Rentals):", result['data'])

def check_q3():
    query = """
    SELECT c.first_name, c.last_name, SUM(p.amount) as total_spent 
    FROM customer c 
    JOIN payment p ON c.customer_id = p.customer_id 
    GROUP BY c.customer_id, c.first_name, c.last_name 
    ORDER BY total_spent DESC 
    LIMIT 3;
    """
    result = sql_engine.execute_query(query)
    print("Q3 (Spending):", result['data'])

if __name__ == "__main__":
    check_q2()
    check_q3()
