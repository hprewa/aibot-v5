from bigquery_client import BigQueryClient

def check_orders_data():
    client = BigQueryClient()
    
    # Check CFC names
    cfc_query = """
    SELECT DISTINCT cfc 
    FROM `text-to-sql-dev.chatbotdb.orders_drt`
    """
    
    # Check date range
    date_query = """
    SELECT MIN(delivery_date) as min_date, MAX(delivery_date) as max_date
    FROM `text-to-sql-dev.chatbotdb.orders_drt`
    """
    
    try:
        print("Checking CFC names...")
        cfc_result = client.client.query(cfc_query).result()
        print("\nAvailable CFCs:")
        for row in cfc_result:
            print(f"- {row.cfc}")
            
        print("\nChecking date range...")
        date_result = client.client.query(date_query).result()
        for row in date_result:
            print(f"\nDate range:")
            print(f"- Earliest date: {row.min_date}")
            print(f"- Latest date: {row.max_date}")
            
    except Exception as e:
        print(f"Error querying data: {str(e)}")

if __name__ == "__main__":
    check_orders_data() 