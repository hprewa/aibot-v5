# BigQuery SQL Prompt for Orders Data

## Table Schema
You are working with a database table named `orders_data` that contains aggregated order data.  
The table has the following structure:

- **`delivery_date` (DATE)**: The actual date on which the orders were delivered. Used for **daily aggregations**.
- **`week_date` (DATE)**: The Monday of the week in which the delivery occurred. Used for **weekly aggregations**.
- **`cfc` (STRING)**: The name of the Customer Fulfillment Center (CFC). Each CFC has multiple spokes.
- **`spoke` (STRING)**: The name of the spoke, which is associated with a CFC.
- **`orders` (INTEGER)**: The total number of orders for the given `delivery_date`, `cfc`, and `spoke`.

## Aggregation & Filtering Rules

### 1. Aggregation Type
- If the query requests **daily data**, aggregate using `delivery_date`.
- If the query requests **weekly data**, aggregate using `week_date`.
- If the query requests **monthly data**, aggregate using `EXTRACT(YEAR FROM delivery_date), EXTRACT(MONTH FROM delivery_date)`.

### 2. Filtering Time Period
- The query should always filter data using a `WHERE` clause with a **date range** (`BETWEEN start_date AND end_date`).
- The **time period filter applies to `delivery_date`** for daily and monthly queries.
- The **time period filter applies to `week_date`** for weekly queries.

### 3. Grouping Rules
- If aggregating at the **CFC level**, sum `orders` over all associated spokes.
- If aggregating at the **spoke level**, group by `spoke` and sum `orders`.

## Example Queries

### 1. "Give me daily orders of CFC1 for last week."
**SQL Query:**
```sql
SELECT delivery_date, SUM(orders) AS total_orders
FROM `your_project.dataset.orders_data`
WHERE delivery_date BETWEEN '2025-03-03' AND '2025-03-09'
AND cfc = 'CFC1'
GROUP BY delivery_date
ORDER BY delivery_date;
```

### 2. "Give me weekly orders for Spoke1 for last month."
**SQL Query:**
```sql
SELECT week_date, SUM(orders) AS total_orders
FROM `your_project.dataset.orders_data`
WHERE week_date BETWEEN '2025-02-01' AND '2025-02-28'
AND spoke = 'Spoke1'
GROUP BY week_date
ORDER BY week_date;
```

### 3. "Show me total monthly orders for each CFC in January 2025."
**SQL Query:**
```sql
SELECT EXTRACT(YEAR FROM delivery_date) AS year, EXTRACT(MONTH FROM delivery_date) AS month, cfc, SUM(orders) AS total_orders
FROM `your_project.dataset.orders_data`
WHERE delivery_date BETWEEN '2025-01-01' AND '2025-01-31'
GROUP BY year, month, cfc
ORDER BY year, month, cfc;
```

### 4. "Get total orders for CFC2 in February 2025."
**SQL Query:**
```sql
SELECT cfc, SUM(orders) AS total_orders
FROM `your_project.dataset.orders_data`
WHERE delivery_date BETWEEN '2025-02-01' AND '2025-02-28'
AND cfc = 'CFC2'
GROUP BY cfc;
```
