# Cursor Grounding Document

## **Overview**
This document describes the architecture and workflow of an AI agent designed to process analytical queries via Slack. The agent extracts constraints, generates SQL queries, retrieves data from BigQuery, and returns summarized insights.

## **Workflow**

### **Step 0: System Initialization**
- Load KPI table schemas and definitions.
- Populate DSPy mapping table with constraint-to-query mappings.

### **Step 1: User Sends a Query**
- A user posts a message in Slack, triggering the bot.
- Example questions:
  - *"What were the orders for London last week?"*
  - *"Compare last week's eaches with last to last week’s eaches."*

### **Step 2: Strategy Agent Plans the Query**
- Uses **Gemini 2.0 Thinking** to determine:
  - **KPI** (e.g., orders, revenue).
  - **Time and location constraints**.
  - **SQL query structure** (single or multiple queries).
- Updates the **Session Log** with the extracted constraints.

### **Step 3: Extract Constraints for SQL Query**
- Extracts key parameters:
  ```json
  {
    "kpi": ["orders"],
    "time_granularity": "Weekly",
    "location_granularity": "Cfc",
    "start_date_filter": "2025-03-01",
    "end_date_filter": "2025-03-07"
  }
  ```
- Logs the extracted constraints in the **Session Table**.

### **Step 4: Store Query Execution State**
- Updates session tracking with pending tool calls:
  ```json
  {
    "session_id": "abc-123",
    "tool_calls": ["BigQuery_orders"],
    "tool_call_status": ["pending"]
  }
  ```

### **Step 5: DSPy Agent Generates SQL**
- Searches **DSPy Mapping Table** for a pre-existing SQL template.
- If a match exists → Uses it.
- If no match → Generates and stores a new SQL query.
- Example SQL query:
  ```sql
  SELECT COUNT(*) AS orders
  FROM orders_table
  WHERE location = 'London' 
    AND order_date BETWEEN '2025-03-01' AND '2025-03-07';
  ```

### **Step 6: Execute SQL on BigQuery**
- Runs the SQL query and retrieves results.
- Stores the results in the **Session Table**.

### **Step 7: Check Query Completion**
- If all queries are completed, move to summarization.

### **Step 8: Generate Summary**
- **Gemini 2.0 Pro** generates a summary based on:
  - The retrieved data.
  - The strategy outlined in **Step 2**.
- Example summary:
  > "The total number of orders in London last week was **15,345**, a **5% increase** from the previous week."

### **Step 9: Return Summary to Slack**
- Posts the summarized response in Slack.

---

## **Database Schema**

### **Session Log Table**
```json
{
  "session_id": "abc-123",
  "user_id": "U12345",
  "question": "What were the orders for London last week?",
  "constraints": {
    "kpi": ["orders"],
    "time_granularity": "Weekly",
    "location_granularity": "City",
    "start_date_filter": "2025-03-01",
    "end_date_filter": "2025-03-07"
  },
  "strategy": "Fetch 'orders' KPI for London for last week and summarize.",
  "final_query": ["SELECT COUNT(*) AS orders FROM orders_table WHERE location = 'London' AND order_date BETWEEN '2025-03-01' AND '2025-03-07'"],
  "tool_calls": ["BigQuery_orders"],
  "tool_call_status": ["completed"],
  "summary": "The total number of orders in London last week was 15,345, a 5% increase from the previous week.",
  "status": "completed",
  "updated_at": "2025-03-10T12:15:00Z"
}
```

### **DSPy Mapping Table**
```json
{
  "user_question": "What were the orders for London last week?",
  "kpi": "orders",
  "time_granularity": "Weekly",
  "location_granularity": "City",
  "sql_query": "SELECT COUNT(*) AS orders FROM orders_table WHERE location = 'London' AND order_date BETWEEN '2025-03-01' AND '2025-03-07';",
  "validated": true,
  "uploaded_timestamp": "2025-03-10T12:00:00Z"
}
```

---

## **Enhancements & Future Considerations**
1. **Cache Optimization**
   - Store previous query results to avoid redundant computations.

2. **Error Handling & Retries**
   - Implement auto-retries for failed queries.

3. **Multi-Turn Conversations**
   - Allow follow-up questions for deeper insights.

---

## **Conclusion**
This document outlines the structured approach for handling Slack-based analytical queries using a **Strategy Agent, DSPy for SQL generation, BigQuery for execution, and Gemini for summarization**. The system ensures efficient query handling, tracking, and response generation.
