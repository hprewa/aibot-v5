# Session Management with BigQuery

This document describes the implementation of session management using BigQuery for storing and retrieving session data.

## Overview

The session management system provides a way to:
- Create and store user sessions with questions
- Track session state and updates
- Store constraints extracted from questions
- Store detailed strategy and response plans
- Track tool call status and results
- Query session history

## Table Schema

The sessions table in BigQuery has the following schema:

```sql
CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.${DATASET_ID}.sessions` (
  session_id STRING NOT NULL,
  user_id STRING NOT NULL,
  question STRING NOT NULL,
  constraints JSON,
  response_plan JSON,
  strategy JSON,
  summary STRING,
  status STRING NOT NULL,
  tool_calls ARRAY<STRING>,
  tool_call_status ARRAY<STRING>,
  tool_call_results JSON,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
)
PARTITION BY DATE(created_at)
CLUSTER BY session_id;
```

## Implementation Details

### Append-Only Pattern

Due to BigQuery's streaming buffer limitations (which prevent updating recently inserted rows), we use an append-only pattern for session updates:

1. When a session is created, a new row is inserted with the initial session data
2. When a session is updated, a new row is inserted with the updated session data
3. When querying a session, we get the most recent row for that session ID

This approach allows us to:
- Avoid the streaming buffer limitation
- Maintain a history of session updates
- Query the latest state of a session efficiently

### Session Manager

The `SessionManagerV2` class provides the following methods:

- `create_session(user_id, question)`: Creates a new session with a unique ID
- `get_session(session_id)`: Retrieves the latest state of a session
- `update_session(session_id, updates)`: Updates a session by inserting a new row
- `update_constraints(session_id, constraints)`: Updates the constraints for a session
- `update_strategy(session_id, strategy)`: Updates the strategy for a session
- `update_response_plan(session_id, response_plan)`: Updates the response plan for a session
- `update_summary(session_id, summary)`: Updates the summary for a session
- `update_tool_call_status(session_id, tool_name, status, result)`: Updates the status and result of a tool call
- `is_session_complete(session_id)`: Checks if all tool calls in a session are completed

### Enhanced Fields

The enhanced session management system includes the following additional fields:

- `strategy`: Stores the detailed strategy for processing the question
- `tool_call_results`: Stores the results of tool calls, including SQL queries and data

### Utility Scripts

- `create_sessions_table.py`: Creates the sessions table in BigQuery
- `test_session_v2.py`: Tests the session manager implementation
- `test_enhanced_session.py`: Tests the enhanced session manager with additional fields
- `create_test_session.py`: Creates a test session with a specific question
- `query_sessions.py`: Queries and displays session data from BigQuery

## Usage Examples

### Creating a Session

```python
from session_manager_v2 import SessionManagerV2

# Initialize session manager
session_manager = SessionManagerV2()

# Create a new session
session_id = session_manager.create_session(
    user_id="user123",
    question="What were the total orders for CFC1 in the last week?"
)
```

### Updating a Session with Constraints

```python
# Update session with constraints
constraints = {
    "kpi": ["orders"],
    "time_aggregation": "Weekly",
    "time_filter": {
        "start_date": "2025-02-09",
        "end_date": "2025-03-11"
    },
    "cfc": ["CFC-1"],
    "spokes": [],
    "comparison_type": "trend",
    "tool_calls": [
        {
            "name": "Orders Trend for CFC-1",
            "description": "This query retrieves order trends for CFC-1 over the past month",
            "tables": ["orders"],
            "result_id": "orders_trend_cfc1"
        }
    ]
}
session_manager.update_constraints(session_id, constraints)
```

### Updating a Session with Strategy

```python
# Update session with strategy
strategy = """
## Comprehensive Strategy

### 1. Data Collection Plan:
* **Desired Data**: Weekly order counts for CFC-1
* **Data Source**: Orders table
* **Fetch Specifics**: Select columns: `week_date`, `orders`, `cfc`

### 2. Processing Steps:
* **Calculate weekly order aggregates**: Group orders by `cfc` and `week_date`

### 3. Calculations Required:
* **Sum of orders by week for CFC**: `SUM(orders) GROUP BY cfc, week_date`
"""
session_manager.update_strategy(session_id, strategy)
```

### Updating a Session with Response Plan

```python
# Update session with response plan
response_plan = {
    "data_connections": [
        {
            "result_id": "orders_trend_cfc1",
            "purpose": "Analyze order trends for CFC-1",
            "processing_steps": ["Aggregate orders by week"],
            "outputs": ["Weekly order trends for CFC-1"]
        }
    ],
    "insights": [
        {
            "type": "trend",
            "description": "Analyze weekly order trends for CFC-1",
            "source_result_ids": ["orders_trend_cfc1"]
        }
    ],
    "response_structure": {
        "introduction": "This analysis examines order trends for CFC-1 over the past month.",
        "main_points": ["Key trends in orders for CFC-1"],
        "conclusion": "The analysis provides insights into order patterns for CFC-1."
    }
}
session_manager.update_response_plan(session_id, response_plan)
```

### Updating Tool Call Status with Results

```python
# Update tool call status with results
tool_name = "Orders Trend for CFC-1"
tool_result = {
    "sql": "SELECT week_date, SUM(orders) AS total_orders FROM orders WHERE cfc = 'CFC-1' GROUP BY week_date",
    "data": [
        {"week_date": "2025-02-09", "total_orders": 1200},
        {"week_date": "2025-02-16", "total_orders": 1350},
        {"week_date": "2025-02-23", "total_orders": 1100}
    ]
}
session_manager.update_tool_call_status(session_id, tool_name, "completed", tool_result)
```

### Updating Session with Summary

```python
# Update session with summary
summary = """
## Order Trends for CFC-1

This analysis examines order trends for CFC-1 over the past month, broken down by week.

### Key Findings
- Orders peaked in the week of February 16 with 1,350 orders
- The lowest order volume was in the week of February 23 with 1,100 orders
- The average weekly order volume was 1,217 orders
"""
session_manager.update_summary(session_id, summary)
```

### Querying Sessions

```bash
# Query all sessions (limited to 10)
python query_sessions.py

# Query a specific session
python query_sessions.py <session_id>

# Query a specific number of sessions
python query_sessions.py --limit 5

# Query all sessions with detailed information
python query_sessions.py --details
```

## Best Practices

1. Always use the `get_session` method to retrieve the latest state of a session
2. Use the `update_session` method to update a session, which handles the append-only pattern
3. Use the specialized update methods (`update_constraints`, `update_strategy`, etc.) for specific fields
4. Use the `update_tool_call_status` method to update tool call status and results
5. Query sessions using the provided utility scripts, which handle the append-only pattern 

## Querying the Latest State of a Session

If you want to get only the latest state of each session (rather than all historical versions), you can use a query similar to what's in the `query_sessions.py` file:

```sql
WITH latest_sessions AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY updated_at DESC) as row_num
    FROM `${PROJECT_ID}.${DATASET_ID}.sessions`
)
SELECT * FROM latest_sessions WHERE row_num = 1
```

This query uses a window function to identify the most recent row for each session ID and returns only those rows. 