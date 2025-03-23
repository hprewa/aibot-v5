# Query Agent

This document provides an overview of the Query Agent, which is responsible for generating SQL queries based on tool calls created by the Strategy Agent.

## Overview

The Query Agent is a component of the Analytics Bot that generates SQL queries for different KPIs (Key Performance Indicators) based on tool calls created by the Strategy Agent. It can use either:

1. **DSPy Tables**: For standard queries with well-defined structures
2. **Gemini**: For more complex queries not covered by DSPy tables

## Architecture

The Query Agent is integrated with the existing pipeline as follows:

```
User Question
    ↓
Strategy Agent (generates strategy and tool calls)
    ↓
Query Agent (generates SQL queries for each tool call)
    ↓
BigQuery Client (executes the queries)
    ↓
Response Agent (generates a response based on the results)
```

## Components

### DSPy Tables

DSPy tables are defined in `dspy_tables.py` and provide a structured way to generate SQL queries for specific KPIs. Each table is defined as a class with:

- Table schema (column definitions)
- Methods for generating SQL queries based on parameters

Currently implemented tables:
- `OrdersTable`: For generating queries related to order data

### Query Agent

The Query Agent (`query_agent.py`) is responsible for:

1. Determining the KPI type from a tool call
2. Routing the query generation to the appropriate handler based on the KPI type
3. Falling back to Gemini for KPIs without specific handlers

## KPI Handlers

Each KPI has a specific handler method in the Query Agent:

- `_handle_orders_query`: Generates queries for the orders KPI using the OrdersTable

## Adding New KPIs

To add support for a new KPI:

1. Create a new DSPy table definition in `dspy_tables.py`
2. Add a new handler method in the Query Agent
3. Register the handler in the `kpi_handlers` dictionary in the Query Agent's `__init__` method
4. Add documentation for the KPI in a markdown file (e.g., `kpi_name sql prompt.md`)

## Testing

The Query Agent can be tested using:

- `test_query_agent.py`: Tests the Query Agent in isolation
- `test_integration.py`: Tests the integration of the Query Agent with the existing pipeline

## Example Usage

```python
# Initialize the Query Agent
gemini_client = GeminiClient()
query_agent = QueryAgent(gemini_client)

# Create a tool call
tool_call = {
    "name": "Daily Orders for CFC-1",
    "description": "Fetch daily orders for CFC-1 for the last week",
    "tables": ["orders_data"],
    "result_id": "daily_orders_cfc1"
}

# Create constraints
constraints = {
    "time_filter": {
        "start_date": "2025-03-01",
        "end_date": "2025-03-07"
    },
    "cfc": ["CFC-1"],
    "time_granularity": "daily"
}

# Generate SQL
sql = query_agent.generate_query(tool_call, constraints)
print(sql)
```

## Future Enhancements

1. Add support for more KPIs (slot availability, inventory, etc.)
2. Implement caching for frequently used queries
3. Add query validation and optimization
4. Support for more complex query patterns (joins, subqueries, etc.)
5. Integration with a query library for reusing common query patterns 