# Model Context Protocol (MCP) for Analytics Bot

This document describes the Model Context Protocol (MCP) implementation for the Analytics Bot project.

## Overview

The Model Context Protocol (MCP) provides a standardized way to manage context flow between components in the Analytics Bot system, enabling better traceability, debugging, and testing.

MCP introduces several benefits over the previous approach:

1. **Standardized Context Flow**: Context flows through the system in a consistent, typed manner.
2. **Reduced Redundancy**: Eliminates repeated JSON parsing/serialization code throughout the components.
3. **Better Error Handling**: Errors are captured within the context objects themselves.
4. **Improved Traceability**: The transformation of context between components is explicit.
5. **Enhanced Debugging**: The exact state of context at each step can be inspected.

## Architecture

The MCP implementation consists of three main parts:

1. **Protocol Layer (`mcp/protocol.py`)**: Core protocol classes like `Context`, `ContextProcessor`, `ContextFlow`, and `ContextRegistry`.

2. **Data Models (`mcp/models.py`)**: Typed data models for each component in the system, e.g., `QueryData`, `ConstraintData`, `StrategyData`, etc.

3. **Processors (`mcp/processors.py`)**: MCP wrappers around the original components, implementing the `ContextProcessor` interface.

### Components

- **MCPQueryProcessor**: Extracts constraints from user queries.
- **MCPQueryAgent**: Generates SQL queries from constraints.
- **MCPQueryExecutor**: Executes SQL queries using BigQuery.
- **MCPResponseGenerator**: Generates natural language responses from query results.
- **MCPSessionManager**: Manages session state using the context objects.
- **MCPQueryFlowOrchestrator**: Orchestrates the flow of contexts between processors.

## How It Works

1. **Context Creation**: When a query is received, a `Context[QueryData]` is created.
2. **Context Transformation**: As the query flows through the system, the context is transformed:
   - `Context[QueryData]` → `Context[ConstraintData]` → `Context[StrategyData]` → `Context[QueryExecutionData]` → `Context[ResponseData]` → `Context[SessionData]`
3. **Immutability**: Each transformation creates a new context object, preserving the lineage.
4. **Error Handling**: Errors are captured in the context metadata, allowing for graceful failure handling.
5. **Session Management**: All contexts related to a query are tied to a single session.

## Usage Example

```python
# Initialize components
bigquery_client = BigQueryClient()
gemini_client = GeminiClient()
query_processor = QueryProcessor(gemini_client, bigquery_client)
query_agent = QueryAgent(gemini_client)
response_agent = ResponseAgent(gemini_client)
session_manager = SessionManagerV2()

# Initialize MCP wrappers
mcp_query_processor = MCPQueryProcessor(query_processor)
mcp_query_agent = MCPQueryAgent(query_agent)
mcp_query_executor = MCPQueryExecutor(bigquery_client)
mcp_response_generator = MCPResponseGenerator(response_agent)
mcp_session_manager = MCPSessionManager(session_manager)

# Create the flow orchestrator
orchestrator = MCPQueryFlowOrchestrator(
    mcp_query_processor,
    mcp_query_agent,
    mcp_query_executor,
    mcp_response_generator,
    mcp_session_manager
)

# Create a query data object
query_data = QueryData(
    user_id="test_user",
    question="How many orders did we have last week?",
    session_id=str(uuid.uuid4()),
    created_at=datetime.utcnow()
)

# Process the query
result_context = orchestrator.process_query(query_data)

# Check the result
if result_context.metadata.status == "success":
    print(f"Query processed successfully: {result_context.data.response.summary}")
else:
    print(f"Error processing query: {result_context.metadata.error}")
```

## Testing

To test the MCP implementation, run the `test_mcp.py` script:

```bash
python test_mcp.py
```

This will test both individual components and the full query flow.

## Benefits Over Previous Implementation

1. **Type Safety**: The previous implementation relied heavily on untyped dictionaries and JSON strings, leading to potential runtime errors. MCP uses typed Pydantic models.

2. **Explicit Data Flow**: The previous implementation had implicit data flow between components. MCP makes these transformations explicit and traceable.

3. **Consistent Error Handling**: The previous implementation had inconsistent error handling. MCP provides a standardized way to capture and propagate errors.

4. **Immutability**: The previous implementation mutated state. MCP creates new context objects for each transformation, preserving the history.

5. **Debugging**: The previous implementation made debugging difficult. MCP allows for inspection of the exact state at each step.

## Future Improvements

1. **Context Registry**: Implement a central registry to store all context objects, enabling lookup by ID.

2. **Visualization**: Create a visualization tool to display the context lineage graph.

3. **Persistence**: Store context objects in a database for post-hoc analysis.

4. **Parallel Processing**: Enable parallel processing of contexts where appropriate.

5. **Integration with Monitoring**: Integrate with monitoring systems to track context flow and errors.

## Conclusion

The Model Context Protocol provides a robust framework for managing context flow in the Analytics Bot system. By standardizing how context is passed between components, MCP improves traceability, debugging, and testing while reducing redundancy and making the system more maintainable. 