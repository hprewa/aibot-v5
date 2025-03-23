"""
Model Context Protocol (MCP) - Processors

This module defines the processors for each component in the Analytics Bot.
Each processor implements the ContextProcessor interface and transforms
contexts between components.
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, cast
import json
import traceback
from datetime import datetime

from mcp.protocol import Context, ContextProcessor
from mcp.models import (
    QueryData, 
    ConstraintData, 
    StrategyData, 
    QueryExecutionData,
    ResponseData, 
    SessionData,
    ToolCallData
)

# Import original components to wrap
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from session_manager_v2 import SessionManagerV2

T = TypeVar('T')
U = TypeVar('U')

class MCPQueryProcessor(ContextProcessor):
    """MCP wrapper for the QueryProcessor"""
    
    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
        
    def process(self, context: Context[QueryData]) -> Context[ConstraintData]:
        """Extract constraints from the query"""
        try:
            # Extract the query data
            query_data = context.data
            
            # Extract constraints using the original query processor
            extracted_constraints = self.query_processor.extract_constraints(query_data.question)
            
            # Create constraint data model
            if isinstance(extracted_constraints, str):
                try:
                    extracted_constraints = json.loads(extracted_constraints)
                except json.JSONDecodeError:
                    extracted_constraints = {}
            
            constraint_data = ConstraintData(
                kpi=extracted_constraints.get("kpi", []),
                time_aggregation=extracted_constraints.get("time_aggregation", "Daily"),
                time_filter=extracted_constraints.get("time_filter", {}),
                cfc=extracted_constraints.get("cfc", []),
                spokes=extracted_constraints.get("spokes", []),
                comparison_type=extracted_constraints.get("comparison_type"),
                tool_calls=extracted_constraints.get("tool_calls", []),
                response_plan=extracted_constraints.get("response_plan", {})
            )
            
            # Return updated context
            return Context.create(
                data=constraint_data,
                component="QueryProcessor",
                operation="extract_constraints",
                parent_id=context.metadata.context_id
            ).success()
        except Exception as e:
            traceback.print_exc()
            return cast(Context[ConstraintData], context.error(f"Error extracting constraints: {str(e)}"))
    
    def generate_strategy(self, context: Context[ConstraintData]) -> Context[StrategyData]:
        """Generate a strategy from the constraints"""
        try:
            # Extract the constraint data
            constraint_data = context.data
            constraint_dict = constraint_data.dict()
            
            # Use the session to get the original question
            session_id = context.metadata.parent_id
            if not session_id:
                raise ValueError("No parent session ID found in metadata")
            
            # We need the query data to get the question
            # For now, assuming it's passed in a parent context
            # In a real implementation, we would fetch it from a registry or session storage
            # Workaround for this demo: reconstruct from available info
            original_question = "What is the query for these constraints?"  # Default fallback
            
            # Generate strategy using the original query processor
            raw_strategy = self.query_processor.generate_strategy(original_question, constraint_dict)
            
            # Create strategy data model 
            # In a full implementation, we would parse the raw strategy into structured fields
            strategy_data = StrategyData(
                raw_strategy=raw_strategy,
                # These would be parsed from the raw strategy in a full implementation
                data_collection_plan=[],
                processing_steps=[],
                calculations=[],
                response_structure=[],
                tool_calls=[]
            )
            
            # Return updated context
            return Context.create(
                data=strategy_data,
                component="QueryProcessor",
                operation="generate_strategy",
                parent_id=context.metadata.context_id
            ).success()
        except Exception as e:
            traceback.print_exc()
            return cast(Context[StrategyData], context.error(f"Error generating strategy: {str(e)}"))

class MCPQueryAgent(ContextProcessor):
    """MCP wrapper for the QueryAgent"""
    
    def __init__(self, query_agent: QueryAgent):
        self.query_agent = query_agent
    
    def process(self, context: Context) -> Context:
        """Process a combined context of ConstraintData and StrategyData to generate queries"""
        try:
            # In a typical implementation, we'd have a more robust way to combine contexts
            # For simplicity, we'll assume this processor receives a ConstraintData context
            # and generates a QueryExecutionData context
            
            if not isinstance(context.data, ConstraintData):
                return context.error("Expected ConstraintData as input")
            
            constraint_data = context.data
            constraint_dict = constraint_data.dict()
            
            # Initialize query execution data
            execution_data = QueryExecutionData(
                tool_calls=[],
                results={},
                execution_start=datetime.utcnow()
            )
            
            # Process each tool call
            for tool_call in constraint_data.tool_calls:
                try:
                    # Generate SQL for this tool call
                    sql = self.query_agent.generate_query(tool_call, constraint_dict)
                    
                    # Create a tool call data object
                    tool_call_data = ToolCallData(
                        name=tool_call.get("name", "Unnamed Query"),
                        description=tool_call.get("description", ""),
                        result_id=tool_call.get("result_id", f"result_{len(execution_data.tool_calls)}"),
                        tables=tool_call.get("tables", []),
                        status="generated",
                        sql=sql
                    )
                    
                    # Add to execution data
                    execution_data.tool_calls.append(tool_call_data)
                    
                except Exception as e:
                    # Create a failed tool call data object
                    tool_call_data = ToolCallData(
                        name=tool_call.get("name", "Unnamed Query"),
                        description=tool_call.get("description", ""),
                        result_id=tool_call.get("result_id", f"result_{len(execution_data.tool_calls)}"),
                        tables=tool_call.get("tables", []),
                        status="failed",
                        error=str(e)
                    )
                    
                    # Add to execution data
                    execution_data.tool_calls.append(tool_call_data)
            
            # Return updated context
            return Context.create(
                data=execution_data,
                component="QueryAgent",
                operation="generate_queries",
                parent_id=context.metadata.context_id
            ).success()
        except Exception as e:
            traceback.print_exc()
            return cast(Context[QueryExecutionData], context.error(f"Error generating queries: {str(e)}"))

class MCPQueryExecutor(ContextProcessor):
    """MCP processor for executing queries"""
    
    def __init__(self, bigquery_client: BigQueryClient):
        self.bigquery_client = bigquery_client
    
    def process(self, context: Context[QueryExecutionData]) -> Context[QueryExecutionData]:
        """Execute queries in the execution data"""
        try:
            # Extract the execution data
            execution_data = context.data
            
            # Execute each query
            for tool_call in execution_data.tool_calls:
                # Skip failed or already executed tool calls
                if tool_call.status == "failed" or tool_call.status == "completed":
                    continue
                
                try:
                    # Execute the query
                    if tool_call.sql:
                        result = self.bigquery_client.client.query(tool_call.sql).to_dataframe()
                        
                        # Store the result
                        execution_data.results[tool_call.result_id] = result.to_dict('records')
                        
                        # Update the tool call status
                        tool_call.status = "completed"
                        tool_call.result = result.to_dict('records')
                    else:
                        tool_call.status = "failed"
                        tool_call.error = "No SQL query provided"
                except Exception as e:
                    # Update the tool call status
                    tool_call.status = "failed"
                    tool_call.error = str(e)
            
            # Set the execution end time
            execution_data.execution_end = datetime.utcnow()
            
            # Return updated context
            return context.update(
                data=execution_data,
                component="QueryExecutor",
                operation="execute_queries",
                status="success"
            )
        except Exception as e:
            traceback.print_exc()
            return context.error(f"Error executing queries: {str(e)}")

class MCPResponseGenerator(ContextProcessor):
    """MCP wrapper for the ResponseAgent"""
    
    def __init__(self, response_agent: ResponseAgent):
        self.response_agent = response_agent
    
    def process(self, context: Context) -> Context:
        """Generate a response from query results"""
        try:
            # In a typical implementation, we'd have a more robust way to combine contexts
            # For simplicity, we'll assume this processor receives a QueryExecutionData context
            # with results, and the original QueryData context
            
            if not isinstance(context.data, QueryExecutionData):
                return context.error("Expected QueryExecutionData as input")
            
            execution_data = context.data
            
            # Ideally, we would have the original question from the QueryData context
            # and the constraints from the ConstraintData context
            # For this demo, we'll use placeholders
            original_question = "What is the query for these results?"  # Default fallback
            constraints = {}
            
            # Generate response
            summary = self.response_agent.generate_response(
                original_question, 
                execution_data.results, 
                constraints
            )
            
            # Create response data
            response_data = ResponseData(
                summary=summary,
                status="completed",
                generated_at=datetime.utcnow()
            )
            
            # Return updated context
            return Context.create(
                data=response_data,
                component="ResponseGenerator",
                operation="generate_response",
                parent_id=context.metadata.context_id
            ).success()
        except Exception as e:
            traceback.print_exc()
            response_data = ResponseData(
                status="failed",
                error=str(e),
                generated_at=datetime.utcnow()
            )
            
            return Context.create(
                data=response_data,
                component="ResponseGenerator",
                operation="generate_response",
                parent_id=context.metadata.context_id if context else None
            ).error(str(e))

class MCPSessionManager(ContextProcessor):
    """MCP wrapper for the SessionManager"""
    
    def __init__(self, session_manager: SessionManagerV2):
        self.session_manager = session_manager
        # Store session contexts for reference
        self._session_contexts = {}
    
    def process(self, context: Context) -> Context[SessionData]:
        """Update the session with the latest context data"""
        try:
            # The session manager accepts any type of context and updates the appropriate
            # field in the session based on the type of data
            
            data = context.data
            
            # Handle different types of context data
            if isinstance(data, QueryData):
                # Create a new session
                session_id = data.session_id
                print(f"Processing QueryData with session_id: {session_id}")
                
                # Store the session context for future reference
                self._session_contexts[session_id] = context
                
                # Initialize session data
                session_data = SessionData(
                    session_id=session_id,
                    user_id=data.user_id,
                    question=data.question,
                    created_at=data.created_at,
                    updated_at=datetime.utcnow()
                )
                
                # Store in BigQuery
                try:
                    print(f"Creating new session in BigQuery for session_id: {session_id}")
                    self.session_manager.create_session(data.user_id, data.question)
                    print(f"Successfully created session in BigQuery: {session_id}")
                except Exception as e:
                    print(f"Error creating session in BigQuery: {str(e)}")
                    traceback.print_exc()
                    # Continue anyway, don't let DB errors block the flow
                
            elif isinstance(data, ConstraintData):
                # Update constraints
                # First, get the current session
                parent_id = context.metadata.parent_id
                if not parent_id:
                    print("No parent_id found for ConstraintData, using context_id instead")
                    parent_id = context.metadata.context_id
                
                print(f"Processing ConstraintData with parent_id: {parent_id}")
                
                # Try to find the original QueryData context
                query_context = self._find_query_context(parent_id)
                if query_context:
                    session_id = query_context.data.session_id
                    user_id = query_context.data.user_id
                    question = query_context.data.question
                else:
                    # Fallback to using parent_id as session_id
                    session_id = parent_id
                    user_id = "unknown"
                    question = "unknown"
                
                # Initialize session data
                session_data = SessionData(
                    session_id=session_id,
                    user_id=user_id,
                    question=question,
                    constraints=data,
                    updated_at=datetime.utcnow()
                )
                
                # Update in BigQuery
                try:
                    print(f"Updating constraints in BigQuery for session_id: {session_id}")
                    # Convert to dict and verify it's properly serializable
                    constraints_dict = data.dict()
                    # Ensure the session is properly updated with a new record
                    self.session_manager.update_constraints(session_id, constraints_dict)
                    print(f"Successfully updated constraints in BigQuery: {session_id}")
                    
                    # Also update the status field to show progress
                    self.session_manager.update_session(session_id, {
                        "status": "processing_constraints"
                    })
                except Exception as e:
                    print(f"Error updating constraints in BigQuery: {str(e)}")
                    traceback.print_exc()
                    # Continue anyway, don't let DB errors block the flow
                
            elif isinstance(data, StrategyData):
                # Update strategy
                parent_id = context.metadata.parent_id
                if not parent_id:
                    print("No parent_id found for StrategyData, using context_id instead")
                    parent_id = context.metadata.context_id
                
                print(f"Processing StrategyData with parent_id: {parent_id}")
                
                # Try to find the original QueryData context
                query_context = self._find_query_context(parent_id)
                if query_context:
                    session_id = query_context.data.session_id
                    user_id = query_context.data.user_id
                    question = query_context.data.question
                else:
                    # Fallback to using parent_id as session_id
                    session_id = parent_id
                    user_id = "unknown"
                    question = "unknown"
                
                # Initialize session data
                session_data = SessionData(
                    session_id=session_id,
                    user_id=user_id,
                    question=question,
                    strategy=data,
                    updated_at=datetime.utcnow()
                )
                
                # Update in BigQuery
                try:
                    print(f"Updating strategy in BigQuery for session_id: {session_id}")
                    self.session_manager.update_strategy(session_id, data.raw_strategy)
                    print(f"Successfully updated strategy in BigQuery: {session_id}")
                    
                    # Also update the status field to show progress
                    self.session_manager.update_session(session_id, {
                        "status": "generating_queries"
                    })
                except Exception as e:
                    print(f"Error updating strategy in BigQuery: {str(e)}")
                    traceback.print_exc()
                    # Continue anyway, don't let DB errors block the flow
                
            elif isinstance(data, QueryExecutionData):
                # Update execution data
                parent_id = context.metadata.parent_id
                if not parent_id:
                    print("No parent_id found for QueryExecutionData, using context_id instead")
                    parent_id = context.metadata.context_id
                
                print(f"Processing QueryExecutionData with parent_id: {parent_id}")
                
                # Try to find the original QueryData context
                query_context = self._find_query_context(parent_id)
                if query_context:
                    session_id = query_context.data.session_id
                    user_id = query_context.data.user_id
                    question = query_context.data.question
                else:
                    # Fallback to using parent_id as session_id
                    session_id = parent_id
                    user_id = "unknown"
                    question = "unknown"
                
                # Initialize session data
                session_data = SessionData(
                    session_id=session_id,
                    user_id=user_id,
                    question=question,
                    execution=data,
                    updated_at=datetime.utcnow()
                )
                
                # Update in BigQuery
                try:
                    print(f"Updating execution data in BigQuery for session_id: {session_id}")
                    
                    # Update the status first to show progress
                    self.session_manager.update_session(session_id, {
                        "status": "executing_queries"
                    })
                    
                    # Update each tool call status
                    for tool_call in data.tool_calls:
                        print(f"Updating tool call status for {tool_call.name} to {tool_call.status}")
                        self.session_manager.update_tool_call_status(
                            session_id, 
                            tool_call.name, 
                            tool_call.status,
                            tool_call.result or tool_call.error
                        )
                    
                    # Update results
                    print(f"Updating query results for session_id: {session_id}")
                    self.session_manager.update_session(session_id, {
                        "results": data.results,
                        "status": "generating_response"
                    })
                    print(f"Successfully updated execution data in BigQuery: {session_id}")
                except Exception as e:
                    print(f"Error updating execution data in BigQuery: {str(e)}")
                    traceback.print_exc()
                    # Continue anyway, don't let DB errors block the flow
                
            elif isinstance(data, ResponseData):
                # Update response
                parent_id = context.metadata.parent_id
                if not parent_id:
                    print("No parent_id found for ResponseData, using context_id instead")
                    parent_id = context.metadata.context_id
                
                print(f"Processing ResponseData with parent_id: {parent_id}")
                
                # Try to find the original QueryData context
                query_context = self._find_query_context(parent_id)
                if query_context:
                    session_id = query_context.data.session_id
                    user_id = query_context.data.user_id
                    question = query_context.data.question
                else:
                    # Fallback to using parent_id as session_id
                    session_id = parent_id
                    user_id = "unknown"
                    question = "unknown"
                
                # Initialize session data
                session_data = SessionData(
                    session_id=session_id,
                    user_id=user_id,
                    question=question,
                    response=data,
                    status="completed" if data.status == "completed" else "failed",
                    updated_at=datetime.utcnow()
                )
                
                # Update in BigQuery
                try:
                    print(f"Updating response in BigQuery for session_id: {session_id}")
                    self.session_manager.update_summary(session_id, data.summary or "")
                    self.session_manager.update_session(session_id, {
                        "status": "completed" if data.status == "completed" else "failed"
                    })
                    print(f"Successfully updated response in BigQuery, session marked as {session_data.status}: {session_id}")
                except Exception as e:
                    print(f"Error updating response in BigQuery: {str(e)}")
                    traceback.print_exc()
                    # Continue anyway, don't let DB errors block the flow
                
            else:
                # Unknown data type
                return cast(Context[SessionData], context.error(f"Unknown data type: {type(data).__name__}"))
            
            # Return updated context
            return Context.create(
                data=session_data,
                component="SessionManager",
                operation="update_session",
                parent_id=context.metadata.context_id
            ).success()
        
        except Exception as e:
            traceback.print_exc()
            return cast(Context[SessionData], context.error(f"Error updating session: {str(e)}"))
    
    def _find_query_context(self, context_id: str) -> Optional[Context]:
        """Find the original QueryData context by traversing context parent IDs"""
        # First, check if this is a QueryData context itself
        for session_id, query_context in self._session_contexts.items():
            if query_context.metadata.context_id == context_id:
                return query_context
            
            # Otherwise, check if this is a valid session ID
            if session_id == context_id:
                return query_context
        
        # If not found, return None
        return None

# An example of a flow orchestrator that combines all processors
class MCPQueryFlowOrchestrator:
    """Orchestrates the full query processing flow using MCP"""
    
    def __init__(
        self,
        query_processor: MCPQueryProcessor,
        query_agent: MCPQueryAgent,
        query_executor: MCPQueryExecutor,
        response_generator: MCPResponseGenerator,
        session_manager: MCPSessionManager
    ):
        self.query_processor = query_processor
        self.query_agent = query_agent
        self.query_executor = query_executor
        self.response_generator = response_generator
        self.session_manager = session_manager
        
    def process_query(self, query_data: QueryData) -> Context[SessionData]:
        """Process a query from start to finish"""
        try:
            print(f"\n🔄 Starting query processing flow for session {query_data.session_id}")
            
            # Initialize the context flow with the query data
            query_context = Context.create(
                data=query_data,
                component="QueryFlow",
                operation="start"
            )
            
            # Update session with initial query data
            print(f"📝 Initializing session {query_data.session_id}")
            session_context = self.session_manager.process(query_context)
            if session_context.metadata.status == "error":
                print(f"❌ Error initializing session: {session_context.metadata.error}")
                return session_context
            
            try:
                # Extract constraints
                print(f"🔍 Extracting constraints for query: {query_data.question}")
                constraint_context = self.query_processor.process(query_context)
                if constraint_context.metadata.status == "error":
                    print(f"❌ Error extracting constraints: {constraint_context.metadata.error}")
                    # Update session with error
                    return self.session_manager.process(constraint_context)
                
                # Update session with constraints
                print(f"💾 Updating session with constraints")
                session_context = self.session_manager.process(constraint_context)
                if session_context.metadata.status == "error":
                    print(f"❌ Error updating session with constraints: {session_context.metadata.error}")
                    return session_context
                
                # Generate strategy
                print(f"🧠 Generating strategy")
                strategy_context = self.query_processor.generate_strategy(constraint_context)
                if strategy_context.metadata.status == "error":
                    print(f"❌ Error generating strategy: {strategy_context.metadata.error}")
                    # Update session with error
                    return self.session_manager.process(strategy_context)
                
                # Update session with strategy
                print(f"💾 Updating session with strategy")
                session_context = self.session_manager.process(strategy_context)
                if session_context.metadata.status == "error":
                    print(f"❌ Error updating session with strategy: {session_context.metadata.error}")
                    return session_context
                
                # Generate queries
                print(f"📊 Generating SQL queries")
                execution_context = self.query_agent.process(constraint_context)
                execution_context.metadata.parent_id = query_data.session_id  # Set parent ID to session ID
                if execution_context.metadata.status == "error":
                    print(f"❌ Error generating queries: {execution_context.metadata.error}")
                    # Update session with error
                    return self.session_manager.process(execution_context)
                
                # Execute queries
                print(f"🚀 Executing SQL queries")
                execution_context = self.query_executor.process(execution_context)
                execution_context.metadata.parent_id = query_data.session_id  # Set parent ID to session ID
                if execution_context.metadata.status == "error":
                    print(f"❌ Error executing queries: {execution_context.metadata.error}")
                    # Update session with error
                    return self.session_manager.process(execution_context)
                
                # Update session with execution results
                print(f"💾 Updating session with query results")
                session_context = self.session_manager.process(execution_context)
                if session_context.metadata.status == "error":
                    print(f"❌ Error updating session with results: {session_context.metadata.error}")
                    return session_context
                
                # Generate response
                print(f"💬 Generating response")
                response_context = self.response_generator.process(execution_context)
                response_context.metadata.parent_id = query_data.session_id  # Set parent ID to session ID
                if response_context.metadata.status == "error":
                    print(f"❌ Error generating response: {response_context.metadata.error}")
                    # Update session with error
                    return self.session_manager.process(response_context)
                
                # Update session with response
                print(f"💾 Updating session with response")
                session_context = self.session_manager.process(response_context)
                if session_context.metadata.status == "error":
                    print(f"❌ Error updating session with response: {session_context.metadata.error}")
                
                print(f"✅ Query processing flow completed for session {query_data.session_id}")
                return session_context
                
            except Exception as e:
                print(f"❌ Unexpected error in processing flow: {str(e)}")
                traceback.print_exc()
                
                # Create error context
                error_context = query_context.error(f"Unexpected error: {str(e)}")
                
                # Update session with error
                return self.session_manager.process(error_context)
                
        except Exception as outer_e:
            print(f"❌ Critical error in process_query: {str(outer_e)}")
            traceback.print_exc()
            
            # Create minimal error context
            error_context = Context.create(
                data=query_data,
                component="QueryFlow",
                operation="start"
            ).error(f"Critical error: {str(outer_e)}")
            
            # Try to update session with error
            try:
                return self.session_manager.process(error_context)
            except Exception:
                # If even that fails, return the error context directly
                return error_context 