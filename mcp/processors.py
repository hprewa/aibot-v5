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
import uuid
import logging

from .protocol import Context, ContextProcessor, ContextMetadata
from .models import (
    QueryData, 
    ClassificationData,
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
from question_classifier import QuestionClassifier

T = TypeVar('T')
U = TypeVar('U')

class MCPQuestionClassifier(ContextProcessor):
    """MCP wrapper for the QuestionClassifier"""
    
    def __init__(self, question_classifier: QuestionClassifier):
        self.question_classifier = question_classifier
        
    def process(self, context: Context[QueryData]) -> Context[ClassificationData]:
        """Classify the query"""
        try:
            # Extract the query data
            query_data = context.data
            
            # Classify the question and pass the session_id
            classification_result = self.question_classifier.classify_question(
                question=query_data.question,
                session_id=query_data.session_id
            )
            
            # Create classification data model
            classification_data = ClassificationData(
                question=query_data.question,
                question_type=classification_result.get("question_type", "Unknown"),
                confidence=classification_result.get("confidence", 0.0),
                requires_sql=classification_result.get("requires_sql", True),
                requires_summary=classification_result.get("requires_summary", True),
                classification_metadata=classification_result.get("classification_metadata", {}),
                created_at=datetime.now()
            )
            
            # Create new metadata for the classification context, including session_id
            metadata = ContextMetadata(
                component="QuestionClassifier",
                operation="classify_question",
                parent_id=context.metadata.context_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="success",
                session_id=query_data.session_id
            )
            
            # Return context with classification data and metadata
            return Context(data=classification_data, metadata=metadata)
        except Exception as e:
            traceback.print_exc()
            return cast(Context[ClassificationData], context.error(f"Error classifying question: {str(e)}"))

class MCPQueryProcessor(ContextProcessor):
    """MCP wrapper for the QueryProcessor"""
    
    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
    
    def process(self, context: Context) -> Context[ConstraintData]:
        """Extract constraints from the query"""
        try:
            # Check if this is a ClassificationData context
            if isinstance(context.data, ClassificationData):
                # Get the parent context to retrieve the original query
                parent_id = context.metadata.parent_id
                if not parent_id:
                    raise ValueError("No parent context ID found")
                
                # In a real implementation, we would fetch the parent context from a registry
                # For now, just create a dummy ConstraintData
                return Context.create(
                    data=ConstraintData(),
                    component="QueryProcessor",
                    operation="extract_constraints",
                    parent_id=context.metadata.context_id
                ).success()
            
            # If this is a QueryData context (backward compatibility)
            elif isinstance(context.data, QueryData):
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
            else:
                return cast(Context[ConstraintData], context.error(f"Unexpected context data type: {type(context.data)}"))
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
    """MCP wrapper for the QueryAgent (placeholder)"""
    
    def __init__(self, query_agent=None):
        self.query_agent = query_agent
    
    def process(self, context: Context) -> Context:
        """Placeholder process method"""
        return context

class MCPQueryExecutor(ContextProcessor):
    """MCP processor for executing queries"""
    
    def __init__(self, bigquery_client: BigQueryClient):
        self.bigquery_client = bigquery_client
        self.logger = logging.getLogger(__name__)
    
    def process(self, context: Context[QueryExecutionData]) -> Context[QueryExecutionData]:
        """Execute queries in the execution data"""
        try:
            # Extract the execution data
            execution_data = context.data
            
            # Set execution start time
            execution_data.execution_start = datetime.now()
            
            # Execute each query
            for tool_call in execution_data.tool_calls:
                # Skip failed or already executed tool calls
                if tool_call.status == "failed" or tool_call.status == "completed":
                    continue
                
                try:
                    # Execute the query
                    if tool_call.sql:
                        self.logger.info(f"\nExecuting query for tool call {tool_call.name}:")
                        self.logger.info(f"SQL: {tool_call.sql}")
                        
                        # Execute query and get results
                        results = self.bigquery_client.query(tool_call.sql)
                        
                        # Format results for response agent
                        formatted_results = []
                        for row in results:
                            # Convert any datetime objects to ISO format strings
                            formatted_row = {}
                            for key, value in row.items():
                                if isinstance(value, (datetime.date, datetime.datetime)):
                                    formatted_row[key] = value.isoformat()
                                else:
                                    formatted_row[key] = value
                            formatted_results.append(formatted_row)
                        
                        # Store the formatted result
                        execution_data.results[tool_call.result_id] = formatted_results
                        
                        self.logger.info(f"Query returned {len(formatted_results)} rows")
                        if formatted_results:
                            self.logger.info(f"First row: {json.dumps(formatted_results[0], indent=2)}")
                        
                        # Update the tool call status
                        tool_call.status = "completed"
                        tool_call.result = formatted_results
                    else:
                        self.logger.error(f"No SQL query provided for tool call {tool_call.name}")
                        tool_call.status = "failed"
                        tool_call.error = "No SQL query provided"
                        
                except Exception as e:
                    self.logger.error(f"Error executing query for tool call {tool_call.name}: {str(e)}")
                    tool_call.status = "failed"
                    tool_call.error = str(e)
                    continue
            
            # Set execution end time
            execution_data.execution_end = datetime.now()
            
            # Log final results
            self.logger.info("\nFinal execution results:")
            self.logger.info(f"Number of tool calls: {len(execution_data.tool_calls)}")
            self.logger.info(f"Results keys: {list(execution_data.results.keys())}")
            for result_id, result in execution_data.results.items():
                self.logger.info(f"Result {result_id}: {len(result)} rows")
                if result:
                    self.logger.info(f"Sample row: {json.dumps(result[0], indent=2)}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error in MCPQueryExecutor: {str(e)}")
            traceback.print_exc()
            raise

class MCPResponseGenerator(ContextProcessor):
    """MCP processor for generating responses"""
    
    def __init__(self, response_agent: ResponseAgent):
        self.response_agent = response_agent
        self.logger = logging.getLogger(__name__)
    
    def process(self, context: Context[QueryExecutionData]) -> Context[ResponseData]:
        """Generate a response from the query results"""
        try:
            # Extract the execution data
            execution_data = context.data
            
            # Get the original question from the parent context
            parent_id = context.metadata.parent_id
            if not parent_id:
                raise ValueError("No parent context ID found")
            
            # Get the original question from the parent context
            # For now, we'll use a default if we can't get the original
            question = "What are the results?"  # Default fallback
            
            # Get the constraints from the parent context
            constraints = {}
            if hasattr(execution_data, 'constraints'):
                constraints = execution_data.constraints
            
            # Generate the response
            self.logger.info("\nGenerating response...")
            self.logger.info(f"Question: {question}")
            self.logger.info(f"Results: {json.dumps(execution_data.results, indent=2)}")
            self.logger.info(f"Constraints: {json.dumps(constraints, indent=2)}")
            
            response_text = self.response_agent.generate_response(
                question=question,
                results=execution_data.results,
                constraints=constraints
            )
            
            if not response_text:
                raise ValueError("No response generated")
            
            # Calculate execution time
            execution_time = 0.0
            if execution_data.execution_start and execution_data.execution_end:
                execution_time = (execution_data.execution_end - execution_data.execution_start).total_seconds()
            
            # Create response data
            response_data = ResponseData(
                query=question,
                summary=response_text,
                results=execution_data.results,
                execution_time=execution_time,
                created_at=datetime.now()
            )
            
            # Return updated context
            return Context.create(
                data=response_data,
                component="ResponseGenerator",
                operation="generate_response",
                parent_id=context.metadata.context_id
            ).success()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            traceback.print_exc()
            return cast(Context[ResponseData], context.error(f"Error generating response: {str(e)}"))

class MCPSessionManager(ContextProcessor):
    """MCP wrapper for the SessionManager"""
    
    def __init__(self, session_manager: SessionManagerV2):
        self.session_manager = session_manager
        
    def process(
        self, 
        query_context: Optional[Context] = None,
        classification_context: Optional[Context] = None,
        constraint_context: Optional[Context] = None,
        strategy_context: Optional[Context] = None,
        execution_context: Optional[Context] = None,
        response_context: Optional[Context] = None
    ) -> Context[SessionData]:
        """Update the session with data from various contexts"""
        try:
            # Find the query context or use the most recent context
            if query_context is None:
                if classification_context is not None:
                    query_context = self._find_query_context(classification_context.metadata.parent_id)
                elif constraint_context is not None:
                    query_context = self._find_query_context(constraint_context.metadata.parent_id)
                elif strategy_context is not None:
                    query_context = self._find_query_context(strategy_context.metadata.parent_id)
                elif execution_context is not None:
                    query_context = self._find_query_context(execution_context.metadata.parent_id)
                elif response_context is not None:
                    query_context = self._find_query_context(response_context.metadata.parent_id)
                    
            if query_context is None:
                raise ValueError("No query context available")
                
            # Extract the query data
            query_data = None
            if isinstance(query_context.data, QueryData):
                query_data = query_context.data
            else:
                raise ValueError(f"Expected QueryData, got {type(query_context.data)}")
                
            # Get the session ID
            session_id = query_data.session_id
            
            # Create or update session data
            session_data = self._get_or_create_session_data(session_id, query_data)
            
            # Update with classification data
            if classification_context is not None and isinstance(classification_context.data, ClassificationData):
                session_data.classification = classification_context.data
                
            # Update with constraint data
            if constraint_context is not None and isinstance(constraint_context.data, ConstraintData):
                session_data.constraints = constraint_context.data
                
            # Update with strategy data
            if strategy_context is not None and isinstance(strategy_context.data, StrategyData):
                session_data.strategy = strategy_context.data
                
            # Update with execution data
            if execution_context is not None and isinstance(execution_context.data, QueryExecutionData):
                session_data.execution = execution_context.data
                
            # Update with response data
            if response_context is not None and isinstance(response_context.data, ResponseData):
                session_data.response = response_context.data
                if session_data.response.status == "completed":
                    session_data.status = "completed"
                elif session_data.response.status == "error":
                    session_data.status = "error"
                    session_data.error = session_data.response.error
                
            # Check for errors in context
            context_with_error = None
            error_message = None
            
            if classification_context is not None and classification_context.metadata.has_error:
                context_with_error = classification_context
                error_message = classification_context.metadata.error_message
            elif constraint_context is not None and constraint_context.metadata.has_error:
                context_with_error = constraint_context
                error_message = constraint_context.metadata.error_message
            elif strategy_context is not None and strategy_context.metadata.has_error:
                context_with_error = strategy_context
                error_message = strategy_context.metadata.error_message
            elif execution_context is not None and execution_context.metadata.has_error:
                context_with_error = execution_context
                error_message = execution_context.metadata.error_message
            elif response_context is not None and response_context.metadata.has_error:
                context_with_error = response_context
                error_message = response_context.metadata.error_message
                
            if context_with_error is not None:
                session_data.status = "error"
                session_data.error = error_message
                
            # Update timestamp
            session_data.updated_at = datetime.now()
            
            # Update session in the session manager
            self.session_manager.update_session(
                session_id=session_id,
                updates=session_data.to_bq_format()
            )
            
            # Create session context
            session_context = Context.create(
                data=session_data,
                component="SessionManager",
                operation="update_session",
                parent_id=query_context.metadata.context_id
            )
            
            if context_with_error is not None:
                return cast(Context[SessionData], session_context.error(error_message))
            else:
                return session_context.success()
                
        except Exception as e:
            traceback.print_exc()
            
            # Try to extract query data for minimal session update
            query_data = None
            if query_context is not None and isinstance(query_context.data, QueryData):
                query_data = query_context.data
                
                # Try to update session with error
                try:
                    session_id = query_data.session_id
                    self.session_manager.update_session(
                        session_id=session_id,
                        updates={
                            "status": "error",
                            "error": f"Error updating session: {str(e)}"
                        }
                    )
                except Exception:
                    # If even that fails, just log the error
                    traceback.print_exc()
            
            # Return error context
            error_context = Context.create(
                data=SessionData(
                    session_id=query_data.session_id if query_data else "unknown",
                    user_id=query_data.user_id if query_data else "unknown",
                    question=query_data.question if query_data else "unknown",
                    status="error",
                    error=f"Error updating session: {str(e)}"
                ),
                component="SessionManager",
                operation="update_session"
            )
            
            return cast(Context[SessionData], error_context.error(f"Error updating session: {str(e)}"))
    
    def _get_or_create_session_data(self, session_id: str, query_data: QueryData) -> SessionData:
        """Get or create a session data object for a query"""
        try:
            # Try to get the session from BigQuery
            session_dict = self.session_manager.get_session(session_id)
            
            if session_dict:
                try:
                    # If classification exists in session
                    classification_dict = None
                    if "classification" in session_dict and session_dict["classification"]:
                        # Try to parse classification as JSON
                        try:
                            classification_dict = json.loads(session_dict["classification"])
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, just use the raw data
                            classification_dict = session_dict["classification"]
                    
                    # Create classification data if it exists
                    classification = None
                    if classification_dict:
                        try:
                            classification = ClassificationData(
                                question=query_data.question,
                                question_type=classification_dict.get("question_type", "Unknown"),
                                confidence=classification_dict.get("confidence", 0.0),
                                requires_sql=classification_dict.get("requires_sql", True),
                                requires_summary=classification_dict.get("requires_summary", True),
                                classification_metadata=classification_dict.get("classification_metadata", {}),
                                created_at=classification_dict.get("created_at", datetime.now())
                            )
                        except (json.JSONDecodeError, TypeError):
                            # If creating classification data fails, just continue without it
                            pass
                    
                    # Create session data from the session dict
                    return SessionData(
                        session_id=session_id,
                        user_id=session_dict.get("user_id", query_data.user_id),
                        question=session_dict.get("question", query_data.question),
                        classification=classification,
                        constraints=session_dict.get("constraints"),
                        response_plan=session_dict.get("response_plan"),
                        strategy=session_dict.get("strategy"),
                        execution=session_dict.get("execution"),
                        response=session_dict.get("response"),
                        results=session_dict.get("results"),
                        slack_channel=session_dict.get("slack_channel"),
                        status=session_dict.get("status", "pending"),
                        created_at=session_dict.get("created_at", query_data.created_at),
                        updated_at=datetime.now(),
                        error=session_dict.get("error")
                    )
                except Exception as inner_e:
                    print(f"Error creating SessionData from session dict: {str(inner_e)}")
                    traceback.print_exc()
                    # Fall back to creating a new session data
            
            # If session doesn't exist or couldn't be parsed, create a new one
            # The session_manager.create_session method already generates a session_id
            created_session_id = self.session_manager.create_session(
                query_data.user_id, 
                query_data.question
            )
            # If received session_id different from input, update it
            if created_session_id != session_id:
                print(f"Note: Generated session ID {created_session_id} different from provided {session_id}")
                session_id = created_session_id
            
            # Create a new session data object
            return SessionData(
                session_id=session_id,
                user_id=query_data.user_id,
                question=query_data.question,
                status="pending",
                created_at=query_data.created_at,
                updated_at=datetime.now()
            )
        except Exception as e:
            print(f"Error in _get_or_create_session_data: {str(e)}")
            traceback.print_exc()
            
            # Return a minimal session data object
            return SessionData(
                session_id=session_id,
                user_id=query_data.user_id,
                question=query_data.question,
                status="pending",
                created_at=query_data.created_at,
                updated_at=datetime.now(),
                error=f"Error getting/creating session: {str(e)}"
            )

# An example of a flow orchestrator that combines all processors
class MCPQueryFlowOrchestrator:
    """Orchestrates the flow of contexts through the MCP pipeline"""
    
    def __init__(
        self,
        question_classifier: MCPQuestionClassifier,
        query_processor: MCPQueryProcessor,
        query_agent: MCPQueryAgent,
        query_executor: MCPQueryExecutor,
        response_generator: MCPResponseGenerator,
        session_manager: MCPSessionManager
    ):
        self.question_classifier = question_classifier
        self.query_processor = query_processor
        self.query_agent = query_agent
        self.query_executor = query_executor
        self.response_generator = response_generator
        self.session_manager = session_manager
        
    def process_query(self, query_data: QueryData) -> Context[ResponseData]:
        """
        Process a query through the full MCP flow.
        
        This method will:
        1. Create a context from the query data
        2. Classify the question
        3. Extract constraints
        4. Generate a strategy
        5. Execute the queries
        6. Generate a response
        7. Update the session
        
        Args:
            query_data: The query data to process
            
        Returns:
            Context with the response data
        """
        # Create error context metadata for reuse
        error_metadata = ContextMetadata(
            component="MCPQueryFlowOrchestrator",
            operation="process_query",
            session_id=query_data.session_id,
            status="error"
        )
        
        try:
            print(f"🔄 Starting processing for question: {query_data.question}")
            session_id = query_data.session_id
            
            # Step 1: Create a context from the query data
            query_context = Context(
                data=query_data,
                metadata=ContextMetadata(
                    component="MCPQueryFlowOrchestrator",
                    operation="process_query",
                    session_id=session_id,
                    status="pending"
                )
            )
            
            # Step 2: Classify the question
            try:
                classification_context = self.question_classifier.process(query_context)
                # Add user_id to metadata to ensure it's available to the router
                classification_context.metadata.user_id = query_data.user_id
                
                # Check for errors
                if classification_context.metadata.status == "error":
                    error_msg = f"Error classifying question: {classification_context.metadata.error_message}"
                    print(f"❌ {error_msg}")
                    return self._create_error_response(query_context, error_msg)
            except Exception as e:
                error_msg = f"Error classifying question: {str(e)}"
                print(f"❌ {error_msg}")
                traceback.print_exc()
                return self._create_error_response(query_context, error_msg)
                
            # Get classification data
            classification_data = classification_context.data
            print(f"📊 Question classified as {classification_data.question_type} with confidence {classification_data.confidence}")
            
            # Step 3: Extract constraints if needed (for some query types)
            constraint_context = None
            strategy_context = None
            execution_context = None
            response_context = None
            
            # Instead of completing the full pipeline, use our router
            # to determine the next steps based on classification
            if hasattr(self, 'router') and self.router:
                try:
                    # Make sure we pass the underlying session_manager object, not the wrapper
                    response_context = self.router.route(
                        classification_context, 
                        self.session_manager.session_manager if hasattr(self.session_manager, 'session_manager') else None
                    )
                except Exception as e:
                    error_msg = f"Error routing question: {str(e)}"
                    print(f"❌ {error_msg}")
                    traceback.print_exc()
                    return self._create_error_response(query_context, error_msg)
            else:
                # Basic fallback when router not available
                response_data = ResponseData(
                    query=query_data.question,
                    summary=f"Your question was classified as {classification_data.question_type}. We're working on a more detailed response."
                )
                
                response_context = Context(
                    data=response_data,
                    metadata=ContextMetadata(
                        component="MCPQueryFlowOrchestrator",
                        operation="generate_response",
                        parent_id=classification_context.metadata.context_id,
                        session_id=session_id,
                        status="success"
                    )
                )
            
            # Step 7: Update session with what we have
            try:
                self._update_session(
                    query_context=query_context,
                    classification_context=classification_context,
                    constraint_context=constraint_context,
                    strategy_context=strategy_context,
                    execution_context=execution_context,
                    response_context=response_context
                )
            except Exception as e:
                print(f"⚠️ Warning: Error updating session: {str(e)}")
                # Continue processing anyway - already have a response
            
            return response_context
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error in query flow: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Create a basic error response
            error_response = ResponseData(
                query=query_data.question if hasattr(query_data, 'question') else "Unknown query",
                summary=f"Sorry, an error occurred while processing your request: {error_msg}"
            )
            
            return Context(
                data=error_response,
                metadata=error_metadata.copy().update(error_message=error_msg)
            )
    
    def _create_error_response(self, query_context: Context, error_message: str) -> Context[ResponseData]:
        """Create an error response context"""
        response_data = ResponseData(
            query=query_context.data.question,
            summary=f"I'm sorry, but I encountered an error while processing your question: {error_message}"
        )
        
        return Context(
            data=response_data,
            metadata=ContextMetadata(
                component="MCPQueryFlowOrchestrator",
                operation="error_response",
                parent_id=query_context.metadata.context_id,
                session_id=query_context.metadata.session_id,
                status="error",
                error_message=error_message
            )
        )
        
    def _update_session(self, query_context, classification_context=None, constraint_context=None,
                        strategy_context=None, execution_context=None, response_context=None):
        """Update the session with available context data"""
        try:
            # Get session ID
            session_id = query_context.metadata.session_id
            if not session_id:
                print("⚠️ No session ID available for update")
                return
                
            # Try to get the session first
            try:
                session = self.session_manager.session_manager.get_session(session_id)
            except Exception as e:
                print(f"⚠️ Session not found, creating new one: {str(e)}")
                # Create new session
                self.session_manager.session_manager.create_session(
                    query_context.data.user_id if hasattr(query_context.data, 'user_id') else "anonymous",
                    query_context.data.question,
                    session_id
                )
                session = self.session_manager.session_manager.get_session(session_id)
                
            if not session:
                print(f"⚠️ Could not create or retrieve session {session_id}")
                return
                
            # Update session with available data
            updates = {
                "status": "processing"
            }
            
            # Add classification data if available
            if classification_context:
                updates["question_type"] = classification_context.data.question_type
                updates["classification"] = {
                    "question": classification_context.data.question,
                    "question_type": classification_context.data.question_type,
                    "confidence": classification_context.data.confidence,
                    "requires_sql": classification_context.data.requires_sql,
                    "requires_summary": classification_context.data.requires_summary
                }
            
            # Add response data if available
            if response_context:
                updates["status"] = "completed"
                updates["summary"] = response_context.data.summary
                
            # Update session
            self.session_manager.session_manager.update_session(session_id, updates)
            print(f"✅ Updated session {session_id}")
            
        except Exception as e:
            print(f"⚠️ Error updating session: {str(e)}")
            traceback.print_exc() 