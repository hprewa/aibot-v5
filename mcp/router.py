"""
Router module for MCP framework.

This module contains the router class that routes questions based on their classification
to the appropriate handlers.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import logging
import uuid
from .protocol import Context, ContextMetadata
from .models import ClassificationData, SessionData, ResponseData
import traceback

# Import required components
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple in-memory cache for testing purposes
session_cache = {}

class MCPRouter:
    """
    Router that determines how to process a question based on its classification.
    """
    
    # Classification categories that require full processing pipeline
    FULL_PIPELINE_CATEGORIES = [
        "KPI Extraction", 
        "Comparitive Analysis",
        "Trend Analysis"
    ]
    
    # Classification categories planned for future implementation
    PLANNED_CATEGORIES = [
        "Forecasting",
        "Anamoly Detection",
        "Operational efficiency",
        "Multi-Intent Questions",
        "Nested or Multi-Step",
        "Constrained Based Optimization"
    ]
    
    # Classification categories that are not supported
    UNSUPPORTED_CATEGORIES = [
        "Unsupported/Random Questions",
        "KPI Constraints & Unknown KPIs",
        "Data Availability Issues",
        "Action-Based Questions"
    ]
    
    # Classification categories that require user clarification
    CLARIFICATION_CATEGORIES = [
        "Unsupported NLP Constructs",
        "Ambiguous Questions"
    ]
    
    # Special handling categories
    SMALL_TALK_CATEGORY = "Small Talk"
    FEEDBACK_CATEGORY = "Feedback"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize required components
        self.gemini_client = GeminiClient()
        self.bigquery_client = BigQueryClient()
        self.query_processor = QueryProcessor(self.gemini_client, self.bigquery_client)
        self.query_agent = QueryAgent(self.gemini_client)
        self.response_agent = ResponseAgent(self.gemini_client)
        
    def get_cached_session(self, session_id):
        """Get a session from the in-memory cache"""
        return session_cache.get(session_id)
        
    def store_cached_session(self, session_id, session_data):
        """Store a session in the in-memory cache"""
        session_cache[session_id] = session_data
        self.logger.info(f"Stored session {session_id} in cache")
        self.logger.info(f"Cache now contains {len(session_cache)} sessions")
    
    def route(self, classification_context: Context[ClassificationData], session_manager=None):
        """
        Route the question to the appropriate handler based on classification.
        
        Args:
            classification_context: Context containing the question classification
            session_manager: Optional session manager for updating session data
            
        Returns:
            Context with appropriate response data
        """
        classification = classification_context.data
        question_type = classification.question_type.lower()
        
        self.logger.info(f"Routing question type: {question_type} with confidence {classification.confidence}")
        
        # Route to appropriate handler based on classification
        if question_type in ["kpi extraction", "comparative analysis", 
                           "trend analysis", "anomaly detection", 
                           "categorical breakdown", "ranking", "other analytics"]:
            return self._handle_full_pipeline(classification_context, session_manager)
        
        elif question_type in ["forecasting"]:
            return self._handle_planned_feature(classification_context, session_manager)
            
        elif question_type in ["unsupported/random questions"]:
            return self._handle_unsupported_feature(classification_context, session_manager)
            
        elif question_type in ["data source", "clarification needed"]:
            return self._handle_clarification_needed(classification_context, session_manager)
            
        elif question_type in ["small talk"]:
            return self._handle_small_talk(classification_context, session_manager)
            
        elif question_type in ["feedback"]:
            return self._handle_feedback(classification_context, session_manager)
            
        else:
            # Default to full pipeline for unknown types
            self.logger.warning(f"Unknown question type: {question_type}. Defaulting to full pipeline.")
            return self._handle_full_pipeline(classification_context, session_manager)
    
    def _create_response_context(self, classification_context: Context[ClassificationData], response_data: ResponseData):
        """Create a response context with appropriate metadata"""
        metadata = ContextMetadata(
            context_id=str(uuid.uuid4()),
            parent_id=classification_context.metadata.context_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            component="MCPRouter",
            operation="route_response",
            status="success",
            session_id=classification_context.metadata.session_id
        )
        
        return Context(data=response_data, metadata=metadata)
    
    def _handle_full_pipeline(self, classification_context: Context[ClassificationData], session_manager=None):
        """Process through the complete MCP pipeline."""
        classification = classification_context.data
        self.logger.info(f"Handling full pipeline for: {classification.question}")
        
        # Get session ID and user ID
        session_id = classification_context.metadata.session_id
        user_id = classification_context.metadata.user_id or getattr(classification_context.data, 'user_id', 'anonymous')
        
        try:
            # First, ensure the session exists with correct user_id and question
            if session_manager:
                current_session = session_manager.get_session(session_id)
                if not current_session:
                    # Create new session with correct user_id and question
                    session_manager.create_session(user_id, classification.question, session_id)
                    self.logger.info(f"Created new session {session_id} for user {user_id} with question: {classification.question}")
            
            # Step 1: Extract constraints
            self.logger.info("Extracting constraints...")
            constraints = self.query_processor.extract_constraints(classification.question)
            self.logger.info(f"Extracted constraints: {json.dumps(constraints, indent=2)}")
            
            # Step 2: Generate SQL queries
            self.logger.info("Generating SQL queries...")
            tool_calls = []
            for tool_call in constraints.get("tool_calls", []):
                try:
                    sql = self.query_agent.generate_query(tool_call, constraints)
                    tool_calls.append({
                        "name": tool_call["name"],
                        "sql": sql,
                        "result_id": tool_call["result_id"]
                    })
                    self.logger.info(f"Generated SQL for {tool_call['name']}: {sql[:100]}...")
                except Exception as e:
                    self.logger.error(f"Error generating SQL for tool call {tool_call['name']}: {str(e)}")
                    continue
            
            # Step 3: Execute queries and collect results
            self.logger.info("Executing queries...")
            results = {}
            tool_call_status = {}
            for tool_call in tool_calls:
                try:
                    self.logger.info(f"Executing query {tool_call['name']} with SQL: {tool_call['sql']}")
                    result = self.bigquery_client.execute_query(tool_call["sql"])
                    self.logger.info(f"Query {tool_call['name']} returned {len(result)} rows")
                    
                    # Format datetime objects in results
                    formatted_result = []
                    for row in result:
                        formatted_row = {}
                        for key, value in row.items():
                            if isinstance(value, datetime):
                                formatted_row[key] = value.isoformat()
                            else:
                                formatted_row[key] = value
                        formatted_result.append(formatted_row)
                    
                    # Store results and update status
                    results[tool_call["result_id"]] = formatted_result
                    tool_call_status[tool_call["name"]] = "completed"
                    
                    # Log the first row of results for debugging
                    if formatted_result:
                        self.logger.info(f"First row of results for {tool_call['name']}: {json.dumps(formatted_result[0], indent=2)}")
                    else:
                        self.logger.warning(f"No results returned for {tool_call['name']}")
                        
                except Exception as e:
                    self.logger.error(f"Error executing query {tool_call['name']}: {str(e)}")
                    tool_call_status[tool_call["name"]] = "failed"
                    continue
            
            # Step 4: Generate response using the response agent
            self.logger.info("Generating response...")
            self.logger.info(f"Results being passed to response agent: {json.dumps(results, indent=2)}")
            
            # Check if we have any results
            if not results:
                self.logger.error("No results were returned from any queries")
                response_text = "I apologize, but I couldn't retrieve any data for your query. Please try again or rephrase your question."
            else:
                response_text = self.response_agent.generate_response(
                    classification.question,
                    results,
                    constraints
                )
            
            if not response_text:
                self.logger.error("No response text generated")
                response_text = "I apologize, but I couldn't generate a response from the data."
            
            # Create response data
            response_data = ResponseData(
                query=classification.question,
                summary=response_text,
                results=results,
                execution_time=0.0,  # TODO: Track actual execution time
                created_at=datetime.now()
            )
            
            # Step 5: Update session with results if we have a session manager
            if session_manager:
                try:
                    # Format updates according to the BigQuery schema
                    updates = {
                        "session_id": session_id,
                        "user_id": user_id,
                        "question": classification.question,
                        "status": "completed",
                        "summary": response_text,
                        "updated_at": datetime.now().isoformat(),
                        "results": results,
                        "tool_calls": tool_calls,
                        "tool_call_status": tool_call_status,
                        "constraints": constraints
                    }
                    
                    # Update the session with all results
                    session_manager.update_session(session_id, updates)
                    self.logger.info(f"Updated session {session_id} with response")
                except Exception as e:
                    self.logger.warning(f"Error updating session: {str(e)}")
                    traceback.print_exc()
            
            # Create response context with the thread ID
            response_context = self._create_response_context(classification_context, response_data)
            response_context.metadata.session_id = session_id  # Ensure session_id is set in metadata
            return response_context
            
        except Exception as e:
            self.logger.error(f"Error in full pipeline: {str(e)}")
            traceback.print_exc()
            return self._create_error_response(classification_context, str(e))
    
    def _handle_planned_feature(self, classification_context: Context[ClassificationData], session_manager=None):
        """Handle a feature that is planned but not yet implemented."""
        classification = classification_context.data
        self.logger.info(f"Routing planned feature: {classification.question_type}")
        
        response_data = ResponseData(
            query=classification.question,
            summary=f"This type of question ({classification.question_type}) is a planned feature that will be available soon. We're working on it!"
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_unsupported_feature(self, classification_context: Context[ClassificationData], session_manager=None):
        """Handle an unsupported feature request."""
        classification = classification_context.data
        self.logger.info(f"Routing unsupported feature: {classification.question}")
        
        response_data = ResponseData(
            query=classification.question,
            summary="I'm sorry, but this type of question isn't currently supported by our system. Please try asking about KPIs, trends, comparisons, or anomalies in your data."
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_clarification_needed(self, classification_context: Context[ClassificationData], session_manager=None):
        """Handle a question that needs clarification."""
        classification = classification_context.data
        self.logger.info(f"Routing clarification needed: {classification.question}")
        
        response_data = ResponseData(
            query=classification.question,
            summary="To better assist you, could you please provide more details about what specific data or metrics you're interested in?"
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_small_talk(self, classification_context: Context[ClassificationData], session_manager=None):
        """Handle small talk with a friendly response."""
        classification = classification_context.data
        self.logger.info(f"Routing small talk: {classification.question}")
        
        response_data = ResponseData(
            query=classification.question,
            summary="I'm your analytics assistant. I'm here to help you analyze your data. How can I assist you with your business metrics today?"
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_feedback(self, classification_context: Context[ClassificationData], session_manager=None):
        """Handle feedback by thanking the user and storing the feedback."""
        classification = classification_context.data
        self.logger.info(f"Routing feedback: {classification.question}")
        
        # Create feedback data to store
        feedback_data = {
            "question": classification.question,
            "feedback_type": classification.question_type,
            "confidence": classification.confidence,
            "received_at": datetime.now().isoformat()
        }
        
        # Create response data
        response_data = ResponseData(
            query=classification.question,
            summary="Thank you for your feedback! We appreciate your input and will use it to improve our system.",
            feedback_data=feedback_data
        )
        
        # Get session ID from context metadata or generate a new Slack thread ID
        session_id = classification_context.metadata.session_id
        if not session_id:
            # Generate a Slack-style thread ID
            timestamp = int(datetime.utcnow().timestamp())
            thread_id = str(int(datetime.utcnow().timestamp() * 1000))[-6:]  # Last 6 digits of millisecond timestamp
            session_id = f"{timestamp}.{thread_id}"
            self.logger.info(f"No session ID in context metadata, generated new thread ID: {session_id}")
        
        # Create session data
        session_data = {
            "session_id": session_id,
            "user_id": "anonymous",
            "question": classification.question,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "summary": response_data.summary, 
            "feedback": [feedback_data],
            "classification": {
                "question": classification.question,
                "question_type": classification.question_type,
                "confidence": classification.confidence
            },
            "response": {
                "summary": response_data.summary,
                "feedback_data": feedback_data
            }
        }
        
        # Store in cache
        self.store_cached_session(session_id, session_data)
        
        # If we have a session manager, update the session with feedback
        if session_manager:
            try:
                # Log session manager type
                self.logger.info(f"Session manager type: {type(session_manager)}")
                
                # Get current session or create if it doesn't exist
                self.logger.info(f"Getting session with thread ID: {session_id}")
                
                try:
                    session = session_manager.get_session(session_id)
                    self.logger.info(f"Retrieved session: {session}")
                except Exception as e:
                    self.logger.error(f"Error retrieving session: {str(e)}")
                    # Session might not exist, create it
                    self.logger.info(f"Creating new session {session_id} for feedback")
                    user_id = "anonymous"  # Default user ID
                    try:
                        session_manager.create_session(user_id, classification.question, session_id)
                        self.logger.info(f"Created new session with thread ID: {session_id}")
                        session = session_manager.get_session(session_id)
                        self.logger.info(f"Retrieved newly created session: {session}")
                    except Exception as e:
                        self.logger.error(f"Error creating session: {str(e)}")
                        session = None
                
                # Handle none or dict session
                if not session:
                    self.logger.info(f"No session found, creating a SessionData object")
                    session = session_data
                elif isinstance(session, dict):
                    self.logger.info(f"Session is a dict, updating it")
                    # Add feedback to the session
                    if "feedback" not in session:
                        session["feedback"] = []
                    session["feedback"].append(feedback_data)
                    
                    # Add classification data
                    session["classification"] = {
                        "question": classification.question,
                        "question_type": classification.question_type,
                        "confidence": classification.confidence
                    }
                    
                    # Add response to session
                    session["response"] = {
                        "summary": response_data.summary,
                        "feedback_data": feedback_data
                    }
                    
                    # Update session status and summary
                    session["status"] = "completed"
                    session["summary"] = response_data.summary
                    session["updated_at"] = datetime.now().isoformat()
                    session["session_id"] = session_id  # Ensure session_id is set
                else:
                    self.logger.info(f"Session is an object of type {type(session)}, updating it")
                    # Add feedback to the session object
                    if not hasattr(session, "feedback") or session.feedback is None:
                        session.feedback = []
                    session.feedback.append(feedback_data)
                    
                    # Add classification data
                    session.classification = {
                        "question": classification.question,
                        "question_type": classification.question_type,
                        "confidence": classification.confidence
                    }
                    
                    # Add response to session
                    session.response = {
                        "summary": response_data.summary,
                        "feedback_data": feedback_data
                    }
                    
                    # Update session status and summary
                    session.status = "completed"
                    session.summary = response_data.summary
                    session.updated_at = datetime.now()
                    session.session_id = session_id  # Ensure session_id is set
                    
                    # Convert object to dict
                    if hasattr(session, "to_dict"):
                        session = session.to_dict()
                    else:
                        # Try manual conversion
                        self.logger.info("Manual conversion of session object to dict")
                        session = {
                            "session_id": session_id,
                            "user_id": getattr(session, "user_id", "anonymous"),
                            "question": getattr(session, "question", classification.question),
                            "status": getattr(session, "status", "completed"),
                            "summary": getattr(session, "summary", response_data.summary),
                            "created_at": getattr(session, "created_at", datetime.now().isoformat()),
                            "updated_at": datetime.now().isoformat(),
                            "feedback": getattr(session, "feedback", []),
                            "classification": getattr(session, "classification", {}),
                            "response": getattr(session, "response", {})
                        }
                
                # Update session
                self.logger.info(f"Updating session: {session}")
                try:
                    session_manager.update_session(session_id, session)
                    self.logger.info(f"Successfully updated session {session_id}")
                except Exception as e:
                    self.logger.error(f"Error updating session: {str(e)}")
                    traceback.print_exc()
            except Exception as e:
                self.logger.error(f"Failed to store feedback: {str(e)}")
                traceback.print_exc()
        
        # Include session_id in the response context metadata
        metadata = ContextMetadata(
            context_id=str(uuid.uuid4()),
            parent_id=classification_context.metadata.context_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            component="MCPRouter",
            operation="route_response",
            status="success",
            session_id=session_id
        )
        
        return Context(data=response_data, metadata=metadata)
    
    def _create_error_response(self, context: Context[ClassificationData], error_message: str) -> Context[ResponseData]:
        """Create an error response context"""
        self.logger.error(f"Creating error response: {error_message}")
        
        # Create error response data
        error_response = ResponseData(
            query=context.data.question,
            summary=f"I apologize, but I encountered an error while processing your request: {error_message}",
            results={},
            execution_time=0.0,
            created_at=datetime.now()
        )
        
        # Create response context
        response_context = Context(
            data=error_response,
            metadata=context.metadata
        )
        
        return response_context 