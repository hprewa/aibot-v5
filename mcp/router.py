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
        
        elif question_type in ["forecasting", "delivery status", "order management"]:
            return self._handle_planned_feature(classification_context, session_manager)
            
        elif question_type in ["unsupported"]:
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
        
        # Get session ID
        session_id = classification_context.metadata.session_id
        
        # For now, we create a more informative response instead of running the full pipeline
        # Later, this method would call the full query processor, executor, and response generator
        
        # Create a response based on the question type
        if classification.question_type.lower() == "kpi extraction":
            response_text = (
                f"I'm currently processing your request for KPI data about '{classification.question}'. "
                f"This would normally run a database query for the requested KPIs and return the results. "
                f"The analytics engine is still being integrated. Your session ID is {session_id}."
            )
        elif classification.question_type.lower() == "comparative analysis":
            response_text = (
                f"I'll analyze the comparative data for '{classification.question}'. "
                f"This would perform a comparison between the requested metrics or time periods. "
                f"The analytics engine is still being integrated. Your session ID is {session_id}."
            )
        elif classification.question_type.lower() == "trend analysis":
            response_text = (
                f"I'm analyzing trends for '{classification.question}'. "
                f"This would extract time-series data and identify patterns or trends. "
                f"The analytics engine is still being integrated. Your session ID is {session_id}."
            )
        else:
            response_text = (
                f"I'll process your analytics request for '{classification.question}'. "
                f"The request has been classified as '{classification.question_type}' with confidence {classification.confidence}. "
                f"The analytics engine is still being integrated. Your session ID is {session_id}."
            )
        
        # Create response data
        response_data = ResponseData(
            query=classification.question,
            summary=response_text,
            question_type=classification.question_type,
            confidence=classification.confidence,
            execution_time=0.0
        )
        
        # If we have a session manager, first try to create the session
        if session_manager:
            try:
                # First check if the session exists
                existing_session = None
                try:
                    existing_session = session_manager.get_session(session_id)
                except Exception as e:
                    self.logger.info(f"Session {session_id} not found, will create: {str(e)}")
                
                # If session doesn't exist, create it first
                if not existing_session:
                    try:
                        # Get user_id from metadata or fallback to query data
                        user_id = classification_context.metadata.user_id
                        if not user_id:
                            user_id = getattr(classification_context.data, 'user_id', 'anonymous')
                            
                        self.logger.info(f"Creating new session with ID {session_id} for user {user_id}")
                        # Note: session_manager.create_session will return a new session_id if successful
                        new_session_id = session_manager.create_session(
                            user_id, 
                            classification.question
                        )
                        self.logger.info(f"Created new session with ID: {new_session_id}")
                    except Exception as e:
                        self.logger.warning(f"Error creating session: {str(e)}")
                
                # Now try to update the session with the response
                try:
                    # Update session with response
                    updates = {
                        "status": "completed",
                        "summary": response_text,
                        "question_type": classification.question_type,
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    session_manager.update_session(session_id, updates)
                    self.logger.info(f"Updated session {session_id} with response")
                except Exception as e:
                    self.logger.warning(f"Error updating session: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error in session management: {str(e)}")
        
        return self._create_response_context(classification_context, response_data)
    
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
        
        # Get session ID from context metadata
        session_id = classification_context.metadata.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            self.logger.info(f"No session ID in context metadata, generated new ID: {session_id}")
        
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
                self.logger.info(f"Getting session with ID: {session_id}")
                
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
                        self.logger.info(f"Created new session with ID: {session_id}")
                        session = session_manager.get_session(session_id)
                        self.logger.info(f"Retrieved newly created session: {session}")
                    except Exception as e:
                        self.logger.error(f"Error creating session: {str(e)}")
                        session = None
                
                # Handle none or dict session
                if not session:
                    self.logger.info(f"No session found, creating a SessionData object")
                    session = {
                        "session_id": session_id,
                        "user_id": "anonymous",
                        "question": classification.question,
                        "status": "completed",
                        "summary": response_data.summary,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
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
                    
                    # Convert object to dict
                    if hasattr(session, "to_dict"):
                        session = session.to_dict()
                    else:
                        # Try manual conversion
                        self.logger.info("Manual conversion of session object to dict")
                        session = {
                            "session_id": getattr(session, "session_id", session_id),
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