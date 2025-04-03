"""
MCP-powered server for the Analytics Bot.

This server uses the Model Context Protocol (MCP) to process analytical queries.
It provides the same API as the original server but uses MCP internally for better
traceability, debugging, and error handling.

NOTE: For graph functionality to work properly, ensure your BigQuery sessions table
has a 'graph_path' column of type STRING. If it doesn't exist, you can add it with:

ALTER TABLE your_dataset.sessions 
ADD COLUMN graph_path STRING;

Also consider adding a 'has_graph' column of type BOOLEAN:

ALTER TABLE your_dataset.sessions 
ADD COLUMN has_graph BOOLEAN;
"""

import os
import json
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, Callable
from dotenv import load_dotenv
import uuid
from datetime import datetime
import time
import traceback
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import original components
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent
from session_manager_v2 import SessionManagerV2
from question_classifier import QuestionClassifier

# Import MCP components
from mcp import (
    QueryData,
    Context,
    MCPQuestionClassifier,
    MCPQueryProcessor,
    MCPQueryExecutor,
    MCPResponseGenerator,
    MCPSessionManager,
    MCPQueryFlowOrchestrator
)
from mcp.processors import (
    MCPQuestionClassifier, 
    MCPQueryProcessor,
    MCPQueryExecutor,
    MCPResponseGenerator,
    MCPSessionManager
)
from mcp.router import MCPRouter
from mcp.protocol import ContextMetadata

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="MCP Analytics Bot API", description="API for the MCP-powered Analytics Bot")

# Pydantic models for API requests (same as original server)
class QueryRequest(BaseModel):
    user_id: str
    question: str

class QueryResponse(BaseModel):
    session_id: str
    summary: Optional[str] = None
    status: str = "processing"

# Additional model for the process endpoint
class ProcessRequest(BaseModel):
    question: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None

# Initialize components
bigquery_client = BigQueryClient()
gemini_client = GeminiClient()
query_processor = QueryProcessor(gemini_client, bigquery_client)
query_agent = QueryAgent(gemini_client)
response_agent = ResponseAgent(gemini_client)
session_manager = SessionManagerV2()
question_classifier = QuestionClassifier()

# Initialize MCP wrappers
mcp_question_classifier = MCPQuestionClassifier(question_classifier)
mcp_query_processor = MCPQueryProcessor(query_processor)
mcp_query_executor = MCPQueryExecutor(bigquery_client)
mcp_response_generator = MCPResponseGenerator(response_agent)
mcp_session_manager = MCPSessionManager(session_manager)

# Create a router
mcp_router = MCPRouter()

# Create the flow orchestrator
orchestrator = MCPQueryFlowOrchestrator(
    mcp_question_classifier,
    mcp_query_processor,
    None,  # Set to None since MCPQueryAgent is not defined
    mcp_query_executor,
    mcp_response_generator,
    mcp_session_manager
)

# Set the router property on the orchestrator
orchestrator.router = mcp_router

# Initialize components if not already initialized
mcp_components = None

def initialize_mcp():
    """Initialize MCP components"""
    try:
        logger.info("Initializing MCP components")
        bigquery_client = BigQueryClient()
        logger.info("BigQueryClient initialized")
        
        gemini_client = GeminiClient()
        logger.info("GeminiClient initialized")
        
        query_processor = QueryProcessor(gemini_client, bigquery_client)
        logger.info("QueryProcessor initialized")
        
        response_agent = ResponseAgent(gemini_client)
        logger.info("ResponseAgent initialized")
        
        session_manager = SessionManagerV2()
        logger.info("SessionManagerV2 initialized")
        
        # Initialize question classifier
        question_classifier = QuestionClassifier()
        logger.info("QuestionClassifier initialized")
        
        # Create MCP Components
        mcp_classifier = MCPQuestionClassifier(question_classifier)
        logger.info("MCPQuestionClassifier initialized")
        
        mcp_query_processor = MCPQueryProcessor(query_processor)
        logger.info("MCPQueryProcessor initialized")
        
        mcp_executor = MCPQueryExecutor(bigquery_client)
        logger.info("MCPQueryExecutor initialized")
        
        mcp_response_generator = MCPResponseGenerator(response_agent)
        logger.info("MCPResponseGenerator initialized")
        
        mcp_session_manager = MCPSessionManager(session_manager)
        logger.info("MCPSessionManager initialized")
        
        # Create MCP Router
        mcp_router = MCPRouter()
        logger.info("MCPRouter initialized")
        
        components = {
            "mcp_classifier": mcp_classifier,
            "mcp_query_processor": mcp_query_processor,
            "mcp_executor": mcp_executor,
            "mcp_response_generator": mcp_response_generator,
            "mcp_session_manager": mcp_session_manager,
            "mcp_router": mcp_router
        }
        logger.info(f"MCP components initialized: {list(components.keys())}")
        return components
    except Exception as e:
        logger.error(f"Error initializing MCP components: {str(e)}")
        traceback.print_exc()
        raise

# Background task to process queries using the MCP flow
async def process_query_task(user_id: str, question: str, session_id: str, send_callback: Optional[Callable] = None):
    """Process a query using the MCP flow"""
    try:
        # Create query data
        query_data = QueryData(
            user_id=user_id,
            question=question,
            session_id=session_id,
            created_at=datetime.now()
        )
        
        # Process the query through the MCP flow
        orchestrator.process_query(query_data, send_callback=send_callback)
        
    except Exception as e:
        # In case of errors, we still want to update the session status
        session_manager.update_session(session_id, {
            "status": "failed",
            "error": str(e)
        })

@app.post("/api/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a query for processing"""
    try:
        # Create a new session
        session_id = session_manager.create_session(request.user_id, request.question)
        
        # Define a callback function for graph rendering
        def graph_callback(cb_session_id: str, graph_filepath: str):
            """Callback function to handle graph generation results"""
            logger.info(f"Graph generated for session {cb_session_id}: {graph_filepath}")
            # Note: In API mode we can only store the graph path; client needs to fetch it separately
            # or we'd need to implement a /api/graph/{session_id} endpoint to serve the graph
            try:
                session_manager.update_session(cb_session_id, {
                    "graph_path": graph_filepath,
                    "has_graph": True
                })
                logger.info(f"Updated session {cb_session_id} with graph_path: {graph_filepath}")
            except Exception as e:
                logger.error(f"Failed to update session with graph_path: {str(e)}")
                logger.error(f"This may be due to missing 'graph_path' column in the sessions table.")
                logger.error(f"Consider adding this column or modifying the callback to use an existing column.")
        
        # Process the query in the background
        background_tasks.add_task(
            process_query_task,
            request.user_id,
            request.question,
            session_id,
            send_callback=graph_callback
        )
        
        return QueryResponse(
            session_id=session_id,
            status="processing"
        )
    except Exception as e:
        # Handle any errors during session creation
        return JSONResponse(
            status_code=500,
            content={"error": f"Error submitting query: {str(e)}"}
        )

@app.get("/api/query/{session_id}", response_model=QueryResponse)
async def get_query_status(session_id: str):
    """Get the status of a query"""
    try:
        # First try to get the session from the router's cache
        if 'mcp_components' in globals() and mcp_components is not None:
            mcp_router = mcp_components.get('mcp_router')
            if mcp_router:
                cached_session = mcp_router.get_cached_session(session_id)
                if cached_session:
                    logger.info(f"Retrieved session {session_id} from cache")
                    return QueryResponse(
                        session_id=session_id,
                        summary=cached_session.get("summary"),
                        status=cached_session.get("status", "processing")
                    )
        
        # If not in cache, try to get from BigQuery
        # Get the session from BigQuery
        session = session_manager.get_session(session_id)
        if not session:
            return JSONResponse(
                status_code=404,
                content={"error": f"Session {session_id} not found"}
            )
        
        # Return the response
        return QueryResponse(
            session_id=session_id,
            summary=session.get("summary"),
            status=session.get("status", "processing")
        )
    except Exception as e:
        # Handle any errors during status retrieval
        logger.error(f"Error getting query status: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Error getting query status: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "version": "1.0.0-mcp"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MCP Analytics Bot API",
        "description": "API for the MCP-powered Analytics Bot",
        "version": "1.0.0",
        "endpoints": [
            "/api/query",
            "/api/query/{session_id}",
            "/health"
        ]
    }

@app.post("/process")
async def process(request: ProcessRequest):
    """Process a question and return a response."""
    session_id = None
    try:
        # Validate request data
        if not request.question:
            return {"error": "Missing required field: question"}
        
        # Initialize components if needed
        global mcp_components
        if mcp_components is None:
            logger.info("Initializing MCP components")
            mcp_components = initialize_mcp()
            
        # Extract components - use correct component names from initialize_mcp
        mcp_question_classifier = mcp_components.get('mcp_classifier')
        if mcp_question_classifier is None:
            logger.error("mcp_classifier component is missing")
            return {"error": "Server configuration error: mcp_classifier not found"}
            
        mcp_router = mcp_components.get('mcp_router')
        if mcp_router is None:
            logger.error("mcp_router component is missing")
            return {"error": "Server configuration error: mcp_router not found"}
            
        mcp_session_manager = mcp_components.get('mcp_session_manager')
        if mcp_session_manager is None:
            logger.error("mcp_session_manager component is missing")
            return {"error": "Server configuration error: mcp_session_manager not found"}
        
        logger.info(f"Processing question: {request.question}")
        logger.info(f"Using components: {list(mcp_components.keys())}")
        
        # Create or use session_id
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        user_id = request.user_id if request.user_id else "anonymous"
        
        logger.info(f"Session ID: {session_id}")
        logger.info(f"User ID: {user_id}")
        
        # Define a callback function for graph rendering
        def graph_callback(cb_session_id: str, graph_filepath: str):
            """Callback function to handle graph generation results"""
            logger.info(f"Graph generated for session {cb_session_id}: {graph_filepath}")
            # Store the graph path in the session for API client to retrieve
            if hasattr(mcp_session_manager, 'session_manager') and mcp_session_manager.session_manager:
                try:
                    mcp_session_manager.session_manager.update_session(cb_session_id, {
                        "graph_path": graph_filepath,
                        "has_graph": True
                    })
                    logger.info(f"Updated session {cb_session_id} with graph_path: {graph_filepath}")
                except Exception as e:
                    logger.error(f"Failed to update session with graph_path: {str(e)}")
                    logger.error(f"This may be due to missing 'graph_path' column in the sessions table.")
                    logger.error(f"Consider adding this column or modifying the callback to use an existing column.")
        
        # Create query data with required fields
        query_data = QueryData(
            user_id=user_id,
            question=request.question,
            session_id=session_id,
            created_at=datetime.now()
        )
        
        logger.info(f"Created QueryData: {query_data}")
        
        # Create metadata for context
        metadata = ContextMetadata(
            context_id=str(uuid.uuid4()),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            component="mcp_server",
            operation="process_question",
            status="pending",
            session_id=session_id
        )
        
        # Create context for query
        query_context = Context(data=query_data, metadata=metadata)
        
        # Classify question
        logger.info("Classifying question")
        start_time = time.time()
        classification_context = mcp_question_classifier.process(query_context)
        
        # Route based on classification
        question_type = classification_context.data.question_type
        confidence = classification_context.data.confidence
        
        logger.info(f"Question classified as {question_type} with confidence {confidence}")
        
        # Store classification in context metadata
        metadata.component = "mcp_router"
        metadata.operation = "route_question"
        metadata.updated_at = datetime.now()
        
        # Use the router to direct the question based on type
        logger.info(f"Routing question to appropriate handler for type: {question_type}")
        
        # Log session manager details
        logger.info(f"Session manager type: {type(mcp_session_manager)}")
        logger.info(f"Session manager has session_manager attribute: {hasattr(mcp_session_manager, 'session_manager')}")
        if hasattr(mcp_session_manager, 'session_manager'):
            logger.info(f"Inner session_manager type: {type(mcp_session_manager.session_manager)}")
        
        response_context = mcp_router.route(
            classification_context, 
            session_manager=mcp_session_manager.session_manager,
            send_callback=graph_callback
        )
        
        # Extract response data
        logger.info("Processing response")
        response_data = response_context.data
        
        # Create response with execution time and classification info
        execution_time = time.time() - start_time
        response = {
            "session_id": session_id,
            "query": request.question,
            "result": response_data.summary,
            "execution_time": round(execution_time, 2),
            "question_type": question_type,
            "confidence": confidence
        }
        
        # Add additional information if available
        if hasattr(response_data, 'feedback_data') and response_data.feedback_data:
            response["feedback_data"] = response_data.feedback_data
            logger.info(f"Feedback data: {response_data.feedback_data}")
        
        logger.info(f"Processing completed in {execution_time:.2f} seconds")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "session_id": session_id}

if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 