"""
MCP-powered server for the Analytics Bot.

This server uses the Model Context Protocol (MCP) to process analytical queries.
It provides the same API as the original server but uses MCP internally for better
traceability, debugging, and error handling.
"""

import os
import json
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Import original components
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent
from session_manager_v2 import SessionManagerV2

# Import MCP components
from mcp import (
    QueryData,
    Context,
    MCPQueryProcessor,
    MCPQueryAgent,
    MCPQueryExecutor,
    MCPResponseGenerator,
    MCPSessionManager,
    MCPQueryFlowOrchestrator
)

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

# Background task to process queries using the MCP flow
async def process_query_task(user_id: str, question: str, session_id: str):
    """Process a query using the MCP flow"""
    try:
        # Create query data
        query_data = QueryData(
            user_id=user_id,
            question=question,
            session_id=session_id,
            created_at=datetime.utcnow()
        )
        
        # Process the query through the MCP flow
        orchestrator.process_query(query_data)
        
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
        
        # Process the query in the background
        background_tasks.add_task(
            process_query_task,
            request.user_id,
            request.question,
            session_id
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

if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 