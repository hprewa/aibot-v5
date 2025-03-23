"""
Model Context Protocol (MCP) - Data Models

This module defines the data models for the various components of the Analytics Bot.
Each model represents the contextual data for a specific component and is used
with the Context class to create context objects.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

class QueryData(BaseModel):
    """Data model for the initial query"""
    user_id: str
    question: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConstraintData(BaseModel):
    """Data model for the extracted constraints"""
    kpi: List[str] = Field(default_factory=list)
    time_aggregation: str = "Daily"
    time_filter: Dict[str, str] = Field(default_factory=dict)
    cfc: List[str] = Field(default_factory=list)
    spokes: List[str] = Field(default_factory=list)
    comparison_type: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    response_plan: Dict[str, Any] = Field(default_factory=dict)

class ToolCallData(BaseModel):
    """Data model for a tool call"""
    name: str
    description: str
    result_id: str
    tables: List[str] = Field(default_factory=list)
    status: str = "pending"
    sql: Optional[str] = None
    error: Optional[str] = None
    result: Any = None

class StrategyData(BaseModel):
    """Data model for the generated strategy"""
    raw_strategy: str
    data_collection_plan: List[str] = Field(default_factory=list)
    processing_steps: List[str] = Field(default_factory=list)
    calculations: List[str] = Field(default_factory=list)
    response_structure: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCallData] = Field(default_factory=list)

class QueryExecutionData(BaseModel):
    """Data model for query execution data"""
    tool_calls: List[ToolCallData] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    execution_start: datetime = Field(default_factory=datetime.utcnow)
    execution_end: Optional[datetime] = None
    
class ResponseData(BaseModel):
    """Data model for the generated response"""
    summary: Optional[str] = None
    status: str = "processing"
    error: Optional[str] = None
    generated_at: Optional[datetime] = None

class SessionData(BaseModel):
    """Data model for the complete session"""
    session_id: str
    user_id: str
    question: str
    constraints: Optional[ConstraintData] = None
    strategy: Optional[StrategyData] = None
    execution: Optional[QueryExecutionData] = None
    response: Optional[ResponseData] = None
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    slack_channel: Optional[str] = None
    error: Optional[str] = None

    def to_bq_format(self) -> Dict[str, Any]:
        """Convert the session data to a format suitable for BigQuery"""
        # This method helps with serializing complex objects for BigQuery
        from mcp.protocol import DateTimeEncoder
        import json
        
        # Create a dictionary that can be serialized to JSON
        bq_data = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "question": self.question,
            "constraints": json.dumps(self.constraints.dict() if self.constraints else None, cls=DateTimeEncoder),
            "response_plan": json.dumps(self.constraints.response_plan if self.constraints else None, cls=DateTimeEncoder),
            "strategy": json.dumps(self.strategy.raw_strategy if self.strategy else None, cls=DateTimeEncoder),
            "summary": self.response.summary if self.response else None,
            "status": self.status,
            "tool_calls": [json.dumps(tc.dict(), cls=DateTimeEncoder) for tc in 
                          (self.execution.tool_calls if self.execution else [])],
            "tool_call_status": json.dumps({tc.name: tc.status for tc in 
                                          (self.execution.tool_calls if self.execution else [])}, 
                                         cls=DateTimeEncoder),
            "tool_call_results": json.dumps(self.execution.results if self.execution else None, cls=DateTimeEncoder),
            "results": json.dumps(self.execution.results if self.execution else None, cls=DateTimeEncoder),
            "slack_channel": self.slack_channel,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        return bq_data 