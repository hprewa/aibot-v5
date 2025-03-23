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
import json

class QueryData(BaseModel):
    """Data model for the initial query"""
    user_id: str
    question: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ClassificationData(BaseModel):
    """Data model for the question classification"""
    question: str
    question_type: str
    confidence: float
    requires_sql: bool = True
    requires_summary: bool = True
    classification_metadata: Dict[str, Any] = Field(default_factory=dict)
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
    """Data model for the response"""
    query: str
    summary: Optional[str] = None
    sql: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    feedback_data: Optional[Dict[str, Any]] = None  # Added for storing feedback
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SessionData(BaseModel):
    """Data model for the session"""
    session_id: str
    user_id: str
    question: str
    slack_channel: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    classification: Optional[ClassificationData] = None
    response_plan: Optional[Dict[str, Any]] = None
    strategy: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = None
    feedback: Optional[List[Dict[str, Any]]] = Field(default_factory=list)  # Added for storing feedback
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for BigQuery storage"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "question": self.question,
            "slack_channel": self.slack_channel,
            "constraints": json.dumps(self.constraints, cls=DateTimeEncoder) if self.constraints else None,
            "classification": json.dumps(self.classification.dict() if self.classification else None, cls=DateTimeEncoder),
            "question_type": self.classification.question_type if self.classification else None,
            "requires_sql": self.classification.requires_sql if self.classification else True,
            "requires_summary": self.classification.requires_summary if self.classification else True,
            "response_plan": json.dumps(self.response_plan, cls=DateTimeEncoder) if self.response_plan else None,
            "strategy": json.dumps(self.strategy, cls=DateTimeEncoder) if self.strategy else None,
            "execution": json.dumps(self.execution, cls=DateTimeEncoder) if self.execution else None,
            "response": json.dumps(self.response, cls=DateTimeEncoder) if self.response else None,
            "feedback": json.dumps(self.feedback, cls=DateTimeEncoder) if self.feedback else None,  # Added for storing feedback
            "results": json.dumps(self.results, cls=DateTimeEncoder) if self.results else None,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    def to_bq_format(self) -> Dict[str, Any]:
        """Convert the session data to a format suitable for BigQuery"""
        # This method helps with serializing complex objects for BigQuery
        from mcp.protocol import DateTimeEncoder
        
        # Create a dictionary that can be serialized to JSON
        bq_data = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "question": self.question,
            "classification": json.dumps(self.classification.dict() if self.classification else None, cls=DateTimeEncoder),
            "question_type": self.classification.question_type if self.classification else None,
            "requires_sql": self.classification.requires_sql if self.classification else True,
            "requires_summary": self.classification.requires_summary if self.classification else True,
            "constraints": json.dumps(self.constraints if self.constraints else None, cls=DateTimeEncoder),
            "response_plan": json.dumps(self.response_plan if self.response_plan else None, cls=DateTimeEncoder),
            "strategy": json.dumps(self.strategy if self.strategy else None, cls=DateTimeEncoder),
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