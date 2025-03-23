"""
Model Context Protocol (MCP) - Core Protocol classes

This module defines the core protocol classes for the Model Context Protocol (MCP).
MCP standardizes how context flows between components in a modular AI system.
"""

from typing import Dict, List, Any, Optional, TypeVar, Generic, Type
from abc import ABC, abstractmethod
import json
import datetime
from pydantic import BaseModel, Field, create_model
import uuid

# Custom JSON encoder to handle date objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

class ContextMetadata(BaseModel):
    """Metadata for the Context class, containing information about the context flow"""
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    component: str
    operation: str
    status: str = "pending"  # pending, success, error
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # Added for user identification
    
    @property
    def has_error(self) -> bool:
        """Check if the context has an error"""
        return self.status == "error"

T = TypeVar('T', bound=BaseModel)

class Context(Generic[T]):
    """
    Context class for the MCP protocol, containing data and metadata.
    
    This class is used to pass data between components in the MCP protocol.
    It contains both the data being processed and metadata about the processing.
    """
    
    data: T
    metadata: ContextMetadata
    
    def __init__(self, data: T, metadata: ContextMetadata):
        """
        Initialize a new Context object.
        
        Args:
            data: The data for this context
            metadata: The metadata for this context
        """
        self.data = data
        self.metadata = metadata
    
    @classmethod
    def create(cls, data: T, component: str, operation: str, parent_id: Optional[str] = None, 
              session_id: Optional[str] = None) -> 'Context[T]':
        """
        Create a new Context object with generated metadata.
        
        Args:
            data: The data for this context
            component: The component that created this context
            operation: The operation that created this context
            parent_id: The parent context ID, if any
            session_id: The session ID, if any
            
        Returns:
            A new Context object
        """
        metadata = ContextMetadata(
            component=component,
            operation=operation,
            parent_id=parent_id,
            session_id=session_id,
            status="pending"
        )
        
        return cls(data, metadata)
    
    def update(self, **kwargs) -> 'Context[T]':
        """
        Update the metadata of this context.
        
        Args:
            **kwargs: The metadata fields to update
            
        Returns:
            The updated context
        """
        # Create a copy of the metadata and update it
        metadata_dict = self.metadata.dict()
        metadata_dict.update(kwargs)
        metadata_dict["updated_at"] = datetime.datetime.utcnow()
        
        # Create a new metadata object with the updated fields
        self.metadata = ContextMetadata(**metadata_dict)
        
        return self
    
    def error(self, error_message: str) -> 'Context[T]':
        """Mark context as errored with an error message"""
        return self.update(status="error", error_message=error_message)
    
    def success(self) -> 'Context[T]':
        """Mark context as successful"""
        return self.update(status="success")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to a dictionary"""
        return {
            "data": self.data.dict(),
            "metadata": self.metadata.dict()
        }
    
    def to_json(self) -> str:
        """Convert context to a JSON string"""
        return json.dumps(self.to_dict(), cls=DateTimeEncoder)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Create a context from a dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Context':
        """Create a context from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

class ContextProcessor(ABC):
    """Abstract base class for components that process contexts"""
    
    @abstractmethod
    def process(self, context: Context) -> Context:
        """Process a context and return a new context"""
        pass

class ContextFlow:
    """Manages the flow of contexts between processors"""
    
    def __init__(self, processors: List[ContextProcessor]):
        self.processors = processors
    
    def execute(self, initial_context: Context) -> Context:
        """Execute the flow with an initial context"""
        current_context = initial_context
        
        for processor in self.processors:
            try:
                current_context = processor.process(current_context)
                if current_context.metadata.status == "error":
                    # Stop processing if an error occurred
                    break
            except Exception as e:
                # Catch any exceptions and mark the context as errored
                current_context = current_context.error(str(e))
                break
        
        return current_context

class ContextRegistry:
    """Registry for tracking context transformations"""
    
    def __init__(self):
        self.contexts: Dict[str, Context] = {}
    
    def register(self, context: Context) -> str:
        """Register a context and return its ID"""
        context_id = context.metadata.context_id
        self.contexts[context_id] = context
        return context_id
    
    def get(self, context_id: str) -> Optional[Context]:
        """Get a context by its ID"""
        return self.contexts.get(context_id)
    
    def get_lineage(self, context_id: str) -> List[Context]:
        """Get the lineage of a context, i.e., all contexts in its ancestry"""
        lineage = []
        current_id = context_id
        
        while current_id and current_id in self.contexts:
            context = self.contexts[current_id]
            lineage.append(context)
            current_id = context.metadata.parent_id
        
        return lineage 