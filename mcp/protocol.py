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
    """Metadata for tracking context transformations"""
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    component: str
    operation: str
    status: str = "created"
    error: Optional[str] = None

T = TypeVar('T', bound=BaseModel)

class Context(BaseModel, Generic[T]):
    """Base context class that wraps data with metadata"""
    data: T
    metadata: ContextMetadata
    
    @classmethod
    def create(cls, data: T, component: str, operation: str, parent_id: Optional[str] = None) -> 'Context[T]':
        """Create a new context with the given data"""
        metadata = ContextMetadata(
            component=component,
            operation=operation,
            parent_id=parent_id
        )
        return cls(data=data, metadata=metadata)
    
    def update(self, data: T = None, **kwargs) -> 'Context[T]':
        """Update the context with new data and metadata"""
        # Create a new context object to maintain immutability
        new_data = data if data is not None else self.data
        
        # Create new metadata with the parent ID set to the current context ID
        new_metadata = ContextMetadata(
            context_id=str(uuid.uuid4()),
            parent_id=self.metadata.context_id,
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow(),
            component=kwargs.get('component', self.metadata.component),
            operation=kwargs.get('operation', self.metadata.operation),
            status=kwargs.get('status', "updated")
        )
        
        # Apply any additional metadata updates
        for key, value in kwargs.items():
            if hasattr(new_metadata, key):
                setattr(new_metadata, key, value)
        
        return Context(data=new_data, metadata=new_metadata)
    
    def error(self, error_message: str) -> 'Context[T]':
        """Mark context as errored with an error message"""
        return self.update(status="error", error=error_message)
    
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