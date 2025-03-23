"""
Model Context Protocol (MCP) Package

This package provides a standardized way to manage context flow between components
in the Analytics Bot system, enabling better traceability, debugging, and testing.
"""

from mcp.protocol import (
    Context, 
    ContextProcessor, 
    ContextFlow, 
    ContextRegistry, 
    ContextMetadata,
    DateTimeEncoder
)

from mcp.models import (
    QueryData,
    ConstraintData,
    StrategyData,
    QueryExecutionData,
    ResponseData,
    SessionData,
    ToolCallData
)

from mcp.processors import (
    MCPQueryProcessor,
    MCPQueryAgent,
    MCPQueryExecutor,
    MCPResponseGenerator,
    MCPSessionManager,
    MCPQueryFlowOrchestrator
)

__all__ = [
    # Protocol classes
    'Context',
    'ContextProcessor',
    'ContextFlow',
    'ContextRegistry',
    'ContextMetadata',
    'DateTimeEncoder',
    
    # Model classes
    'QueryData',
    'ConstraintData',
    'StrategyData',
    'QueryExecutionData',
    'ResponseData',
    'SessionData',
    'ToolCallData',
    
    # Processor classes
    'MCPQueryProcessor',
    'MCPQueryAgent',
    'MCPQueryExecutor',
    'MCPResponseGenerator',
    'MCPSessionManager',
    'MCPQueryFlowOrchestrator'
] 