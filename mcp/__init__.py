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
    ClassificationData,
    ConstraintData,
    StrategyData,
    QueryExecutionData,
    ResponseData,
    SessionData,
    ToolCallData
)

from mcp.processors import (
    MCPQuestionClassifier,
    MCPQueryProcessor,
    MCPQueryAgent,
    MCPQueryExecutor,
    MCPResponseGenerator,
    MCPSessionManager,
    MCPQueryFlowOrchestrator
)

# Integrations
try:
    from mcp.slack_app_home import MCPSlackAppHome
except ImportError:
    MCPSlackAppHome = None

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
    'ClassificationData',
    'ConstraintData',
    'StrategyData',
    'QueryExecutionData',
    'ResponseData',
    'SessionData',
    'ToolCallData',
    
    # Processor classes
    'MCPQuestionClassifier',
    'MCPQueryProcessor',
    'MCPQueryAgent',
    'MCPQueryExecutor',
    'MCPResponseGenerator',
    'MCPSessionManager',
    'MCPQueryFlowOrchestrator',
    
    # Integrations
    'MCPSlackAppHome'
] 