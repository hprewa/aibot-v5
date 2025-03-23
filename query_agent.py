"""
Query Agent for generating SQL queries based on tool calls.

This module provides a Query Agent that can generate SQL queries for different KPIs
using either DSPy table definitions or Gemini for more complex queries.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
import dspy
from dspy_tables import OrdersTable, OrdersQueryBuilder
from gemini_client import GeminiClient

class QueryAgent:
    """
    Query Agent for generating SQL queries based on tool calls.
    
    This agent can generate SQL queries for different KPIs using either:
    1. DSPy table definitions (for standard queries)
    2. Gemini (for more complex queries not covered by DSPy tables)
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize the Query Agent.
        
        Args:
            gemini_client: An instance of GeminiClient for generating complex queries
        """
        self.gemini_client = gemini_client
        self.kpi_handlers = {
            "orders": self._handle_orders_query,
            # Add more KPI handlers as they are implemented
        }
        
        # Load KPI documentation
        self.kpi_docs = {}
        self._load_kpi_documentation()
    
    def _load_kpi_documentation(self):
        """Load documentation for each KPI from markdown files"""
        # Load orders documentation
        try:
            with open("orders sql prompt.md", "r") as f:
                self.kpi_docs["orders"] = f.read()
        except Exception as e:
            print(f"Warning: Could not load orders documentation: {str(e)}")
    
    def generate_query(self, tool_call: Dict[str, Any], constraints: Dict[str, Any]) -> str:
        """
        Generate a SQL query based on the tool call and constraints.
        
        Args:
            tool_call: The tool call definition from the strategy
            constraints: The constraints extracted from the user's question
            
        Returns:
            A SQL query string
        """
        # Extract KPI type from the tool call
        kpi_type = self._extract_kpi_type(tool_call)
        
        # Check if we have a handler for this KPI type
        if kpi_type in self.kpi_handlers:
            # Use the specific handler for this KPI
            return self.kpi_handlers[kpi_type](tool_call, constraints)
        else:
            # Fall back to Gemini for KPIs without specific handlers
            return self._generate_query_with_gemini(kpi_type, tool_call, constraints)
    
    def _extract_kpi_type(self, tool_call: Dict[str, Any]) -> str:
        """
        Extract the KPI type from the tool call.
        
        Args:
            tool_call: The tool call definition
            
        Returns:
            The KPI type as a string (e.g., "orders", "slot_availability")
        """
        # Try to extract from tables list
        tables = tool_call.get("tables", [])
        for table in tables:
            if "order" in table.lower():
                return "orders"
            elif "slot" in table.lower() or "availability" in table.lower():
                return "slot_availability"
        
        # Try to extract from name or description
        name = tool_call.get("name", "").lower()
        description = tool_call.get("description", "").lower()
        
        if "order" in name or "order" in description:
            return "orders"
        elif "slot" in name or "slot" in description or "availability" in name or "availability" in description:
            return "slot_availability"
        
        # Default to unknown
        return "unknown"
    
    def _handle_orders_query(self, tool_call: Dict[str, Any], constraints: Dict[str, Any]) -> str:
        """
        Handle queries for the orders KPI using the OrdersTable DSPy definition.
        
        Args:
            tool_call: The tool call definition
            constraints: The constraints extracted from the user's question
            
        Returns:
            A SQL query string
        """
        # Extract parameters from constraints and tool call
        time_filter = constraints.get("time_filter", {})
        start_date = time_filter.get("start_date", "")
        end_date = time_filter.get("end_date", "")
        
        # Determine aggregation type
        aggregation_type = self._determine_aggregation_type(tool_call, constraints)
        
        # Extract location filters
        cfc = constraints.get("cfc", [])
        spokes = constraints.get("spokes", [])
        
        # Determine group by fields
        group_by_fields = self._determine_group_by_fields(tool_call, constraints)
        
        # Use the OrdersQueryBuilder to build the query
        return OrdersQueryBuilder.build_query(
            aggregation_type=aggregation_type,
            start_date=start_date,
            end_date=end_date,
            cfc=cfc,
            spoke=spokes,
            group_by_fields=group_by_fields
        )
    
    def _determine_aggregation_type(self, tool_call: Dict[str, Any], constraints: Dict[str, Any]) -> str:
        """
        Determine the aggregation type based on the tool call and constraints.
        
        Args:
            tool_call: The tool call definition
            constraints: The constraints extracted from the user's question
            
        Returns:
            The aggregation type as a string ('daily', 'weekly', or 'monthly')
        """
        # Check if explicitly specified in the tool call
        description = tool_call.get("description", "").lower()
        
        if "daily" in description:
            return "daily"
        elif "weekly" in description or "week" in description:
            return "weekly"
        elif "monthly" in description or "month" in description:
            return "monthly"
        
        # Check if specified in constraints
        time_granularity = constraints.get("time_granularity", "").lower()
        
        if time_granularity:
            if "day" in time_granularity:
                return "daily"
            elif "week" in time_granularity:
                return "weekly"
            elif "month" in time_granularity:
                return "monthly"
        
        # Default to daily
        return "daily"
    
    def _determine_group_by_fields(self, tool_call: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        """
        Determine the fields to group by based on the tool call and constraints.
        
        Args:
            tool_call: The tool call definition
            constraints: The constraints extracted from the user's question
            
        Returns:
            A list of field names to group by
        """
        group_by_fields = []
        
        # Check if we need to group by CFC
        if constraints.get("location_type") == "CFC" or "cfc" in tool_call.get("description", "").lower():
            group_by_fields.append("cfc")
        
        # Check if we need to group by spoke
        if constraints.get("location_type") == "Spoke" or "spoke" in tool_call.get("description", "").lower():
            group_by_fields.append("spoke")
        
        return group_by_fields
    
    def _generate_query_with_gemini(self, kpi_type: str, tool_call: Dict[str, Any], constraints: Dict[str, Any]) -> str:
        """
        Generate a SQL query using Gemini for KPIs without specific handlers.
        
        Args:
            kpi_type: The type of KPI
            tool_call: The tool call definition
            constraints: The constraints extracted from the user's question
            
        Returns:
            A SQL query string
        """
        # Get KPI documentation if available
        kpi_doc = self.kpi_docs.get(kpi_type, "")
        
        # Extract parameters from constraints and tool call
        time_filter = constraints.get("time_filter", {})
        start_date = time_filter.get("start_date", "")
        end_date = time_filter.get("end_date", "")
        
        # Extract location filters
        cfc = constraints.get("cfc", [])
        spokes = constraints.get("spokes", [])
        
        # Create a prompt for Gemini
        prompt = f"""
You are a SQL query generator for a BigQuery database.

Tool Call:
- Name: {tool_call.get('name', 'Unnamed Query')}
- Description: {tool_call.get('description', '')}
- Tables: {', '.join(tool_call.get('tables', []))}

Constraints:
- Time Period: {start_date} to {end_date}
- CFCs: {', '.join(cfc) if cfc else 'All'}
- Spokes: {', '.join(spokes) if isinstance(spokes, list) else spokes}

{kpi_doc}

Your task is to:
1. Generate a precise BigQuery SQL query that fetches EXACTLY the data needed for this specific tool call
2. Include appropriate filters for time period, locations (CFCs and spokes)
3. Select only necessary columns to avoid excessive data transfer
4. Use proper BigQuery syntax and optimize for performance
5. Include appropriate comments to explain complex parts of the query

Return ONLY the SQL query, no explanation.
"""
        
        # Generate the query using Gemini
        response = self.gemini_client.generate_content(prompt)
        if not response:
            return "SELECT 1 -- Error generating SQL query"
        
        return response 