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
import traceback

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
    
    def _handle_orders_query(self, tool_call: Dict[str, Any], constraints: Dict[str, Any]) -> Optional[str]:
        """
        Handle queries for the orders KPI using the OrdersTable DSPy definition.
        
        Args:
            tool_call: The tool call definition
            constraints: The constraints extracted from the user's question
            
        Returns:
            A SQL query string or None if an error occurs.
        """
        try:
            print(f"\nGenerating orders query with:")
            print(f"Tool call: {json.dumps(tool_call, indent=2)}")
            print(f"Constraints: {json.dumps(constraints, indent=2)}")
            
            # Extract parameters from constraints and tool call
            time_filter = constraints.get("time_filter", {})
            start_date = time_filter.get("start_date", "")
            end_date = time_filter.get("end_date", "")
            
            # Determine aggregation type
            aggregation_type = self._determine_aggregation_type(tool_call, constraints)
            print(f"Using aggregation type: {aggregation_type}")
            
            # Extract location filters
            cfc = constraints.get("cfc", [])
            spokes = constraints.get("spokes", [])
            print(f"Location filters - CFC: {cfc}, Spokes: {spokes}")

            # Determine group by fields based *only* on the tool call description/name if needed
            # We avoid grouping by the full list of constraints['cfc'] during comparisons.
            group_by_fields = []
            tool_call_desc = tool_call.get("description", "").lower()
            tool_call_name = tool_call.get("name", "").lower()
            # Example: Group by CFC if the tool call explicitly asks for CFC-level aggregation
            if "cfc" in tool_call_desc or "cfc" in tool_call_name:
                 # Check if 'spoke' is also mentioned for grouping - unlikely for comparisons?
                 if "spoke" in tool_call_desc or "spoke" in tool_call_name:
                      group_by_fields.append("spoke")
                 else: # Default to grouping by cfc if mentioned
                      group_by_fields.append("cfc")
            # Only add spoke grouping if explicitly requested in the tool call itself
            elif "spoke" in tool_call_desc or "spoke" in tool_call_name:
                 group_by_fields.append("spoke")

            print(f"Group by fields determined from tool call: {group_by_fields}")

            # Handle different types of comparisons - Filter location based on THIS tool call
            comparison_type = constraints.get("comparison_type")
            cfc_filter_for_this_call = []
            spokes_filter_for_this_call = []

            if comparison_type == "between_locations":
                # Extract the specific location from this specific tool call name/description
                # Example: "get_stevenage_orders_jan_2024" -> target "stevenage"
                target_location = None
                # A simple heuristic: check if known CFCs/spokes are in the name
                known_locations = cfc + (spokes if isinstance(spokes, list) else []) # Combine known CFCs/spokes
                for loc in known_locations:
                    if loc in tool_call_name:
                         target_location = loc
                         break # Take the first match

                if target_location:
                    print(f"Identified target location '{target_location}' for tool call '{tool_call_name}'")
                    # Determine if the target is a CFC or Spoke (using original constraints list for lookup)
                    if target_location in cfc:
                        cfc_filter_for_this_call = [target_location]
                        # Ensure spokes are empty unless explicitly grouped by spoke in this tool call
                        if "spoke" not in group_by_fields:
                            spokes_filter_for_this_call = [] # Override if not grouping by spoke
                        else:
                            # If grouping by spoke, we might need all spokes for that CFC (TBD if needed)
                            spokes_filter_for_this_call = [] # Default to empty for now
                    elif isinstance(spokes, list) and target_location in spokes:
                         # If the target is a spoke, we usually need its parent CFC too for some queries
                         # Find the parent CFC for this spoke from the mapping (requires BQ client access or cached map)
                         # For simplicity here, let's assume the builder handles this or we pass the single spoke
                         spokes_filter_for_this_call = [target_location]
                         # Keep the original CFC list from constraints *unless* grouping is only by spoke
                         if "cfc" in group_by_fields:
                              cfc_filter_for_this_call = cfc # Keep original CFC list if grouping by CFC
                         else:
                              cfc_filter_for_this_call = [] # Clear CFC filter if only grouping by spoke
                    else:
                         # If target location not clearly identified as CFC/Spoke, fallback
                         print(f"Warning: Could not determine type for target location '{target_location}'. Falling back.")
                         cfc_filter_for_this_call = cfc # Fallback to original list
                         spokes_filter_for_this_call = spokes if isinstance(spokes, list) else []
                else:
                    print(f"Warning: Could not identify single target location from tool call name '{tool_call_name}'. Falling back.")
                    # Fallback if location cannot be extracted from tool call name
                    cfc_filter_for_this_call = cfc
                    spokes_filter_for_this_call = spokes if isinstance(spokes, list) else []
            else:
                 # Not a location comparison, use filters from main constraints
                 cfc_filter_for_this_call = cfc
                 spokes_filter_for_this_call = spokes if isinstance(spokes, list) else []

            # Use the OrdersQueryBuilder to build the query with SPECIFIC filters for this call
            query = OrdersQueryBuilder().build_query(
                aggregation_type=aggregation_type,
                start_date=start_date,
                end_date=end_date,
                cfc=cfc_filter_for_this_call, # Use filtered list for this specific call
                spoke=spokes_filter_for_this_call, # Use filtered list for this specific call
                group_by_fields=group_by_fields
            )
            
            print(f"\nGenerated SQL query:\n{query}")
            return query
            
        except Exception as e:
            print(f"!!! Error generating orders query: {str(e)}")
            traceback.print_exc() # Print full traceback for debugging
            return None # Return None to indicate failure
    
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