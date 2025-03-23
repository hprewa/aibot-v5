"""
DSPy table definitions for the analytics bot.

This module defines the DSPy table schemas for different KPIs that can be queried.
Each table schema is defined as a class with attributes representing the columns.
"""

import dspy
from typing import List, Dict, Any, Optional
from datetime import date, datetime

class OrdersTable(dspy.Signature):
    """
    DSPy table definition for the orders_drt table.
    
    This table contains aggregated order data with the following structure:
    - delivery_date (DATE): The actual date on which the orders were delivered. Used for daily aggregations.
    - week_date (DATE): The Monday of the week in which the delivery occurred. Used for weekly aggregations.
    - cfc (STRING): The name of the Customer Fulfillment Center (CFC). Each CFC has multiple spokes.
    - spoke (STRING): The name of the spoke, which is associated with a CFC.
    - orders (INTEGER): The total number of orders for the given delivery_date, cfc, and spoke.
    """
    
    # Column definitions using InputField and OutputField
    delivery_date: date = dspy.InputField(description="The actual date on which the orders were delivered")
    week_date: date = dspy.InputField(description="The Monday of the week in which the delivery occurred")
    cfc: str = dspy.InputField(description="The name of the Customer Fulfillment Center")
    spoke: str = dspy.InputField(description="The name of the spoke, which is associated with a CFC")
    orders: int = dspy.OutputField(description="The total number of orders for the given delivery_date, cfc, and spoke")


class OrdersQueryBuilder:
    """
    Query builder for the orders_drt table.
    
    This class provides methods for building SQL queries for the orders_drt table.
    """
    
    # Table name
    _table_name = "text-to-sql-dev.chatbotdb.orders_drt"
    
    @staticmethod
    def get_date_column_for_aggregation(aggregation_type: str) -> str:
        """
        Get the appropriate date column for the specified aggregation type.
        
        Args:
            aggregation_type: The type of aggregation ('daily', 'weekly', or 'monthly')
            
        Returns:
            The column name to use for date filtering and grouping
        """
        if aggregation_type == 'daily':
            return 'delivery_date'
        elif aggregation_type == 'weekly':
            return 'week_date'
        elif aggregation_type == 'monthly':
            return 'delivery_date'  # For monthly, we'll use EXTRACT in the query
        else:
            return 'delivery_date'  # Default to delivery_date
    
    @staticmethod
    def build_query(
        aggregation_type: str,
        start_date: str,
        end_date: str,
        cfc: Optional[List[str]] = None,
        spoke: Optional[List[str]] = None,
        group_by_fields: Optional[List[str]] = None
    ) -> str:
        """
        Build a SQL query for the orders_drt table based on the provided parameters.
        
        Args:
            aggregation_type: The type of aggregation ('daily', 'weekly', or 'monthly')
            start_date: The start date for filtering (inclusive)
            end_date: The end date for filtering (inclusive)
            cfc: Optional list of CFCs to filter by
            spoke: Optional list of spokes to filter by
            group_by_fields: Optional list of fields to group by
            
        Returns:
            A SQL query string
        """
        # Determine the date column for filtering and grouping
        date_column = OrdersQueryBuilder.get_date_column_for_aggregation(aggregation_type)
        
        # Build the SELECT clause
        select_clause = []
        group_by_clause = []
        
        if aggregation_type == 'daily':
            select_clause.append(f"{date_column} AS date")
            group_by_clause.append(date_column)
        elif aggregation_type == 'weekly':
            select_clause.append(f"{date_column} AS week_start_date")
            group_by_clause.append(date_column)
        elif aggregation_type == 'monthly':
            select_clause.append(f"EXTRACT(YEAR FROM {date_column}) AS year")
            select_clause.append(f"EXTRACT(MONTH FROM {date_column}) AS month")
            group_by_clause.append(f"EXTRACT(YEAR FROM {date_column})")
            group_by_clause.append(f"EXTRACT(MONTH FROM {date_column})")
        
        # Add CFC and/or spoke to SELECT and GROUP BY if needed
        if group_by_fields:
            for field in group_by_fields:
                if field not in group_by_clause and field in ['cfc', 'spoke']:
                    select_clause.append(field)
                    group_by_clause.append(field)
        
        # Always include the sum of orders
        select_clause.append("SUM(orders) AS total_orders")
        
        # Build the WHERE clause
        where_conditions = [f"{date_column} BETWEEN '{start_date}' AND '{end_date}'"]
        
        if cfc:
            cfc_list = "', '".join(cfc)
            where_conditions.append(f"cfc IN ('{cfc_list}')")
        
        if spoke:
            spoke_list = "', '".join(spoke)
            where_conditions.append(f"spoke IN ('{spoke_list}')")
        
        # Construct the final query
        query = f"""
        SELECT {', '.join(select_clause)}
        FROM `{OrdersQueryBuilder._table_name}`
        WHERE {' AND '.join(where_conditions)}
        GROUP BY {', '.join(group_by_clause)}
        ORDER BY {', '.join(group_by_clause)}
        """
        
        return query

# Add more table definitions for other KPIs here 