"""
DSPy table definitions for the analytics bot.

This module defines the DSPy table schemas for different KPIs that can be queried.
Each table schema is defined as a class with attributes representing the columns.
"""

import dspy
from typing import List, Dict, Any, Optional
from datetime import date, datetime


class KPITable(dspy.Signature):
    """Base class for all KPI tables"""
    pass

class KPIQueryBuilder:
    """Base class for all KPI query builders"""
    
    def __init__(self, project_id: str = "text-to-sql-dev", dataset_id: str = "chatbotdb"):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_class = None  # Set by subclasses
        
    def build_query(self, **params) -> str:
        """Build a SQL query based on parameters"""
        raise NotImplementedError("Subclasses must implement build_query")

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

    def __init__(self):
        super().__init__()
        self.table_class = OrdersTable
        self._table_name = "orders_drt"
        self.project_id = "text-to-sql-dev"
        self.dataset_id = "chatbotdb"

    

    
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
        
        # Only add spoke filter if spoke list is not empty
        if spoke and isinstance(spoke, list):
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

class ATPTable(KPITable):
    """
    DSPy table definition for the slot_availability_drt table.
    
    This table contains ATP (Available to Promise) data with the following structure:
    - slot_date (DATE): The date of the delivery slot
    - week_date (DATE): The Monday of the week
    - cfc (STRING): The name of the Customer Fulfillment Center
    - spoke (STRING): The name of the spoke
    - atp_score (FLOAT): The ATP score (0-100)
    - total_slots (INTEGER): Total number of slots
    - available_slots (INTEGER): Number of available slots
    """
    
    slot_date: date = dspy.InputField(description="The date of the delivery slot")
    week_date: date = dspy.InputField(description="The Monday of the week")
    cfc: str = dspy.InputField(description="The name of the Customer Fulfillment Center")
    spoke: str = dspy.InputField(description="The name of the spoke")
    atp_score: float = dspy.OutputField(description="The ATP score (0-100)")
    total_slots: int = dspy.OutputField(description="Total number of slots")
    available_slots: int = dspy.OutputField(description="Number of available slots")

class ATPQueryBuilder(KPIQueryBuilder):
    """Query builder for the slot_availability_drt table"""
    
    def __init__(self):
        super().__init__()
        self.table_class = ATPTable
        self._table_name = "slot_availability_drt"
    
    def build_query(self, **params) -> str:
        """
        Build a SQL query for ATP data based on parameters
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            cfc: Optional CFC name to filter by
            spoke: Optional spoke name to filter by
            aggregation: Time aggregation level ('daily', 'weekly', 'monthly')
            
        Returns:
            SQL query string
        """
        # Get date column based on aggregation
        date_col = self._get_date_column(params.get('aggregation', 'daily'))
        
        # Build location filter
        location_filter = self._build_location_filter(params)
        
        # Build group by clause
        group_by = self._build_group_by(params)
        
        # Build the query
        query = f"""
        WITH location_data AS (
            SELECT 
                s.*,
                COALESCE(m.cfc, s.location) as cfc,
                m.spoke
            FROM `{self.project_id}.{self.dataset_id}.{self._table_name}` s
            LEFT JOIN `{self.project_id}.{self.dataset_id}.cfc_spoke_mapping` m
            ON s.location = m.spoke
            WHERE {date_col} BETWEEN @start_date AND @end_date
        )
        SELECT 
            {date_col},
            {group_by},
            AVG(atp_score) as avg_atp_score,
            SUM(total_slots) as total_slots,
            SUM(available_slots) as available_slots
        FROM location_data
        WHERE {location_filter}
        GROUP BY {date_col}, {group_by}
        ORDER BY {date_col}
        """
        
        return query.strip()
    
    def _get_date_column(self, aggregation: str) -> str:
        """Get the appropriate date column for aggregation"""
        if aggregation == 'weekly':
            return 'week_date'
        elif aggregation == 'monthly':
            return 'DATE_TRUNC(slot_date, MONTH)'
        else:
            return 'slot_date'
    
    def _build_location_filter(self, params: Dict[str, Any]) -> str:
        """Build the location filter clause"""
        filters = []
        
        if 'cfc' in params:
            filters.append("LOWER(cfc) = LOWER(@cfc)")
        if 'spoke' in params and params['spoke'] != 'all':
            filters.append("LOWER(spoke) = LOWER(@spoke)")
            
        return ' AND '.join(filters) if filters else '1=1'
    
    def _build_group_by(self, params: Dict[str, Any]) -> str:
        """Build the group by clause"""
        group_by = []
        
        if 'cfc' in params:
            group_by.append('cfc')
        if 'spoke' in params and params['spoke'] != 'all':
            group_by.append('spoke')
            
        return ', '.join(group_by) if group_by else 'cfc'

# Add more table definitions for other KPIs here 