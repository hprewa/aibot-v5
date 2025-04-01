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

class OrdersTable(KPITable):
    """
    DSPy table definition for the orders_drt table.
    
    This table contains aggregated order data with the following structure:
    - delivery_date (DATE): The actual date on which the orders were delivered
    - week_date (DATE): The Monday of the week in which the delivery occurred
    - cfc (STRING): The name of the Customer Fulfillment Center
    - spoke (STRING): The name of the spoke, which is associated with a CFC
    - orders (INTEGER): The total number of orders
    """
    
    delivery_date: date = dspy.InputField(description="The actual date on which the orders were delivered")
    week_date: date = dspy.InputField(description="The Monday of the week in which the delivery occurred")
    cfc: str = dspy.InputField(description="The name of the Customer Fulfillment Center")
    spoke: str = dspy.InputField(description="The name of the spoke, which is associated with a CFC")
    orders: int = dspy.OutputField(description="The total number of orders")

class OrdersQueryBuilder(KPIQueryBuilder):
    """Query builder for the orders_drt table"""
    
    def __init__(self):
        super().__init__()
        self.table_class = OrdersTable
        self._table_name = "orders_drt"
        self.project_id = "text-to-sql-dev"
        self.dataset_id = "chatbotdb"
    
    def build_query(self, **params) -> str:
        """
        Build a SQL query for orders data based on parameters
        """
        try:
            print(f"\nBuilding orders query with parameters: {params}")
            
            # Validate required parameters
            if 'start_date' not in params or 'end_date' not in params:
                raise ValueError("start_date and end_date are required parameters")
            
            # Get date column based on aggregation
            date_col = self._get_date_column(params.get('aggregation', 'daily'))
            print(f"Using date column: {date_col}")
            
            # Build location filter
            location_filter = self._build_location_filter(params)
            print(f"Location filter: {location_filter}")
            
            # Build group by clause
            group_by = self._build_group_by(params)
            print(f"Group by clause: {group_by}")
            
            # Build the query
            query = f"""
            WITH location_data AS (
                SELECT 
                    o.*,
                    COALESCE(LOWER(m.cfc), LOWER(o.cfc)) as normalized_cfc,
                    LOWER(m.spoke) as normalized_spoke,
                    COALESCE(m.cfc, o.cfc) as display_cfc,
                    m.spoke as display_spoke
                FROM `{self.project_id}.{self.dataset_id}.{self._table_name}` o
                LEFT JOIN `{self.project_id}.{self.dataset_id}.cfc_spoke_mapping` m
                ON LOWER(o.cfc) = LOWER(m.spoke)
                WHERE DATE({date_col}) BETWEEN DATE(@start_date) AND DATE(@end_date)
            )
            SELECT 
                {date_col} as delivery_date,
                {self._build_select_columns(params)},
                SUM(orders) as total_orders
            FROM location_data
            WHERE {location_filter}
            GROUP BY {date_col}, {group_by}
            ORDER BY {date_col}
            """
            
            print(f"\nGenerated query:\n{query}")
            return query.strip()
            
        except Exception as e:
            print(f"Error building orders query: {str(e)}")
            raise
    
    def _get_date_column(self, aggregation: str) -> str:
        """Get the appropriate date column for aggregation"""
        if aggregation == 'weekly':
            return 'week_date'
        elif aggregation == 'monthly':
            return 'DATE_TRUNC(delivery_date, MONTH)'
        else:
            return 'delivery_date'
    
    def _build_location_filter(self, params: Dict[str, Any]) -> str:
        """Build the location filter clause"""
        filters = []
        
        if 'cfc' in params and params['cfc']:
            if isinstance(params['cfc'], list):
                # For list of CFCs, use IN clause with case-insensitive comparison
                filters.append(f"normalized_cfc IN UNNEST(@cfc)")
            else:
                # For single CFC, use direct comparison
                filters.append(f"normalized_cfc = @cfc")
                
        if 'spoke' in params and params['spoke'] and params['spoke'] != 'all':
            if isinstance(params['spoke'], list):
                # For list of spokes, use IN clause with case-insensitive comparison
                filters.append(f"normalized_spoke IN UNNEST(@spoke)")
            else:
                # For single spoke, use direct comparison
                filters.append(f"normalized_spoke = @spoke")
            
        print(f"Built location filters: {filters}")
        return ' AND '.join(filters) if filters else '1=1'
    
    def _build_select_columns(self, params: Dict[str, Any]) -> str:
        """Build the columns to select"""
        columns = []
        
        if 'cfc' in params and params['cfc']:
            columns.append('display_cfc as cfc')
        if 'spoke' in params and params['spoke'] and params['spoke'] != 'all':
            columns.append('display_spoke as spoke')
            
        return ', '.join(columns) if columns else 'display_cfc as cfc'
    
    def _build_group_by(self, params: Dict[str, Any]) -> str:
        """Build the group by clause"""
        group_by = []
        
        if 'cfc' in params and params['cfc']:
            group_by.append('display_cfc')
        if 'spoke' in params and params['spoke'] and params['spoke'] != 'all':
            group_by.append('display_spoke')
            
        return ', '.join(group_by) if group_by else 'display_cfc'

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