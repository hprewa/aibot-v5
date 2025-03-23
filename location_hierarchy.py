"""Location Hierarchy Manager for handling location data and generating SQL conditions. 
Manages location levels (Network, CFC, Spoke) and facilitates location-based 
query filtering in the Analytics Bot."""

from google.cloud import bigquery
from typing import Dict, List, Set, Optional
import pandas as pd
from functools import lru_cache



class LocationHierarchyManager:
    """Manages location hierarchy and SQL generation for different aggregation levels"""
    
    def __init__(self, client: bigquery.Client):
        self.client = client
        self._refresh_cache()

    @lru_cache(maxsize=1, ttl=3600)  # Cache for 1 hour
    def _refresh_cache(self) -> Dict[str, Set]:
        """Fetch and cache location hierarchies from BigQuery"""
        query = """
        SELECT DISTINCT
            network_id,
            cfc_id,
            spoke_id
        FROM `{}.location_hierarchy`
        WHERE is_active = TRUE
        """
        
        df = self.client.query(query).to_dataframe()
        
        self.hierarchy = {
            'networks': set(df['network_id'].unique()),
            'cfcs': set(df['cfc_id'].unique()),
            'spokes': set(df['spoke_id'].unique()),
            'cfc_to_spokes': df.groupby('cfc_id')['spoke_id'].agg(set).to_dict(),
            'spoke_to_cfc': df.set_index('spoke_id')['cfc_id'].to_dict()
        }
        return self.hierarchy

    def detect_location_level(self, query: str) -> Dict[str, str]:
        """Detect location hierarchy level from query"""
        query_lower = query.lower()
        
        # Check for specific location mentions
        if 'spoke' in query_lower:
            return {'level': 'SPOKE'}
        elif 'cfc' in query_lower:
            # Extract CFC ID if present (e.g., "CFC-1")
            import re
            cfc_match = re.search(r'cfc-?\d+', query_lower)
            if cfc_match:
                return {
                    'level': 'CFC',
                    'location_id': cfc_match.group().upper()
                }
            return {'level': 'CFC'}
        
        # Default to network level
        return {'level': 'NETWORK'}

    @lru_cache(maxsize=100)
    def get_sql_conditions(self, location_info: Dict) -> str:
        """Generate SQL WHERE conditions based on location info"""
        level = location_info.get('level', 'NETWORK')
        location_id = location_info.get('location_id')
        
        if level == 'NETWORK':
            return "1=1"  # No filtering needed
        elif level == 'CFC':
            return f"cfc_id = '{location_id}'" if location_id else "1=1"
        elif level == 'SPOKE':
            return f"spoke_id = '{location_id}'" if location_id else "1=1"
        return "1=1"

    @lru_cache(maxsize=100)
    def get_grouping_columns(self, location_info: Dict) -> List[str]:
        """Get columns to group by based on location level"""
        level = location_info.get('level', 'NETWORK')
        
        if level == 'NETWORK':
            return []
        elif level == 'CFC':
            return ['cfc_id']
        elif level == 'SPOKE':
            return ['spoke_id', 'cfc_id']
        return []

    def validate_location(self, location_id: str, level: str) -> bool:
        """Validate if a location ID exists at the specified level"""
        if not location_id:
            return True
            
        query = f"""
        SELECT COUNT(1) as count
        FROM `{self.client.project}.{self.client.dataset_id}.cfc_spoke_mapping`
        WHERE {'cfc_id' if level == 'CFC' else 'spoke_id'} = '{location_id}'
        """
        
        try:
            results = self.client.query(query).result()
            return next(results).count > 0
        except Exception:
            return False  # If query fails, assume invalid location 