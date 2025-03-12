import os
import json
from google.cloud import bigquery
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from typing import Optional, Dict, List, Any
from google.api_core import retry
from datetime import datetime, timedelta
import pandas as pd

class BigQueryClient:
    """Client for interacting with BigQuery tables"""
    
    def __init__(self):
        self.project_id = os.getenv('PROJECT_ID')
        self.dataset_id = os.getenv('DATASET_ID')
        self.client = bigquery.Client()
        
        print("\nInitializing BigQueryClient...")
        print(f"Project ID: {self.project_id}")
        print(f"Dataset ID: {self.dataset_id}")
        print("Initializing BigQuery client...")
        
        # Define table references
        self.tables = {
            "orders": f"{self.project_id}.{self.dataset_id}.orders_drt",
            "slot_availability": f"{self.project_id}.{self.dataset_id}.slot_availability_drt",
            "cfc_spoke_mapping": f"{self.project_id}.{self.dataset_id}.cfc_spoke_mapping"
        }
        
        print("Table references:")
        for table_name, table_ref in self.tables.items():
            print(f"- {table_name}: {table_ref}")
        
        print("\nInitializing schema cache...")
        # Initialize schema cache
        self.schema_cache = {}
        self._load_table_schemas()
        
    def _load_table_schemas(self):
        """Load and cache schemas for all tables"""
        try:
            for table_name, table_ref in self.tables.items():
                print(f"\nLoading schema for {table_name} ({table_ref})")
                table = self.client.get_table(table_ref)
                self.schema_cache[table_name] = table.schema
                print(f"Successfully loaded schema with {len(table.schema)} columns")
                print(f"Columns: {[field.name for field in table.schema]}")
        except Exception as e:
            print(f"Error loading table schemas: {str(e)}")
            
    def _get_table_schema(self, table_name: str, force_refresh: bool = False) -> Optional[List[bigquery.SchemaField]]:
        """Get schema for a specific table with optional refresh"""
        if force_refresh or table_name not in self.schema_cache:
            try:
                table = self.client.get_table(self.tables[table_name])
                self.schema_cache[table_name] = table.schema
            except Exception as e:
                print(f"Error fetching schema for table {table_name}: {str(e)}")
                return None
                
        return self.schema_cache.get(table_name)
        
    def _validate_columns(self, table_name: str, columns: List[str]) -> bool:
        """Validate if columns exist in table schema"""
        schema = self._get_table_schema(table_name)
        if not schema:
            return False
            
        schema_columns = {field.name for field in schema}
        return all(col in schema_columns for col in columns)
        
    def get_column_type(self, table_name: str, column_name: str) -> Optional[str]:
        """Get the data type of a specific column"""
        schema = self._get_table_schema(table_name)
        if not schema:
            return None
            
        for field in schema:
            if field.name == column_name:
                return field.field_type
        return None

    def insert_row(self, table_id: str, row: Dict[str, Any]) -> None:
        """Insert a single row into a BigQuery table"""
        # Process JSON fields
        processed_row = {}
        for key, value in row.items():
            if key in ["constraints", "response_plan", "strategy", "tool_call_results"] and value is not None:
                # For JSON fields, convert to JSON string
                processed_row[key] = json.dumps(value)
            else:
                processed_row[key] = value
                
        errors = self.client.insert_rows_json(table_id, [processed_row])
        if errors:
            raise Exception(f"Error inserting row into {table_id}: {errors}")

    def update_row(self, table_id: str, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update a row in a BigQuery table.
        Note: BigQuery doesn't support updates directly in the same way as transactional databases.
        You typically perform a MERGE operation or overwrite the row.
        """
        # For JSON fields, we need to use JSON_EXTRACT_SCALAR or TO_JSON_STRING
        set_clauses = []
        for key, value in updates.items():
            if key in ["constraints", "response_plan", "strategy", "tool_call_results"] and value is not None:
                # For JSON fields, use TO_JSON_STRING
                json_value = json.dumps(value)
                set_clauses.append(f"{key} = JSON '{json_value}'")
            elif isinstance(value, (dict, list)):
                # For other dict/list values, convert to JSON string
                json_value = json.dumps(value)
                set_clauses.append(f"{key} = '{json_value}'")
            else:
                # For scalar values, use string literals
                if isinstance(value, str):
                    set_clauses.append(f"{key} = '{value}'")
                elif value is None:
                    set_clauses.append(f"{key} = NULL")
                else:
                    set_clauses.append(f"{key} = {value}")
        
        set_clause_str = ", ".join(set_clauses)
        query = f"""
            UPDATE `{table_id}`
            SET {set_clause_str}
            WHERE session_id = '{session_id}'
        """
        
        print(f"Executing update query: {query}")
        query_job = self.client.query(query)
        query_job.result()  # Wait for the job to complete

    def query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return the results as a list of dictionaries"""
        query_job = self.client.query(query)
        results = query_job.result()
        return [dict(row.items()) for row in results]

     