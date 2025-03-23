"""BigQuery client for interacting with Google BigQuery. Manages table schemas, 
executes queries, and handles data insertion and updates."""

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
            "cfc_spoke_mapping": f"{self.project_id}.{self.dataset_id}.cfc_spoke_mapping",
            "sessions": f"{self.project_id}.{self.dataset_id}.sessions"  # Explicitly add sessions table
        }
        
        print("Table references:")
        for table_name, table_ref in self.tables.items():
            print(f"- {table_name}: {table_ref}")
        
        print("\nInitializing schema cache...")
        # Initialize schema cache
        self.schema_cache = {}
        self._load_table_schemas()
        
        # Ensure sessions table exists
        self._ensure_sessions_table_exists()
        
    def _ensure_sessions_table_exists(self):
        """Ensure the sessions table exists, create it if it doesn't"""
        try:
            # Check if sessions table exists
            sessions_table_id = self.tables["sessions"]
            table_exists = False
            
            try:
                self.client.get_table(sessions_table_id)
                print(f"Sessions table {sessions_table_id} exists")
                table_exists = True
                
                # Check if we need to update the schema
                self._update_sessions_table_schema()
                
            except Exception as e:
                print(f"Sessions table {sessions_table_id} does not exist, creating it...")
                table_exists = False
            
            if not table_exists:
                # Define schema for sessions table
                schema = [
                    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("question", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("constraints", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("response_plan", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("strategy", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("summary", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("tool_calls", "STRING", mode="REPEATED"),  # Changed to REPEATED
                    bigquery.SchemaField("tool_call_status", "STRING", mode="NULLABLE"),  # Changed to NULLABLE
                    bigquery.SchemaField("tool_call_results", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("results", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("slack_channel", "STRING", mode="NULLABLE"),  # Added for Slack integration
                    bigquery.SchemaField("error", "STRING", mode="NULLABLE"),  # Added for error tracking
                    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED")
                ]
                
                # Create table
                table = bigquery.Table(sessions_table_id, schema=schema)
                
                try:
                    self.client.create_table(table)
                    print(f"Created sessions table {sessions_table_id}")
                    
                    # Add to schema cache
                    self.schema_cache["sessions"] = schema
                except Exception as create_error:
                    print(f"Error creating sessions table: {str(create_error)}")
                    raise
                
        except Exception as e:
            print(f"Error ensuring sessions table exists: {str(e)}")
            raise
            
    def _update_sessions_table_schema(self):
        """Update the sessions table schema if needed"""
        try:
            sessions_table_id = self.tables["sessions"]
            table = self.client.get_table(sessions_table_id)
            
            # Check if we need to update the schema
            needs_update = False
            field_names = [field.name for field in table.schema]
            
            # Check if any fields are missing
            missing_fields = []
            if "slack_channel" not in field_names:
                missing_fields.append(bigquery.SchemaField("slack_channel", "STRING", mode="NULLABLE"))
                needs_update = True
            if "error" not in field_names:
                missing_fields.append(bigquery.SchemaField("error", "STRING", mode="NULLABLE"))
                needs_update = True
                
            # Check if tool_calls and tool_call_status need to be updated to REPEATED
            for field in table.schema:
                if field.name == "tool_calls" and field.mode != "REPEATED":
                    needs_update = True
                    print(f"Field tool_calls needs to be updated to REPEATED mode")
                if field.name == "tool_call_status" and field.mode == "REPEATED":
                    needs_update = True
                    print(f"Field tool_call_status needs to be updated to NULLABLE mode")
            
            if needs_update:
                print(f"Updating sessions table schema...")
                
                # We need to recreate the table with the new schema
                # First, create a backup of the existing table
                backup_table_id = f"{sessions_table_id}_backup_{int(datetime.now().timestamp())}"
                print(f"Creating backup table {backup_table_id}")
                
                # Create a copy job
                job_config = bigquery.QueryJobConfig()
                sql = f"CREATE TABLE `{backup_table_id}` AS SELECT * FROM `{sessions_table_id}`"
                query_job = self.client.query(sql, job_config=job_config)
                query_job.result()  # Wait for the job to complete
                
                # Delete the existing table
                print(f"Deleting existing table {sessions_table_id}")
                self.client.delete_table(sessions_table_id)
                
                # Create a new table with the updated schema
                schema = [
                    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("question", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("constraints", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("response_plan", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("strategy", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("summary", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("tool_calls", "STRING", mode="REPEATED"),  # Keep as REPEATED
                    bigquery.SchemaField("tool_call_status", "STRING", mode="NULLABLE"),  # Changed to NULLABLE
                    bigquery.SchemaField("tool_call_results", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("results", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("slack_channel", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("error", "STRING", mode="NULLABLE"),
                    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED")
                ]
                
                # Create table
                table = bigquery.Table(sessions_table_id, schema=schema)
                self.client.create_table(table)
                print(f"Created sessions table with updated schema")
                
                # Check if the backup table has any data
                count_query = f"SELECT COUNT(*) as count FROM `{backup_table_id}`"
                count_job = self.client.query(count_query)
                count_result = list(count_job.result())[0]
                row_count = count_result.get('count', 0)
                
                if row_count > 0:
                    # Copy data back from the backup table
                    print(f"Copying {row_count} rows from backup table {backup_table_id}")
                    
                    # Get the column names from the backup table
                    schema_query = f"SELECT * FROM `{backup_table_id}` LIMIT 0"
                    schema_job = self.client.query(schema_query)
                    schema_result = schema_job.result()
                    backup_columns = [field.name for field in schema_result.schema]
                    
                    # Get the column names from the new table
                    new_columns = [field.name for field in schema]
                    
                    # Find common columns
                    common_columns = [col for col in backup_columns if col in new_columns]
                    
                    # Create the INSERT query with only common columns
                    columns_str = ", ".join(common_columns)
                    sql = f"""
                    INSERT INTO `{sessions_table_id}` ({columns_str})
                    SELECT {columns_str}
                    FROM `{backup_table_id}`
                    """
                    
                    print(f"Executing query: {sql}")
                    query_job = self.client.query(sql)
                    query_job.result()  # Wait for the job to complete
                    print(f"Successfully copied data from backup table")
                else:
                    print(f"Backup table {backup_table_id} is empty, no data to copy")
                
                # Update the schema cache
                self.schema_cache["sessions"] = schema
                
                print(f"Successfully updated sessions table schema")
        except Exception as e:
            print(f"Error updating sessions table schema: {str(e)}")
            # If there was an error, make sure the sessions table exists
            self._ensure_sessions_table_exists()
        
    def _load_table_schemas(self):
        """Load and cache schemas for all tables"""
        try:
            for table_name, table_ref in self.tables.items():
                print(f"\nLoading schema for {table_name} ({table_ref})")
                try:
                    table = self.client.get_table(table_ref)
                    self.schema_cache[table_name] = table.schema
                    print(f"Successfully loaded schema with {len(table.schema)} columns")
                    print(f"Columns: {[field.name for field in table.schema]}")
                except Exception as e:
                    print(f"Error loading schema for {table_name}: {str(e)}")
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
        try:
            # Process JSON fields
            processed_row = {}
            for key, value in row.items():
                if key in ["constraints", "response_plan", "strategy", "tool_call_results", "results", "tool_call_status"] and value is not None:
                    # For JSON fields, convert to JSON string
                    if isinstance(value, (dict, list)):
                        processed_row[key] = json.dumps(value)
                    else:
                        processed_row[key] = value
                elif key in ["tool_calls"] and value is not None:
                    # For REPEATED fields, ensure they are lists of strings
                    if isinstance(value, list):
                        # Convert each item to string if needed
                        processed_row[key] = [str(item) if not isinstance(item, str) else item for item in value]
                    else:
                        # If not a list, make it a list with one item
                        processed_row[key] = [str(value)]
                else:
                    processed_row[key] = value
            
            # Convert datetime objects to ISO format strings
            for key, value in processed_row.items():
                if isinstance(value, datetime):
                    processed_row[key] = value.isoformat()
                    
            print(f"Inserting row into {table_id}: {json.dumps(processed_row, default=str)[:200]}...")
            errors = self.client.insert_rows_json(table_id, [processed_row])
            if errors:
                raise Exception(f"Error inserting row into {table_id}: {errors}")
            else:
                print(f"Successfully inserted row into {table_id}")
        except Exception as e:
            print(f"Error in insert_row: {str(e)}")
            raise

    def get_cfc_spoke_mapping(self) -> Dict[str, List[str]]:
        """
        Fetch CFC-Spoke mapping from the database
        
        Returns:
            Dict[str, List[str]]: A dictionary mapping CFC names to lists of spoke names
        """
        try:
            print("Fetching CFC-Spoke mapping from database...")
            query = """
                SELECT cfc, spoke
                FROM `text-to-sql-dev.chatbotdb.cfc_spoke_mapping`
            """
            
            # Execute the query
            query_job = self.client.query(query)
            results = query_job.result()
            
            # Organize the results into a dictionary mapping CFCs to spokes
            cfc_to_spokes = {}
            all_cfcs = set()
            all_spokes = set()
            
            for row in results:
                cfc = row['cfc']
                spoke = row['spoke']
                
                all_cfcs.add(cfc)
                all_spokes.add(spoke)
                
                if cfc not in cfc_to_spokes:
                    cfc_to_spokes[cfc] = []
                    
                cfc_to_spokes[cfc].append(spoke)
            
            print(f"Fetched mapping for {len(cfc_to_spokes)} CFCs and {len(all_spokes)} Spokes")
            return cfc_to_spokes
            
        except Exception as e:
            print(f"Error fetching CFC-Spoke mapping: {str(e)}")
            # Return empty dictionary if there's an error
            return {}

    def update_row(self, table_id: str, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update a row in a BigQuery table.
        Note: BigQuery doesn't support updates directly in the same way as transactional databases.
        You typically perform a MERGE operation or overwrite the row.
        """
        # For JSON fields, we need to use JSON_EXTRACT_SCALAR or TO_JSON_STRING
        set_clauses = []
        for key, value in updates.items():
            if key in ["constraints", "response_plan", "strategy", "tool_call_results", "results"] and value is not None:
                # For JSON fields, use TO_JSON_STRING
                if isinstance(value, (dict, list)):
                    json_value = json.dumps(value)
                    set_clauses.append(f"{key} = JSON '{json_value}'")
                else:
                    set_clauses.append(f"{key} = '{value}'")
            elif key in ["tool_calls", "tool_call_status"] and value is not None:
                # For REPEATED fields, we need to use ARRAY
                if isinstance(value, list):
                    array_items = [f"'{item}'" for item in value]
                    array_str = f"[{', '.join(array_items)}]"
                    set_clauses.append(f"{key} = {array_str}")
                else:
                    set_clauses.append(f"{key} = ['{value}']")
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

     