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
import uuid
import traceback

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
            "sessions": f"{self.project_id}.{self.dataset_id}.sessions",  # Explicitly add sessions table
            "question_classifications": f"{self.project_id}.{self.dataset_id}.question_classifications",  # Add question classifications table
            "question_categories": f"{self.project_id}.{self.dataset_id}.question_categories"  # Add question categories table
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
        
        # Ensure question classifications table exists
        self._ensure_question_classifications_table_exists()
        
        # Ensure question categories table exists
        self._ensure_question_categories_table_exists()
        
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

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return the results as a list of dictionaries"""
        try:
            print(f"Executing query: {query}")
            query_job = self.client.query(query)
            
            # Wait for the query to complete
            query_job.result()
            
            # Check if there were any errors
            if query_job.errors:
                print(f"Query execution errors: {query_job.errors}")
                return []
            
            # Get the results
            results = list(query_job.result())
            print(f"Query returned {len(results)} rows")
            
            # Convert to list of dictionaries
            formatted_results = []
            for row in results:
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        formatted_row[key] = value.isoformat()
                    elif hasattr(value, 'isoformat'):  # Handle date objects
                        formatted_row[key] = value.isoformat()
                    else:
                        formatted_row[key] = value
                formatted_results.append(formatted_row)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            traceback.print_exc()
            return []

    def _ensure_question_classifications_table_exists(self):
        """Ensure the question_classifications table exists, create it if it doesn't"""
        try:
            # Check if classifications table exists
            classifications_table_id = self.tables["question_classifications"]
            table_exists = False
            
            try:
                table = self.client.get_table(classifications_table_id)
                print(f"Question classifications table {classifications_table_id} exists")
                table_exists = True
                
                # Update schema cache
                self.schema_cache["question_classifications"] = table.schema
            except Exception as e:
                print(f"Question classifications table {classifications_table_id} does not exist, creating it...")
                table_exists = False
            
            if not table_exists:
                # Define schema for question_classifications table
                schema = [
                    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),  # Unique ID for each classification entry
                    bigquery.SchemaField("question", "STRING", mode="REQUIRED"),  # The original question text
                    bigquery.SchemaField("question_type", "STRING", mode="REQUIRED"),  # The classified question type
                    bigquery.SchemaField("confidence", "FLOAT", mode="REQUIRED"),  # Confidence score of classification
                    bigquery.SchemaField("requires_sql", "BOOLEAN", mode="REQUIRED"),  # Whether SQL generation is needed
                    bigquery.SchemaField("requires_summary", "BOOLEAN", mode="REQUIRED"),  # Whether summary generation is needed
                    bigquery.SchemaField("classification_metadata", "STRING", mode="NULLABLE"),  # Additional classification metadata as JSON
                    bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),  # Associated session ID if any
                    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),  # When the classification was created
                    bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED")   # When the classification was last updated
                ]
                
                # Create table
                table = bigquery.Table(classifications_table_id, schema=schema)
                
                try:
                    self.client.create_table(table)
                    print(f"Created question classifications table {classifications_table_id}")
                    
                    # Add to schema cache
                    self.schema_cache["question_classifications"] = schema
                except Exception as create_error:
                    print(f"Error creating question classifications table: {str(create_error)}")
                    raise
                
        except Exception as e:
            print(f"Error ensuring question classifications table exists: {str(e)}")
            raise
    
    def get_question_classification(self, question: str) -> Optional[Dict[str, Any]]:
        """Get an existing classification for a question if available"""
        query = f"""
            SELECT id, question, question_type, confidence, requires_sql, requires_summary, 
                   classification_metadata, session_id, created_at, updated_at
            FROM `{self.tables["question_classifications"]}`
            WHERE question = @question
            ORDER BY updated_at DESC
            LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("question", "STRING", question),
            ]
        )
        
        try:
            result = self.client.query(query, job_config=job_config).to_dataframe()
            
            if result.empty:
                return None
                
            # Convert the first row to a dictionary
            classification = result.iloc[0].to_dict()
            
            # Parse JSON fields
            if classification.get("classification_metadata") and isinstance(classification["classification_metadata"], str):
                try:
                    classification["classification_metadata"] = json.loads(classification["classification_metadata"])
                except json.JSONDecodeError:
                    classification["classification_metadata"] = {}
            
            return classification
        except Exception as e:
            print(f"Error getting question classification: {str(e)}")
            return None
    
    def save_question_classification(self, classification_data: Dict[str, Any]) -> str:
        """Save a question classification to BigQuery"""
        # Generate a unique ID if not provided
        if "id" not in classification_data:
            classification_data["id"] = str(uuid.uuid4())
            
        # Set timestamps if not provided
        current_time = datetime.utcnow().isoformat()
        if "created_at" not in classification_data:
            classification_data["created_at"] = current_time
        if "updated_at" not in classification_data:
            classification_data["updated_at"] = current_time
            
        # Serialize classification_metadata if it's a dictionary
        if "classification_metadata" in classification_data and isinstance(classification_data["classification_metadata"], dict):
            classification_data["classification_metadata"] = json.dumps(classification_data["classification_metadata"])
            
        # Insert the classification data
        self.insert_row(self.tables["question_classifications"], classification_data)
        
        return classification_data["id"]

    def get_question_categories(self) -> List[Dict[str, Any]]:
        """Retrieve question categories from BigQuery
        
        Returns:
            List of dictionaries containing question categories with fields:
            - category_id: Numeric ID of the category
            - question_type: Category name
            - description: Category description
            - example: Example question for this category
        """
        try:
            # Check if question_categories table exists
            categories_table = self.tables["question_categories"]
            try:
                self.client.get_table(categories_table)
            except Exception as e:
                print(f"Question categories table does not exist. Creating it with default categories...")
                self._create_default_question_categories_table()
            
            # Query the categories
            query = f"""
                SELECT 
                    category_id, 
                    question_type,
                    description,
                    example
                FROM `{self.project_id}.{self.dataset_id}.question_categories`
                ORDER BY category_id ASC
            """
            
            results = self.client.query(query).result()
            categories = [dict(row) for row in results]
            
            print(f"Retrieved {len(categories)} question categories from BigQuery")
            return categories
        except Exception as e:
            print(f"Error retrieving question categories: {str(e)}")
            # Return empty list as fallback
            return []
        
    def _create_default_question_categories_table(self):
        """Create the question_categories table with default categories"""
        try:
            # Define schema
            categories_table = self.tables["question_categories"]
            schema = [
                bigquery.SchemaField("category_id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("question_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("example", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            # Create table
            table = bigquery.Table(categories_table, schema=schema)
            self.client.create_table(table)
            print(f"Created question categories table {categories_table}")
            
            # Insert default categories based on the original CSV
            default_categories = [
                {
                    "category_id": 1,
                    "question_type": "KPI Extraction",
                    "description": "These questions request a single KPI value for a specific time range or location.",
                    "example": "What were perfect orders for Purfleet last week",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 2,
                    "question_type": "Comparitive Analysis",
                    "description": "These involve comparison between time periods, locations, or categories.",
                    "example": "Compare last week ATP with last to last week",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 3, 
                    "question_type": "Trend Analysis",
                    "description": "These ask for patterns and trends over time.",
                    "example": "Show me the trend of orders over the last 6 months? What are the peak ATP months in 2024",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 4,
                    "question_type": "Forecasting",
                    "description": "These ask for future trends, requiring predictive modeling via BigQuery ML",
                    "example": "Will orders per week exceed 600000 next year?",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 5,
                    "question_type": "Anamoly Detection",
                    "description": "Users want to understand reasons behind KPI fluctuations.",
                    "example": "Why did ATP drop in January 2025",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 6,
                    "question_type": "Operational efficiency",
                    "description": "Users want performance insights beyond just KPI values",
                    "example": "Which CFC had highest Perfect order values?",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 7,
                    "question_type": "Constrained Based Optimization",
                    "description": "Users need optimal actions based on constraints",
                    "example": "How should I allocate inventory across warehouses? What's the best way to reduce delivery delays?",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 8,
                    "question_type": "Exception Handling and Alert",
                    "description": "Users need for threshold-based monitoring.",
                    "example": "Alert me if perfect orders drops below 83%. Are there any KPI anamolies today",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 9,
                    "question_type": "Metadata and Schema",
                    "description": "Users asks about data structure or ai chatbot capabilities instead of KPI values.",
                    "example": "What KPIs can I query? Which locations are available in the database?",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 10,
                    "question_type": "Unsupported/Random Questions",
                    "description": "Questions that don't fit the system's purpose.",
                    "example": "Who is the CEO of Tesla? Tell me a joke.",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 11, 
                    "question_type": "Ambiguous Questions",
                    "description": "Questions with incomplete information, unclear User Intent",
                    "example": "Show me last week's report. (Report of what?) How did we do in Q4? (What KPI?)",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 12,
                    "question_type": "Multi-Intent Questions",
                    "description": "User combines two category of questions",
                    "example": "Compare last week's orders with last month and predict next week's trend.",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 13,
                    "question_type": "KPI Constraints & Unknown KPIs",
                    "description": "Users might request a derived KPI that isn't stored but can be computed.",
                    "example": "Find profit margin for last quarter. (What if 'profit margin' isn't stored as a KPI?)",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 14,
                    "question_type": "Nested or Multi-Step",
                    "description": "User questions that require multiple dependent steps",
                    "example": "Get last month's revenue, then compare it with last year's same month and show percentage growth.",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 15, 
                    "question_type": "Data Availability Issues",
                    "description": "User may ask about kpi, cfc, spoke or timeperiod which doesn't exist",
                    "example": "Find customer churn rate (but we don't store churn data).",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 16,
                    "question_type": "Unsupported NLP Constructs",
                    "description": "Complex Language patterns that may be difficult to parse",
                    "example": "Could you kindly tell me the average sales for Q1? Between London and Paris, which city sold more units last quarter?",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 17,
                    "question_type": "Action-Based Questions",
                    "description": "Users might request actions rather than just data.",
                    "example": "Email me last month's sales report. Generate PowerPoint from these insights.",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 18,
                    "question_type": "Small Talk",
                    "description": "Casual conversation",
                    "example": "How are you today?",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                {
                    "category_id": 19,
                    "question_type": "Feedback",
                    "description": "User provides comment on their experience",
                    "example": "You are doing a great job. This wasn't helpful at all",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            ]
            
            # Insert each category
            for category in default_categories:
                self.client.insert_rows_json(categories_table, [category])
            
            print(f"Inserted {len(default_categories)} default question categories")
        except Exception as e:
            print(f"Error creating question categories table: {str(e)}")
            raise

    def _ensure_question_categories_table_exists(self):
        """Ensure the question_categories table exists, create it if it doesn't"""
        try:
            # Check if categories table exists
            categories_table_id = self.tables["question_categories"]
            table_exists = False
            
            try:
                self.client.get_table(categories_table_id)
                print(f"Question categories table {categories_table_id} exists")
                table_exists = True
            except Exception as e:
                print(f"Question categories table {categories_table_id} does not exist, creating it...")
                table_exists = False
            
            if not table_exists:
                # Create the table with default categories
                self._create_default_question_categories_table()
            
        except Exception as e:
            print(f"Error ensuring question categories table exists: {str(e)}")
            raise

     