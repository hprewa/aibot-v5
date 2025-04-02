"""
Model Context Protocol (MCP) - Processors

This module defines the processors for each component in the Analytics Bot.
Each processor implements the ContextProcessor interface and transforms
contexts between components.
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, cast
import json
import traceback
from datetime import datetime
import uuid
import logging
import re

from .protocol import Context, ContextProcessor, ContextMetadata
from .models import (
    QueryData, 
    ClassificationData,
    ConstraintData, 
    StrategyData, 
    QueryExecutionData,
    ResponseData, 
    SessionData,
    ToolCallData
)

# Import original components to wrap
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from session_manager_v2 import SessionManagerV2
from question_classifier import QuestionClassifier

T = TypeVar('T')
U = TypeVar('U')

class MCPQuestionClassifier(ContextProcessor):
    """MCP wrapper for the QuestionClassifier"""
    
    def __init__(self, question_classifier: QuestionClassifier):
        self.question_classifier = question_classifier
        
    def process(self, context: Context[QueryData]) -> Context[ClassificationData]:
        """Classify the query"""
        try:
            # Extract the query data
            query_data = context.data
            
            # Classify the question and pass the session_id
            classification_result = self.question_classifier.classify_question(
                question=query_data.question,
                session_id=query_data.session_id
            )
            
            # Create classification data model
            classification_data = ClassificationData(
                question=query_data.question,
                question_type=classification_result.get("question_type", "Unknown"),
                confidence=classification_result.get("confidence", 0.0),
                requires_sql=classification_result.get("requires_sql", True),
                requires_summary=classification_result.get("requires_summary", True),
                classification_metadata=classification_result.get("classification_metadata", {}),
                created_at=datetime.now()
            )
            
            # Create new metadata for the classification context, including session_id
            metadata = ContextMetadata(
                component="QuestionClassifier",
                operation="classify_question",
                parent_id=context.metadata.context_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status="success",
                session_id=query_data.session_id
            )
            
            # Return context with classification data and metadata
            return Context(data=classification_data, metadata=metadata)
        except Exception as e:
            traceback.print_exc()
            return cast(Context[ClassificationData], context.error(f"Error classifying question: {str(e)}"))

class MCPQueryProcessor(ContextProcessor):
    """MCP wrapper for the QueryProcessor"""
    
    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
    
    def process(self, context: Context) -> Context[ConstraintData]:
        """Extract constraints from the query"""
        try:
            # Check if this is a ClassificationData context
            if isinstance(context.data, ClassificationData):
                # Get the parent context to retrieve the original query
                parent_id = context.metadata.parent_id
                if not parent_id:
                    raise ValueError("No parent context ID found")
                
                # In a real implementation, we would fetch the parent context from a registry
                # For now, just create a dummy ConstraintData
                return Context.create(
                    data=ConstraintData(),
                    component="QueryProcessor",
                    operation="extract_constraints",
                    parent_id=context.metadata.context_id
                ).success()
            
            # If this is a QueryData context (backward compatibility)
            elif isinstance(context.data, QueryData):
                # Extract the query data
                query_data = context.data
                
                # Extract constraints using the original query processor
                extracted_constraints = self.query_processor.extract_constraints(query_data.question)
                
                # Create constraint data model
                if isinstance(extracted_constraints, str):
                    try:
                        extracted_constraints = json.loads(extracted_constraints)
                    except json.JSONDecodeError:
                        extracted_constraints = {}
                
                constraint_data = ConstraintData(
                    kpi=extracted_constraints.get("kpi", []),
                    time_aggregation=extracted_constraints.get("time_aggregation", "Daily"),
                    time_filter=extracted_constraints.get("time_filter", {}),
                    cfc=extracted_constraints.get("cfc", []),
                    spokes=extracted_constraints.get("spokes", []),
                    comparison_type=extracted_constraints.get("comparison_type"),
                    tool_calls=extracted_constraints.get("tool_calls", []),
                    response_plan=extracted_constraints.get("response_plan", {})
                )
                
                # Return updated context
                return Context.create(
                    data=constraint_data,
                    component="QueryProcessor",
                    operation="extract_constraints",
                    parent_id=context.metadata.context_id
                ).success()
            else:
                return cast(Context[ConstraintData], context.error(f"Unexpected context data type: {type(context.data)}"))
        except Exception as e:
            traceback.print_exc()
            return cast(Context[ConstraintData], context.error(f"Error extracting constraints: {str(e)}"))
    
    def generate_strategy(self, context: Context[ConstraintData]) -> Context[StrategyData]:
        """Generate a strategy from the constraints"""
        try:
            # Extract the constraint data
            constraint_data = context.data
            constraint_dict = constraint_data.dict()
            
            # Use the session to get the original question
            session_id = context.metadata.parent_id
            if not session_id:
                raise ValueError("No parent session ID found in metadata")
            
            # We need the query data to get the question
            # For now, assuming it's passed in a parent context
            # In a real implementation, we would fetch it from a registry or session storage
            # Workaround for this demo: reconstruct from available info
            original_question = "What is the query for these constraints?"  # Default fallback
            
            # Generate strategy using the original query processor
            raw_strategy = self.query_processor.generate_strategy(original_question, constraint_dict)
            
            # Create strategy data model 
            # In a full implementation, we would parse the raw strategy into structured fields
            strategy_data = StrategyData(
                raw_strategy=raw_strategy,
                # These would be parsed from the raw strategy in a full implementation
                data_collection_plan=[],
                processing_steps=[],
                calculations=[],
                response_structure=[],
                tool_calls=[]
            )
            
            # Return updated context
            return Context.create(
                data=strategy_data,
                component="QueryProcessor",
                operation="generate_strategy",
                parent_id=context.metadata.context_id
            ).success()
        except Exception as e:
            traceback.print_exc()
            return cast(Context[StrategyData], context.error(f"Error generating strategy: {str(e)}"))

class MCPQueryAgent(ContextProcessor):
    """MCP wrapper for the QueryAgent (placeholder)"""
    
    def __init__(self, query_agent=None):
        self.query_agent = query_agent
    
    def process(self, context: Context) -> Context:
        """Placeholder process method"""
        return context

class MCPQueryExecutor(ContextProcessor):
    """MCP processor for executing queries"""
    
    def __init__(self, bigquery_client: BigQueryClient):
        self.bigquery_client = bigquery_client
        self.logger = logging.getLogger(__name__)
    
    def process(self, context: Context[QueryExecutionData]) -> Context[QueryExecutionData]:
        """Execute queries in the execution data"""
        try:
            # Extract the execution data
            execution_data = context.data
            
            # Add query field from parent context if available
            if not hasattr(execution_data, 'query') and context.metadata.parent_id:
                # Try to get query from the parent context
                from mcp.protocol import Context
                parent_context = Context.get_context(context.metadata.parent_id)
                if parent_context and hasattr(parent_context.data, 'question'):
                    execution_data.query = parent_context.data.question
            
            # Set execution start time
            execution_data.execution_start = datetime.now()
            
            # Check data availability in the database
            available_date_range = self._get_data_availability()
            
            # Execute each query
            for tool_call in execution_data.tool_calls:
                # Skip failed or already executed tool calls
                if tool_call.status == "failed" or tool_call.status == "completed":
                    continue
                
                try:
                    # Execute the query
                    if tool_call.sql:
                        self.logger.info(f"\nExecuting query for tool call {tool_call.name}:")
                        self.logger.info(f"SQL: {tool_call.sql}")
                        
                        # Extract parameters from the SQL query
                        param_pattern = r'@([a-zA-Z0-9_]+)'
                        params_in_sql = re.findall(param_pattern, tool_call.sql)
                        self.logger.info(f"Parameters referenced in SQL: {params_in_sql}")
                        
                        # Extract constraints if available to use for parameters
                        params = {}
                        if hasattr(execution_data, 'constraints') and execution_data.constraints:
                            constraints = execution_data.constraints
                            self.logger.info(f"Extracting parameters from constraints: {constraints}")
                            
                            # Extract parameters from constraints
                            if isinstance(constraints, dict):
                                # Handle time filter parameters
                                if 'time_filter' in constraints and isinstance(constraints['time_filter'], dict):
                                    time_filter = constraints['time_filter']
                                    if 'start_date' in time_filter and 'start_date' in params_in_sql:
                                        params['start_date'] = time_filter['start_date']
                                    if 'end_date' in time_filter and 'end_date' in params_in_sql:
                                        params['end_date'] = time_filter['end_date']
                                
                                # Handle location parameters
                                if 'cfc' in constraints and 'cfc' in params_in_sql:
                                    params['cfc'] = constraints['cfc']
                                if 'spokes' in constraints and 'spoke' in params_in_sql:
                                    params['spoke'] = constraints['spokes']
                            else:
                                # Try to access dictionary methods if available
                                if hasattr(constraints, 'get'):
                                    time_filter = constraints.get('time_filter', {})
                                    if hasattr(time_filter, 'get'):
                                        if 'start_date' in params_in_sql:
                                            params['start_date'] = time_filter.get('start_date')
                                        if 'end_date' in params_in_sql:
                                            params['end_date'] = time_filter.get('end_date')
                                    
                                    if 'cfc' in params_in_sql:
                                        params['cfc'] = constraints.get('cfc')
                                    if 'spoke' in params_in_sql:
                                        params['spoke'] = constraints.get('spokes')
                            
                            self.logger.info(f"Parameters extracted from constraints: {params}")
                            
                        # Try to extract parameters from the tool call as a fallback
                        if tool_call.name and any(p not in params for p in params_in_sql):
                            self.logger.info(f"Extracting additional parameters from tool call name: {tool_call.name}")
                            # Look for date patterns in the tool call name
                            if 'start_date' in params_in_sql and 'start_date' not in params:
                                # Look for year patterns like "2023" or "jan_2023"
                                year_matches = re.findall(r'(\d{4})', tool_call.name)
                                month_matches = re.findall(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:_|-)(\d{4})', tool_call.name.lower())
                                
                                if month_matches:
                                    month_dict = {'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
                                                 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'}
                                    month, year = month_matches[0]
                                    start_date = f"{year}-{month_dict[month]}-01"
                                    params['start_date'] = start_date
                                    # Set end_date to the end of the month if not already set
                                    if 'end_date' in params_in_sql and 'end_date' not in params:
                                        month_int = int(month_dict[month])
                                        year_int = int(year)
                                        # Get next month
                                        if month_int == 12:
                                            next_month = 1
                                            next_year = year_int + 1
                                        else:
                                            next_month = month_int + 1
                                            next_year = year_int
                                        # Last day is one day before first day of next month
                                        end_date = f"{year if month_int < 12 else year_int+1}-{next_month:02d}-01"
                                        from datetime import datetime, timedelta
                                        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)
                                        params['end_date'] = end_date_obj.strftime('%Y-%m-%d')
                                    
                                elif year_matches and 'start_date' not in params:
                                    year = year_matches[0]
                                    params['start_date'] = f"{year}-01-01"
                                    if 'end_date' in params_in_sql and 'end_date' not in params:
                                        params['end_date'] = f"{year}-12-31"
                            
                            # Look for location patterns like "london" or "stevenage"
                            if 'cfc' in params_in_sql and 'cfc' not in params:
                                location_matches = re.findall(r'(london|stevenage|bristol|dordon|hatfield)', tool_call.name.lower())
                                if location_matches:
                                    params['cfc'] = [location_matches[0]]
                            
                            self.logger.info(f"Parameters after extraction from tool call: {params}")
                        
                        # Check if we have all required parameters
                        missing_params = [p for p in params_in_sql if p not in params or params[p] is None]
                        if missing_params:
                            self.logger.error(f"Missing required parameters: {missing_params}")
                            tool_call.status = "failed"
                            tool_call.error = f"Missing required parameters: {missing_params}"
                            continue
                        
                        # Execute query and get results
                        self.logger.info(f"Executing query with parameters: {params}")
                        self.logger.info(f"SQL QUERY EXECUTION START -----------------")
                        try:
                            # EXPLICITLY TRACE THE PROCESS
                            self.logger.info(f"ABOUT TO CALL execute_query with SQL: {tool_call.sql[:100]}...")
                            self.logger.info(f"Parameters that will be sent: {json.dumps(params, default=str)}")
                            
                            # Convert constraints object to dict if needed
                            if hasattr(execution_data, 'constraints') and not isinstance(execution_data.constraints, dict):
                                if hasattr(execution_data.constraints, 'dict'):
                                    params_from_dict = execution_data.constraints.dict()
                                    if 'time_filter' in params_from_dict and isinstance(params_from_dict['time_filter'], dict):
                                        if 'start_date' in params_from_dict['time_filter']:
                                            params['start_date'] = params_from_dict['time_filter']['start_date']
                                        if 'end_date' in params_from_dict['time_filter']:
                                            params['end_date'] = params_from_dict['time_filter']['end_date']
                                    if 'cfc' in params_from_dict:
                                        params['cfc'] = params_from_dict['cfc']
                                    self.logger.info(f"Updated parameters from constraints dict: {json.dumps(params, default=str)}")
                            
                            # Add a direct debug command call
                            import inspect
                            self.logger.info(f"BigQueryClient class structure: {inspect.getmembers(self.bigquery_client)}")
                            self.logger.info(f"Does execute_query exist? {'execute_query' in dir(self.bigquery_client)}")
                            
                            # DIRECT CALL TO EXECUTE THE QUERY - try both methods
                            try:
                                # Add special debug info to verify parameters
                                self.logger.info(f"\n*** DEBUG: ABOUT TO EXECUTE QUERY ***")
                                self.logger.info(f"Parameters being passed: {json.dumps(params, default=str)}")
                                
                                # Verify job_config creation and parameter attachment
                                from google.cloud import bigquery
                                job_config = bigquery.QueryJobConfig()
                                query_params = []
                                for name, value in params.items():
                                    self.logger.info(f"Setting parameter {name}={value} ({type(value)})")
                                    if isinstance(value, list):
                                        param = bigquery.ArrayQueryParameter(name, "STRING", value)
                                        self.logger.info(f"Created ArrayQueryParameter: {name}={value}")
                                    else:
                                        param = bigquery.ScalarQueryParameter(name, "STRING", str(value))
                                        self.logger.info(f"Created ScalarQueryParameter: {name}={value}")
                                    query_params.append(param)
                                
                                # Set parameters on job_config
                                job_config.query_parameters = query_params
                                self.logger.info(f"Set {len(query_params)} parameters on job_config")
                                
                                # Test the parameters with the BigQuery client directly
                                self.logger.info(f"*** DIRECT BIGQUERY TEST WITH PARAMETERS ***")
                                client = bigquery.Client(project="text-to-sql-dev")
                                # Create simple test query first
                                test_query = "SELECT @start_date as start, @end_date as end, @cfc[OFFSET(0)] as cfc"
                                self.logger.info(f"Test query: {test_query}")
                                test_job = client.query(test_query, job_config=job_config)
                                test_results = list(test_job.result())
                                self.logger.info(f"Test query results: {test_results}")
                                self.logger.info(f"First row: {dict(test_results[0].items()) if test_results else 'No results'}")
                                
                                # Now execute the real query with the verified job_config
                                self.logger.info(f"*** EXECUTING REAL QUERY WITH VERIFIED PARAMETERS ***")
                                if hasattr(self.bigquery_client, 'execute_query'):
                                    results = self.bigquery_client.execute_query(tool_call.sql, params)
                                    self.logger.info("execute_query method was used successfully")
                                else:
                                    self.logger.error("execute_query method not found, trying to call query method")
                                    results = self.bigquery_client.query(tool_call.sql, params)
                                    self.logger.info("query method was used successfully")
                            except Exception as specific_error:
                                self.logger.error(f"Direct method call failed: {str(specific_error)}")
                                # Try with client.query as a final attempt
                                try:
                                    self.logger.info("\n*** FINAL ATTEMPT - DIRECT BIGQUERY CLIENT ***")
                                    from google.cloud import bigquery
                                    
                                    # Create client with specific project
                                    project_id = "text-to-sql-dev"
                                    self.logger.info(f"Creating BigQuery client with project_id={project_id}")
                                    client = bigquery.Client(project=project_id)
                                    
                                    # Set up job configuration
                                    self.logger.info(f"Creating job configuration with parameters")
                                    job_config = bigquery.QueryJobConfig()
                                    query_params = []
                                    
                                    # Debug the SQL query
                                    self.logger.info(f"SQL query to execute:\n{tool_call.sql}")
                                    
                                    # Process parameters
                                    for name, value in params.items():
                                        self.logger.info(f"Processing parameter {name}={value} ({type(value)})")
                                        if isinstance(value, list):
                                            self.logger.info(f"Creating array parameter for {name}")
                                            # For array parameters (used in IN UNNEST clauses)
                                            param = bigquery.ArrayQueryParameter(name, "STRING", value)
                                            self.logger.info(f"Created array parameter: {name} = {value}")
                                        else:
                                            # For scalar parameters based on their type
                                            if isinstance(value, (datetime, datetime.date)):
                                                self.logger.info(f"Creating DATE parameter for {name}")
                                                param = bigquery.ScalarQueryParameter(name, "DATE", value)
                                            elif isinstance(value, int):
                                                self.logger.info(f"Creating INT64 parameter for {name}")
                                                param = bigquery.ScalarQueryParameter(name, "INT64", value)
                                            elif isinstance(value, float):
                                                self.logger.info(f"Creating FLOAT64 parameter for {name}")
                                                param = bigquery.ScalarQueryParameter(name, "FLOAT64", value)
                                            else:
                                                self.logger.info(f"Creating STRING parameter for {name}")
                                                # Convert to string for all other types
                                                string_value = str(value)
                                                param = bigquery.ScalarQueryParameter(name, "STRING", string_value)
                                                self.logger.info(f"Created string parameter: {name} = {string_value}")
                                        
                                        query_params.append(param)
                                    
                                    # Set parameters on job_config
                                    job_config.query_parameters = query_params
                                    self.logger.info(f"Set {len(query_params)} parameters on job_config")
                                    self.logger.info(f"Job config: {job_config}")
                                    
                                    # First run a simple test query to verify parameters
                                    self.logger.info(f"*** RUNNING TEST QUERY TO VERIFY PARAMETERS ***")
                                    test_query = "SELECT "
                                    test_parts = []
                                    for name in params.keys():
                                        if isinstance(params[name], list):
                                            test_parts.append(f"@{name}[OFFSET(0)] as {name}")
                                        else:
                                            test_parts.append(f"@{name} as {name}")
                                    test_query += ", ".join(test_parts)
                                    
                                    self.logger.info(f"Test query: {test_query}")
                                    try:
                                        test_job = client.query(test_query, job_config=job_config)
                                        test_results = list(test_job.result())
                                        self.logger.info(f"Test query succeeded with {len(test_results)} rows")
                                        if test_results:
                                            self.logger.info(f"Test result: {dict(test_results[0])}")
                                    except Exception as test_error:
                                        self.logger.error(f"Test query failed: {str(test_error)}")
                                    
                                    # Execute actual query
                                    self.logger.info(f"*** EXECUTING ACTUAL QUERY WITH BIGQUERY CLIENT ***")
                                    self.logger.info(f"SQL: {tool_call.sql}")
                                    self.logger.info(f"Parameters: {params}")
                                    
                                    # Execute query
                                    query_job = client.query(tool_call.sql, job_config=job_config)
                                    
                                    # Log job details
                                    self.logger.info(f"Query job created - ID: {query_job.job_id}")
                                    self.logger.info(f"Query job state: {query_job.state}")
                                    
                                    # Wait for results
                                    self.logger.info("Waiting for query results...")
                                    results = list(query_job.result())
                                    
                                    # Convert to dictionaries
                                    self.logger.info(f"Query returned {len(results)} rows")
                                    results = [dict(row.items()) for row in results]
                                    
                                    # Log sample result
                                    if results:
                                        self.logger.info(f"First result row: {results[0]}")
                                    
                                    self.logger.info("Direct BigQuery client.query call was successful")
                                except Exception as direct_error:
                                    self.logger.error(f"Direct BigQuery client call failed: {str(direct_error)}")
                                    self.logger.error(f"Error type: {type(direct_error)}")
                                    self.logger.error(f"Error traceback: {traceback.format_exc()}")
                                    
                                    # Log detailed error information
                                    self.logger.error(f"*** DETAILED ERROR INFORMATION ***")
                                    self.logger.error(f"Error message: {str(direct_error)}")
                                    self.logger.error(f"Error class: {direct_error.__class__.__name__}")
                                    
                                    # Set detailed error message on tool_call
                                    error_message = f"BigQuery query execution failed: {str(direct_error)}"
                                    tool_call.error = error_message
                                    tool_call.status = "failed"
                                    
                                    # LAST RESORT: Try with raw API call
                                    try:
                                        self.logger.info("*** ATTEMPTING ABSOLUTE LAST RESORT WITH SUBPROCESS ***")
                                        import subprocess
                                        import tempfile
                                        import os
                                        
                                        # Create a temporary file with the SQL and parameters
                                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                                            param_json = {
                                                "query": tool_call.sql,
                                                "params": params
                                            }
                                            json.dump(param_json, f)
                                            param_file = f.name
                                        
                                        # Create a Python script that will execute the query
                                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                                            f.write("""
import json
import sys
from google.cloud import bigquery

# Load query and params
with open(sys.argv[1], 'r') as f:
    data = json.load(f)
    
query = data['query']
params_dict = data['params']

# Set up BigQuery client
client = bigquery.Client(project="text-to-sql-dev")

# Set up job config with parameters
job_config = bigquery.QueryJobConfig()
query_params = []

# Convert parameters
for name, value in params_dict.items():
    if isinstance(value, list):
        param = bigquery.ArrayQueryParameter(name, "STRING", value)
    else:
        param = bigquery.ScalarQueryParameter(name, "STRING", str(value))
    query_params.append(param)

# Set parameters on job config
job_config.query_parameters = query_params

# Execute query
print(f"Executing query with parameters: {params_dict}")
query_job = client.query(query, job_config=job_config)

# Wait for results and print
results = list(query_job.result())
result_list = [dict(row.items()) for row in results]
print(f"Query returned {len(result_list)} rows")
if result_list:
    print(f"First row: {result_list[0]}")

# Write results to file
with open(sys.argv[2], 'w') as f:
    json.dump(result_list, f)
""")
                                            script_file = f.name
                                        
                                        # Create a file for results
                                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                                            results_file = f.name
                                        
                                        # Run the script
                                        self.logger.info(f"Running script: {script_file}")
                                        process = subprocess.run(
                                            ["python", script_file, param_file, results_file], 
                                            capture_output=True, 
                                            text=True
                                        )
                                        
                                        # Log the output
                                        self.logger.info(f"Script output: {process.stdout}")
                                        self.logger.info(f"Script errors: {process.stderr}")
                                        
                                        # Load results if successful
                                        if process.returncode == 0 and os.path.exists(results_file):
                                            with open(results_file, 'r') as f:
                                                results = json.load(f)
                                            self.logger.info(f"Loaded {len(results)} results from subprocess")
                                        else:
                                            raise Exception(f"Subprocess failed with code {process.returncode}")
                                        
                                        # Clean up temp files
                                        for file in [param_file, script_file, results_file]:
                                            try:
                                                os.remove(file)
                                            except:
                                                pass
                                    except Exception as subprocess_error:
                                        self.logger.error(f"Subprocess execution failed: {str(subprocess_error)}")
                                        self.logger.error(f"Error type: {type(subprocess_error)}")
                                        self.logger.error(f"Error traceback: {traceback.format_exc()}")
                                        
                                        # Set detailed error message on tool_call
                                        subprocess_error_msg = f"All execution methods failed. Last error: {str(subprocess_error)}"
                                        tool_call.error = subprocess_error_msg
                                        tool_call.status = "failed"
                                        
                                        # We've tried everything - just re-raise
                                        raise
                            
                            self.logger.info(f"SQL QUERY EXECUTION COMPLETE -------------")
                            self.logger.info(f"Results returned: {len(results) if results else 0} rows")
                        except Exception as query_error:
                            self.logger.error(f"Exception during execute_query: {str(query_error)}")
                            self.logger.error(f"Exception type: {type(query_error)}")
                            import traceback
                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                            # Re-raise to let the outer exception handler deal with it
                            raise
                        
                        # Format results for response agent
                        formatted_results = []
                        for row in results:
                            # Convert any datetime objects to ISO format strings
                            formatted_row = {}
                            for key, value in row.items():
                                if isinstance(value, (datetime.date, datetime.datetime)):
                                    formatted_row[key] = value.isoformat()
                                else:
                                    formatted_row[key] = value
                            formatted_results.append(formatted_row)
                        
                        # Check if we have zero results and provide a helpful message
                        if not formatted_results:
                            self.logger.info("Query returned zero results, checking for the reason")
                            
                            # Check for future dates in parameters
                            future_date_detected = False
                            current_date = datetime.now().date()
                            
                            if 'start_date' in params:
                                try:
                                    start_date_str = params['start_date']
                                    if isinstance(start_date_str, str):
                                        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                                        if start_date > current_date:
                                            future_date_detected = True
                                            self.logger.info(f"Future start date detected: {start_date} > {current_date}")
                                except (ValueError, TypeError) as date_error:
                                    self.logger.error(f"Error parsing start_date: {date_error}")
                            
                            if 'end_date' in params:
                                try:
                                    end_date_str = params['end_date']
                                    if isinstance(end_date_str, str):
                                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                                        if end_date > current_date:
                                            future_date_detected = True
                                            self.logger.info(f"Future end date detected: {end_date} > {current_date}")
                                except (ValueError, TypeError) as date_error:
                                    self.logger.error(f"Error parsing end_date: {date_error}")
                            
                            if future_date_detected:
                                self.logger.info("This query is for future dates which have no data")
                                # Still mark as completed but with a note
                                tool_call.status = "completed"
                                tool_call.result = []
                                tool_call.note = "This query is for future dates. No data is available for future time periods."
                            else:
                                # Check if we have date range info to provide helpful message
                                date_range_info = ""
                                out_of_range = False
                                available_range_message = ""
                                
                                # Get the available date range we fetched earlier
                                if available_date_range:
                                    min_date = available_date_range.get('min_date')
                                    max_date = available_date_range.get('max_date')
                                    
                                    if min_date and max_date:
                                        available_range_message = f" The database currently contains data from {min_date} to {max_date}."
                                        
                                        # Check if the query date range is outside available data
                                        if 'start_date' in params and 'end_date' in params:
                                            try:
                                                start_date_str = params['start_date']
                                                end_date_str = params['end_date']
                                                
                                                # Parse dates
                                                if isinstance(start_date_str, str):
                                                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                                                else:
                                                    start_date = start_date_str
                                                    
                                                if isinstance(end_date_str, str):
                                                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                                                else:
                                                    end_date = end_date_str
                                                
                                                # Convert min/max dates if needed
                                                if isinstance(min_date, str):
                                                    try:
                                                        min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
                                                    except ValueError:
                                                        # Handle other date formats
                                                        pass
                                                        
                                                if isinstance(max_date, str):
                                                    try:
                                                        max_date = datetime.strptime(max_date, '%Y-%m-%d').date()
                                                    except ValueError:
                                                        # Handle other date formats
                                                        pass
                                                
                                                # Check if dates are out of range
                                                if isinstance(end_date, date) and isinstance(min_date, date) and end_date < min_date:
                                                    out_of_range = True
                                                    date_range_info = f" for the period {start_date_str} to {end_date_str}, which is before our available data range"
                                                elif isinstance(start_date, date) and isinstance(max_date, date) and start_date > max_date:
                                                    out_of_range = True
                                                    date_range_info = f" for the period {start_date_str} to {end_date_str}, which is after our available data range"
                                                else:
                                                    date_range_info = f" for the period {start_date_str} to {end_date_str}"
                                            except (ValueError, TypeError) as parse_error:
                                                self.logger.error(f"Error parsing dates for range check: {parse_error}")
                                                date_range_info = f" for the period {params['start_date']} to {params['end_date']}"
                                
                                if not date_range_info and 'start_date' in params and 'end_date' in params:
                                    date_range_info = f" for the period {params['start_date']} to {params['end_date']}"

                                # Include location info if available
                                location_info = ""
                                if 'cfc' in params:
                                    location_values = params['cfc']
                                    if isinstance(location_values, list) and location_values:
                                        location_info = f" in {', '.join(location_values)}"
                                    elif isinstance(location_values, str):
                                        location_info = f" in {location_values}"

                                # Store the empty result but with an informative note
                                self.logger.info(f"Query returned zero results for specified parameters")
                                
                                # Update the tool call status
                                tool_call.status = "completed"
                                tool_call.result = formatted_results
                                
                                if out_of_range:
                                    tool_call.note = f"No data found{location_info}{date_range_info}.{available_range_message} Please try querying within the available data range."
                                else:
                                    tool_call.note = f"No data found{location_info}{date_range_info}. This could be due to no data being available for the specified time period or location, or because the data hasn't been loaded yet.{available_range_message}"
                        else:
                            # Store the formatted result
                            execution_data.results[tool_call.result_id] = formatted_results
                            
                            self.logger.info(f"Query returned {len(formatted_results)} rows")
                            if formatted_results:
                                self.logger.info(f"First row: {json.dumps(formatted_results[0], indent=2)}")
                            
                            # Update the tool call status
                            tool_call.status = "completed"
                            tool_call.result = formatted_results
                except Exception as e:
                    self.logger.error(f"Error executing query for tool call {tool_call.name}: {str(e)}")
                    tool_call.status = "failed"
                    tool_call.error = str(e)
                    
                    # Log error details
                    error_traceback = traceback.format_exc()
                    self.logger.error(f"Error traceback: {error_traceback}")
                    
                    # If it's a BigQuery error, try to get more details
                    try:
                        if hasattr(e, 'errors'):
                            self.logger.error(f"BigQuery errors: {e.errors}")
                        if hasattr(e, 'error_result'):
                            self.logger.error(f"Error result: {e.error_result}")
                    except:
                        pass
                    
                    continue
            
            # Set execution end time
            execution_data.execution_end = datetime.now()
            
            # Log final results
            self.logger.info("\nFinal execution results:")
            self.logger.info(f"Number of tool calls: {len(execution_data.tool_calls)}")
            self.logger.info(f"Results keys: {list(execution_data.results.keys())}")
            for result_id, result in execution_data.results.items():
                self.logger.info(f"Result {result_id}: {len(result)} rows")
                if result:
                    self.logger.info(f"Sample row: {json.dumps(result[0], indent=2)}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error in MCPQueryExecutor: {str(e)}")
            traceback.print_exc()
            raise

    def _get_data_availability(self):
        """Get the available date range in the database"""
        try:
            # Query to get min and max dates for orders
            date_range_query = """
            SELECT 
                MIN(DATE(delivery_date)) as min_date, 
                MAX(DATE(delivery_date)) as max_date 
            FROM `text-to-sql-dev.chatbotdb.orders_drt`
            """
            
            # Execute query
            self.logger.info("Checking available data range in the database...")
            try:
                date_results = self.bigquery_client.execute_query(date_range_query)
                
                if date_results and len(date_results) > 0:
                    min_date = date_results[0].get('min_date')
                    max_date = date_results[0].get('max_date')
                    
                    self.logger.info(f"Available data range: {min_date} to {max_date}")
                    return {
                        'min_date': min_date,
                        'max_date': max_date
                    }
                else:
                    self.logger.warning("No data range information available")
                    return None
            except Exception as e:
                self.logger.error(f"Error getting data availability: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Error in _get_data_availability: {e}")
            return None

class MCPResponseGenerator(ContextProcessor):
    """MCP processor for generating responses"""
    
    def __init__(self, response_agent: ResponseAgent):
        self.response_agent = response_agent
        self.logger = logging.getLogger(__name__)
    
    def process(self, context: Context[QueryExecutionData]) -> Context[ResponseData]:
        """Generate a response for the user based on the query execution results"""
        try:
            # Extract the execution data
            execution_data = context.data
            
            # Check if we have tool calls with notes about future dates
            future_date_notes = []
            no_data_notes = []
            
            for tool_call in execution_data.tool_calls:
                if hasattr(tool_call, 'note') and tool_call.note:
                    if 'future date' in tool_call.note.lower():
                        future_date_notes.append(tool_call.note)
                    elif 'no data found' in tool_call.note.lower():
                        no_data_notes.append(tool_call.note)
            
            if future_date_notes:
                # We have a future date query, create a specific response
                self.logger.info(f"Found future date query notes: {future_date_notes}")
                
                # Get cleaner date information for the response
                date_info = ""
                for tool_call in execution_data.tool_calls:
                    if tool_call.sql and '@start_date' in tool_call.sql and '@end_date' in tool_call.sql:
                        # Find the parameters in the execution data
                        if hasattr(execution_data, 'constraints') and execution_data.constraints:
                            constraints = execution_data.constraints
                            if isinstance(constraints, dict) and 'time_filter' in constraints:
                                time_filter = constraints['time_filter']
                                if 'start_date' in time_filter and 'end_date' in time_filter:
                                    # Format dates for display
                                    start_date = time_filter['start_date']
                                    end_date = time_filter['end_date']
                                    
                                    # Try to parse and format dates
                                    try:
                                        from datetime import datetime
                                        start = datetime.strptime(start_date, '%Y-%m-%d')
                                        end = datetime.strptime(end_date, '%Y-%m-%d')
                                        date_info = f"from {start.strftime('%B %d, %Y')} to {end.strftime('%B %d, %Y')}"
                                    except (ValueError, TypeError):
                                        date_info = f"from {start_date} to {end_date}"
                
                # Create a user-friendly response
                if date_info:
                    summary = f"I don't have any data {date_info} as this time period is in the future. I can only provide data for past time periods. Please try your query with dates in the past."
                else:
                    summary = f"I don't have any data for this query because it refers to a future time period. I can only provide data for past time periods. Please try your query with dates in the past."
                
                # Create the response data
                response_data = ResponseData(
                    query=execution_data.query if hasattr(execution_data, 'query') else "",
                    summary=summary,
                    results={},
                    execution_time=(execution_data.execution_end - execution_data.execution_start).total_seconds() if execution_data.execution_end else 0.0,
                    created_at=datetime.utcnow(),
                    status="completed"
                )
                
                return Context(data=response_data)
            
            elif no_data_notes:
                # We have a query that returned no data, but it's not a future date issue
                self.logger.info(f"Found no data notes: {no_data_notes}")
                
                # Extract useful information from the note
                note = no_data_notes[0]
                
                # Create a user-friendly response
                summary = f"I couldn't find any data for your query. {note}"
                
                # Create the response data
                response_data = ResponseData(
                    query=execution_data.query if hasattr(execution_data, 'query') else "",
                    summary=summary,
                    results={},
                    execution_time=(execution_data.execution_end - execution_data.execution_start).total_seconds() if execution_data.execution_end else 0.0,
                    created_at=datetime.utcnow(),
                    status="completed"
                )
                
                return Context(data=response_data)
            
            # Check if we have a failed query
            has_failed_tool_calls = any(tool_call.status == "failed" for tool_call in execution_data.tool_calls)
            
            if has_failed_tool_calls or not execution_data.results:
                # We have a failed query
                self.logger.info("Generating response for failed query")
                
                # Extract error messages from failed tool calls
                error_messages = []
                for tool_call in execution_data.tool_calls:
                    if tool_call.status == "failed" and tool_call.error:
                        error_messages.append(tool_call.error)
                
                error_summary = "; ".join(error_messages) if error_messages else "Unknown error"
                
                # Create a response data object with the error information
                response_data = ResponseData(
                    query=execution_data.query if hasattr(execution_data, 'query') else "",
                    summary="I apologize, but I couldn't retrieve any data for your query. Please try again or rephrase your question.",
                    error=error_summary,
                    results={},
                    execution_time=(execution_data.execution_end - execution_data.execution_start).total_seconds() if execution_data.execution_end else 0.0,
                    created_at=datetime.utcnow(),
                    status="failed"
                )
                
                # Return a context with the response data
                return Context(data=response_data)
            
            # Generate the response
            self.logger.info("\nGenerating response...")
            self.logger.info(f"Results: {json.dumps(execution_data.results, indent=2)}")
            
            response_text = self.response_agent.generate_response(
                question=execution_data.query,
                results=execution_data.results
            )
            
            if not response_text:
                raise ValueError("No response generated")
            
            # Calculate execution time
            execution_time = 0.0
            if execution_data.execution_start and execution_data.execution_end:
                execution_time = (execution_data.execution_end - execution_data.execution_start).total_seconds()
            
            # Create response data
            response_data = ResponseData(
                query=execution_data.query,
                summary=response_text,
                results=execution_data.results,
                execution_time=execution_time,
                created_at=datetime.now()
            )
            
            # Return updated context
            return Context.create(
                data=response_data,
                component="ResponseGenerator",
                operation="generate_response",
                parent_id=context.metadata.context_id
            ).success()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            traceback.print_exc()
            return cast(Context[ResponseData], context.error(f"Error generating response: {str(e)}"))

class MCPSessionManager(ContextProcessor):
    """MCP wrapper for the SessionManager"""
    
    def __init__(self, session_manager: SessionManagerV2):
        self.session_manager = session_manager
        
    def process(
        self, 
        query_context: Optional[Context] = None,
        classification_context: Optional[Context] = None,
        constraint_context: Optional[Context] = None,
        strategy_context: Optional[Context] = None,
        execution_context: Optional[Context] = None,
        response_context: Optional[Context] = None
    ) -> Context[SessionData]:
        """Update the session with data from various contexts"""
        try:
            # Find the query context or use the most recent context
            if query_context is None:
                if classification_context is not None:
                    query_context = self._find_query_context(classification_context.metadata.parent_id)
                elif constraint_context is not None:
                    query_context = self._find_query_context(constraint_context.metadata.parent_id)
                elif strategy_context is not None:
                    query_context = self._find_query_context(strategy_context.metadata.parent_id)
                elif execution_context is not None:
                    query_context = self._find_query_context(execution_context.metadata.parent_id)
                elif response_context is not None:
                    query_context = self._find_query_context(response_context.metadata.parent_id)
                    
            if query_context is None:
                raise ValueError("No query context available")
                
            # Extract the query data
            query_data = None
            if isinstance(query_context.data, QueryData):
                query_data = query_context.data
            else:
                raise ValueError(f"Expected QueryData, got {type(query_context.data)}")
                
            # Get the session ID
            session_id = query_data.session_id
            
            # Create or update session data
            session_data = self._get_or_create_session_data(session_id, query_data)
            
            # Update with classification data
            if classification_context is not None and isinstance(classification_context.data, ClassificationData):
                session_data.classification = classification_context.data
                
            # Update with constraint data
            if constraint_context is not None and isinstance(constraint_context.data, ConstraintData):
                session_data.constraints = constraint_context.data
                
            # Update with strategy data
            if strategy_context is not None and isinstance(strategy_context.data, StrategyData):
                session_data.strategy = strategy_context.data
                
            # Update with execution data
            if execution_context is not None and isinstance(execution_context.data, QueryExecutionData):
                session_data.execution = execution_context.data
                
            # Update with response data
            if response_context is not None and isinstance(response_context.data, ResponseData):
                session_data.response = response_context.data
                if session_data.response.status == "completed":
                    session_data.status = "completed"
                elif session_data.response.status == "error":
                    session_data.status = "error"
                    session_data.error = session_data.response.error
                
            # Check for errors in context
            context_with_error = None
            error_message = None
            
            if classification_context is not None and classification_context.metadata.has_error:
                context_with_error = classification_context
                error_message = classification_context.metadata.error_message
            elif constraint_context is not None and constraint_context.metadata.has_error:
                context_with_error = constraint_context
                error_message = constraint_context.metadata.error_message
            elif strategy_context is not None and strategy_context.metadata.has_error:
                context_with_error = strategy_context
                error_message = strategy_context.metadata.error_message
            elif execution_context is not None and execution_context.metadata.has_error:
                context_with_error = execution_context
                error_message = execution_context.metadata.error_message
            elif response_context is not None and response_context.metadata.has_error:
                context_with_error = response_context
                error_message = response_context.metadata.error_message
                
            if context_with_error is not None:
                session_data.status = "error"
                session_data.error = error_message
                
            # Update timestamp
            session_data.updated_at = datetime.now()
            
            # Update session in the session manager
            self.session_manager.update_session(
                session_id=session_id,
                updates=session_data.to_bq_format()
            )
            
            # Create session context
            session_context = Context.create(
                data=session_data,
                component="SessionManager",
                operation="update_session",
                parent_id=query_context.metadata.context_id
            )
            
            if context_with_error is not None:
                return cast(Context[SessionData], session_context.error(error_message))
            else:
                return session_context.success()
                
        except Exception as e:
            traceback.print_exc()
            
            # Try to extract query data for minimal session update
            query_data = None
            if query_context is not None and isinstance(query_context.data, QueryData):
                query_data = query_context.data
                
                # Try to update session with error
                try:
                    session_id = query_data.session_id
                    self.session_manager.update_session(
                        session_id=session_id,
                        updates={
                            "status": "error",
                            "error": f"Error updating session: {str(e)}"
                        }
                    )
                except Exception:
                    # If even that fails, just log the error
                    traceback.print_exc()
            
            # Return error context
            error_context = Context.create(
                data=SessionData(
                    session_id=query_data.session_id if query_data else "unknown",
                    user_id=query_data.user_id if query_data else "unknown",
                    question=query_data.question if query_data else "unknown",
                    status="error",
                    error=f"Error updating session: {str(e)}"
                ),
                component="SessionManager",
                operation="update_session"
            )
            
            return cast(Context[SessionData], error_context.error(f"Error updating session: {str(e)}"))
    
    def _get_or_create_session_data(self, session_id: str, query_data: QueryData) -> SessionData:
        """Get or create a session data object for a query"""
        try:
            # Try to get the session from BigQuery
            session_dict = self.session_manager.get_session(session_id)
            
            if session_dict:
                try:
                    # If classification exists in session
                    classification_dict = None
                    if "classification" in session_dict and session_dict["classification"]:
                        # Try to parse classification as JSON
                        try:
                            classification_dict = json.loads(session_dict["classification"])
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, just use the raw data
                            classification_dict = session_dict["classification"]
                    
                    # Create classification data if it exists
                    classification = None
                    if classification_dict:
                        try:
                            classification = ClassificationData(
                                question=query_data.question,
                                question_type=classification_dict.get("question_type", "Unknown"),
                                confidence=classification_dict.get("confidence", 0.0),
                                requires_sql=classification_dict.get("requires_sql", True),
                                requires_summary=classification_dict.get("requires_summary", True),
                                classification_metadata=classification_dict.get("classification_metadata", {}),
                                created_at=classification_dict.get("created_at", datetime.now())
                            )
                        except (json.JSONDecodeError, TypeError):
                            # If creating classification data fails, just continue without it
                            pass
                    
                    # Create session data from the session dict
                    return SessionData(
                        session_id=session_id,
                        user_id=session_dict.get("user_id", query_data.user_id),
                        question=session_dict.get("question", query_data.question),
                        classification=classification,
                        constraints=session_dict.get("constraints"),
                        response_plan=session_dict.get("response_plan"),
                        strategy=session_dict.get("strategy"),
                        execution=session_dict.get("execution"),
                        response=session_dict.get("response"),
                        results=session_dict.get("results"),
                        slack_channel=session_dict.get("slack_channel"),
                        status=session_dict.get("status", "pending"),
                        created_at=session_dict.get("created_at", query_data.created_at),
                        updated_at=datetime.now(),
                        error=session_dict.get("error")
                    )
                except Exception as inner_e:
                    print(f"Error creating SessionData from session dict: {str(inner_e)}")
                    traceback.print_exc()
                    # Fall back to creating a new session data
            
            # If session doesn't exist or couldn't be parsed, create a new one
            # The session_manager.create_session method already generates a session_id
            created_session_id = self.session_manager.create_session(
                query_data.user_id, 
                query_data.question
            )
            # If received session_id different from input, update it
            if created_session_id != session_id:
                print(f"Note: Generated session ID {created_session_id} different from provided {session_id}")
                session_id = created_session_id
            
            # Create a new session data object
            return SessionData(
                session_id=session_id,
                user_id=query_data.user_id,
                question=query_data.question,
                status="pending",
                created_at=query_data.created_at,
                updated_at=datetime.now()
            )
        except Exception as e:
            print(f"Error in _get_or_create_session_data: {str(e)}")
            traceback.print_exc()
            
            # Return a minimal session data object
            return SessionData(
                session_id=session_id,
                user_id=query_data.user_id,
                question=query_data.question,
                status="pending",
                created_at=query_data.created_at,
                updated_at=datetime.now(),
                error=f"Error getting/creating session: {str(e)}"
            )

# An example of a flow orchestrator that combines all processors
class MCPQueryFlowOrchestrator:
    """Orchestrates the flow of contexts through the MCP pipeline"""
    
    def __init__(
        self,
        question_classifier: MCPQuestionClassifier,
        query_processor: MCPQueryProcessor,
        query_agent: MCPQueryAgent,
        query_executor: MCPQueryExecutor,
        response_generator: MCPResponseGenerator,
        session_manager: MCPSessionManager
    ):
        self.question_classifier = question_classifier
        self.query_processor = query_processor
        self.query_agent = query_agent
        self.query_executor = query_executor
        self.response_generator = response_generator
        self.session_manager = session_manager
        
    def process_query(self, query_data: QueryData) -> Context[ResponseData]:
        """
        Process a query through the full MCP flow.
        
        This method will:
        1. Create a context from the query data
        2. Classify the question
        3. Extract constraints
        4. Generate a strategy
        5. Execute the queries
        6. Generate a response
        7. Update the session
        
        Args:
            query_data: The query data to process
            
        Returns:
            Context with the response data
        """
        # Create error context metadata for reuse
        error_metadata = ContextMetadata(
            component="MCPQueryFlowOrchestrator",
            operation="process_query",
            session_id=query_data.session_id,
            status="error"
        )
        
        try:
            print(f"🔄 Starting processing for question: {query_data.question}")
            session_id = query_data.session_id
            
            # Step 1: Create a context from the query data
            query_context = Context(
                data=query_data,
                metadata=ContextMetadata(
                    component="MCPQueryFlowOrchestrator",
                    operation="process_query",
                    session_id=session_id,
                    status="pending"
                )
            )
            
            # Step 2: Classify the question
            try:
                classification_context = self.question_classifier.process(query_context)
                # Add user_id to metadata to ensure it's available to the router
                classification_context.metadata.user_id = query_data.user_id
                
                # Check for errors
                if classification_context.metadata.status == "error":
                    error_msg = f"Error classifying question: {classification_context.metadata.error_message}"
                    print(f"❌ {error_msg}")
                    return self._create_error_response(query_context, error_msg)
            except Exception as e:
                error_msg = f"Error classifying question: {str(e)}"
                print(f"❌ {error_msg}")
                traceback.print_exc()
                return self._create_error_response(query_context, error_msg)
                
            # Get classification data
            classification_data = classification_context.data
            print(f"📊 Question classified as {classification_data.question_type} with confidence {classification_data.confidence}")
            
            # --- Add logic to fetch previous session context for follow-up questions ---
            previous_context = {}
            if classification_data.question_type == "Follow-up question":
                print(f"🔍 Detected follow-up question for session {session_id}. Fetching previous context...")
                try:
                    # Use the *underlying* session manager to fetch data directly
                    if hasattr(self.session_manager, 'session_manager') and self.session_manager.session_manager:
                        previous_session = self.session_manager.session_manager.get_session(session_id)
                        if previous_session:
                            previous_context['previous_question'] = previous_session.get('question')
                            previous_context['previous_summary'] = previous_session.get('summary')
                            previous_context['previous_results'] = previous_session.get('results') # Assumes results are stored appropriately
                            print("✅ Successfully fetched previous context.")
                            # Log fetched context previews
                            print(f"  Prev Q: {previous_context['previous_question'][:100]}...")
                            print(f"  Prev S: {previous_context['previous_summary'][:100]}...")
                            # Note: Printing results can be large, skip detailed log here
                        else:
                            print(f"⚠️ Previous session {session_id} not found.")
                    else:
                        print("⚠️ Cannot fetch previous session: underlying session manager not available.")
                except Exception as e:
                    print(f"❌ Error fetching previous session context: {str(e)}")
                    traceback.print_exc()
            # -----------------------------------------------------------------------

            # Step 3: Extract constraints if needed (for some query types)
            constraint_context = None
            strategy_context = None
            execution_context = None
            response_context = None
            
            # Update the query processor call to include previous context if available
            # This assumes the query processor is called before the router for relevant types
            # Note: The current orchestrator logic relies heavily on the router. 
            # We need to adjust the flow to ensure constraints (with potential previous context) 
            # are extracted *before* routing, or adjust the router itself.
            # For now, we'll modify the assumption that the router handles the pipeline.
            # Let's modify the router's _handle_full_pipeline to accept previous_context
            
            # Instead of completing the full pipeline here, use our router
            # to determine the next steps based on classification
            if hasattr(self, 'router') and self.router:
                try:
                    # Make sure we pass the underlying session_manager object, not the wrapper
                    # Also pass the fetched previous_context to the router
                    response_context = self.router.route(
                        classification_context, 
                        self.session_manager.session_manager if hasattr(self.session_manager, 'session_manager') else None,
                        previous_context=previous_context # Pass the fetched context
                    )
                except Exception as e:
                    error_msg = f"Error routing question: {str(e)}"
                    print(f"❌ {error_msg}")
                    traceback.print_exc()
                    return self._create_error_response(query_context, error_msg)
            else:
                # Basic fallback when router not available
                response_data = ResponseData(
                    query=query_data.question,
                    summary=f"Your question was classified as {classification_data.question_type}. We're working on a more detailed response."
                )
                
                response_context = Context(
                    data=response_data,
                    metadata=ContextMetadata(
                        component="MCPQueryFlowOrchestrator",
                        operation="generate_response",
                        parent_id=classification_context.metadata.context_id,
                        session_id=session_id,
                        status="success"
                    )
                )
            
            return response_context
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Error in query flow: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Create a basic error response
            error_response = ResponseData(
                query=query_data.question if hasattr(query_data, 'question') else "Unknown query",
                summary=f"Sorry, an error occurred while processing your request: {error_msg}"
            )
            
            return Context(
                data=error_response,
                metadata=error_metadata.copy().update(error_message=error_msg)
            )
    
    def _create_error_response(self, query_context: Context, error_message: str) -> Context[ResponseData]:
        """Create an error response context"""
        response_data = ResponseData(
            query=query_context.data.question,
            summary=f"I'm sorry, but I encountered an error while processing your question: {error_message}"
        )
        
        return Context(
            data=response_data,
            metadata=ContextMetadata(
                component="MCPQueryFlowOrchestrator",
                operation="error_response",
                parent_id=query_context.metadata.context_id,
                session_id=query_context.metadata.session_id,
                status="error",
                error_message=error_message
            )
        ) 