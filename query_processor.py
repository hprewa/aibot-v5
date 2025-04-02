"""Query Processor for transforming natural language questions into actionable constraints and strategies. Integrates with Gemini and BigQuery to extract constraints, generate strategies, and manage SQL query templates in the Analytics Bot."""
from typing import Dict, List, Optional, Any
from gemini_client import GeminiClient
from bigquery_client import BigQueryClient
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, MO, SU
import os
import traceback
from google.cloud import bigquery
import dspy
from dspy_tables import OrdersTable, OrdersQueryBuilder, ATPTable, ATPQueryBuilder, KPIQueryBuilder

class QueryProcessor:
    def __init__(self, gemini_client: GeminiClient, bigquery_client: BigQueryClient):
        self.gemini_client = gemini_client
        self.bigquery_client = bigquery_client
        self._initialize_sql_templates()
        self.kpi_query_builders = {
            'orders': OrdersQueryBuilder(),
            'atp': ATPQueryBuilder(),
            # Add more KPI query builders as needed
        }
        
    def _initialize_sql_templates(self):
        """Initialize SQL query templates"""
        self.sql_templates = {
            "simple_metric": """
                WITH location_data AS (
                    SELECT 
                        d.*,
                        COALESCE(m.cfc, d.location) as cfc,
                        m.spoke
                    FROM `{project}.{dataset}.{table}` d
                    LEFT JOIN `{project}.{dataset}.cfc_spoke_mapping` m
                    ON d.location = m.spoke
                    WHERE {date_col} BETWEEN @start_date AND @end_date
                )
                SELECT 
                    {date_col},
                    {group_by_cols},
                    {metric_cols}
                FROM location_data
                WHERE {location_filter}
                GROUP BY {date_col}, {group_by_cols}
                ORDER BY {date_col}
            """
        }
        
    def extract_constraints(self, question: str, previous_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract constraints using Gemini Flash thinking 2.0"""
        try:
            # Get current date and calculate time ranges
            current_date = datetime.now()
            last_week_start = (current_date - timedelta(days=7)).strftime('%Y-%m-%d')
            last_week_end = current_date.strftime('%Y-%m-%d')
            past_month_start = (current_date - timedelta(days=30)).strftime('%Y-%m-%d')
            past_month_end = current_date.strftime('%Y-%m-%d')
            
            # Get CFC-Spoke mapping
            cfc_spoke_mapping = self.bigquery_client.get_cfc_spoke_mapping()
            cfc_list = list(cfc_spoke_mapping.keys())
            spoke_list = []
            for spokes in cfc_spoke_mapping.values():
                spoke_list.extend(spokes)
            spoke_list = list(set(spoke_list))  # Remove duplicates
            
            # Get schema context
            schema_context = self._get_schema_context()

            # --- Prepare previous context string for the prompt ---
            previous_context_str = ""
            if previous_context:
                prev_q = previous_context.get('previous_question', 'N/A')
                prev_s = previous_context.get('previous_summary', 'N/A')
                # Avoid adding large results to the prompt, just mention they exist
                prev_r_exists = "Yes" if previous_context.get('previous_results') else "No"
                previous_context_str = f"""
Previous conversation turn:
- Previous Question: "{prev_q}"
- Previous Summary: "{prev_s}"
- Previous Data Exists: {prev_r_exists}

Please consider this previous context when extracting constraints for the current question. If the current question modifies the previous one (e.g., asks for a different breakdown, time period, or location), adjust the constraints accordingly. If it refers to the previous data, reflect that in the response plan.
"""
            # ----------------------------------------------------

            prompt = f"""You are a Flash Thinking 2.0 constraint extractor for a SQL query generator.
Given this question: "{question}"

{previous_context_str} # Include previous context here

Available tables and their schemas:
{schema_context}

Location hierarchy rules:
1. A location can be either a CFC or a spoke
2. Each spoke belongs to exactly one CFC
3. All CFCs together form the network
4. Use cfc_spoke_mapping table to resolve relationships

Available CFCs: {', '.join(cfc_list)}
Available Spokes: {', '.join(spoke_list)}

Time-related rules:
1. Time filters (for WHERE clause):
   - "last week" = {last_week_start} to {last_week_end}
   - "past month" = {past_month_start} to {past_month_end}
   - "yesterday" = {(current_date - timedelta(days=1)).strftime('%Y-%m-%d')}
   - "this quarter" = {(current_date - timedelta(days=90)).strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}
   - "last month" = {past_month_start} to {past_month_end}
   - "last year" = {(current_date - timedelta(days=365)).strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}
   - "this year" = {(current_date - timedelta(days=365)).strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}
   - "next month" = {(current_date + relativedelta(day=1, month=current_date.month+1)).strftime('%Y-%m-%d')} to {(current_date + relativedelta(day=1, month=current_date.month+1)).strftime('%Y-%m-%d')}
   - "next year" = {(current_date + relativedelta(day=1, month=current_date.month+1)).strftime('%Y-%m-%d')} to {(current_date + relativedelta(day=1, month=current_date.month+1)).strftime('%Y-%m-%d')}
   - For specific years (e.g., "2025"), use the full year range: "2025-01-01" to "2025-12-31"
2. Time aggregation (for GROUP BY):
   - If not explicitly specified, default to "Daily"
   - Valid values: "Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"

Please extract the constraints and format them EXACTLY as a JSON object with these fields:
{{
    "kpi": ["orders"],  # List of KPIs mentioned in the question
    "time_aggregation": "Daily/Weekly/Monthly",  # Default to "Daily" if not specified
    "time_filter": {{
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD"
    }},
    "cfc": [],  # List of CFCs mentioned, empty if none mentioned, empty for network-level questions
    "spokes": [],  # List of spokes mentioned, empty if none mentioned, "all" if all spokes needed
    "comparison_type": "null/trend/between_locations/time_periods",
    "tool_calls": [  # List of BigQuery queries that need to be executed
        {{
            "name": "string",  # Descriptive name of the query
            "description": "string",  # What this query will fetch
            "tables": ["string"],  # List of tables this query will use
            "result_id": "string"  # Unique identifier to match results back to this call
        }}
    ],
    "response_plan": {{  # Plan for Response agent to use when summarizing results
        "data_connections": [  # How data points connect
            {{
                "result_id": "string",  # ID matching a tool_call's result_id
                "purpose": "string",  # What this data is used for
                "processing_steps": ["string"],  # Steps to process this data
                "outputs": ["string"]  # What insights should be derived
            }}
        ],
        "insights": [  # Key insights to highlight
            {{
                "type": "string",  # Type of insight (trend, comparison, etc.)
                "description": "string",  # What the insight will show
                "source_result_ids": ["string"]  # Which result_ids this insight uses
            }}
        ],
        "response_structure": {{  # Structure for the final response
            "introduction": "string",  # How to introduce the response
            "main_points": ["string"],  # Main points to cover
            "context": "string",  # Additional context to provide
            "conclusion": "string"  # How to conclude the response
        }}
    }}
}}

IMPORTANT:
1. **Crucially, if `Previous Data Exists: Yes`, examine the structure/content of the previous data based on the `Previous Summary` and `Previous Question`. If the current question asks for information *already contained* within that previous data (e.g., asking for a monthly breakdown when the previous results *were* monthly, or asking for details about a specific location already present), then **DO NOT** generate a new `tool_call`. Instead, set `tool_calls: []` and update the `response_plan` to indicate the answer should be derived *directly* from the previous results.**
2.  Only generate new `tool_calls` if the current question genuinely requires fetching *new* data (e.g., a different KPI, a different primary location, a different time range, or a breakdown not present previously).
3.  Only include fields that are explicitly or implicitly mentioned in the question
4.  Return ONLY the JSON object, no other text
5.  Ensure the JSON is properly formatted with double quotes
6.  For dates, use the provided date calculations above
7.  For comparison_type:
    - Use "trend" for time-based analysis
    - Use "between_locations" for location comparisons
    - Use "null" for simple queries
8.  For cfc and spokes:
    - cfc should be empty list for network-level questions
    - spokes should be "all" if all spokes are needed
    - Both should be empty lists if not mentioned
    - Only use CFCs and spokes from the provided lists
9.  For tool_calls (if generated):
    - List all BigQuery queries that will be needed
    - Each query should have a descriptive name and purpose
    - Include all tables that will be needed for each query
    - Assign a unique result_id to each tool call
    - **IMPORTANT FOR COMPARISONS:** If `comparison_type` is "between_locations" or "time_periods", create a *separate tool call for each location or time period being compared*. Each tool call should fetch data for only *one* specific entity (e.g., one CFC, one time range). The `result_id` should clearly indicate which entity it belongs to (e.g., `orders_london_2024`, `orders_stevenage_2024`).
10. For response_plan:
    - Create a clear plan for how the Response agent should handle results
    - Link each tool call result (if any) to specific insights
    - If no tool calls are needed because the answer is in previous results, specify that in the plan (e.g., in `processing_steps` or `purpose` for a dummy `data_connection` referencing previous data).
    - Provide a structured outline for the final response
"""
            
            # --- Try calling Gemini and parsing --- 
            try:
                print(f"Extract Constraints: Using GeminiClient ID = {id(self.gemini_client)}") # LOGGING
                response = self.gemini_client.generate_content(prompt)
                if not response:
                    raise Exception("No response from Gemini")
                    
                # Clean the response to ensure it's valid JSON
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # Parse the JSON
                constraints = json.loads(cleaned_response)
                
            except json.JSONDecodeError as json_e:
                print(f"!!! Invalid JSON response from Gemini during constraint extraction: {response}")
                print(f"JSON Decode Error: {str(json_e)}")
                # Raise the exception to be caught by the outer block
                raise Exception(f"Invalid constraint format from Gemini: {str(json_e)}") 
            except Exception as gemini_e:
                print(f"!!! Error during Gemini call or response cleaning: {str(gemini_e)}")
                raise # Re-raise to be caught by outer block
            # --------------------------------------
            
            # --- Try validating and processing constraints ---    
            try:
                # Validate required fields and apply defaults
                required_fields = ["kpi", "time_aggregation", "time_filter", "cfc", "spokes", "tool_calls", "response_plan"]
                current_date = datetime.now() # Re-define for default date logic
                for field in required_fields:
                    if field not in constraints:
                        print(f"Warning: Field '{field}' missing from constraints, applying default.")
                        if field == "time_aggregation":
                            constraints[field] = "Daily"  # Default time aggregation
                        elif field == "time_filter":
                            constraints[field] = {
                                "start_date": (current_date - timedelta(days=7)).strftime('%Y-%m-%d'),
                                "end_date": current_date.strftime('%Y-%m-%d')
                            }
                        elif field in ["cfc", "spokes", "kpi", "tool_calls"]:
                            constraints[field] = []
                        elif field == "response_plan":
                            constraints[field] = {
                                "data_connections": [],
                                "insights": [],
                                "response_structure": {
                                    "introduction": "Provide an overview of the analysis",
                                    "main_points": ["Present the key findings"],
                                    "context": "Add relevant context",
                                    "conclusion": "Summarize the analysis"
                                }
                            }
                        else:
                             constraints[field] = None # Or some other appropriate default
                
                # Validate CFCs and spokes against the mapping
                if constraints.get("cfc"):
                    valid_cfcs = [cfc for cfc in constraints["cfc"] if cfc in cfc_list]
                    if valid_cfcs != constraints["cfc"]:
                        print(f"Warning: Some CFCs were not found in the mapping: {set(constraints['cfc']) - set(valid_cfcs)}")
                    constraints["cfc"] = valid_cfcs
                
                if constraints.get("spokes"):
                    if constraints["spokes"] == "all":
                        constraints["spokes"] = spoke_list
                    elif isinstance(constraints["spokes"], list):
                        valid_spokes = [spoke for spoke in constraints["spokes"] if spoke in spoke_list]
                        if valid_spokes != constraints["spokes"]:
                            print(f"Warning: Some spokes were not found in the mapping: {set(constraints['spokes']) - set(valid_spokes)}")
                        constraints["spokes"] = valid_spokes
                    else:
                        print(f"Warning: 'spokes' field has unexpected type: {type(constraints['spokes'])}, setting to empty list.")
                        constraints["spokes"] = [] # Reset if not a list or 'all'
                
                print("Constraint validation successful.")
                return constraints

            except Exception as validation_e:
                print(f"!!! Error during constraint validation/processing: {str(validation_e)}")
                # Re-raise to be caught by the outer block
                raise 
            # ----------------------------------------------
            
        except Exception as outer_e:
            print(f"!!! FAILED to extract constraints for question: '{question}'")
            print(f"Error: {str(outer_e)}")
            traceback.print_exc() # Log the full traceback here
            # Return a default empty/error structure instead of crashing
            return {
                "kpi": [],
                "time_aggregation": "Daily",
                "time_filter": {"start_date": "", "end_date": ""},
                "cfc": [],
                "spokes": [],
                "comparison_type": None,
                "tool_calls": [],
                "response_plan": {},
                "error": f"Failed to extract constraints: {str(outer_e)}"
            }
        
    def _get_schema_context(self) -> str:
        """Get schema context for all relevant tables"""
        schema_context = []
        
        # Add schema for each table
        for table_name in self.bigquery_client.tables:
            schema = self.bigquery_client.get_schema(table_name)  # Use get_schema method instead
            if schema:
                schema_context.append(f"Table {table_name}:")
                schema_context.append("Columns: " + ", ".join(schema))
                schema_context.append("")  # Empty line for readability
                
        return "\n".join(schema_context)
        
    def generate_strategy(self, question: str, constraints: Dict[str, Any]) -> str:
        """Generate a strategy using Gemini Flash thinking 2.0"""
        schema_context = self._get_schema_context()
        
        prompt = f"""You are a Flash Thinking 2.0 strategy generator for SQL queries.
Given this analytical question: "{question}"
And these constraints: {json.dumps(constraints)}

Available tables and their schemas:
{schema_context}

Generate a comprehensive strategy that will be used both for data collection and response generation.
DO NOT generate any SQL queries.
Instead, provide a structured plan in the following format:

1. Data Collection Plan
   - List each piece of data that needs to be collected
   - For each data point, specify:
     * What needs to be fetched
     * Why it's needed
     * Dependencies on other data points (if any)

2. Processing Steps
   - Detailed steps for processing the collected data
   - Any intermediate calculations or transformations
   - Order of operations and dependencies

3. Calculations Required
   - List all calculations needed
   - Formulas or methods to be used
   - Expected output format for each calculation

4. Response Structure
   - How the final response should be structured
   - Key insights to highlight
   - Comparisons to emphasize
   - Context to include
   - Suggested visualization types (if applicable)

5. Tool Calls
   For each required BigQuery query:
   - Name: A descriptive name for the query
   - Purpose: What this query will fetch and why
   - Tables: List of tables needed
   - Dependencies: Any dependencies on other queries
   - Output: Expected structure of the query results
   - Usage: How the results will be used in the final response

Format each section with clear bullet points and maintain a logical flow between sections.
The Response Agent will use this structure to generate a comprehensive and well-organized response.
"""
        
        response = self.gemini_client.generate_content(prompt)
        return response if response else "No strategy generated"
        
    def generate_sql(self, question: str, constraints: Dict[str, Any]) -> List[str]:
        """Generate SQL queries based on the question and constraints"""
        # Get schema context for the prompt
        schema_context = self._get_schema_context()
        
        # Extract KPIs and tables needed
        kpis = constraints.get("kpi", [])
        tables = set()
        for kpi in kpis:
            # Get table for each KPI
            table = self._get_table_for_kpi(kpi)
            if table:
                tables.add(table)
        
        prompt = f"""You are a Flash Thinking 2.0 SQL generator.
Generate a BigQuery SQL query to answer this question: "{question}"
Using these constraints: {json.dumps(constraints)}

Available tables and their schemas:
{schema_context}

KPIs requested: {', '.join(kpis)}
Tables needed: {', '.join(tables)}

The query should:
1. Use proper BigQuery syntax
2. Include appropriate date filters using BETWEEN
3. Handle the location hierarchy using cfc_spoke_mapping table
4. Handle necessary aggregations
5. Include ORDER BY for any time series or rankings
6. Use appropriate parameter placeholders (@param_name)
7. Be optimized for performance
8. Handle multiple KPIs if needed
9. Use appropriate joins between fact and dimension tables
10. Include proper column aliases for clarity

Return ONLY the SQL query, no explanation.
"""
        
        response = self.gemini_client.generate_content(prompt)
        if not response:
            raise Exception("Failed to generate SQL query")
            
        return [response]
        
    def _get_table_for_kpi(self, kpi: str) -> Optional[str]:
        """Get the appropriate table name for a given KPI"""
        # This mapping should be moved to a configuration file or database
        kpi_to_table = {
            "orders": "orders_drt",
            "atp": "slot_availability_drt",
            # Add more KPI to table mappings as needed
        }
        return kpi_to_table.get(kpi)
        
    def _fill_sql_template(self, template: str, constraints: Dict[str, Any]) -> str:
        """Fill in SQL template with specific constraints"""
        # Get project and dataset
        project = self.project_id
        dataset = self.dataset_id
        
        # Get KPI details
        kpi = constraints.get("kpi", [])[0] if constraints.get("kpi") else None
        if not kpi:
            raise ValueError("No KPI specified in constraints")
        
        table = self._get_table_for_kpi(kpi)
        if not table:
            raise ValueError(f"No table found for KPI: {kpi}")
        
        # Build location filter
        location_filter = self._build_location_filter(constraints)
        
        # Get date column and metric columns
        date_col = self._get_date_column(table)
        metric_cols = self._get_metric_columns(table, kpi)
        
        # Determine group by columns
        group_by_cols = self._get_group_by_columns(constraints)
        
        # Fill in the template
        query = template.format(
            project=project,
            dataset=dataset,
            table=table,
            date_col=date_col,
            location_filter=location_filter,
            group_by_cols=group_by_cols,
            metric_cols=metric_cols
        )
        
        return query.strip()
        
    def _build_location_filter(self, constraints: Dict[str, Any]) -> str:
        """Build the location filter clause based on constraints"""
        cfcs = constraints.get("cfc", [])
        spokes = constraints.get("spokes", [])
        
        filters = []
        if cfcs:
            filters.append(f"cfc IN UNNEST(@cfcs)")
        if spokes and spokes != "all":
            filters.append(f"spoke IN UNNEST(@spokes)")
        
        return " AND ".join(filters) if filters else "1=1"
        
    def _get_date_column(self, table: str) -> str:
        """Get the date column name for a table"""
        # This mapping should be moved to a configuration file or database
        date_columns = {
            "orders_drt": "delivery_date",
            "slot_availability_drt": "slot_date",
            # Add more table to date column mappings as needed
        }
        return date_columns.get(table, "date")
        
    def _get_metric_columns(self, table: str, kpi: str) -> str:
        """Get the metric columns to select for a KPI"""
        # This mapping should be moved to a configuration file or database
        metric_columns = {
            "orders": "SUM(orders) as total_orders",
            "atp": "AVG(atp_score) as avg_atp, COUNT(*) as total_slots",
            # Add more KPI to metric column mappings as needed
        }
        return metric_columns.get(kpi, f"COUNT(*) as total_{kpi}")
        
    def _get_group_by_columns(self, constraints: Dict[str, Any]) -> str:
        """Get the group by columns based on constraints"""
        group_by = []
        
        # Add location-based grouping
        if constraints.get("cfc"):
            group_by.append("cfc")
        if constraints.get("spokes") and constraints["spokes"] != "all":
            group_by.append("spoke")
        
        # Add time-based grouping if specified
        time_agg = constraints.get("time_aggregation", "Daily")
        if time_agg != "Daily":
            group_by.append(f"DATE_TRUNC({time_agg})")
        
        return ", ".join(group_by) if group_by else "1"
        
    def generate_summary(self, question: str, results: Dict[str, list], constraints: Dict[str, Any]) -> str:
        """Generate summary using structured response plan"""
        # Extract the response plan from constraints
        response_plan = constraints.get("response_plan", {})
        tool_calls = constraints.get("tool_calls", [])
        
        # Map result_ids to their actual results
        result_mapping = {}
        for i, (result_id, result_data) in enumerate(results.items()):
            result_mapping[result_id] = result_data
        
        # Create context for the response agent
        response_context = {
            "question": question,
            "results": result_mapping,
            "response_plan": response_plan,
            "tool_calls": tool_calls
        }
        
        prompt = f"""You are a Flash Thinking 2.0 data analyst responsible for summarizing query results.
Given this analytical question: "{question}"

The following data has been retrieved:
{json.dumps(result_mapping, indent=2)}

Please follow this response plan to generate a comprehensive summary:

Data Connections:
{json.dumps(response_plan.get("data_connections", []), indent=2)}

Key Insights to Highlight:
{json.dumps(response_plan.get("insights", []), indent=2)}

Response Structure:
{json.dumps(response_plan.get("response_structure", {}), indent=2)}

Original Tool Calls:
{json.dumps(tool_calls, indent=2)}

Your task is to:
1. Analyze the data according to the provided connections and processing steps
2. Extract the key insights as outlined in the response plan
3. Structure your response according to the provided template
4. Include relevant metrics, comparisons, and context
5. Make the response clear, concise, and actionable for a business audience

Format your response as a well-structured summary with sections, bullet points where appropriate, and clear explanations of trends and insights.
"""
        
        response = self.gemini_client.generate_content(prompt)
        return response if response else "No summary could be generated"
        
    def generate_sql_for_tool_call(self, question: str, constraints: Dict[str, Any], tool_call: Dict[str, Any]) -> str:
        """Generate SQL for a specific tool call"""
        schema_context = self._get_schema_context()
        
        # Extract tool call information
        name = tool_call.get("name", "Unnamed Query")
        description = tool_call.get("description", "")
        tables = tool_call.get("tables", [])
        
        # Extract other constraints
        time_filter = constraints.get("time_filter", {})
        start_date = time_filter.get("start_date", "")
        end_date = time_filter.get("end_date", "")
        cfcs = constraints.get("cfc", [])
        spokes = constraints.get("spokes", [])
        
        prompt = f"""You are a Flash Thinking 2.0 SQL generator.
Generate a specific BigQuery SQL query for the following tool call:

Original Question: "{question}"

Tool Call:
- Name: {name}
- Description: {description}
- Tables: {', '.join(tables)}

Constraints:
- Time Period: {start_date} to {end_date}
- CFCs: {', '.join(cfcs) if cfcs else 'All'}
- Spokes: {', '.join(spokes) if isinstance(spokes, list) else spokes}

Available tables and their schemas:
{schema_context}

Your task is to:
1. Generate a precise SQL query that fetches EXACTLY the data needed for this specific tool call
2. Include appropriate filters for time period, locations (CFCs and spokes)
3. Select only necessary columns to avoid excessive data transfer
4. Use proper BigQuery syntax and optimize for performance
5. Include appropriate comments to explain complex parts of the query

Return ONLY the SQL query, no explanation.
"""
        
        response = self.gemini_client.generate_content(prompt)
        return response if response else "SELECT 1 -- Error generating SQL query"

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        try:
            print(f"Starting update_session for {session_id} with updates: {list(updates.keys())}")
            
            # Get the current session data
            current_session = self.get_session(session_id)
            if not current_session:
                print(f"ERROR: Session {session_id} not found")
                raise ValueError(f"Session {session_id} not found")
            
            print(f"Retrieved current session: {session_id}")
            
            # Create a new session data dictionary with the updated information
            updated_session = current_session.copy()
            updated_session.update(updates)
            
            # Set the updated_at timestamp
            updated_session["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert complex objects to JSON strings for storage
            for key, value in list(updated_session.items()):
                if key in ["constraints", "response_plan", "strategy", "tool_call_results", "results"]:
                    if value is not None and not isinstance(value, str):
                        try:
                            print(f"Serializing {key} of type {type(value)}")
                            updated_session[key] = json.dumps(value, cls=DateTimeEncoder)
                            print(f"Successfully serialized {key}")
                        except Exception as e:
                            print(f"ERROR serializing {key}: {str(e)}")
                            traceback.print_exc()
                            # Store a simplified version
                            updated_session[key] = json.dumps({"error": f"Failed to serialize: {str(e)}"})
            
            # Insert the updated session data as a new row
            print(f"Inserting updated session {session_id}")
            
            # Call the BigQuery client to insert the row
            insert_result = self.bigquery_client.insert_row(self.table_id, updated_session)
            if insert_result:
                print(f"ERROR: Session update insert failed: {insert_result}")
            else:
                print(f"Successfully inserted updated session {session_id}")
            
        except Exception as e:
            print(f"CRITICAL ERROR updating session {session_id}: {str(e)}")
            traceback.print_exc()

    def process_kpi_query(self, kpi: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a KPI query using DSPy query builders
        """
        try:
            print(f"\nProcessing {kpi} query with parameters:", json.dumps(params, indent=2))
            
            # Get the appropriate query builder
            query_builder = self.kpi_query_builders.get(kpi)
            if not query_builder:
                error_msg = f"Unsupported KPI type: {kpi}"
                print(f"ERROR: {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg,
                    "data": None
                }
            
            # Prepare query parameters
            query_params = {}
            
            # Handle date parameters
            if 'time_filter' in params:
                time_filter = params['time_filter']
                # Convert string dates to datetime.date objects
                if isinstance(time_filter.get('start_date'), str):
                    query_params['start_date'] = datetime.strptime(time_filter['start_date'], '%Y-%m-%d').date()
                else:
                    query_params['start_date'] = time_filter.get('start_date')
                    
                if isinstance(time_filter.get('end_date'), str):
                    query_params['end_date'] = datetime.strptime(time_filter['end_date'], '%Y-%m-%d').date()
                else:
                    query_params['end_date'] = time_filter.get('end_date')
            
            # Handle location parameters
            if 'cfc' in params and params['cfc']:
                if isinstance(params['cfc'], list):
                    # Ensure all values are strings
                    query_params['cfc'] = [str(cfc).lower() for cfc in params['cfc']]
                else:
                    query_params['cfc'] = str(params['cfc']).lower()
                    
            if 'spokes' in params and params['spokes']:
                if params['spokes'] == 'all':
                    query_params['spoke'] = 'all'
                elif isinstance(params['spokes'], list):
                    # Ensure all values are strings
                    query_params['spoke'] = [str(spoke).lower() for spoke in params['spokes']]
                else:
                    query_params['spoke'] = str(params['spokes']).lower()
            
            # Handle aggregation
            if 'time_aggregation' in params:
                query_params['aggregation'] = str(params['time_aggregation']).lower()
            
            print(f"Prepared query parameters:", json.dumps(query_params, default=str, indent=2))
            
            # Get date column based on aggregation
            date_col = query_builder._get_date_column(query_params.get('aggregation', 'daily'))
            print(f"Using date column: {date_col}")
            
            # Build location filter
            try:
                location_filter = query_builder._build_location_filter(query_params)
                print(f"Built location filters: {location_filter}")
            except Exception as e:
                print(f"Error building location filter: {str(e)}")
                raise

            # Execute count query first
            try:
                count_query = f"""
                    WITH location_data AS (
                        SELECT
                            o.*,
                            COALESCE(LOWER(m.cfc), LOWER(o.cfc)) as normalized_cfc,
                            LOWER(m.spoke) as normalized_spoke
                        FROM `{query_builder.project_id}.{query_builder.dataset_id}.{query_builder._table_name}` o
                        LEFT JOIN `{query_builder.project_id}.{query_builder.dataset_id}.cfc_spoke_mapping` m
                        ON LOWER(o.cfc) = LOWER(m.spoke)
                        WHERE DATE({date_col}) BETWEEN DATE(@start_date) AND DATE(@end_date)
                    )
                    SELECT COUNT(*) as count
                    FROM location_data
                    WHERE {location_filter}
                """
                
                print("\nExecuting count query:\n")
                print(count_query)
                print("\nWith parameters:", json.dumps(query_params, default=str))
                
                count_result = self.bigquery_client.execute_query(count_query, query_params)
                if not count_result or count_result[0]['count'] == 0:
                    error_msg = f"No {kpi} data found for the specified parameters"
                    print(f"WARNING: {error_msg}")
                    return {
                        "status": "error",
                        "message": error_msg,
                        "data": None
                    }
                
                print(f"Found {count_result[0]['count']} matching records")

                # Now execute the main query
                query = query_builder.build_query(**query_params)
                print(f"\nExecuting main query:\n{query}")
                
                # Execute query
                results = self.bigquery_client.execute_query(query, query_params)
                print(f"Main query returned {len(results) if results else 0} rows")
                
                if results:
                    # Calculate summary statistics
                    summary = self._calculate_summary_stats(results, kpi)
                    
                    result = {
                        "status": "success",
                        "message": f"Successfully retrieved {kpi} data",
                        "data": {
                            "summary": summary,
                            "data": results
                        }
                    }
                    
                    print(f"Processed successfully: {result['message']}")
                    print("Summary statistics:", json.dumps(summary, indent=2))
                    
                    return result
                else:
                    error_msg = f"Main query returned no results despite count > 0"
                    print(f"WARNING: {error_msg}")
                    return {
                        "status": "error",
                        "message": error_msg,
                        "data": None
                    }
                    
            except Exception as query_error:
                error_msg = f"Error executing {kpi} query: {str(query_error)}"
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                return {
                    "status": "error",
                    "message": error_msg,
                    "data": None
                }
                
        except Exception as e:
            error_msg = f"Error processing {kpi} query: {str(e)}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": error_msg,
                "data": None
            }
            
    def _calculate_summary_stats(self, data_points: List[Dict[str, Any]], kpi: str) -> Dict[str, Any]:
        """Calculate summary statistics using DSPy table definitions"""
        try:
            # Get metric fields based on KPI type
            if kpi == 'orders':
                metric_fields = ['total_orders']
            elif kpi == 'atp':
                metric_fields = ['avg_atp_score', 'total_slots', 'available_slots']
            else:
                metric_fields = []
            
            if not metric_fields:
                print(f"WARNING: No metric fields found for {kpi}")
                return {"error": f"No metric fields found for {kpi}"}
            
            # Calculate statistics for each metric
            summary = {}
            for metric in metric_fields:
                values = [float(point[metric]) for point in data_points if metric in point]
                if values:
                    summary[metric] = {
                        "total": sum(values),
                        "average": round(sum(values) / len(values), 2),
                        "max": {
                            "value": max(values),
                            "details": next(p for p in data_points if float(p[metric]) == max(values))
                        },
                        "min": {
                            "value": min(values),
                            "details": next(p for p in data_points if float(p[metric]) == min(values))
                        }
                    }
            
            # Add data points count
            summary["data_points"] = len(data_points)
            
            # Add time range if available
            if kpi == 'orders':
                date_field = 'delivery_date'
            elif kpi == 'atp':
                date_field = 'slot_date'
            else:
                date_field = None
            
            if date_field:
                dates = [point[date_field] for point in data_points if date_field in point]
                if dates:
                    summary["date_range"] = {
                        "start": min(dates),
                        "end": max(dates)
                    }
            
            return summary
            
        except Exception as e:
            print(f"Error calculating summary statistics: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

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