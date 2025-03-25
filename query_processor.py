"""Query Processor for transforming natural language questions into actionable constraints and strategies. Integrates with Gemini and BigQuery to extract constraints, generate strategies, and manage SQL query templates in the Analytics Bot."""
from typing import Dict, List, Optional, Any
from gemini_client import GeminiClient
from bigquery_client import BigQueryClient
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, MO, SU
import os
import traceback

class QueryProcessor:
    def __init__(self, gemini_client: GeminiClient, bigquery_client: BigQueryClient):
        self.gemini_client = gemini_client
        self.bigquery_client = bigquery_client
        self._initialize_sql_templates()
        
    def _initialize_sql_templates(self):
        """Initialize SQL query templates"""
        self.sql_templates = {
            "simple_count": """
                WITH location_data AS (
                    SELECT 
                        o.*,
                        COALESCE(m.cfc, o.location) as cfc,
                        m.spoke
                    FROM `{project}.{dataset}.orders_drt` o
                    LEFT JOIN `{project}.{dataset}.cfc_spoke_mapping` m
                    ON o.location = m.spoke
                    WHERE order_date BETWEEN @start_date AND @end_date
                )
                SELECT 
                    COUNT(*) as order_count
                FROM location_data
                WHERE {location_filter}
            """,
            "location_comparison": """
                WITH location_data AS (
                    SELECT 
                        o.*,
                        COALESCE(m.cfc, o.location) as cfc,
                        m.spoke
                    FROM `{project}.{dataset}.orders_drt` o
                    LEFT JOIN `{project}.{dataset}.cfc_spoke_mapping` m
                    ON o.location = m.spoke
                    WHERE order_date BETWEEN @start_date AND @end_date
                )
                SELECT 
                    {group_by_cols},
                    COUNT(*) as order_count
                FROM location_data
                WHERE {location_filter}
                GROUP BY {group_by_cols}
                ORDER BY order_count DESC
            """,
            "daily_trend": """
                WITH location_data AS (
                    SELECT 
                        o.*,
                        COALESCE(m.cfc, o.location) as cfc,
                        m.spoke
                    FROM `{project}.{dataset}.orders_drt` o
                    LEFT JOIN `{project}.{dataset}.cfc_spoke_mapping` m
                    ON o.location = m.spoke
                    WHERE order_date BETWEEN @start_date AND @end_date
                )
                SELECT 
                    DATE(order_date) as date,
                    {group_by_cols},
                    COUNT(*) as order_count
                FROM location_data
                WHERE {location_filter}
                GROUP BY date, {group_by_cols}
                ORDER BY date
            """
        }
        
    def extract_constraints(self, question: str) -> Dict[str, Any]:
        """Extract constraints using Gemini Flash thinking 2.0"""
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
        
        prompt = f"""You are a Flash Thinking 2.0 constraint extractor for a SQL query generator.
Given this question: "{question}"

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
1. Only include fields that are explicitly or implicitly mentioned in the question
2. Return ONLY the JSON object, no other text
3. Ensure the JSON is properly formatted with double quotes
4. For dates, use the provided date calculations above
5. For comparison_type:
   - Use "trend" for time-based analysis
   - Use "between_locations" for location comparisons
   - Use "null" for simple queries
6. For cfc and spokes:
   - cfc should be empty list for network-level questions
   - spokes should be "all" if all spokes are needed
   - Both should be empty lists if not mentioned
   - Only use CFCs and spokes from the provided lists
7. For tool_calls:
   - List all BigQuery queries that will be needed
   - Each query should have a descriptive name and purpose
   - Include all tables that will be needed for each query
   - Assign a unique result_id to each tool call
8. For response_plan:
   - Create a clear plan for how the Response agent should handle results
   - Link each tool call result to specific insights
   - Provide a structured outline for the final response
"""
        
        try:
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
            
            # Validate required fields
            required_fields = ["kpi", "time_aggregation", "time_filter", "cfc", "spokes", "tool_calls", "response_plan"]
            for field in required_fields:
                if field not in constraints:
                    if field == "time_aggregation":
                        constraints[field] = "Daily"  # Default time aggregation
                    elif field == "time_filter":
                        constraints[field] = {
                            "start_date": (current_date - timedelta(days=7)).strftime('%Y-%m-%d'),
                            "end_date": current_date.strftime('%Y-%m-%d')
                        }
                    elif field in ["cfc", "spokes"]:
                        constraints[field] = []
                    elif field == "tool_calls":
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
                        constraints[field] = []
            
            # Validate CFCs and spokes against the mapping
            if constraints["cfc"]:
                valid_cfcs = [cfc for cfc in constraints["cfc"] if cfc in cfc_list]
                if valid_cfcs != constraints["cfc"]:
                    print(f"Warning: Some CFCs were not found in the mapping: {set(constraints['cfc']) - set(valid_cfcs)}")
                constraints["cfc"] = valid_cfcs
            
            if constraints["spokes"]:
                if constraints["spokes"] == "all":
                    constraints["spokes"] = spoke_list
                else:
                    valid_spokes = [spoke for spoke in constraints["spokes"] if spoke in spoke_list]
                    if valid_spokes != constraints["spokes"]:
                        print(f"Warning: Some spokes were not found in the mapping: {set(constraints['spokes']) - set(valid_spokes)}")
                    constraints["spokes"] = valid_spokes
            
            return constraints
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response from Gemini: {response}")
            raise Exception("Invalid constraint format returned")
        except Exception as e:
            print(f"Error extracting constraints: {str(e)}")
            raise
            
    def _get_schema_context(self) -> str:
        """Get formatted schema information for all relevant tables"""
        schema_info = []
        
        for table_name in ["orders", "slot_availability", "cfc_spoke_mapping"]:
            schema = self.bigquery_client.schema_cache.get(table_name)
            if schema:
                columns = [f"- {field.name} ({field.field_type})" for field in schema]
                schema_info.append(f"{table_name} table:\n" + "\n".join(columns))
        
        return "\n\n".join(schema_info)
        
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
        # First check if we can use a template
        template_name = self._get_template_name(constraints)
        if template_name:
            return [self._fill_sql_template(self.sql_templates[template_name], constraints)]
            
        # If no template fits, generate new SQL using Gemini
        schema_context = self._get_schema_context()
        
        prompt = f"""You are a Flash Thinking 2.0 SQL generator.
Generate a BigQuery SQL query to answer this question: "{question}"
Using these constraints: {json.dumps(constraints)}

Available tables and their schemas:
{schema_context}

The query should:
1. Use proper BigQuery syntax
2. Include appropriate date filters using BETWEEN
3. Handle the location hierarchy using cfc_spoke_mapping table
4. Handle necessary aggregations
5. Include ORDER BY for any time series or rankings
6. Use appropriate parameter placeholders (@param_name)
7. Be optimized for performance

Return ONLY the SQL query, no explanation.
"""
        
        response = self.gemini_client.generate_content(prompt)
        if not response:
            raise Exception("Failed to generate SQL query")
            
        return [response]
        
    def _get_template_name(self, constraints: Dict[str, Any]) -> Optional[str]:
        """Determine which SQL template to use based on constraints"""
        if not constraints.get("comparison_type"):
            return "simple_count"
        elif constraints["comparison_type"] == "between_locations":
            return "location_comparison"
        elif constraints["comparison_type"] == "trend":
            return "daily_trend"
        return None
        
    def _fill_sql_template(self, template: str, constraints: Dict[str, Any]) -> str:
        """Fill in SQL template with specific constraints"""
        # Get project and dataset from environment
        project = os.getenv('PROJECT_ID')
        dataset = os.getenv('DATASET_ID')
        
        # Build location filter based on type
        location_type = constraints.get('location_type', 'CFC')
        locations = constraints.get('locations', [])
        
        if location_type == 'Network':
            location_filter = "1=1"  # No filtering needed
        elif location_type == 'CFC':
            location_filter = "cfc IN UNNEST(@locations)" if locations else "1=1"
        else:  # Spoke
            location_filter = "spoke IN UNNEST(@locations)" if locations else "1=1"
            
        # Determine group by columns
        group_by_cols = ", ".join(constraints.get('group_by_cols', []))
        if not group_by_cols:
            if location_type == 'CFC':
                group_by_cols = "cfc"
            elif location_type == 'Spoke':
                group_by_cols = "spoke, cfc"
                
        # Fill in the template
        query = template.format(
            project=project,
            dataset=dataset,
            location_filter=location_filter,
            group_by_cols=group_by_cols
        )
        
        return query.strip()
        
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