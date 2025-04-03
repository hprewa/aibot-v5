"""
Router module for MCP framework.

This module contains the router class that routes questions based on their classification
to the appropriate handlers.
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
import json
import logging
import uuid
from .protocol import Context, ContextMetadata
from .models import ClassificationData, SessionData, ResponseData
import traceback
import concurrent.futures
import threading
import sqlite3
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import required components
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent, DateTimeEncoder
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from graph_generator import GraphGenerator
from session_manager_v2 import SessionManagerV2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple in-memory cache for testing purposes
session_cache = {}

# Global variable to maintain warm instances
_warm_clients = None
_warm_lock = threading.Lock()

class LazyBigQueryClient:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = BigQueryClient()
        return cls._instance

class PersistentSchemaCache:
    CACHE_FILE = "schema_cache.sqlite"
    
    @classmethod
    def get_schema(cls, table_name):
        with sqlite3.connect(cls.CACHE_FILE) as conn:
            cursor = conn.cursor()
            schema = cursor.execute(
                "SELECT schema FROM schemas WHERE table_name = ?",
                (table_name,)
            ).fetchone()
            if schema:
                return json.loads(schema[0])
        return None

class Clients:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.bigquery = None
                cls._instance.gemini = None
            return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize(cls):
        instance = cls.get_instance()
        with cls._lock:
            if instance.bigquery is None:
                print("Initializing shared BigQuery client...")
                instance.bigquery = BigQueryClient()
            if instance.gemini is None:
                print("Initializing shared Gemini client...")
                instance.gemini = GeminiClient()
        return instance

class MCPRouter:
    """
    Router that determines how to process a question based on its classification.
    """
    
    # Classification categories that require full processing pipeline
    FULL_PIPELINE_CATEGORIES = [
        "KPI Extraction", 
        "Comparitive Analysis",
        "Trend Analysis"
    ]
    
    # Classification categories planned for future implementation
    PLANNED_CATEGORIES = [
        "Forecasting",
        "Anamoly Detection",
        "Operational efficiency",
        "Multi-Intent Questions",
        "Nested or Multi-Step",
        "Constrained Based Optimization"
    ]
    
    # Classification categories that are not supported
    UNSUPPORTED_CATEGORIES = [
        "Unsupported/Random Questions",
        "KPI Constraints & Unknown KPIs",
        "Data Availability Issues",
        "Action-Based Questions"
    ]
    
    # Classification categories that require user clarification
    CLARIFICATION_CATEGORIES = [
        "Unsupported NLP Constructs",
        "Ambiguous Questions"
    ]
    
    # Special handling categories
    SMALL_TALK_CATEGORY = "Small Talk"
    FEEDBACK_CATEGORY = "Feedback"
    AMBIGUOUS_CATEGORY = "Ambiguous Questions"
    UNSUPPORTED_CONSTRUCT_CATEGORY = "Unsupported NLP Constructs"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Get shared client instances
        clients = Clients.get_instance()
        if not clients.bigquery or not clients.gemini:
            clients.initialize()
        
        # Use shared clients
        self.query_processor = QueryProcessor(clients.gemini, clients.bigquery)
        self.query_agent = QueryAgent(clients.gemini)
        self.response_agent = ResponseAgent(clients.gemini)
        self.graph_generator = GraphGenerator()
        
        # Log instance IDs during initialization
        self.logger.info(f"Router Init: Router ID = {id(self)}")
        self.logger.info(f"Router Init: QueryProcessor ID = {id(self.query_processor)}")
        self.logger.info(f"Router Init: QueryAgent ID = {id(self.query_agent)}")
        self.logger.info(f"Router Init: ResponseAgent ID = {id(self.response_agent)}")
        self.logger.info(f"Router Init: GraphGenerator ID = {id(self.graph_generator)}")
        if hasattr(self.query_processor, 'gemini_client'):
             self.logger.info(f"Router Init: QP GeminiClient ID = {id(self.query_processor.gemini_client)}")
        if hasattr(self.query_agent, 'gemini_client'):
             self.logger.info(f"Router Init: QA GeminiClient ID = {id(self.query_agent.gemini_client)}")
        if hasattr(self.response_agent, 'gemini_client'):
             self.logger.info(f"Router Init: RA GeminiClient ID = {id(self.response_agent.gemini_client)}")
        
        # Define routing map
        self.route_map = {
            "kpi extraction": self._handle_full_pipeline,
            "comparitive analysis": self._handle_full_pipeline,
            "trend analysis": self._handle_full_pipeline,
            "anomaly detection": self._handle_full_pipeline,
            "categorical breakdown": self._handle_full_pipeline,
            "ranking": self._handle_full_pipeline,
            "other analytics": self._handle_full_pipeline,
            "follow-up question": self._handle_follow_up,
            "forecasting": self._handle_planned_feature,
            "unsupported/random questions": self._handle_unsupported_feature,
            "ambiguous questions": self._handle_ambiguous_question,
            "unsupported nlp constructs": self._handle_unsupported_construct,
            "data source": self._handle_clarification_needed,
            "clarification needed": self._handle_clarification_needed,
            "small talk": self._handle_small_talk,
            "feedback": self._handle_feedback
        }
        
    def get_cached_session(self, session_id):
        """Get a session from the in-memory cache"""
        return session_cache.get(session_id)
        
    def store_cached_session(self, session_id, session_data):
        """Store a session in the in-memory cache"""
        session_cache[session_id] = session_data
        self.logger.info(f"Stored session {session_id} in cache")
        self.logger.info(f"Cache now contains {len(session_cache)} sessions")
    
    def route(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable] = None):
        """
        Route the question to the appropriate handler based on classification.
        
        Args:
            classification_context: Context containing the question classification
            session_manager: Optional session manager for updating session data
            previous_context: Optional dictionary containing previous question, summary, results
            send_callback: Optional callback function to send graph filepath
            
        Returns:
            Context with appropriate response data
        """
        classification = classification_context.data
        question_type = classification.question_type.lower()
        
        self.logger.info(f"Routing question type: {question_type} with confidence {classification.confidence}")
        print(f"[DEBUG PRINT] Router received send_callback: {send_callback is not None}")
        if send_callback:
            print(f"[DEBUG PRINT] send_callback type: {type(send_callback)}")
        
        # Route to appropriate handler based on classification
        if question_type in self.route_map:
            # Pass previous_context and send_callback to the handler
            return self.route_map[question_type](classification_context, session_manager, previous_context, send_callback)
            
        else:
            # Default to full pipeline for unknown types
            self.logger.warning(f"Unknown question type: {question_type}. Defaulting to full pipeline.")
            # Pass previous_context and send_callback to the default handler too
            return self._handle_full_pipeline(classification_context, session_manager, previous_context, send_callback)
    
    def _create_response_context(self, classification_context: Context[ClassificationData], response_data: ResponseData):
        """Create a response context with appropriate metadata"""
        metadata = ContextMetadata(
            context_id=str(uuid.uuid4()),
            parent_id=classification_context.metadata.context_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            component="MCPRouter",
            operation="route_response",
            status="success",
            session_id=classification_context.metadata.session_id
        )
        
        return Context(data=response_data, metadata=metadata)
    
    def _handle_full_pipeline(self, classification_context: Context[ClassificationData], session_manager: Optional[SessionManagerV2] = None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable] = None):
        """Process through the complete MCP pipeline."""
        # --- Use print for debugging ---
        session_id = classification_context.metadata.session_id # Get session_id early
        print(f"[ROUTER PRINT DEBUG {session_id}] STARTING _handle_full_pipeline.")
        print(f"[DEBUG PRINT] _handle_full_pipeline received send_callback: {send_callback is not None}")
        
        start_time = time.time()
        classification = classification_context.data
        question_type = classification.question_type.lower()
        self.logger.info(f"Handling full pipeline for: {classification.question} (Type: {question_type})")
        # Log if previous context is present
        if previous_context:
            self.logger.info("Received previous context for full pipeline.")
            self.logger.info(f"  Prev Q: {previous_context.get('previous_question', 'N/A')[:100]}...")
        
        # Get session ID and user ID
        session_id = classification_context.metadata.session_id
        user_id = classification_context.metadata.user_id or getattr(classification_context.data, 'user_id', 'anonymous')
        
        try:
            # --- Re-verify QueryProcessor instance ---
            if not hasattr(self, 'query_processor') or not self.query_processor:
                self.logger.error("!!! QueryProcessor not initialized correctly in router instance! Re-initializing...")
                clients = Clients.get_instance()
                if not clients.bigquery or not clients.gemini:
                    clients.initialize()
                self.query_processor = QueryProcessor(clients.gemini, clients.bigquery)
            elif not hasattr(self.query_processor, 'extract_constraints'):
                 self.logger.error(f"!!! QueryProcessor exists but LACKS extract_constraints method! Type: {type(self.query_processor)}")
                 # Attempt re-initialization as a potential fix
                 clients = Clients.get_instance()
                 if not clients.bigquery or not clients.gemini:
                    clients.initialize()
                 self.query_processor = QueryProcessor(clients.gemini, clients.bigquery)
                 if not hasattr(self.query_processor, 'extract_constraints'):
                     # If still missing after re-init, raise to stop
                     raise AttributeError("QueryProcessor still lacks extract_constraints after re-initialization.")
            # ------------------------------------------
            
            # First, ensure the session exists with correct user_id and question
            if session_manager:
                current_session = session_manager.get_session(session_id)
                if not current_session:
                    # Create new session with correct user_id and question
                    session_manager.create_session(user_id, classification.question, session_id)
                    self.logger.info(f"Created new session {session_id} for user {user_id} with question: {classification.question}")
            
            # Step 1: Extract constraints, passing previous_context
            self.logger.info("Extracting constraints...")
            # Pass previous_context to extract_constraints
            constraints = self.query_processor.extract_constraints(
                classification.question,
                previous_context=previous_context # Pass the argument here
            )
            self.logger.info(f"Extracted constraints: {json.dumps(constraints, indent=2)}")
            
            # Add previous context to the constraints dict itself so it's available later if needed
            if previous_context:
                constraints['previous_question'] = previous_context.get('previous_question')
                constraints['previous_summary'] = previous_context.get('previous_summary')
                constraints['previous_results'] = previous_context.get('previous_results')

            # Step 2: Generate SQL queries
            self.logger.info("Generating SQL queries...")
            tool_calls = []
            tool_call_status = {} # Initialize status dictionary here
            
            for tool_call in constraints.get("tool_calls", []):
                tool_name = tool_call.get("name", "unknown_tool")
                sql = None 
                
                # --- Try generating SQL ---
                try:
                    self.logger.info(f"Attempting SQL generation for tool call: {tool_name}")
                    sql = self.query_agent.generate_query(tool_call, constraints)
                    if not sql:
                        self.logger.error(f"SQL generation for {tool_name} returned empty result (None). Marking as failed.")
                        tool_call_status[tool_name] = "failed"
                        continue # Skip to next tool call
                except Exception as gen_e:
                    self.logger.error(f"!!! Exception during SQL generation for {tool_name}: {str(gen_e)}")
                    self.logger.error(traceback.format_exc()) 
                    tool_call_status[tool_name] = "failed" 
                    continue # Skip to next tool call
                    
                # --- If SQL was generated, try appending ---
                if sql:
                    try:
                        self.logger.info(f"SQL generated for {tool_name}. Appending to tool_calls list...")
                        tool_calls.append({
                            "name": tool_name,
                            "sql": sql,
                            "result_id": tool_call["result_id"]
                        })
                        # If append succeeds, status remains unset (implicitly pending/success unless execution fails later)
                        self.logger.info(f"Append successful for {tool_name}.")
                    except Exception as append_e:
                        self.logger.error(f"!!! Exception during append for {tool_name}: {str(append_e)}")
                        self.logger.error(traceback.format_exc()) 
                        tool_call_status[tool_name] = "failed" # Mark as failed if append fails
                        continue
                    
                    # --- Try logging the generated SQL (non-critical) ---
                    try:
                        # Simplify logging to reduce potential errors
                        self.logger.info(f"Successfully generated and appended SQL for {tool_name}.") 
                    except Exception as log_e:
                        self.logger.error(f"!!! Exception while logging success for {tool_name}: {str(log_e)}")
                        # Do not mark as failed or continue just for logging error
            
            # --- After loop ---
            self.logger.info(f"Completed SQL generation loop. Tool calls list count: {len(tool_calls)}")
            self.logger.info(f"Tool call status after Step 2 (only failures marked): {json.dumps(tool_call_status)}")
            
            # Step 3: Execute queries and collect results
            self.logger.info("Executing queries...")
            results = {}
            # tool_call_status is already initialized and updated in Step 2 loop
            for tool_call in tool_calls: # Iterate over successfully generated SQL
                tool_name = tool_call["name"]
                # Skip execution if already marked as failed in Step 2
                if tool_call_status.get(tool_name) == "failed":
                    self.logger.warning(f"Skipping execution for {tool_name} as it failed during generation.")
                    continue
                
                try:
                    self.logger.info(f"Executing query {tool_name} with SQL: {tool_call['sql']}")
                    
                    # Prepare parameters for the query
                    query_params = {}
                    time_filter = constraints.get("time_filter", {})
                    if time_filter.get("start_date"):
                        query_params["start_date"] = time_filter["start_date"]
                    if time_filter.get("end_date"):
                        query_params["end_date"] = time_filter["end_date"]
                    if constraints.get("cfc"):
                        query_params["cfcs"] = constraints["cfc"] # Use 'cfcs' as expected by BQ client parameter logic for lists
                    if constraints.get("spokes") and constraints["spokes"] != "all":
                        query_params["spokes"] = constraints["spokes"] # Use 'spokes' as expected by BQ client
                    
                    self.logger.info(f"Executing query {tool_name} via BigQueryClient with params: {query_params}")
                    # Call the correct execute_query method on the BigQueryClient instance
                    result = self.query_processor.bigquery_client.execute_query(
                        tool_call["sql"],
                        params=query_params
                    )
                    self.logger.info(f"Query {tool_name} returned {len(result)} rows")
                    
                    # Format datetime objects and convert NumPy types in results
                    formatted_result = []
                    for row in result:
                        formatted_row = {}
                        for key, value in row.items():
                            if isinstance(value, datetime):
                                formatted_row[key] = value.isoformat()
                            # Explicitly convert potential NumPy types
                            elif hasattr(value, 'item'): # Check if it has numpy's item() method
                                try:
                                    formatted_row[key] = value.item() # Convert numpy types (int64, float64, etc.) to standard python types
                                except Exception:
                                    # Fallback if item() fails for some reason
                                    formatted_row[key] = str(value) if value is not None else None
                            elif isinstance(value, (int, float, str, bool)) or value is None:
                                # Keep standard types as is
                                formatted_row[key] = value
                            else:
                                # Convert other non-standard types to string as a fallback
                                formatted_row[key] = str(value)
                        formatted_result.append(formatted_row)
                    
                    # Determine the correct location for *this* tool_call's summary
                    current_tool_location = None
                    tool_call_name = tool_call.get("name", "").lower()
                    # Simple heuristic: Check if known locations from constraints are in the tool call name
                    known_cfcs = constraints.get("cfc", [])
                    known_spokes = constraints.get("spokes", [])
                    possible_locations = known_cfcs + (known_spokes if isinstance(known_spokes, list) else [])
                    for loc in possible_locations:
                        if loc in tool_call_name:
                            current_tool_location = loc
                            break # Use first match
                    # Fallback if not found in name (should ideally not happen with good tool call names)
                    if not current_tool_location:
                         # Check the actual data if possible (more robust)
                         if formatted_result and 'cfc' in formatted_result[0]:
                              current_tool_location = formatted_result[0]['cfc']
                         elif formatted_result and 'spoke' in formatted_result[0]:
                              current_tool_location = formatted_result[0]['spoke']
                         else:
                              current_tool_location = "Unknown Location"

                    # Store results and update status
                    results[tool_call["result_id"]] = {
                        "status": "success",
                        "data": {
                            "data": formatted_result,
                            "summary": {
                                "total_records": len(formatted_result),
                                "time_period": f"{constraints.get('time_filter', {}).get('start_date')} to {constraints.get('time_filter', {}).get('end_date')}",
                                "location": current_tool_location, # Use the determined location for this tool
                                "total_orders": {
                                    "total": sum(int(row['total_orders']) for row in formatted_result if row.get('total_orders') is not None),
                                    "average": sum(int(row['total_orders']) for row in formatted_result if row.get('total_orders') is not None) / len(formatted_result) if formatted_result else 0,
                                    "max": max(int(row['total_orders']) for row in formatted_result if row.get('total_orders') is not None) if any(row.get('total_orders') is not None for row in formatted_result) else 0,
                                    "min": min(int(row['total_orders']) for row in formatted_result if row.get('total_orders') is not None) if any(row.get('total_orders') is not None for row in formatted_result) else 0
                                }
                            }
                        }
                    }
                    tool_call_status[tool_name] = "completed"
                    
                    # Log the first row of results for debugging
                    if formatted_result:
                        self.logger.info(f"First row of results for {tool_name}: {json.dumps(formatted_result[0], indent=2)}")
                    else:
                        self.logger.warning(f"No results returned for {tool_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error executing query {tool_name}: {str(e)}")
                    tool_call_status[tool_name] = "failed"
                    continue
            
            # --- ADD THIS PRINT STATEMENT ---
            print(f"[ROUTER PRINT DEBUG {session_id}] Reached point just before graph condition check.")
            # --------------------------------

            # If results are available and it's a suitable question type, generate graph
            # --- NEW LOG ---
            self.logger.info(f"[ROUTER {session_id}] EVALUATING graph condition: has_results={bool(results)}, type='{question_type}'")
            # Add more explicit debug for question type
            print(f"[ROUTER DEBUG] Raw question_type: '{question_type}', transformed: '{question_type.lower().strip()}'")
            self.logger.info(f"[ROUTER {session_id}] Exact question_type check: raw='{question_type}', transformed='{question_type.lower().strip()}'")
            self.logger.info(f"[ROUTER {session_id}] Check result: {question_type.lower().strip() in ['trend analysis', 'comparitive analysis', 'trend', 'compare']}")
            
            # Check constraint comparison_type for additional fallback
            comparison_type = constraints.get('comparison_type', '').lower().strip() if constraints else ''
            self.logger.info(f"[ROUTER {session_id}] Constraint comparison_type: '{comparison_type}'")
            
            # Expanded condition to check question_type OR constraint comparison_type
            should_generate_graph = (
                question_type.lower().strip() in ["trend analysis", "comparitive analysis", "trend", "compare", "kpi extraction"] or
                comparison_type in ["trend", "compare", "trend analysis", "comparitive analysis"]
            )
            self.logger.info(f"[ROUTER {session_id}] Final graph decision: {should_generate_graph}")
            
            if results and should_generate_graph:
                # --- NEW LOG ---
                self.logger.info(f"[ROUTER {session_id}] Graph condition MET. Proceeding to check constraints/session_id.")
                if constraints and session_id:
                     # --- NEW LOG ---
                     self.logger.info(f"[ROUTER {session_id}] Constraints and session_id PRESENT. Creating Thread.")
                     print(f"[DEBUG PRINT] About to create thread with send_callback: {send_callback is not None}")
                     try:
                         graph_thread = threading.Thread(
                             target=self._generate_and_save_graph,
                             args=(results, constraints, session_id, session_manager, send_callback),
                             daemon=True
                         )
                         # --- NEW LOG ---
                         self.logger.info(f"[ROUTER {session_id}] Thread object CREATED successfully.")
                         graph_thread.start()
                         # --- NEW LOG ---
                         self.logger.info(f"[ROUTER {session_id}] Background graph generation thread STARTED for session {session_id}.")
                     except Exception as thread_err:
                         # --- NEW LOG ---
                         self.logger.error(f"[ROUTER {session_id}] !!! EXCEPTION during Thread creation/start: {thread_err}", exc_info=True)

                else:
                    # --- NEW LOG ---
                    self.logger.warning(f"[ROUTER {session_id}] Skipping graph task: constraints_present={bool(constraints)}, session_id_present={bool(session_id)}")
            elif not results:
                 # --- NEW LOG ---
                 self.logger.info(f"[ROUTER {session_id}] Graph condition FAILED: No results.")
            else:
                 # --- NEW LOG ---
                 self.logger.info(f"[ROUTER {session_id}] Graph condition FAILED: Question type '{question_type}' not suitable for graphing. Supported types: trend analysis, comparitive analysis")
            
            # Step 4: Generate response using the response agent
            self.logger.info("Generating response...")
            self.logger.info(f"Results being passed to response agent: {json.dumps(results, indent=2, cls=DateTimeEncoder)}")
            self.logger.info(f"Constraints being passed to response agent: {json.dumps(constraints, indent=2)}")
            
            # Check if we have any results
            if not results:
                self.logger.error("No results were returned from any queries")
                response_text = "I apologize, but I couldn't retrieve any data for your query. Please try again or rephrase your question."
            else:
                try:
                    # Generate response using response agent - pass all results and constraints
                    self.logger.info("Attempting to generate response via ResponseAgent...")
                    response_text = self.response_agent.generate_response(
                        classification.question,
                        results, # Pass the entire results dictionary
                        constraints # Pass the constraints which include the response_plan
                    )
                    self.logger.info(f"Response received from ResponseAgent: '{response_text[:100]}...'")

                    # If response is empty or error, generate a default response (less likely needed now)
                    if not response_text or response_text.strip() == "":
                        self.logger.error("Empty response from response agent, generating fallback.")
                        response_text = "I was able to retrieve the data, but encountered an issue generating the summary. Please check the raw results if available."
                    else:
                        self.logger.info("ResponseAgent returned a valid response.")

                except Exception as e:
                    self.logger.error(f"!!! Exception during response generation: {str(e)}")
            
            self.logger.info(f"Final response_text before creating ResponseData: '{response_text}'")
            # Create response data
            response_data = ResponseData(
                query=classification.question,
                summary=response_text,
                results=results,
                execution_time=0.0,  # TODO: Track actual execution time
                created_at=datetime.now()
            )
            
            # Step 5: Update session with results if we have a session manager
            # NOTE: This update happens *before* the graph generation background task finishes.
            # The background task will perform a *separate* update later if it succeeds.
            if session_manager:
                try:
                    # Format updates according to the BigQuery schema
                    updates = {
                        "session_id": session_id,
                        "user_id": user_id,
                        "question": classification.question,
                        "status": "completed", # Mark as completed for text response, graph is async
                        "summary": response_text,
                        "updated_at": datetime.now().isoformat(),
                        "results": results,
                        "tool_calls": tool_calls,
                        "tool_call_status": tool_call_status,
                        "constraints": constraints
                        # "graph_path": None # Explicitly set to None initially if the column exists
                    }
                    
                    session_manager.update_session(session_id, updates)
                    self.logger.info(f"Updated session {session_id} with primary response (graph pending).")
                except Exception as e:
                    self.logger.warning(f"Error updating session with primary response: {str(e)}")
                    traceback.print_exc()
            
            # Create response context with the thread ID
            response_context = self._create_response_context(classification_context, response_data)
            response_context.metadata.session_id = session_id  # Ensure session_id is set in metadata

            end_time = time.time()
            print(f"[ROUTER PRINT DEBUG {session_id}] FINISHING _handle_full_pipeline. Duration: {end_time - start_time:.2f} seconds.")
            return response_context
            
        except Exception as e:
            self.logger.error(f"Error in full pipeline: {str(e)}")
            traceback.print_exc()
            return self._create_error_response(classification_context, str(e))
    
    def _generate_and_save_graph(self, results: Dict[str, Dict[str, Any]], constraints: Dict[str, Any], session_id: str, session_manager: Optional[SessionManagerV2], send_callback: Optional[Callable] = None):
        """Generates graph in background, saves it, and triggers callback."""
        # --- NEW LOG ---
        print(f"[DEBUG PRINT] _generate_and_save_graph STARTED with session_id: {session_id}")
        print(f"[DEBUG PRINT] send_callback type: {type(send_callback)}, callable: {callable(send_callback) if send_callback else False}")
        self.logger.info(f"[GraphBG {session_id}] STARTING _generate_and_save_graph background task.")
        graph_filepath = None # Initialize filepath
        
        try:
            self.logger.info(f"[GraphBG {session_id}] Initializing GraphGenerator...")
            # Create a new instance to avoid potential thread safety issues
            graph_generator = GraphGenerator()
            
            # Log what we're about to do with our input data
            result_keys = list(results.keys()) if isinstance(results, dict) else "NOT A DICT"
            self.logger.info(f"[GraphBG {session_id}] Input validation - results keys: {result_keys}")
            self.logger.info(f"[GraphBG {session_id}] Input validation - has constraints: {bool(constraints)}")
            
            self.logger.info(f"[GraphBG {session_id}] Calling generate_line_graph...")
            # --- Generate Graph ---
            graph_filepath = graph_generator.generate_line_graph(results, constraints, session_id)
            
            # --- NEW LOG ---
            self.logger.info(f"[GraphBG {session_id}] generate_line_graph finished. Filepath: {graph_filepath}")

            if graph_filepath:
                self.logger.info(f"[GraphBG {session_id}] Graph generated successfully at: {graph_filepath}")
                
                # Verify file exists
                if os.path.exists(graph_filepath):
                    self.logger.info(f"[GraphBG {session_id}] Verified graph file exists on disk: {graph_filepath}")
                else:
                    self.logger.error(f"[GraphBG {session_id}] !!! File reported by generate_line_graph DOES NOT EXIST: {graph_filepath}")
                    raise FileNotFoundError(f"Generated graph file not found: {graph_filepath}")
                
                # Update session with graph filepath (optional, but can be useful)
                if session_manager:
                     try:
                          self.logger.info(f"[GraphBG {session_id}] Attempting to update session with graph_path...")
                          # Use the correct session manager method
                          if hasattr(session_manager, 'update_session'):
                               session_manager.update_session(session_id, {
                                   "graph_path": graph_filepath,
                                   "has_graph": True
                               }) # Use graph_path to match BigQuery schema
                               self.logger.info(f"[GraphBG {session_id}] Session update with graph_path and has_graph=True attempted.")
                          else:
                               self.logger.warning(f"[GraphBG {session_id}] Cannot update session graph_path, session_manager has no 'update_session' method.")
                     except Exception as update_err:
                          self.logger.error(f"[GraphBG {session_id}] Failed to update session with graph path: {update_err}", exc_info=True)

                # --- CRITICAL: Check and Call Callback ---
                print(f"[DEBUG PRINT] Checking send_callback exists: {send_callback is not None}")
                if send_callback:
                    print(f"[DEBUG PRINT] send_callback.__name__={getattr(send_callback, '__name__', 'no-name')}")
                self.logger.info(f"[GraphBG {session_id}] Checking if send_callback exists: {send_callback is not None}")
                
                if send_callback:
                    # --- NEW LOG ---
                    print(f"[DEBUG PRINT] About to call send_callback({session_id}, {graph_filepath})")
                    self.logger.info(f"[GraphBG {session_id}] send_callback exists. Preparing to call send_callback with filepath: {graph_filepath}")
                    
                    # Check callback is callable
                    if not callable(send_callback):
                        self.logger.error(f"[GraphBG {session_id}] !!! send_callback is not callable: {type(send_callback)}")
                        return
                        
                    try:
                        # Trigger the callback to send the graph with explicit try/except
                        print(f"[DEBUG PRINT] Calling send_callback NOW")
                        send_callback(session_id, graph_filepath)
                        print(f"[DEBUG PRINT] send_callback executed successfully")
                        self.logger.info(f"[GraphBG {session_id}] send_callback executed successfully.")
                    except Exception as callback_err:
                        print(f"[DEBUG PRINT] EXCEPTION in send_callback: {type(callback_err).__name__} - {str(callback_err)}")
                        self.logger.error(f"[GraphBG {session_id}] !!! EXCEPTION during send_callback: {callback_err}", exc_info=True)
                else:
                    print(f"[DEBUG PRINT] No send_callback provided. Cannot send graph.")
                    self.logger.warning(f"[GraphBG {session_id}] No send_callback provided. Cannot send graph.")
            else:
                self.logger.warning(f"[GraphBG {session_id}] Graph generation did not return a filepath.")

        # --- UPDATED EXCEPTION LOGGING ---
        except Exception as e:
            print(f"[DEBUG PRINT] EXCEPTION in _generate_and_save_graph: {type(e).__name__} - {str(e)}")
            self.logger.error(f"[GraphBG {session_id}] EXCEPTION in _generate_and_save_graph: {type(e).__name__} - {str(e)}")
            self.logger.error(traceback.format_exc()) # Log the full traceback
            # Optionally update session with graph error
            if session_manager:
                try:
                    update_payload = {"graph_error": f"Graph generation failed: {type(e).__name__} - {str(e)}"}
                    # Use the correct session manager method if it exists
                    if hasattr(session_manager, 'update_session'):
                         session_manager.update_session(session_id, update_payload)
                         self.logger.info(f"[GraphBG {session_id}] Session updated with graph_error.")
                    else:
                         self.logger.warning(f"[GraphBG {session_id}] Cannot update session with graph_error, session_manager has no 'update_session' method.")
                except Exception as update_err:
                    self.logger.error(f"[GraphBG {session_id}] Failed to update session with graph error: {update_err}")
        finally:
             # --- NEW LOG ---
             print(f"[DEBUG PRINT] _generate_and_save_graph FINISHED with session_id: {session_id}")
             self.logger.info(f"[GraphBG {session_id}] FINISHING _generate_and_save_graph background task.")
    
    def _handle_planned_feature(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable[[str, str], None]] = None):
        """Handle planned features with a standard message."""
        classification = classification_context.data
        self.logger.info(f"Routing planned feature: {classification.question_type}")
        
        response_data = ResponseData(
            query=classification.question,
            summary=f"This type of question ({classification.question_type}) is a planned feature that will be available soon. We're working on it!"
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_unsupported_feature(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable[[str, str], None]] = None):
        """Handle an unsupported feature request."""
        classification = classification_context.data
        self.logger.info(f"Routing unsupported feature: {classification.question}")
        
        response_data = ResponseData(
            query=classification.question,
            summary="I'm sorry, but this type of question isn't currently supported by our system. Please try asking about KPIs, trends, comparisons, or anomalies in your data."
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_clarification_needed(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable[[str, str], None]] = None):
        """Handle a question that needs clarification."""
        classification = classification_context.data
        self.logger.info(f"Routing clarification needed: {classification.question}")
        
        response_data = ResponseData(
            query=classification.question,
            summary="To better assist you, could you please provide more details about what specific data or metrics you're interested in?"
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_small_talk(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable[[str, str], None]] = None):
        """Handle small talk with a friendly response."""
        classification = classification_context.data
        self.logger.info(f"Routing small talk: {classification.question}")
        
        response_data = ResponseData(
            query=classification.question,
            summary="I'm your analytics assistant. I'm here to help you analyze your data. How can I assist you with your business metrics today?"
        )
        return self._create_response_context(classification_context, response_data)
    
    def _handle_feedback(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable[[str, str], None]] = None):
        """Handle feedback by thanking the user and storing the feedback."""
        classification = classification_context.data
        self.logger.info(f"Routing feedback: {classification.question}")
        
        # Create feedback data to store
        feedback_data = {
            "question": classification.question,
            "feedback_type": classification.question_type,
            "confidence": classification.confidence,
            "received_at": datetime.now().isoformat()
        }
        
        # Create response data
        response_data = ResponseData(
            query=classification.question,
            summary="Thank you for your feedback! We appreciate your input and will use it to improve our system.",
            feedback_data=feedback_data
        )
        
        # Get session ID from context metadata or generate a new Slack thread ID
        session_id = classification_context.metadata.session_id
        if not session_id:
            # Generate a Slack-style thread ID
            timestamp = int(datetime.utcnow().timestamp())
            thread_id = str(int(datetime.utcnow().timestamp() * 1000))[-6:]  # Last 6 digits of millisecond timestamp
            session_id = f"{timestamp}.{thread_id}"
            self.logger.info(f"No session ID in context metadata, generated new thread ID: {session_id}")
        
        # Create session data
        session_data = {
            "session_id": session_id,
            "user_id": "anonymous",
            "question": classification.question,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "summary": response_data.summary, 
            "feedback": [feedback_data],
            "classification": {
                "question": classification.question,
                "question_type": classification.question_type,
                "confidence": classification.confidence
            },
            "response": {
                "summary": response_data.summary,
                "feedback_data": feedback_data
            }
        }
        
        # Store in cache
        self.store_cached_session(session_id, session_data)
        
        # If we have a session manager, update the session with feedback
        if session_manager:
            try:
                # Log session manager type
                self.logger.info(f"Session manager type: {type(session_manager)}")
                
                # Get current session or create if it doesn't exist
                self.logger.info(f"Getting session with thread ID: {session_id}")
                
                try:
                    session = session_manager.get_session(session_id)
                    self.logger.info(f"Retrieved session: {session}")
                except Exception as e:
                    self.logger.error(f"Error retrieving session: {str(e)}")
                    # Session might not exist, create it
                    self.logger.info(f"Creating new session {session_id} for feedback")
                    user_id = "anonymous"  # Default user ID
                    try:
                        session_manager.create_session(user_id, classification.question, session_id)
                        self.logger.info(f"Created new session with thread ID: {session_id}")
                        session = session_manager.get_session(session_id)
                        self.logger.info(f"Retrieved newly created session: {session}")
                    except Exception as e:
                        self.logger.error(f"Error creating session: {str(e)}")
                        session = None
                
                # Handle none or dict session
                if not session:
                    self.logger.info(f"No session found, creating a SessionData object")
                    session = session_data
                elif isinstance(session, dict):
                    self.logger.info(f"Session is a dict, updating it")
                    # Add feedback to the session
                    if "feedback" not in session:
                        session["feedback"] = []
                    session["feedback"].append(feedback_data)
                    
                    # Add classification data
                    session["classification"] = {
                        "question": classification.question,
                        "question_type": classification.question_type,
                        "confidence": classification.confidence
                    }
                    
                    # Add response to session
                    session["response"] = {
                        "summary": response_data.summary,
                        "feedback_data": feedback_data
                    }
                    
                    # Update session status and summary
                    session["status"] = "completed"
                    session["summary"] = response_data.summary
                    session["updated_at"] = datetime.now().isoformat()
                    session["session_id"] = session_id  # Ensure session_id is set
                else:
                    self.logger.info(f"Session is an object of type {type(session)}, updating it")
                    # Add feedback to the session object
                    if not hasattr(session, "feedback") or session.feedback is None:
                        session.feedback = []
                    session.feedback.append(feedback_data)
                    
                    # Add classification data
                    session.classification = {
                        "question": classification.question,
                        "question_type": classification.question_type,
                        "confidence": classification.confidence
                    }
                    
                    # Add response to session
                    session.response = {
                        "summary": response_data.summary,
                        "feedback_data": feedback_data
                    }
                    
                    # Update session status and summary
                    session.status = "completed"
                    session.summary = response_data.summary
                    session.updated_at = datetime.now()
                    session.session_id = session_id  # Ensure session_id is set
                    
                    # Convert object to dict
                    if hasattr(session, "to_dict"):
                        session = session.to_dict()
                    else:
                        # Try manual conversion
                        self.logger.info("Manual conversion of session object to dict")
                        session = {
                            "session_id": session_id,
                            "user_id": getattr(session, "user_id", "anonymous"),
                            "question": getattr(session, "question", classification.question),
                            "status": getattr(session, "status", "completed"),
                            "summary": getattr(session, "summary", response_data.summary),
                            "created_at": getattr(session, "created_at", datetime.now().isoformat()),
                            "updated_at": datetime.now().isoformat(),
                            "feedback": getattr(session, "feedback", []),
                            "classification": getattr(session, "classification", {}),
                            "response": getattr(session, "response", {})
                        }
                
                # Update session
                self.logger.info(f"Updating session: {session}")
                try:
                    session_manager.update_session(session_id, session)
                    self.logger.info(f"Successfully updated session {session_id}")
                except Exception as e:
                    self.logger.error(f"Error updating session: {str(e)}")
                    traceback.print_exc()
            except Exception as e:
                self.logger.error(f"Failed to store feedback: {str(e)}")
                traceback.print_exc()
        
        # Include session_id in the response context metadata
        metadata = ContextMetadata(
            context_id=str(uuid.uuid4()),
            parent_id=classification_context.metadata.context_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            component="MCPRouter",
            operation="route_response",
            status="success",
            session_id=session_id
        )
        
        return Context(data=response_data, metadata=metadata)
    
    def _handle_ambiguous_question(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable[[str, str], None]] = None):
        """Handle ambiguous questions by asking for clarification."""
        classification = classification_context.data
        question = classification.question
        session_id = classification_context.metadata.session_id
        user_id = classification_context.metadata.user_id or getattr(classification_context.data, 'user_id', 'anonymous')
        
        self.logger.info(f"Handling ambiguous question: {question}")
        
        clarification_needed = []
        partial_constraints = None
        
        try:
            # Attempt to extract constraints to see what's missing
            # Pass previous_context if available
            partial_constraints = self.query_processor.extract_constraints(
                question,
                previous_context=previous_context
            )
            self.logger.info(f"Partial constraints extracted: {json.dumps(partial_constraints, indent=2)}")
            
            # --- Refined Checks for Missing Information ---
            
            # Check 1: Missing KPI
            # Force missing KPI if the question used generic term "kpi" or similar,
            # OR if the extractor returned an empty list.
            generic_kpi_term_used = any(term in question.lower() for term in [" kpi", " metric", " measure"])
            if generic_kpi_term_used or not partial_constraints.get("kpi"):
                clarification_needed.append("which specific KPI (e.g., orders, ATP)")
            
            # Check 2: Missing Time Period
            time_filter = partial_constraints.get("time_filter", {})
            if not time_filter or not time_filter.get("start_date") or not time_filter.get("end_date"):
                 # Check if specific time keywords were actually used in the question
                 # This prevents asking for time if the extractor defaulted but user didn't specify
                 time_keywords = ["last week", "month", "january", "february", "march", "april", "may", "june", 
                                  "july", "august", "september", "october", "november", "december", 
                                  "quarter", "year", "today", "yesterday", "\d{4}"] # Basic check for year
                 if any(keyword in question.lower() for keyword in time_keywords):
                     # User likely mentioned time, but extractor failed
                     clarification_needed.append("the time period (e.g., last week, January 2024)")
                 # If no time keywords found, maybe user didn't intend to specify time - don't ask? 
                 # Let's still ask for now, as most KPIs need time.
                 elif "the time period (e.g., last week, January 2024)" not in clarification_needed:
                    clarification_needed.append("the time period (e.g., last week, January 2024)")

            # Check 3: Missing Location
            # Only ask if relevant KPI was extracted OR if the generic term "kpi" was used
            extracted_kpi = partial_constraints.get("kpi", [])
            kpi_requires_location = any(k in ["orders", "atp"] for k in extracted_kpi)
            location_keywords = ["cfc", "spoke", "network"] + [cfc.lower() for cfc in self.query_processor.bigquery_client.get_cfc_spoke_mapping().keys()] # Add known CFCs
            
            if (kpi_requires_location or generic_kpi_term_used) and not partial_constraints.get("cfc") and not partial_constraints.get("spokes"):
                 # Also check if user actually mentioned a location term
                 if any(keyword in question.lower() for keyword in location_keywords):
                    # User mentioned location, but extractor missed it
                    clarification_needed.append("the location (e.g., a specific CFC, spoke, or the entire network)")
                 # If no location term found, and it's required, ask for it.
                 elif "the location (e.g., a specific CFC, spoke, or the entire network)" not in clarification_needed:
                    clarification_needed.append("the location (e.g., a specific CFC, spoke, or the entire network)")
            # --- End Refined Checks ---

        except Exception as e:
            self.logger.error(f"Error extracting partial constraints for ambiguous question: {str(e)}")
            # Fallback to generic clarification if extraction fails
            clarification_needed = ["more details about what specific data or metrics you're interested in"]

        # Formulate clarification message
        if clarification_needed:
            # Check if the list is non-empty BEFORE joining
            if clarification_needed:
                missing_info = " and ".join(clarification_needed)
                summary = f"To help me answer your question, could you please specify {missing_info}?"
            else:
                 # Should not happen if the initial check is done right, but as a fallback:
                 summary = "I understand you're asking about something, but could you please provide a bit more detail? For example, specify the KPI, time period, and location."
        else:
            # If extraction succeeded but somehow nothing was marked as needed (unlikely but possible)
            summary = "I seem to have all the details, but I'm still unsure how to proceed. Could you please rephrase your request?"

        self.logger.info(f"Generated clarification message: {summary}")

        # Create response data
        response_data = ResponseData(
            query=question,
            summary=summary,
            created_at=datetime.now()
        )

        # Update session status and store partial context
        if session_manager and session_id:
            try:
                update_data = {
                    "status": "awaiting_clarification",
                    "summary": summary, # Store the question asked to the user
                    "updated_at": datetime.now().isoformat()
                }
                # Store partial constraints if extracted
                if partial_constraints:
                    # Ensure constraints are serializable
                    try:
                        update_data["constraints"] = json.loads(json.dumps(partial_constraints, cls=DateTimeEncoder))
                    except Exception as json_err:
                        self.logger.error(f"Could not serialize partial constraints: {json_err}")
                        update_data["constraints"] = {"error": "Could not store partial constraints"}
                
                session_manager.update_session(session_id, update_data)
                self.logger.info(f"Updated session {session_id} status to awaiting_clarification")
            except Exception as e:
                self.logger.error(f"Error updating session for ambiguous question: {str(e)}")
                traceback.print_exc()
        
        # Create and return the response context
        return self._create_response_context(classification_context, response_data)
    
    def _handle_unsupported_construct(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable[[str, str], None]] = None):
        """Handle questions classified as having unsupported NLP constructs."""
        classification = classification_context.data
        self.logger.info(f"Handling unsupported NLP construct: {classification.question}")
        
        # Provide a simple response indicating inability to process
        response_data = ResponseData(
            query=classification.question,
            summary="I'm sorry, I had trouble understanding the structure of your request. Could you please try rephrasing it as a question about specific KPIs, locations, or time periods?"
        )
        
        # Optionally update session status if needed (e.g., mark as failed/unsupported)
        session_id = classification_context.metadata.session_id
        if session_manager and session_id:
            try:
                update_data = {
                    "status": "failed_unsupported_construct",
                    "summary": response_data.summary,
                    "updated_at": datetime.now().isoformat()
                }
                session_manager.update_session(session_id, update_data)
                self.logger.info(f"Updated session {session_id} status to failed_unsupported_construct")
            except Exception as e:
                self.logger.error(f"Error updating session for unsupported construct: {str(e)}")

        return self._create_response_context(classification_context, response_data)
    
    def _handle_follow_up(self, classification_context: Context[ClassificationData], session_manager=None, previous_context: Optional[Dict] = None, send_callback: Optional[Callable] = None):
        """Handle follow-up questions by reusing previous context and data, falling back to full pipeline if needed."""
        classification = classification_context.data
        session_id = classification_context.metadata.session_id
        self.logger.info(f"Handling follow-up question: {classification.question}")
        
        try:
            # Check if we have previous context
            if not previous_context:
                self.logger.warning("No previous context available for follow-up question, falling back to full pipeline")
                return self._handle_full_pipeline(classification_context, session_manager, previous_context, send_callback)
            
            # Get previous results and constraints
            previous_results = previous_context.get('previous_results', {})
            previous_constraints = previous_context.get('constraints', {})
            
            if not previous_results:
                self.logger.warning("No previous results available for follow-up question, falling back to full pipeline")
                return self._handle_full_pipeline(classification_context, session_manager, previous_context, send_callback)
            
            # Extract current constraints from classification metadata
            current_constraints = classification.classification_metadata.get('constraints', {})
            
            # Filter previous results based on current constraints
            filtered_results = {}
            for result_id, result in previous_results.items():
                if result.get('status') == 'success':
                    data = result.get('data', {}).get('data', [])
                    filtered_data = []
                    
                    # Filter data based on time constraints
                    time_filter = current_constraints.get('time_filter', {})
                    if time_filter:
                        start_date = datetime.fromisoformat(time_filter.get('start_date', '1900-01-01'))
                        end_date = datetime.fromisoformat(time_filter.get('end_date', '2100-12-31'))
                        
                        for point in data:
                            point_date = datetime.fromisoformat(point.get('week_start_date', '1900-01-01'))
                            if start_date <= point_date <= end_date:
                                filtered_data.append(point)
                    
                    if filtered_data:
                        filtered_results[result_id] = {
                            'status': 'success',
                            'data': {
                                'data': filtered_data,
                                'summary': result.get('data', {}).get('summary', {})
                            }
                        }
            
            if not filtered_results:
                self.logger.warning("No data matching the follow-up constraints found in previous results, falling back to full pipeline")
                return self._handle_full_pipeline(classification_context, session_manager, previous_context, send_callback)
            
            # Generate response using filtered data
            response_data = ResponseData(
                query=classification.question,
                summary=self._generate_follow_up_summary(filtered_results, current_constraints, previous_context)
            )
            
            # Generate graph if appropriate
            if send_callback and self._should_generate_graph(classification.question_type, current_constraints):
                try:
                    graph_path = self._generate_and_save_graph(
                        filtered_results,
                        current_constraints,
                        session_id,
                        session_manager,
                        send_callback
                    )
                    if graph_path:
                        response_data.graph_path = graph_path
                except Exception as e:
                    self.logger.error(f"Error generating graph for follow-up: {str(e)}")
            
            return self._create_response_context(classification_context, response_data)
            
        except Exception as e:
            self.logger.error(f"Error handling follow-up question: {str(e)}")
            traceback.print_exc()
            return self._create_error_response(
                classification_context,
                f"An error occurred while processing your follow-up question: {str(e)}"
            )
    
    def _generate_follow_up_summary(self, results: Dict, constraints: Dict, previous_context: Dict) -> str:
        """Generate a summary for follow-up question results."""
        try:
            # Extract key information
            time_filter = constraints.get('time_filter', {})
            time_period = f"{time_filter.get('start_date')} to {time_filter.get('end_date')}"
            
            # Build summary
            summary_parts = []
            summary_parts.append(f"Here's the analysis for {time_period} based on your previous question:")
            
            # Add data points
            for result_id, result in results.items():
                if result.get('status') == 'success':
                    data = result.get('data', {}).get('data', [])
                    summary = result.get('data', {}).get('summary', {})
                    
                    if data:
                        location = summary.get('location', result_id)
                        total_orders = sum(point.get('total_orders', 0) for point in data)
                        avg_orders = total_orders / len(data) if data else 0
                        
                        summary_parts.append(f"\n*{location}:*")
                        summary_parts.append(f"• Total Orders: {total_orders:,}")
                        summary_parts.append(f"• Average Weekly Orders: {avg_orders:,.2f}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up summary: {str(e)}")
            return "Here's the analysis for the requested time period based on your previous question."
    
    def _create_error_response(self, context: Context[ClassificationData], error_message: str) -> Context[ResponseData]:
        """Create an error response context"""
        self.logger.error(f"Creating error response: {error_message}")
        
        # Create error response data
        error_response = ResponseData(
            query=context.data.question,
            summary=f"I apologize, but I encountered an error while processing your request: {error_message}",
            results={},
            execution_time=0.0,
            created_at=datetime.now()
        )
        
        # Create response context
        response_context = Context(
            data=error_response,
            metadata=context.metadata
        )
        
        return response_context 

    def _init_schema_cache(self):
        """Initialize schema cache with parallel loading"""
        with ThreadPoolExecutor(max_workers=6) as executor:  # Increased workers
            futures = [executor.submit(load_table_schema, (name, ref)) 
                      for name, ref in self.tables.items()]
            
            # Use as_completed for faster processing
            for future in as_completed(futures):
                name, ref, schema = future.result()
                if schema:
                    self._schema_cache[ref] = schema

def ensure_warm_clients():
    global _warm_clients
    with _warm_lock:
        if _warm_clients is None:
            _warm_clients = initialize_components()
        return _warm_clients

def cloud_function_handler(request):
    # Use warm clients
    clients = ensure_warm_clients()
    # Handle request

async def initialize_components_async():
    async with asyncio.TaskGroup() as group:
        bq_task = group.create_task(init_bigquery())
        gemini_task = group.create_task(init_gemini())
    return bq_task.result(), gemini_task.result()

class ConnectionPool:
    _pools = {}
    
    @classmethod
    def get_connection(cls, service_type):
        if service_type not in cls._pools:
            cls._pools[service_type] = create_connection_pool(service_type)
        return cls._pools[service_type].get_connection() 

class ResourceManager:
    def __init__(self):
        self.resources = {}
        self._locks = {}
        self._init_tasks = {}
    
    async def get_resource(self, resource_type):
        if resource_type not in self.resources:
            if resource_type not in self._init_tasks:
                async with self._get_lock(resource_type):
                    if resource_type not in self._init_tasks:
                        self._init_tasks[resource_type] = asyncio.create_task(
                            self._initialize_resource(resource_type)
                        )
            await self._init_tasks[resource_type]
        return self.resources[resource_type]

class BigQueryConnectionPool:
    _pool = None
    _lock = threading.Lock()
    
    @classmethod
    def get_connection(cls):
        with cls._lock:
            if cls._pool is None:
                cls._pool = bigquery.Client()
            return cls._pool 

class SchemaCache:
    CACHE_FILE = "schema_cache.json"
    
    @classmethod
    def load(cls):
        if os.path.exists(cls.CACHE_FILE):
            with open(cls.CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save(cls, schemas):
        with open(cls.CACHE_FILE, 'w') as f:
            json.dump(schemas, f) 

class QuestionCategoriesCache:
    _cache = None
    _last_refresh = 0
    _refresh_interval = 300  # 5 minutes
    _lock = threading.Lock()
    
    @classmethod
    def get_categories(cls, client):
        current_time = time.time()
        with cls._lock:
            if cls._cache and current_time - cls._last_refresh < cls._refresh_interval:
                return cls._cache.copy()  # Return copy to prevent mutations
            
            cls._cache = client.get_question_categories()
            cls._last_refresh = current_time
            return cls._cache.copy() 