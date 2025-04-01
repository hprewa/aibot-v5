"""Session Manager (V2) for handling user sessions using an append-only pattern in BigQuery. 
Manages session creation, updates, tool call status, and retrieval of session data."""
from typing import Dict, List, Optional, Any
import json
import uuid
import datetime
from bigquery_client import BigQueryClient
import traceback
import inspect

# Custom JSON encoder to handle date objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

class SessionManagerV2:
    """
    Session Manager that handles BigQuery's streaming buffer limitations
    by using an append-only pattern for session updates.
    """
    
    def __init__(self):
        self.bigquery_client = BigQueryClient()
        self.table_id = f"{self.bigquery_client.project_id}.{self.bigquery_client.dataset_id}.sessions"
        
        # Add sessions table to the BigQuery client's tables dictionary
        self.bigquery_client.tables["sessions"] = self.table_id

    def create_session(self, user_id: str, question: str, session_id: Optional[str] = None) -> str:
        """Create a new session for a user query and store it in BigQuery"""
        # If no session_id provided, generate a Slack-style thread ID
        if not session_id:
            # Generate a Slack-style thread ID (timestamp.thread_id)
            timestamp = int(datetime.utcnow().timestamp())
            thread_id = str(int(datetime.utcnow().timestamp() * 1000))[-6:]  # Last 6 digits of millisecond timestamp
            session_id = f"{timestamp}.{thread_id}"
        
        current_time = datetime.datetime.utcnow().isoformat()
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "constraints": None,
            "response_plan": None,
            "strategy": None,
            "summary": None,
            "status": "pending",
            "tool_calls": json.dumps([]),  # Store as JSON string
            "tool_call_status": json.dumps({}),  # Store as JSON string
            "tool_call_results": None,
            "results": None,
            "slack_channel": None,
            "error": None,
            "created_at": current_time,
            "updated_at": current_time
        }
        
        print(f"Creating new session with ID: {session_id} for user: {user_id} with question: {question}")
        self.bigquery_client.insert_row(self.table_id, session_data)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information from BigQuery"""
        query = f"""
            SELECT * FROM `{self.table_id}`
            WHERE session_id = '{session_id}'
            ORDER BY updated_at DESC
            LIMIT 1
        """
        
        try:
            print(f"Fetching latest session record for session_id: {session_id}")
            result = self.bigquery_client.client.query(query).to_dataframe()
            
            if result.empty:
                print(f"No session found with session_id: {session_id}")
                return None
                
            # Convert the first row to a dictionary
            session = result.iloc[0].to_dict()

            # --- LOG RAW VALUES BEFORE PARSING ---
            print("--- Raw session data fetched from BigQuery (before parsing): ---")
            for key, value in session.items():
                # Print type and a preview, especially for complex fields
                preview = str(value)[:200] + ("..." if len(str(value)) > 200 else "")
                print(f"  Raw Key: {key:<20} | Type: {type(value).__name__:<15} | Value Preview: {preview}")
            print("------------------------------------------------------------")
            # --- END LOG RAW VALUES ---

            # Parse JSON fields
            for field in ["constraints", "response_plan", "strategy", "tool_call_results", "results"]:
                if session.get(field) and isinstance(session[field], str):
                    try:
                        session[field] = json.loads(session[field])
                    except json.JSONDecodeError:
                        pass # Keep as string if invalid JSON

            # Parse tool_calls (handle potential array from BQ REPEATED field)
            tool_calls_value = session.get("tool_calls")
            if tool_calls_value is not None:
                parsed_tool_calls = []
                if isinstance(tool_calls_value, str):
                    try:
                        # If it's a JSON string representing a list, parse it
                        parsed_list = json.loads(tool_calls_value)
                        if isinstance(parsed_list, list):
                            parsed_tool_calls = parsed_list # Assume items are already strings or simple types
                        else:
                            # If it's a string but not a JSON list, wrap it
                            parsed_tool_calls = [tool_calls_value]
                    except json.JSONDecodeError:
                        # If it's a string that's not valid JSON, wrap it
                        parsed_tool_calls = [tool_calls_value]
                elif hasattr(tool_calls_value, 'tolist'): # Check for array-like (e.g., numpy array from pandas)
                    try:
                        # Convert array to a standard Python list
                        # We expect items to be strings (serialized JSONs) from BQ
                        parsed_tool_calls = tool_calls_value.tolist()
                    except Exception as e:
                        print(f"Warning: Could not convert tool_calls array to list: {e}")
                        parsed_tool_calls = [] # Fallback to empty list
                elif isinstance(tool_calls_value, list):
                    # If it's already a list, use it directly
                    parsed_tool_calls = tool_calls_value
                else:
                     # Fallback for other unexpected types
                     print(f"Warning: Unexpected type for tool_calls: {type(tool_calls_value)}. Converting to list of string.")
                     parsed_tool_calls = [str(tool_calls_value)]

                # Now, attempt to parse each item in the list if it looks like JSON
                final_tool_calls = []
                for item in parsed_tool_calls:
                    if isinstance(item, str) and item.startswith('{') and item.endswith('}'):
                        try:
                            final_tool_calls.append(json.loads(item))
                        except json.JSONDecodeError:
                            final_tool_calls.append(item) # Keep as string if not valid JSON
                    else:
                        final_tool_calls.append(item) # Keep non-dict strings as is
                session["tool_calls"] = final_tool_calls
            else:
                session["tool_calls"] = [] # Default to empty list if None

            # Parse tool_call_status
            if session.get("tool_call_status") and isinstance(session["tool_call_status"], str):
                try:
                    session["tool_call_status"] = json.loads(session["tool_call_status"])
                except json.JSONDecodeError:
                    session["tool_call_status"] = {}
            
            print(f"Successfully retrieved session {session_id} with updated_at: {session.get('updated_at')}")
            return session
        except Exception as e:
            print(f"Error getting session: {str(e)}")
            traceback.print_exc()
            return None

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update session with new information in BigQuery
        
        Instead of updating the existing row, we insert a new row with the updated information.
        This avoids the streaming buffer limitation in BigQuery.
        """
        # --- DETAILED LOGGING START ---
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame)
        print(f"\n--- Calling update_session for session_id: {session_id} ---")
        print(f"    Called from: {caller_info.filename} - Line {caller_info.lineno} - Function {caller_info.function}")
        print(f"    Updates dictionary keys: {list(updates.keys())}")
        # Optional: Log the summary if present
        if 'summary' in updates:
             print(f"    Summary in updates: {updates['summary'][:100]}...")
        # --- DETAILED LOGGING END ---
        try:
            # Get the current session data
            current_session = self.get_session(session_id)
            if not current_session:
                # If session doesn't exist, log an error and return. 
                # update_session should not create sessions.
                print(f"Error: update_session called for non-existent session_id: {session_id}")
                # Optionally raise an error, or just return to prevent insertion
                # raise ValueError(f"Session {session_id} not found, cannot update.")
                return # Stop processing if session doesn't exist
            
            # Create a new session data dictionary with the updated information
            updated_session = current_session.copy()
            
            # Filter updates to only include fields that exist in the schema
            schema_fields = {
                "session_id", "user_id", "question", "constraints", "response_plan",
                "strategy", "summary", "status", "tool_calls", "tool_call_status",
                "tool_call_results", "results", "slack_channel", "error",
                "created_at", "updated_at"
            }
            
            # Only include fields that exist in the schema
            filtered_updates = {k: v for k, v in updates.items() if k in schema_fields}
            updated_session.update(filtered_updates)
            
            # Set the updated_at timestamp
            updated_session["updated_at"] = datetime.datetime.utcnow().isoformat()
            
            # Convert complex objects to JSON strings or handle REPEATED fields
            for key, value in updated_session.items():
                # Handle fields that should be stored as JSON strings
                if key in ["constraints", "response_plan", "strategy", "tool_call_results", "results", "tool_call_status"]:
                    if value is not None and not isinstance(value, str):
                        try:
                            updated_session[key] = json.dumps(value, cls=DateTimeEncoder)
                        except Exception as e:
                            print(f"Error serializing {key}: {str(e)}")
                            traceback.print_exc()
                            # Store a simplified version that can be serialized
                            updated_session[key] = json.dumps({"error": f"Could not serialize {key}: {str(e)}"})
                # Handle tool_calls specifically for REPEATED STRING field
                elif key == "tool_calls":
                    if value is not None:
                        if isinstance(value, list):
                            # Ensure all items in the list are strings
                            try:
                                # Serialize items if they are not strings (e.g., dicts)
                                updated_session[key] = [json.dumps(item, cls=DateTimeEncoder) if not isinstance(item, str) else item for item in value]
                            except Exception as e:
                                print(f"Error processing tool_calls list items: {str(e)}")
                                traceback.print_exc()
                                # Store error representation if items can't be processed
                                updated_session[key] = [json.dumps({"error": f"Could not process tool_call item: {str(e)}"})]
                        elif isinstance(value, str):
                            # If it's already a string, try parsing it as JSON list first
                            try:
                                parsed_list = json.loads(value)
                                if isinstance(parsed_list, list):
                                     # If parsing succeeds and it's a list, process its items
                                     updated_session[key] = [json.dumps(item, cls=DateTimeEncoder) if not isinstance(item, str) else item for item in parsed_list]
                                else:
                                     # If it's a string but not a JSON list, wrap it in a list
                                     updated_session[key] = [value]
                            except json.JSONDecodeError:
                                 # If it's a string that's not valid JSON, wrap it in a list
                                 updated_session[key] = [value]
                        else:
                             # For other types, try to convert to string and wrap in list
                             try:
                                 # Serialize the whole value as a single string item in the list
                                 updated_session[key] = [json.dumps(value, cls=DateTimeEncoder)]
                             except Exception as e:
                                 print(f"Error converting non-list/non-string tool_calls value to string: {str(e)}")
                                 updated_session[key] = [json.dumps({"error": f"Could not process tool_calls value: {str(e)}"})]
                    else:
                        # Ensure it's an empty list if None
                        updated_session[key] = [] # Use empty list for None

            # Remove any fields that don't exist in the schema
            for key in list(updated_session.keys()):
                if key not in schema_fields:
                    del updated_session[key]
            
            # Insert the updated session data as a new row
            print(f"Updating session {session_id} with new record containing updates: {json.dumps(filtered_updates, cls=DateTimeEncoder)}")
            
            # Call the BigQuery client to insert the row
            insert_result = self.bigquery_client.insert_row(self.table_id, updated_session)
            if insert_result:
                print(f"Warning: Session update insert returned result: {insert_result}")
            
            # Verify the update by retrieving the latest record
            verification = self.get_session(session_id)
            if verification:
                print(f"Verified session update - Latest updated_at: {verification.get('updated_at')}")
            else:
                print(f"Warning: Could not verify session update - session {session_id} not found after update")
                
        except Exception as e:
            print(f"Error updating session: {str(e)}")
            traceback.print_exc()
            # Don't re-raise the exception, just log it
            # This allows the pipeline to continue even if session updates fail

    def update_tool_call_status(self, session_id: str, tool_name: str, status: str, result: Any = None) -> None:
        """
        Update the status of a tool call in the session
        
        Args:
            session_id: The ID of the session
            tool_name: The name of the tool call
            status: The status of the tool call (e.g., "running", "completed", "failed")
            result: Optional result data from the tool call
        """
        try:
            session = self.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Get the current tool call status
            tool_call_status = session.get("tool_call_status", {})
            if not tool_call_status:
                tool_call_status = {}
            elif isinstance(tool_call_status, list):
                # Convert list to dictionary if needed
                tool_call_status = {}
            elif isinstance(tool_call_status, str):
                try:
                    tool_call_status = json.loads(tool_call_status)
                except json.JSONDecodeError:
                    tool_call_status = {}
            
            # Update the tool call status
            tool_call_status[tool_name] = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Add error or result if provided
            if status == "failed" and result is not None:
                tool_call_status[tool_name]["error"] = str(result)
            elif status == "completed" and result is not None:
                # Convert result to a JSON-serializable format
                try:
                    # First try to serialize with the custom encoder to catch any date objects
                    serialized_result = json.dumps(result, cls=DateTimeEncoder)
                    # Then parse it back to get a clean Python object with dates as strings
                    tool_call_status[tool_name]["result"] = json.loads(serialized_result)
                except Exception as e:
                    print(f"Error serializing result for tool call {tool_name}: {str(e)}")
                    traceback.print_exc()
                    # Store a simplified version or error message
                    tool_call_status[tool_name]["result"] = f"Result could not be serialized: {str(e)}"
            
            print(f"Updating tool call status for {tool_name} to {status}")
            
            # Update the session
            self.update_session(session_id, {
                "tool_call_status": tool_call_status
            })
        except Exception as e:
            print(f"Error updating tool call status: {str(e)}")
            traceback.print_exc()

    def is_session_complete(self, session_id: str) -> bool:
        """Check if a session is complete"""
        session = self.get_session(session_id)
        if not session:
            return False
            
        return session.get("status") == "completed"

    def get_all_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all sessions, ordered by creation time"""
        query = f"""
            SELECT DISTINCT session_id, user_id, question, status, created_at
            FROM `{self.table_id}`
            ORDER BY created_at DESC
            LIMIT {limit}
        """
        
        try:
            result = self.bigquery_client.client.query(query).to_dataframe()
            if result.empty:
                return []
                
            # Convert rows to dictionaries
            sessions = result.to_dict('records')
            return sessions
        except Exception as e:
            print(f"Error getting all sessions: {str(e)}")
            return []

    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent sessions for a user"""
        query = f"""
            SELECT DISTINCT session_id, user_id, question, status, created_at
            FROM `{self.table_id}`
            WHERE user_id = '{user_id}'
            ORDER BY created_at DESC
            LIMIT {limit}
        """
        
        try:
            result = self.bigquery_client.client.query(query).to_dataframe()
            if result.empty:
                return []
                
            # Convert rows to dictionaries
            sessions = result.to_dict('records')
            return sessions
        except Exception as e:
            print(f"Error getting user sessions: {str(e)}")
            return []

    def update_constraints(self, session_id: str, constraints: Dict[str, Any]) -> None:
        """Update the constraints for a session"""
        self.update_session(session_id, {"constraints": constraints})
        
    def update_strategy(self, session_id: str, strategy: str) -> None:
        """
        Update the strategy for a session
        
        Args:
            session_id: The ID of the session
            strategy: The strategy text, which may contain multi-line content
        """
        # Store the strategy as a JSON object with a 'content' field
        # This ensures proper handling of multi-line strings in BigQuery
        strategy_json = {"content": strategy}
        self.update_session(session_id, {"strategy": strategy_json})
        
    def update_response_plan(self, session_id: str, response_plan: Dict[str, Any]) -> None:
        """Update the response plan for a session"""
        self.update_session(session_id, {"response_plan": response_plan})
        
    def update_summary(self, session_id: str, summary: str) -> None:
        """
        Update the summary for a session
        
        Args:
            session_id: The ID of the session
            summary: The summary text, which may contain multi-line content
        """
        # For consistency with the strategy field, we'll store the summary as a string directly
        # BigQuery can handle this as the summary field is defined as STRING in the schema
        self.update_session(session_id, {"summary": summary})