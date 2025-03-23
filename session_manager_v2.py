"""Session Manager (V2) for handling user sessions using an append-only pattern in BigQuery. 
Manages session creation, updates, tool call status, and retrieval of session data."""
from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime
from bigquery_client import BigQueryClient
import traceback

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

    def create_session(self, user_id: str, question: str) -> str:
        """Create a new session for a user query and store it in BigQuery"""
        session_id = str(uuid.uuid4())
        current_time = datetime.utcnow().isoformat()
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
        
        print(f"Creating new session with ID: {session_id}")
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
            
            # Parse JSON fields
            for field in ["constraints", "response_plan", "strategy", "tool_call_results", "results"]:
                if session.get(field) and isinstance(session[field], str):
                    try:
                        session[field] = json.loads(session[field])
                    except json.JSONDecodeError:
                        pass
            
            # Parse tool_calls and tool_call_status
            if session.get("tool_calls") and isinstance(session["tool_calls"], str):
                try:
                    session["tool_calls"] = json.loads(session["tool_calls"])
                except json.JSONDecodeError:
                    session["tool_calls"] = []
            
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
        try:
            # Get the current session data
            current_session = self.get_session(session_id)
            if not current_session:
                raise ValueError(f"Session {session_id} not found")
                
            # Create a new session data dictionary with the updated information
            updated_session = current_session.copy()
            updated_session.update(updates)
            
            # Set the updated_at timestamp
            updated_session["updated_at"] = datetime.utcnow().isoformat()
            
            # Convert complex objects to JSON strings for storage
            for key, value in updated_session.items():
                if key in ["constraints", "response_plan", "strategy", "tool_call_results", "results"]:
                    if value is not None and not isinstance(value, str):
                        try:
                            updated_session[key] = json.dumps(value, cls=DateTimeEncoder)
                        except Exception as e:
                            print(f"Error serializing {key}: {str(e)}")
                            traceback.print_exc()
                            # Store a simplified version that can be serialized
                            updated_session[key] = json.dumps({"error": f"Could not serialize {key}: {str(e)}"})
            
            # Convert tool_call_status to JSON if it's a dictionary
            if "tool_call_status" in updated_session and isinstance(updated_session["tool_call_status"], dict):
                updated_session["tool_call_status"] = json.dumps(updated_session["tool_call_status"], cls=DateTimeEncoder)
                
            # Insert the updated session data as a new row
            print(f"Updating session {session_id} with new record containing updates: {json.dumps(updates, cls=DateTimeEncoder)}")
            print(f"Full updated session data to be inserted: {json.dumps({k: str(v)[:100] + '...' if isinstance(v, str) and len(str(v)) > 100 else v for k, v in updated_session.items()}, cls=DateTimeEncoder)}")
            
            # Call the BigQuery client to insert the row
            insert_result = self.bigquery_client.insert_row(self.table_id, updated_session)
            print(f"Session update insert result: {insert_result if insert_result else 'Success'}")
            
            # Verify the update by retrieving the latest record
            verification = self.get_session(session_id)
            if verification:
                print(f"Verified session update - Latest updated_at: {verification.get('updated_at')}")
            else:
                print(f"Warning: Could not verify session update - session {session_id} not found after update")
                
        except Exception as e:
            print(f"Error updating session: {str(e)}")
            traceback.print_exc()

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
        """Get sessions for a specific user, ordered by creation time"""
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