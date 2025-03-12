from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime
from bigquery_client import BigQueryClient

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
            "tool_calls": [],
            "tool_call_status": [],
            "tool_call_results": None,
            "created_at": current_time,
            "updated_at": current_time
        }
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
        results = self.bigquery_client.query(query)
        if not results:
            return None
            
        session = results[0]
        
        # Parse JSON fields
        json_fields = ["constraints", "response_plan", "strategy", "tool_call_results"]
        for field in json_fields:
            if session.get(field) and isinstance(session[field], str):
                try:
                    session[field] = json.loads(session[field])
                except json.JSONDecodeError:
                    pass
                
        return session

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update session with new information in BigQuery
        
        Instead of updating the existing row, we insert a new row with the updated information.
        This avoids the streaming buffer limitation in BigQuery.
        """
        # Get the current session data
        current_session = self.get_session(session_id)
        if not current_session:
            raise ValueError(f"Session {session_id} not found")
            
        # Create a new session data with the updates
        new_session = {}
        
        # Copy all fields from current session, converting datetime objects to ISO format strings
        for key, value in current_session.items():
            if isinstance(value, datetime):
                new_session[key] = value.isoformat()
            else:
                new_session[key] = value
        
        # Apply updates
        for key, value in updates.items():
            new_session[key] = value
            
        # Update timestamp
        new_session["updated_at"] = datetime.utcnow().isoformat()
        
        # Insert the new session data
        self.bigquery_client.insert_row(self.table_id, new_session)

    def update_tool_call_status(self, session_id: str, tool_name: str, status: str, result: Any = None) -> None:
        """
        Update the status of a tool call in the session
        
        Args:
            session_id: The ID of the session
            tool_name: The name of the tool call
            status: The status of the tool call (e.g., "running", "completed", "failed")
            result: Optional result data from the tool call
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Get the current tool calls and status
        tool_calls = session.get("tool_calls", [])
        if isinstance(tool_calls, str):
            tool_calls = json.loads(tool_calls)
        
        tool_call_status = session.get("tool_call_status", [])
        if isinstance(tool_call_status, str):
            tool_call_status = json.loads(tool_call_status)
        
        # Get the current tool call results
        tool_call_results = session.get("tool_call_results", {})
        if not tool_call_results:
            tool_call_results = {}
        elif isinstance(tool_call_results, str):
            tool_call_results = json.loads(tool_call_results)
        
        # Update the tool call status
        try:
            idx = tool_calls.index(tool_name)
            tool_call_status[idx] = status
        except ValueError:
            tool_calls.append(tool_name)
            tool_call_status.append(status)
        
        # Update the tool call result if provided
        if result is not None:
            tool_call_results[tool_name] = result
        
        # Update the session
        updates = {
            "tool_calls": tool_calls,
            "tool_call_status": tool_call_status,
            "tool_call_results": tool_call_results
        }
        self.update_session(session_id, updates)

    def is_session_complete(self, session_id: str) -> bool:
        """Check if all tool calls in a session are completed"""
        session = self.get_session(session_id)
        if not session:
            return False
            
        tool_call_status = session.get("tool_call_status", [])
        if isinstance(tool_call_status, str):
            tool_call_status = json.loads(tool_call_status)
            
        return all(status == "completed" for status in tool_call_status) and tool_call_status
        
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