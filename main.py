import os
from dotenv import load_dotenv
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from session_manager_v2 import SessionManagerV2
from query_processor import QueryProcessor
from slack_bot import SlackBot
from typing import Dict, Any
import json

class AnalyticsBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        print("Initializing components...")
        self.bigquery_client = BigQueryClient()
        self.gemini_client = GeminiClient()
        self.session_manager = SessionManagerV2()
        self.query_processor = QueryProcessor(self.gemini_client)
        
        # Initialize Slack bot with message handler
        self.slack_bot = SlackBot(self.handle_message)
        
    def handle_message(self, user_id: str, message: str) -> str:
        """Handle incoming Slack messages"""
        try:
            # Create new session
            session_id = self.session_manager.create_session(user_id, message)
            
            # Extract constraints
            constraints = self.query_processor.extract_constraints(message)
            self.session_manager.update_session(session_id, {
                "constraints": constraints
            })
            
            # Generate strategy
            strategy = self.query_processor.generate_strategy(message, constraints)
            self.session_manager.update_session(session_id, {
                "strategy": strategy
            })
            
            # Execute BigQuery tool calls
            tool_calls = constraints.get("tool_calls", [])
            results = {}
            
            for tool_call in tool_calls:
                # Log the tool call
                call_name = tool_call.get("name", "Unnamed Query")
                result_id = tool_call.get("result_id", f"result_{len(results)}")
                self.session_manager.update_tool_call_status(session_id, call_name, "running")
                
                # Generate the SQL for this tool call
                sql = self.query_processor.generate_sql_for_tool_call(message, constraints, tool_call)
                
                # Execute the query
                result = self.bigquery_client.client.query(sql).to_dataframe()
                
                # Store the result with its result_id
                results[result_id] = result.to_dict('records')
                
                # Update the session with completion status
                self.session_manager.update_tool_call_status(session_id, call_name, "completed")
            
            # Store all results in the session
            self.session_manager.update_session(session_id, {
                "results": results
            })
            
            # Generate summary using response plan
            summary = self.query_processor.generate_summary(message, results, constraints)
            self.session_manager.update_session(session_id, {
                "summary": summary,
                "status": "completed"
            })
            
            return summary
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
            
    def start(self):
        """Start the bot"""
        print("Starting Analytics Bot...")
        self.slack_bot.start()
        
if __name__ == "__main__":
    bot = AnalyticsBot()
    bot.start()
