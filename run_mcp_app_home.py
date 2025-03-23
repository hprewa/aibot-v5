"""
Run script for the MCP-powered Analytics Bot with Slack App Home integration.

This script initializes all components, creates the MCP orchestrator,
and starts the Slack App Home integration server with FastAPI for HTTP Events API support.
"""

import os
import sys
import argparse
import threading
import time
import subprocess
from dotenv import load_dotenv
from datetime import datetime

# Import MCP components
from mcp import (
    MCPQuestionClassifier,
    MCPQueryProcessor,
    MCPQueryAgent,
    MCPQueryExecutor,
    MCPResponseGenerator,
    MCPSessionManager,
    MCPQueryFlowOrchestrator
)
from mcp.router import MCPRouter

# Import original components
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent
from session_manager_v2 import SessionManagerV2
from question_classifier import QuestionClassifier

# Import Slack App Home integration
from mcp.slack_app_home import MCPSlackAppHome

def setup_ngrok(port=8000):
    """Set up ngrok if available"""
    try:
        # Check if ngrok is available
        ngrok_path = "./ngrok.exe" if os.name == "nt" else "ngrok"
        
        # Start ngrok
        print("\n🌐 Starting ngrok tunnel for port", port)
        ngrok_process = subprocess.Popen(
            [ngrok_path, "http", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ngrok to start
        time.sleep(3)
        
        # Get the public URL
        try:
            import requests
            response = requests.get("http://localhost:4040/api/tunnels")
            data = response.json()
            
            if "tunnels" in data and len(data["tunnels"]) > 0:
                public_url = data["tunnels"][0]["public_url"]
                
                print("\n✅ Ngrok tunnel established!")
                print(f"📡 Public URL: {public_url}")
                print(f"🔗 Slack Events URL: {public_url}/slack/events")
                print("\nUse this URL in your Slack app's Event Subscriptions settings.")
                print("Remember to subscribe to the 'app_home_opened' and 'message.im' events.")
            else:
                print("No active ngrok tunnels found.")
        except Exception as e:
            print(f"Error getting ngrok URL: {str(e)}")
        
        return ngrok_process
    except Exception as e:
        print(f"Error setting up ngrok: {str(e)}")
        return None

def initialize_components():
    """Initialize all components and create the MCP orchestrator"""
    print("\n🔧 Initializing components...")
    
    # Initialize original components
    bigquery_client = BigQueryClient()
    gemini_client = GeminiClient()
    query_processor = QueryProcessor(gemini_client, bigquery_client)
    query_agent = QueryAgent(gemini_client)
    response_agent = ResponseAgent(gemini_client)
    session_manager = SessionManagerV2()
    question_classifier = QuestionClassifier()
    
    # Initialize MCP wrappers
    mcp_question_classifier = MCPQuestionClassifier(question_classifier)
    mcp_query_processor = MCPQueryProcessor(query_processor)
    mcp_query_agent = MCPQueryAgent(query_agent)
    mcp_query_executor = MCPQueryExecutor(bigquery_client)
    mcp_response_generator = MCPResponseGenerator(response_agent)
    mcp_session_manager = MCPSessionManager(session_manager)
    
    # Create router
    mcp_router = MCPRouter()
    
    # Create the flow orchestrator
    orchestrator = MCPQueryFlowOrchestrator(
        question_classifier=mcp_question_classifier,
        query_processor=mcp_query_processor,
        query_agent=mcp_query_agent,
        query_executor=mcp_query_executor,
        response_generator=mcp_response_generator,
        session_manager=mcp_session_manager
    )
    
    # Add router to orchestrator
    orchestrator.router = mcp_router
    
    return orchestrator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the MCP-powered Analytics Bot with Slack App Home integration"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to run the Slack events server on (default: 8000)"
    )
    parser.add_argument(
        "--ngrok", 
        action="store_true", 
        help="Start ngrok tunnel for Slack events"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the MCP-powered Analytics Bot with App Home"""
    # Parse command line arguments
    args = parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set port
    port = args.port
    
    # Print startup banner
    print("\n" + "=" * 70)
    print(" MCP-POWERED ANALYTICS BOT WITH SLACK APP HOME INTEGRATION ".center(70, "="))
    print("=" * 70)
    
    # Initialize components
    orchestrator = initialize_components()
    
    # Set up ngrok if requested
    ngrok_process = None
    if args.ngrok:
        ngrok_process = setup_ngrok(port)
    
    try:
        # Create and start Slack App Home integration
        app_home = MCPSlackAppHome(orchestrator)
        app_home.start(port=port)
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        if ngrok_process:
            ngrok_process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if ngrok_process:
            ngrok_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main() 