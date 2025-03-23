"""
Run script for the MCP-powered Analytics Bot with Slack integration.

This script initializes all components, creates the MCP orchestrator,
and starts the Slack integration server with FastAPI for HTTP Events API support.
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
    MCPQueryProcessor,
    MCPQueryAgent,
    MCPQueryExecutor,
    MCPResponseGenerator,
    MCPSessionManager,
    MCPQueryFlowOrchestrator
)

# Import original components
from bigquery_client import BigQueryClient
from gemini_client import GeminiClient
from query_processor import QueryProcessor
from query_agent import QueryAgent
from response_agent import ResponseAgent
from session_manager_v2 import SessionManagerV2

# Import Slack integration
from mcp.slack_integration import MCPSlackIntegration

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
                print("Remember to subscribe to the 'message.channels' and 'message.im' events.")
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
    
    # Initialize MCP wrappers
    mcp_query_processor = MCPQueryProcessor(query_processor)
    mcp_query_agent = MCPQueryAgent(query_agent)
    mcp_query_executor = MCPQueryExecutor(bigquery_client)
    mcp_response_generator = MCPResponseGenerator(response_agent)
    mcp_session_manager = MCPSessionManager(session_manager)
    
    # Create the flow orchestrator
    orchestrator = MCPQueryFlowOrchestrator(
        mcp_query_processor,
        mcp_query_agent,
        mcp_query_executor,
        mcp_response_generator,
        mcp_session_manager
    )
    
    return orchestrator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the MCP-powered Analytics Bot with Slack integration"
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
    """Main function to run the MCP-powered Analytics Bot"""
    # Parse command line arguments
    args = parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set port
    port = args.port
    
    # Print startup banner
    print("\n" + "=" * 60)
    print(" MCP-POWERED ANALYTICS BOT WITH SLACK INTEGRATION ".center(60, "="))
    print("=" * 60)
    
    # Initialize components
    orchestrator = initialize_components()
    
    # Set up ngrok if requested
    ngrok_process = None
    if args.ngrok:
        ngrok_process = setup_ngrok(port)
    
    try:
        # Create and start Slack integration
        slack_integration = MCPSlackIntegration(orchestrator)
        slack_integration.start(port=port)
        
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