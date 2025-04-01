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
from importlib import import_module
from typing import Any

def lazy_import(module_name: str, class_name: str = None) -> Any:
    """Lazily import a module or class"""
    try:
        module = import_module(module_name)
        return getattr(module, class_name) if class_name else module
    except ImportError as e:
        print(f"Error importing {module_name}: {str(e)}")
        raise

class LazyLoader:
    """Lazy loader for MCP components"""
    _components = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_component(cls, module_name: str, class_name: str = None) -> Any:
        key = f"{module_name}.{class_name}" if class_name else module_name
        if key not in cls._components:
            with cls._lock:
                if key not in cls._components:
                    cls._components[key] = lazy_import(module_name, class_name)
        return cls._components[key]

def setup_ngrok(port=8000):
    """Set up ngrok if available"""
    ngrok_start = time.time()
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
                ngrok_time = time.time() - ngrok_start
                print(f"\n✅ Ngrok tunnel established in {ngrok_time:.2f} seconds!")
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
    component_start = time.time()
    
    print("Initializing BigQuery client...")
    bq_start = time.time()
    BigQueryClient = LazyLoader.get_component("bigquery_client", "BigQueryClient")
    bigquery_client = BigQueryClient()
    bq_time = time.time() - bq_start
    print(f"✅ BigQuery client initialized in {bq_time:.2f} seconds")
    
    print("Initializing Gemini client...")
    gemini_start = time.time()
    GeminiClient = LazyLoader.get_component("gemini_client", "GeminiClient")
    gemini_client = GeminiClient()
    gemini_time = time.time() - gemini_start
    print(f"✅ Gemini client initialized in {gemini_time:.2f} seconds")
    
    print("Initializing query components...")
    query_start = time.time()
    QueryProcessor = LazyLoader.get_component("query_processor", "QueryProcessor")
    QueryAgent = LazyLoader.get_component("query_agent", "QueryAgent")
    ResponseAgent = LazyLoader.get_component("response_agent", "ResponseAgent")
    SessionManagerV2 = LazyLoader.get_component("session_manager_v2", "SessionManagerV2")
    QuestionClassifier = LazyLoader.get_component("question_classifier", "QuestionClassifier")
    
    query_processor = QueryProcessor(gemini_client, bigquery_client)
    query_agent = QueryAgent(gemini_client)
    response_agent = ResponseAgent(gemini_client)
    session_manager = SessionManagerV2()
    question_classifier = QuestionClassifier()
    query_time = time.time() - query_start
    print(f"✅ Query components initialized in {query_time:.2f} seconds")
    
    # Initialize MCP wrappers
    print("Initializing MCP wrappers...")
    mcp_start = time.time()
    
    # Lazy load MCP components
    MCPQuestionClassifier = LazyLoader.get_component("mcp", "MCPQuestionClassifier")
    MCPQueryProcessor = LazyLoader.get_component("mcp", "MCPQueryProcessor")
    MCPQueryAgent = LazyLoader.get_component("mcp", "MCPQueryAgent")
    MCPQueryExecutor = LazyLoader.get_component("mcp", "MCPQueryExecutor")
    MCPResponseGenerator = LazyLoader.get_component("mcp", "MCPResponseGenerator")
    MCPSessionManager = LazyLoader.get_component("mcp", "MCPSessionManager")
    MCPQueryFlowOrchestrator = LazyLoader.get_component("mcp", "MCPQueryFlowOrchestrator")
    MCPRouter = LazyLoader.get_component("mcp.router", "MCPRouter")
    
    mcp_question_classifier = MCPQuestionClassifier(question_classifier)
    mcp_query_processor = MCPQueryProcessor(query_processor)
    mcp_query_agent = MCPQueryAgent(query_agent)
    mcp_query_executor = MCPQueryExecutor(bigquery_client)
    mcp_response_generator = MCPResponseGenerator(response_agent)
    mcp_session_manager = MCPSessionManager(session_manager)
    mcp_time = time.time() - mcp_start
    print(f"✅ MCP wrappers initialized in {mcp_time:.2f} seconds")
    
    # Create router
    print("Creating MCP router...")
    router_start = time.time()
    mcp_router = MCPRouter()
    router_time = time.time() - router_start
    print(f"✅ Router created in {router_time:.2f} seconds")
    
    # Create the flow orchestrator
    print("Creating flow orchestrator...")
    orch_start = time.time()
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
    orch_time = time.time() - orch_start
    print(f"✅ Flow orchestrator created in {orch_time:.2f} seconds")
    
    total_time = time.time() - component_start
    print(f"\n✨ All components initialized in {total_time:.2f} seconds")
    
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
    total_start = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Load environment variables
    env_start = time.time()
    load_dotenv()
    env_time = time.time() - env_start
    print(f"\n✅ Environment variables loaded in {env_time:.2f} seconds")
    
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
        print("\n🚀 Starting Slack App Home integration...")
        server_start = time.time()
        
        # Lazy load MCPSlackAppHome
        MCPSlackAppHome = LazyLoader.get_component("mcp.slack_app_home", "MCPSlackAppHome")
        app_home = MCPSlackAppHome(orchestrator)
        app_home.start(port=port)
        
        server_time = time.time() - server_start
        print(f"✅ Server started in {server_time:.2f} seconds")
        
        total_time = time.time() - total_start
        print(f"\n🎉 Total startup completed in {total_time:.2f} seconds")
        
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