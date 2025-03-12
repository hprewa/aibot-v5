import os
from dotenv import load_dotenv
from gemini_client import GeminiClient
from bigquery_client import BigQueryClient
from query_processor import QueryProcessor
from session_manager import SessionManager
import json
from typing import Dict, Any
import time

def print_analysis(question: str, constraints: Dict[str, Any], strategy: str):
    """Pretty print the analysis results"""
    print("\n" + "="*80)
    print(f"Question: {question}")
    print("-"*80)
    print("Extracted Constraints:")
    print(json.dumps(constraints, indent=2))
    print("-"*80)
    print("Strategy:")
    print(strategy)
    print("="*80 + "\n")

def test_response_agent_flow(question: str, constraints: Dict[str, Any], strategy: str):
    """Test the Response Agent flow with mock results"""
    print("\nTesting Response Agent Flow...")
    print("-"*50)
    
    # Create mock results for each tool call
    mock_results = {}
    for tool_call in constraints.get("tool_calls", []):
        result_id = tool_call.get("result_id", f"result_{len(mock_results)}")
        # Create mock data based on the tool call
        mock_results[result_id] = [
            {"value": 100, "date": "2023-01-01", "cfc": "CFC-1", "spoke": "S001"},
            {"value": 150, "date": "2023-01-02", "cfc": "CFC-1", "spoke": "S001"},
            {"value": 200, "date": "2023-01-03", "cfc": "CFC-1", "spoke": "S001"}
        ]
    
    # Print the response plan
    print("Response Plan:")
    response_plan = constraints.get("response_plan", {})
    print(json.dumps(response_plan, indent=2))
    
    # Print data connections
    print("\nData Connections:")
    for connection in response_plan.get("data_connections", []):
        print(f"- Result ID: {connection.get('result_id')}")
        print(f"  Purpose: {connection.get('purpose')}")
        print(f"  Processing Steps: {', '.join(connection.get('processing_steps', []))}")
    
    # Print insights
    print("\nInsights to be derived:")
    for insight in response_plan.get("insights", []):
        print(f"- Type: {insight.get('type')}")
        print(f"  Description: {insight.get('description')}")
        print(f"  Source Results: {', '.join(insight.get('source_result_ids', []))}")
    
    # Print response structure
    print("\nResponse Structure:")
    structure = response_plan.get("response_structure", {})
    print(f"- Introduction: {structure.get('introduction')}")
    print(f"- Main Points: {', '.join(structure.get('main_points', []))}")
    print(f"- Context: {structure.get('context')}")
    print(f"- Conclusion: {structure.get('conclusion')}")
    
    print("-"*50)
    return mock_results

def test_strategy_agent():
    """Test the Strategy Agent with complex analytical questions"""
    load_dotenv()
    
    # Initialize components
    print("Initializing test components...")
    bigquery_client = BigQueryClient()
    gemini_client = GeminiClient()
    query_processor = QueryProcessor(gemini_client, bigquery_client)
    session_manager = SessionManager()
    
    # Complex test questions involving multiple tool calls
    test_questions = [
        # Multi-KPI Analysis
        "Compare orders and slot availability trends for CFC-1 and CFC-2 over the past month, broken down by week",
        
        # Network vs CFC Analysis
        "How do the total orders for each CFC compare to the network average last week? Show percentage differences",
        
        # Complex Spoke Analysis
        "For CFC-1, identify the top 3 spokes by order volume and analyze their slot availability patterns over the past month"
    ]
    
    # Process each test question
    for question in test_questions:
        try:
            print(f"\nProcessing question: {question}")
            
            # Add delay between questions to avoid rate limiting
            if test_questions.index(question) > 0:
                print("Waiting 30 seconds to avoid rate limiting...")
                time.sleep(30)
            
            # Create a test session
            session_id = session_manager.create_session("test_user", question)
            
            # Extract constraints
            print("Extracting constraints...")
            constraints = query_processor.extract_constraints(question)
            session_manager.update_session(session_id, {"constraints": constraints})
            
            # Add small delay between API calls
            time.sleep(5)
            
            # Generate strategy
            print("Generating strategy...")
            strategy = query_processor.generate_strategy(question, constraints)
            session_manager.update_session(session_id, {"strategy": strategy})
            
            # Print analysis
            print_analysis(question, constraints, strategy)
            
            # Validate tool calls
            tool_calls = constraints.get("tool_calls", [])
            if not tool_calls:
                print("Warning: No tool calls generated for this question")
            else:
                print(f"\nGenerated {len(tool_calls)} tool calls:")
                for i, call in enumerate(tool_calls, 1):
                    print(f"\nTool Call {i}:")
                    print(f"Name: {call.get('name', 'Unnamed')}")
                    print(f"Description: {call.get('description', 'No description')}")
                    print(f"Tables: {', '.join(call.get('tables', []))}")
                    print(f"Result ID: {call.get('result_id', 'No ID')}")
                    
                    # Generate SQL for this tool call
                    print("\nGenerating SQL for this tool call...")
                    sql = query_processor.generate_sql_for_tool_call(question, constraints, call)
                    print(f"SQL: {sql}")
            
            # Test Response Agent flow with mock results
            mock_results = test_response_agent_flow(question, constraints, strategy)
            
            # Generate summary using response plan
            print("\nGenerating summary with response plan...")
            summary = query_processor.generate_summary(question, mock_results, constraints)
            print("\nGenerated Summary:")
            print(summary)
            
        except Exception as e:
            print(f"\nError processing question '{question}': {str(e)}")
            print("Continuing with next question after delay...")
            time.sleep(30)

def validate_strategy_output(strategy: str) -> bool:
    """Validate that strategy output contains all required sections"""
    required_sections = [
        "Data Collection Plan",
        "Processing Steps",
        "Calculations Required",
        "Response Structure"
    ]
    
    return all(section.lower() in strategy.lower() for section in required_sections)

if __name__ == "__main__":
    test_strategy_agent() 