#!/usr/bin/env python
"""
Comprehensive test script for the enhanced session management system.
This script tests all aspects of the session management system, including:
- Creating a session
- Updating constraints, strategy, and response plan
- Updating tool call status with results
- Querying sessions
- Recreating the sessions table (optional)

Usage:
    python test_full_session_system.py [--recreate]
    
Options:
    --recreate    Recreate the sessions table before running tests
"""

import os
import json
import time
import datetime
import argparse
from dotenv import load_dotenv
from session_manager_v2 import SessionManagerV2
import subprocess

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_json(data):
    """Print JSON data with formatting."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            print(data)
            return
    print(json.dumps(data, indent=2, default=str))

def recreate_sessions_table():
    """Recreate the sessions table with the new schema."""
    print_section("Recreating Sessions Table")
    try:
        from create_sessions_table import create_sessions_table
        create_sessions_table(force_recreate=True)
        print("Sessions table recreated successfully.")
    except Exception as e:
        print(f"Error recreating sessions table: {e}")
        return False
    return True

def test_session_creation():
    """Test creating a new session."""
    print_section("Testing Session Creation")
    
    # Initialize session manager
    session_manager = SessionManagerV2()
    
    # Create a test user ID
    user_id = f"test_user_{int(time.time())}"
    
    # Define a test question
    question = """
    What are the order and slot availability trends for CFC-1 and CFC-2 over the past month?
    Can you compare them and identify any patterns or anomalies?
    """
    
    # Create a new session
    session_id = session_manager.create_session(user_id, question)
    print(f"Created session with ID: {session_id}")
    
    # Get the session
    session = session_manager.get_session(session_id)
    print("Initial session data:")
    print_json(session)
    
    return session_manager, session_id, user_id

def test_update_constraints(session_manager, session_id):
    """Test updating session constraints."""
    print_section("Testing Update Constraints")
    
    # Define constraints
    constraints = {
        "kpi": ["orders", "slot_availability"],
        "time_aggregation": "Weekly",
        "time_filter": {
            "start_date": "2025-02-09",
            "end_date": "2025-03-11"
        },
        "cfc": ["CFC-1", "CFC-2"],
        "spokes": [],
        "comparison_type": "trend",
        "tool_calls": [
            {
                "name": "Orders Trend for CFC-1",
                "description": "This query retrieves order trends for CFC-1 over the past month",
                "tables": ["orders"],
                "result_id": "orders_trend_cfc1"
            },
            {
                "name": "Orders Trend for CFC-2",
                "description": "This query retrieves order trends for CFC-2 over the past month",
                "tables": ["orders"],
                "result_id": "orders_trend_cfc2"
            },
            {
                "name": "Slot Availability for CFC-1",
                "description": "This query retrieves slot availability for CFC-1 over the past month",
                "tables": ["slots"],
                "result_id": "slots_cfc1"
            },
            {
                "name": "Slot Availability for CFC-2",
                "description": "This query retrieves slot availability for CFC-2 over the past month",
                "tables": ["slots"],
                "result_id": "slots_cfc2"
            }
        ]
    }
    
    # Update constraints
    session_manager.update_constraints(session_id, constraints)
    print("Updated constraints:")
    session = session_manager.get_session(session_id)
    print_json(session["constraints"])
    
    return constraints

def test_update_strategy(session_manager, session_id):
    """Test updating session strategy."""
    print_section("Testing Update Strategy")
    
    # Define strategy
    strategy = """
## Comprehensive Strategy

### 1. Data Collection Plan:
* **Desired Data**: Weekly order counts and slot availability for CFC-1 and CFC-2
* **Data Sources**: Orders table and Slots table
* **Fetch Specifics**: 
  - Orders: Select columns: `week_date`, `orders`, `cfc`
  - Slots: Select columns: `week_date`, `available_slots`, `total_slots`, `cfc`

### 2. Processing Steps:
* **Calculate weekly order aggregates**: Group orders by `cfc` and `week_date`
* **Calculate weekly slot availability**: Group slots by `cfc` and `week_date`
* **Calculate slot availability percentage**: `available_slots / total_slots * 100`

### 3. Calculations Required:
* **Sum of orders by week for each CFC**: `SUM(orders) GROUP BY cfc, week_date`
* **Average slot availability by week for each CFC**: `AVG(available_slots / total_slots * 100) GROUP BY cfc, week_date`

### 4. Comparison Analysis:
* **Compare order trends between CFCs**: Identify differences in order patterns
* **Compare slot availability between CFCs**: Identify differences in slot availability
* **Correlation analysis**: Check if there's a relationship between orders and slot availability

### 5. Response Structure:
* **Introduction**: Overview of the analysis scope and objectives
* **Order Trends Analysis**: Detailed findings on order trends for both CFCs
* **Slot Availability Analysis**: Detailed findings on slot availability for both CFCs
* **Comparative Analysis**: Comparison between the two CFCs
* **Anomalies and Patterns**: Identification of any unusual patterns or anomalies
* **Conclusion**: Summary of key findings and insights
"""
    
    # Update strategy
    session_manager.update_strategy(session_id, strategy)
    print("Updated strategy:")
    session = session_manager.get_session(session_id)
    
    # Print the strategy text if it's in the new JSON format
    if isinstance(session["strategy"], dict) and "content" in session["strategy"]:
        print(session["strategy"]["content"])
    else:
        print(session["strategy"])
    
    return strategy

def test_update_response_plan(session_manager, session_id):
    """Test updating session response plan."""
    print_section("Testing Update Response Plan")
    
    # Define response plan
    response_plan = {
        "data_connections": [
            {
                "result_id": "orders_trend_cfc1",
                "purpose": "Analyze order trends for CFC-1",
                "processing_steps": ["Aggregate orders by week"],
                "outputs": ["Weekly order trends for CFC-1"]
            },
            {
                "result_id": "orders_trend_cfc2",
                "purpose": "Analyze order trends for CFC-2",
                "processing_steps": ["Aggregate orders by week"],
                "outputs": ["Weekly order trends for CFC-2"]
            },
            {
                "result_id": "slots_cfc1",
                "purpose": "Analyze slot availability for CFC-1",
                "processing_steps": ["Calculate slot availability percentage by week"],
                "outputs": ["Weekly slot availability for CFC-1"]
            },
            {
                "result_id": "slots_cfc2",
                "purpose": "Analyze slot availability for CFC-2",
                "processing_steps": ["Calculate slot availability percentage by week"],
                "outputs": ["Weekly slot availability for CFC-2"]
            }
        ],
        "insights": [
            {
                "type": "trend",
                "description": "Analyze weekly order trends for both CFCs",
                "source_result_ids": ["orders_trend_cfc1", "orders_trend_cfc2"]
            },
            {
                "type": "trend",
                "description": "Analyze weekly slot availability for both CFCs",
                "source_result_ids": ["slots_cfc1", "slots_cfc2"]
            },
            {
                "type": "comparison",
                "description": "Compare order trends between CFCs",
                "source_result_ids": ["orders_trend_cfc1", "orders_trend_cfc2"]
            },
            {
                "type": "comparison",
                "description": "Compare slot availability between CFCs",
                "source_result_ids": ["slots_cfc1", "slots_cfc2"]
            },
            {
                "type": "correlation",
                "description": "Analyze correlation between orders and slot availability",
                "source_result_ids": ["orders_trend_cfc1", "slots_cfc1", "orders_trend_cfc2", "slots_cfc2"]
            }
        ],
        "response_structure": {
            "introduction": "This analysis examines order trends and slot availability for CFC-1 and CFC-2 over the past month.",
            "main_points": [
                "Key trends in orders for both CFCs",
                "Key trends in slot availability for both CFCs",
                "Comparison between CFCs",
                "Identified patterns and anomalies"
            ],
            "conclusion": "The analysis provides insights into order and slot availability patterns for both CFCs."
        }
    }
    
    # Update response plan
    session_manager.update_response_plan(session_id, response_plan)
    print("Updated response plan:")
    session = session_manager.get_session(session_id)
    print_json(session["response_plan"])
    
    return response_plan

def test_update_status(session_manager, session_id):
    """Test updating session status."""
    print_section("Testing Update Status")
    
    # Update session status to processing
    session_manager.update_session(session_id, {"status": "processing"})
    print("Updated session status to 'processing'")
    
    # Get the session
    session = session_manager.get_session(session_id)
    print(f"Current status: {session['status']}")
    
    return session

def test_tool_call_updates(session_manager, session_id):
    """Test updating tool call status with results."""
    print_section("Testing Tool Call Updates")
    
    # Update tool call status for Orders Trend for CFC-1
    tool_name = "Orders Trend for CFC-1"
    tool_result = {
        "sql": "SELECT week_date, SUM(orders) AS total_orders FROM orders WHERE cfc = 'CFC-1' GROUP BY week_date ORDER BY week_date",
        "data": [
            {"week_date": "2025-02-09", "total_orders": 1200},
            {"week_date": "2025-02-16", "total_orders": 1350},
            {"week_date": "2025-02-23", "total_orders": 1100},
            {"week_date": "2025-03-02", "total_orders": 1250},
            {"week_date": "2025-03-09", "total_orders": 1300}
        ]
    }
    session_manager.update_tool_call_status(session_id, tool_name, "completed", tool_result)
    print(f"Updated tool call status for '{tool_name}' to 'completed' with results")
    
    # Update tool call status for Orders Trend for CFC-2
    tool_name = "Orders Trend for CFC-2"
    tool_result = {
        "sql": "SELECT week_date, SUM(orders) AS total_orders FROM orders WHERE cfc = 'CFC-2' GROUP BY week_date ORDER BY week_date",
        "data": [
            {"week_date": "2025-02-09", "total_orders": 950},
            {"week_date": "2025-02-16", "total_orders": 1050},
            {"week_date": "2025-02-23", "total_orders": 900},
            {"week_date": "2025-03-02", "total_orders": 1000},
            {"week_date": "2025-03-09", "total_orders": 1100}
        ]
    }
    session_manager.update_tool_call_status(session_id, tool_name, "completed", tool_result)
    print(f"Updated tool call status for '{tool_name}' to 'completed' with results")
    
    # Update tool call status for Slot Availability for CFC-1
    tool_name = "Slot Availability for CFC-1"
    tool_result = {
        "sql": "SELECT week_date, AVG(available_slots / total_slots * 100) AS availability_percentage FROM slots WHERE cfc = 'CFC-1' GROUP BY week_date ORDER BY week_date",
        "data": [
            {"week_date": "2025-02-09", "availability_percentage": 85.5},
            {"week_date": "2025-02-16", "availability_percentage": 78.2},
            {"week_date": "2025-02-23", "availability_percentage": 82.7},
            {"week_date": "2025-03-02", "availability_percentage": 80.1},
            {"week_date": "2025-03-09", "availability_percentage": 75.8}
        ]
    }
    session_manager.update_tool_call_status(session_id, tool_name, "completed", tool_result)
    print(f"Updated tool call status for '{tool_name}' to 'completed' with results")
    
    # Update tool call status for Slot Availability for CFC-2
    tool_name = "Slot Availability for CFC-2"
    tool_result = {
        "sql": "SELECT week_date, AVG(available_slots / total_slots * 100) AS availability_percentage FROM slots WHERE cfc = 'CFC-2' GROUP BY week_date ORDER BY week_date",
        "data": [
            {"week_date": "2025-02-09", "availability_percentage": 90.2},
            {"week_date": "2025-02-16", "availability_percentage": 88.5},
            {"week_date": "2025-02-23", "availability_percentage": 91.3},
            {"week_date": "2025-03-02", "availability_percentage": 89.7},
            {"week_date": "2025-03-09", "availability_percentage": 87.4}
        ]
    }
    session_manager.update_tool_call_status(session_id, tool_name, "completed", tool_result)
    print(f"Updated tool call status for '{tool_name}' to 'completed' with results")
    
    # Check if session is complete
    is_complete = session_manager.is_session_complete(session_id)
    print(f"Is session complete? {is_complete}")
    
    # Get the session
    session = session_manager.get_session(session_id)
    print("Tool call status:")
    for i, tool_call in enumerate(session["tool_calls"]):
        print(f"  {tool_call}: {session['tool_call_status'][i]}")
    
    print("\nTool call results:")
    print_json(session["tool_call_results"])
    
    return session

def test_update_summary(session_manager, session_id):
    """Test updating session summary."""
    print_section("Testing Update Summary")
    
    # Define summary
    summary = """
## Order and Slot Availability Trends for CFC-1 and CFC-2

This analysis examines order trends and slot availability for CFC-1 and CFC-2 over the past month, broken down by week.

### Order Trends

#### CFC-1
- Orders peaked in the week of February 16 with 1,350 orders
- The lowest order volume was in the week of February 23 with 1,100 orders
- The average weekly order volume was 1,240 orders

#### CFC-2
- Orders peaked in the week of March 9 with 1,100 orders
- The lowest order volume was in the week of February 23 with 900 orders
- The average weekly order volume was 1,000 orders

### Slot Availability

#### CFC-1
- Slot availability was highest in the week of February 9 at 85.5%
- Slot availability was lowest in the week of March 9 at 75.8%
- The average weekly slot availability was 80.5%

#### CFC-2
- Slot availability was highest in the week of February 23 at 91.3%
- Slot availability was lowest in the week of March 9 at 87.4%
- The average weekly slot availability was 89.4%

### Comparative Analysis

- CFC-1 consistently had higher order volumes than CFC-2 (24% higher on average)
- CFC-2 consistently had higher slot availability than CFC-1 (11% higher on average)
- Both CFCs showed a decrease in slot availability in the week of March 9
- There appears to be an inverse relationship between order volume and slot availability

### Key Insights

1. CFC-1 has higher demand but lower slot availability compared to CFC-2
2. The week of February 23 showed a dip in orders for both CFCs
3. Slot availability generally decreases as order volume increases
"""
    
    # Update summary
    session_manager.update_summary(session_id, summary)
    print("Updated summary:")
    session = session_manager.get_session(session_id)
    print(session["summary"])
    
    # Update session status to completed
    session_manager.update_session(session_id, {"status": "completed"})
    print("Updated session status to 'completed'")
    
    return summary

def test_query_sessions(session_id):
    """Test querying sessions."""
    print_section("Testing Query Sessions")
    
    # Query all sessions
    print("Querying all sessions:")
    subprocess.run(["python", "query_sessions.py", "--limit", "5"])
    
    # Query specific session
    print("\nQuerying specific session:")
    subprocess.run(["python", "query_sessions.py", session_id])
    
    # Query all sessions with details
    print("\nQuerying all sessions with details:")
    subprocess.run(["python", "query_sessions.py", "--limit", "5", "--details"])

def main():
    """Main function to run all tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the enhanced session management system')
    parser.add_argument('--recreate', action='store_true', help='Recreate the sessions table before running tests')
    args = parser.parse_args()
    
    print_section("Starting Comprehensive Session Management System Test")
    
    # Recreate sessions table if requested
    if args.recreate:
        if not recreate_sessions_table():
            print("Exiting due to error recreating sessions table.")
            return
    
    # Test session creation
    session_manager, session_id, user_id = test_session_creation()
    
    # Test updating constraints
    constraints = test_update_constraints(session_manager, session_id)
    
    # Test updating strategy
    strategy = test_update_strategy(session_manager, session_id)
    
    # Test updating response plan
    response_plan = test_update_response_plan(session_manager, session_id)
    
    # Test updating status
    session = test_update_status(session_manager, session_id)
    
    # Test tool call updates
    session = test_tool_call_updates(session_manager, session_id)
    
    # Test updating summary
    summary = test_update_summary(session_manager, session_id)
    
    # Test querying sessions
    test_query_sessions(session_id)
    
    print_section("Test Completed Successfully")
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")

if __name__ == "__main__":
    main() 