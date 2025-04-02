# Follow-up Question Pipeline Documentation

## Overview
The follow-up question pipeline is designed to handle questions that reference or build upon previous conversation context within a Slack thread. The system uses session IDs (based on Slack thread IDs) to maintain context and determine whether new SQL queries are needed or existing data can be reused.

## Key Components

### 1. Question Classification
- **Component**: `QuestionClassifier` (question_classifier.py)
- **Purpose**: Identifies if a question is a follow-up and determines SQL requirements
- **Classification Rules**:
  - Looks for references to previous context ("that", "these", "those")
  - Checks for modifications to previous analysis
  - Determines if new SQL is needed or existing data can be used
  - Sets `requires_sql` flag based on analysis

### 2. Context Management
- **Component**: `SessionManagerV2` (session_manager_v2.py)
- **Purpose**: Maintains conversation context within threads
- **Key Functions**:
  - Stores previous question, summary, and results
  - Uses thread_ts as session_id for continuity
  - Retrieves latest context using `ORDER BY updated_at DESC LIMIT 1`

### 3. Constraint Extraction
- **Component**: `QueryProcessor` (query_processor.py)
- **Purpose**: Extracts query constraints considering previous context
- **Processing Steps**:
  1. Receives previous context (question, summary, results)
  2. Analyzes if new data is needed
  3. Generates appropriate tool_calls based on analysis
  4. Updates response_plan to use previous data when applicable

### 4. Response Generation
- **Component**: `ResponseAgent` (response_agent.py)
- **Purpose**: Generates contextual responses
- **Key Features**:
  - Incorporates previous context in prompts
  - References previous results when appropriate
  - Maintains conversation continuity

## Decision Flow

### 1. Follow-up Detection
```
User Question
↓
Question Classification
↓
Check for:
- References to previous data
- Temporal modifiers
- Location changes
- Detail requests
```

### 2. SQL Requirement Analysis
```
Follow-up Detected
↓
Analyze Modification Type:
├─ Time Period Change → New SQL needed
├─ Location Change → New SQL needed
├─ Granularity Change → Check existing data
└─ Clarification → Use existing data
```

### 3. Data Utilization
```
Previous Context Available
↓
Check if needed data exists:
├─ Yes → Use existing results
└─ No → Generate new SQL
```

## Example Scenarios

### 1. Time Period Modification
```
Initial: "Show me orders for London in 2024"
Follow-up: "What about Q1?"
Action: New SQL (different time period)
```

### 2. Detail Request
```
Initial: "What was the total orders last week?"
Follow-up: "Show me day by day"
Action: Check if daily data exists in previous results
```

### 3. Location Addition
```
Initial: "Compare ATP between Stevenage and London"
Follow-up: "Add Manchester to that"
Action: New SQL (additional location)
```

### 4. Clarification
```
Initial: "Compare CFC performance"
Follow-up: "What does ATP mean?"
Action: No SQL (use existing data)
```

## Implementation Details

### 1. Session Context Structure
```json
{
  "session_id": "thread_ts",
  "previous_question": "original question",
  "previous_summary": "previous response",
  "previous_results": {
    "data": [...],
    "summary": {...}
  }
}
```

### 2. Tool Call Decision
```python
if needs_new_data:
    tool_calls = [
        {
            "name": "get_specific_data",
            "description": "Fetch new data for specific requirement",
            "result_id": "unique_identifier"
        }
    ]
else:
    tool_calls = []  # Use existing data
```

### 3. Response Plan Structure
```json
{
  "data_connections": [
    {
      "result_id": "identifier",
      "purpose": "data usage description",
      "processing_steps": ["step1", "step2"],
      "outputs": ["expected insights"]
    }
  ],
  "response_structure": {
    "introduction": "context from previous",
    "main_points": ["key findings"],
    "conclusion": "summary with context"
  }
}
```

## Best Practices

1. **Context Preservation**
   - Always maintain thread context
   - Store relevant previous data
   - Track data granularity

2. **SQL Optimization**
   - Avoid redundant queries
   - Check existing data first
   - Cache common queries

3. **Response Coherence**
   - Reference previous context
   - Explain changes clearly
   - Maintain conversation flow

## Common Patterns

1. **Time-based Follow-ups**
   - Different period
   - Different granularity
   - Historical comparison

2. **Location-based Follow-ups**
   - Additional locations
   - Location comparison
   - Location details

3. **Detail-based Follow-ups**
   - Breakdown requests
   - Specific metrics
   - Clarifications

## Error Handling

1. **Missing Context**
   - Fallback to new query
   - Request clarification
   - Log context miss

2. **Data Granularity Mismatch**
   - Check available granularity
   - Generate new query if needed
   - Explain limitations

3. **Invalid Follow-ups**
   - Validate context relevance
   - Request clarification
   - Maintain thread context 