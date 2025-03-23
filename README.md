# Analytics Bot with BigQuery Session Management

A system that processes analytical queries using Google BigQuery and Vertex AI (Gemini), with a robust session management system that tracks query state and results.

## Features

- Natural language processing of analytical queries
- Automatic SQL query generation
- BigQuery integration for data retrieval
- Advanced session management with append-only pattern
- Intelligent summarization of results
- DSPy-powered Query Agent for structured query generation
- Model Context Protocol (MCP) for standardized context flow
- Slack App Home integration for dedicated bot window experience

## Session Management System

The session management system uses BigQuery for storing and retrieving session data with an append-only pattern:

- **Append-Only Pattern**: Due to BigQuery's streaming buffer limitations (which prevent updating recently inserted rows), we use an append-only pattern for session updates.
- **Session History**: Each update to a session creates a new row, allowing us to maintain a complete history of session updates.
- **Latest State Retrieval**: When querying a session, we get the most recent row for that session ID.

## Query Agent System

The Query Agent system uses DSPy tables and Gemini to generate SQL queries:

- **DSPy Tables**: Structured definitions of KPI tables with methods for generating SQL queries
- **KPI Handlers**: Specialized methods for handling different types of KPIs
- **Gemini Fallback**: For complex queries not covered by DSPy tables
- **Modular Design**: Easy to extend with new KPIs

For more details, see [QUERY_AGENT_README.md](QUERY_AGENT_README.md).

## Model Context Protocol (MCP)

The Analytics Bot now implements the Model Context Protocol, a standardized way to manage context flow between components:

- **Typed Context Objects**: Replaces untyped dictionaries with strongly-typed Pydantic models
- **Context Transformation**: Explicit, traceable flow of context between components
- **Standardized Error Handling**: Consistent error handling throughout the system
- **Immutable Context**: Each transformation creates a new context object, preserving history
- **Enhanced Debugging**: The exact state of context at each step can be inspected

For more details, see [MCP_README.md](MCP_README.md).

## Slack App Home Integration

The Analytics Bot includes a Slack App Home integration that provides a dedicated bot window experience directly within Slack:

- **MCP-powered**: Uses the Model Context Protocol for robust, typed context flow
- **Private Conversations**: Each user has their own private conversation space
- **Per-User History**: Complete conversation history displayed in the App Home tab
- **Status Indicators**: Uses Slack reactions to indicate processing status
- **Threaded Replies**: Responses are organized in threads for better context tracking
- **Native Slack Experience**: No need to leave Slack or learn a new interface

### Slack App Configuration

1. Create a Slack app at https://api.slack.com/apps
2. Under "OAuth & Permissions", add the following scopes:
   - `chat:write` (for sending messages)
   - `im:history` (for reading direct messages)
   - `im:write` (for sending direct messages)
   - `app_home:update` (for updating the App Home tab)
   - `reactions:write` (for adding/removing reactions)

3. Install the app to your workspace

4. Under "Event Subscriptions":
   - Enable events
   - Set the Request URL to your endpoint (e.g., ngrok URL + `/slack/events`)
   - Subscribe to the following events under "Subscribe to bot events":
     - `app_home_opened`
     - `message.im`

5. Under "App Home":
   - Enable the Home Tab
   - Enable the Messages Tab
   - Check "Allow users to send Slash commands and messages from the messages tab"

6. Reinstall the app to your workspace after making changes

For detailed setup instructions, see [README_SLACK_SETUP.md](README_SLACK_SETUP.md) and [APP_HOME_README.md](APP_HOME_README.md).

## Prerequisites

- Python 3.8+
- Google Cloud Platform account with:
  - BigQuery API enabled
  - Vertex AI API enabled
  - Service account with appropriate permissions
- Slack workspace with:
  - Admin privileges to create a Slack app
  - Bot User OAuth Token
  - Signing Secret (for HTTP Events API)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd analytics-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
# Slack API credentials - BOTH required for Events API
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# Google Cloud credentials
PROJECT_ID=your-gcp-project-id
DATASET_ID=your-bigquery-dataset
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Application settings
PORT=8000
```

4. Configure Google Cloud credentials:
- Download your service account key file
- Set the path in GOOGLE_APPLICATION_CREDENTIALS

## Running the System

### Option 1: FastAPI Server Only

Run the MCP-powered FastAPI server:
```bash
python mcp_server.py
```

### Option 2: Slack App Home Integration

Run the MCP-powered Slack App Home integration:
```bash
python run_mcp_app_home.py
```

With ngrok for external access:
```bash
python run_mcp_app_home.py --ngrok
```

This will start:
- The FastAPI-based Slack App Home integration server
- An ngrok tunnel (if --ngrok flag is used) for public access

For Windows users, a PowerShell script is provided:
```powershell
# Run the bot (local only)
.\run_app_home.ps1

# Run with ngrok tunnel for public access
.\run_app_home.ps1 -UseNgrok
```

## Project Structure

The application consists of several components:

- **Core MCP Implementation**:
  - `mcp/protocol.py`: Core protocol classes
  - `mcp/models.py`: Data models for context objects
  - `mcp/processors.py`: MCP wrappers for original components
  - `mcp/slack_app_home.py`: MCP-compatible Slack App Home integration

- **Entry Points**:
  - `mcp_server.py`: MCP-powered FastAPI server
  - `run_mcp_app_home.py`: Integrated server with Slack App Home support
  - `run_app_home.ps1`: PowerShell script for running the Slack App Home integration

- **Original Components** (wrapped by MCP):
  - `bigquery_client.py`: Handles BigQuery interactions
  - `gemini_client.py`: Manages Vertex AI/Gemini interactions
  - `query_processor.py`: Processes queries and extracts constraints
  - `query_agent.py`: Generates SQL queries for different KPIs
  - `response_agent.py`: Generates natural language responses
  - `session_manager_v2.py`: Manages query session state
  - `dspy_tables.py`: Defines DSPy table schemas for different KPIs

- **Documentation**:
  - `README.md`: Main project documentation
  - `README_SLACK_SETUP.md`: Detailed guide for setting up Slack integration
  - `APP_HOME_README.md`: Documentation for the Slack App Home integration
  - `MCP_README.md`: Documentation for the MCP implementation
  - `QUERY_AGENT_README.md`: Documentation for the Query Agent system
  - `SESSIONS_README.md`: Documentation for the Session Management system

- **Utilities**:
  - `get_ngrok_url.py`: Script to get the current ngrok URL
  - `ngrok.exe`: Tunnel tool for exposing local server to the internet

## Maintenance and Testing

### Updating Bot Permissions

If you need to update Slack permissions, follow the troubleshooting section in [README_SLACK_SETUP.md](README_SLACK_SETUP.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.