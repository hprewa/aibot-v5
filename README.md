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
- Slack integration for interactive querying

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

## Slack Integration

The Analytics Bot includes a Slack integration that allows users to interact with the bot directly from Slack:

- **MCP-powered**: Uses the Model Context Protocol for robust, typed context flow
- **Status Indicators**: Uses Slack reactions to indicate processing status
- **Threaded Replies**: Responses are posted as threaded replies to keep channels clean
- **Real-time Processing**: Messages are processed in real-time with status updates

### Slack App Configuration

1. Create a Slack app at https://api.slack.com/apps
2. Under "OAuth & Permissions", add the following scopes:
   - `chat:write` (for sending messages)
   - `reactions:write` (for adding/removing reactions)
   - `reactions:read` (for checking reactions)
   - `channels:history` (for reading channel messages)
   - `im:history` (for reading direct messages)
   - `groups:history` (for reading private channel messages)

3. Install the app to your workspace

4. Under "Event Subscriptions":
   - Enable events
   - Set the Request URL to your endpoint (e.g., ngrok URL + `/slack/events`)
   - Subscribe to the following events under "Subscribe to bot events":
     - `message.channels`
     - `message.im`

5. Reinstall the app to your workspace after making changes

For detailed setup instructions, see [README_SLACK_SETUP.md](README_SLACK_SETUP.md).

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

### Option 2: Slack Integration

Run the MCP-powered Slack integration:
```bash
python run_mcp_slack.py
```

With ngrok for external access:
```bash
python run_mcp_slack.py --ngrok
```

This will start:
- The FastAPI-based Slack integration server
- An ngrok tunnel (if --ngrok flag is used) for public access

For Windows users, a PowerShell script is provided:
```powershell
# Run the bot (local only)
.\run_slack_bot.ps1

# Run with ngrok tunnel for public access
.\run_slack_bot.ps1 -ngrok
```

## Project Structure

The application consists of several components:

- **Core MCP Implementation**:
  - `mcp/protocol.py`: Core protocol classes
  - `mcp/models.py`: Data models for context objects
  - `mcp/processors.py`: MCP wrappers for original components
  - `mcp/slack_integration.py`: MCP-compatible Slack integration

- **Entry Points**:
  - `mcp_server.py`: MCP-powered FastAPI server
  - `run_mcp_slack.py`: Integrated server with Slack support
  - `run_slack_bot.ps1`: PowerShell script for running the Slack integration

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
  - `MCP_README.md`: Documentation for the MCP implementation
  - `QUERY_AGENT_README.md`: Documentation for the Query Agent system
  - `SESSIONS_README.md`: Documentation for the Session Management system

- **Utilities**:
  - `test_slack.py`: Tool for testing Slack API connectivity
  - `ngrok.exe`: Tunnel tool for exposing local server to the internet

## Maintenance and Testing

### Testing Slack Integration

To test Slack API connectivity before running the full application:

```bash
python test_slack.py
```

This script will verify your Slack credentials and permissions.

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