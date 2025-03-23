# MCP-powered Slack Integration for Analytics Bot

This document describes how to use the Model Context Protocol (MCP) integration with Slack for the Analytics Bot project.

## Overview

The MCP Slack integration allows users to interact with the Analytics Bot directly from Slack. The integration leverages the Model Context Protocol to provide a structured, typed flow of context between components, resulting in improved traceability, debugging, and error handling.

## Architecture

The MCP Slack integration consists of these main components:

1. **MCP Framework**: Provides the context protocol and data models.
2. **MCPSlackIntegration**: Handles Slack events and integration with the MCP framework.
3. **MCPQueryFlowOrchestrator**: Processes queries using the MCP framework.
4. **Unified Run Script**: Starts both the FastAPI server and Slack integration.

## Features

- **Real-time Processing**: Messages are processed in real-time with status indicators.
- **Reaction Indicators**: Uses Slack reactions to indicate processing status (⏳, ✅, ❌).
- **Threaded Replies**: Responses are posted as threaded replies to keep channels clean.
- **Error Handling**: Robust error handling with informative error messages.
- **Status Updates**: Processing status updates are provided in real-time.

## Setup

1. **Environment Variables**:

   Make sure you have the following environment variables in your `.env` file:

   ```
   SLACK_BOT_TOKEN=xoxb-your-bot-token
   SLACK_APP_TOKEN=xapp-your-app-token
   SLACK_SIGNING_SECRET=your-signing-secret
   PROJECT_ID=your-gcp-project-id
   DATASET_ID=your-bigquery-dataset-id
   GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
   PORT=8000
   ```

2. **Slack App Configuration**:

   - Create a Slack app at https://api.slack.com/apps
   - Enable Socket Mode
   - Add the following scopes:
     - `channels:history`
     - `chat:write`
     - `reactions:write`
     - `im:history`
   - Subscribe to bot events:
     - `message.channels`
     - `message.im`
   - Install the app to your workspace

3. **Run the Integrated Server**:

   ```bash
   python run_mcp_slack.py
   ```

   This will start:
   - The MCP FastAPI server
   - The Slack integration
   - An ngrok tunnel (if available) for local development

## How It Works

1. **Message Reception**:
   - A user sends a message in a channel or DM where the bot is present
   - The bot receives the message through the Slack API

2. **Processing Indicators**:
   - The bot adds a ⏳ reaction to indicate processing
   - The bot replies in a thread with "Processing your question..."

3. **Query Processing**:
   - The message is converted to a `QueryData` context object
   - The context flows through the MCP orchestrator:
     - Extracting constraints
     - Generating a strategy
     - Creating SQL queries
     - Executing queries
     - Generating a response

4. **Response Delivery**:
   - The response is posted as a thread reply
   - The bot adds a ✅ reaction to the original message
   - The ⏳ reaction is removed

5. **Error Handling**:
   - If an error occurs, the bot adds a ❌ reaction
   - An error message is posted as a thread reply
   - The ⏳ reaction is removed

## Example Usage

1. **User sends a message**:
   ```
   How many orders did we have last week by CFC?
   ```

2. **Bot indicates processing**:
   - Adds ⏳ reaction
   - Replies: "Processing your question: 'How many orders did we have last week by CFC?'..."

3. **Bot delivers response**:
   - Posts detailed response as thread reply
   - Adds ✅ reaction
   - Removes ⏳ reaction

## Troubleshooting

- **Bot not responding**: Check if the bot is in the channel and has the correct permissions.
- **Error reactions**: Look at the thread reply for detailed error information.
- **Connection issues**: Make sure your Slack app is correctly configured and tokens are valid.
- **Ngrok issues**: If using ngrok, ensure it's running and the URL is updated in your Slack app configuration.

## Future Improvements

- **Asynchronous processing**: Implement fully asynchronous processing for better performance.
- **Interactive components**: Add Slack interactive components for better user experience.
- **Message pagination**: Add pagination for large responses.
- **Message formatting**: Improve message formatting for better readability.
- **File attachments**: Support for file attachments like charts and data exports.

## Conclusion

The MCP-powered Slack integration provides a robust, type-safe way to interact with the Analytics Bot from Slack. By leveraging the Model Context Protocol, the integration offers improved traceability, error handling, and debugging compared to traditional approaches. 