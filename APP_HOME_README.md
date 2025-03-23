# MCP Slack App Home Bot Window

This document explains how to set up and use the Slack App Home bot window interface for the MCP-powered Analytics Bot.

## Overview

The App Home implementation provides a dedicated conversation interface within Slack using the App Home tab. This gives each user a private, persistent space to interact with the bot, while keeping all the advantages of staying within the Slack ecosystem.

## Key Features

- **Dedicated Bot Window**: Clean interface in the App Home tab for each user
- **Private Conversations**: All conversations are private to each user
- **Contextual History**: Full conversation history displayed in the App Home tab
- **Direct Message Support**: Send questions directly to the bot via DM
- **Persistent Context**: The bot maintains conversation context across sessions
- **Slack Native**: No need to leave Slack or learn a new interface

## Setup Instructions

### 1. Slack App Configuration

Make sure your Slack app has the following settings:

#### Bot Token Scopes

The following OAuth scopes are required:
- `app_home:update` - Update the App Home tab
- `chat:write` - Send messages as the bot
- `im:history` - View messages in direct message channels
- `im:write` - Send messages in direct message channels

#### Event Subscriptions

Enable the following events:
- `app_home_opened` - When a user opens the App Home tab
- `message.im` - When a message is sent to the bot in a direct message

#### App Home Setup

In your Slack app settings:
1. Go to "App Home"
2. Enable "Home Tab"
3. Turn on "Messages Tab" 
4. Check "Allow users to send Slash commands and messages from the messages tab"

### 2. Environment Variables

Set the following environment variables in your `.env` file:
```
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
```

### 3. Running the Bot

#### Using PowerShell (Windows)

```powershell
./run_app_home.ps1 [-Port <port_number>] [-UseNgrok]
```

Parameters:
- `-Port`: Port to run the server on (default: 8000)
- `-UseNgrok`: Use ngrok to expose your local server (optional)

#### Using Python Directly

```bash
python run_mcp_app_home.py [--port PORT] [--ngrok]
```

## Using the App Home Bot Window

### 1. Accessing the Bot Window

There are two ways to interact with the bot:

1. **App Home Tab**: 
   - Click on the bot's name in your Slack sidebar
   - Select the "Home" tab at the top

2. **Direct Messages**:
   - Start or continue a conversation with the bot in a direct message channel
   - The App Home tab will update with the conversation history

### 2. Asking Questions

- Type your question in the direct message channel with the bot
- The bot will process your question and update both the DM channel and the App Home tab
- Your conversation history is displayed in the App Home tab

### 3. Viewing Conversation History

The App Home tab shows:
- All your past questions
- The bot's responses
- Timestamps for each message
- Any errors that occurred

The conversation history persists between sessions, so you can always resume where you left off.

## Comparison to Channel-based Integration

Feature | App Home | Channel-based
--- | --- | ---
Privacy | Private to each user | Visible to all channel members
Interface | Clean, dedicated window | Mixed with other channel messages
Notification noise | Low (only your messages) | High (all channel messages)
Thread management | Simple linear conversation | Multiple overlapping threads
Context preservation | Persistent per user | Thread-based
Collaboration | Private to individual | Team can see and collaborate
Implementation | Slack App Home tab | Slack channel messages

## Troubleshooting

### Common Issues

1. **App Home Not Updating**:
   - Ensure your bot has the `app_home:update` scope
   - Check that your Event Subscription URL is correct
   - Verify the bot is properly installed in the workspace

2. **Direct Messages Not Working**:
   - Ensure your bot has the `im:history` and `im:write` scopes
   - Check that the `message.im` event is subscribed

3. **Event Subscription URL Issues**:
   - Use ngrok to expose your local server (`./run_app_home.ps1 -UseNgrok`)
   - Ensure the URL points to `/slack/events` endpoint
   - Check that your signing secret is correctly set in the environment

For detailed logs, check the console output when running the bot. 