# Slack Integration Setup Guide

This guide walks you through setting up the Slack integration for the Analytics Bot. Follow these steps to get your bot working with Slack.

## 1. Configure Slack App

First, you need to create and configure a Slack app with the correct permissions:

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" > "From scratch"
3. Enter a name (e.g., "Analytics Bot") and select your workspace
4. In the left sidebar, click on "OAuth & Permissions"
5. Under "Bot Token Scopes", add the following scopes:
   - `chat:write` (for sending messages)
   - `reactions:write` (for adding/removing reactions)
   - `reactions:read` (for checking reactions)
   - `channels:history` (for reading channel messages)
   - `im:history` (for reading direct messages)
   - `groups:history` (for reading private channel messages)
6. Scroll to the top and click "Install to Workspace"
7. Authorize the app when prompted
8. Copy the "Bot User OAuth Token" (starts with `xoxb-`)
9. In the left sidebar, click on "Basic Information"
10. Under "App Credentials", copy the "Signing Secret"

## 2. Configure Environment Variables

Update your `.env` file with the Slack credentials:

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

## 3. Test Slack API Connectivity

Before setting up the full integration, test your Slack API connectivity:

```bash
# Run the test script
python test_slack.py
```

This will verify that your bot token is valid and has the correct permissions.

## 4. Run the Bot

### Option 1: PowerShell Script (Windows)

```powershell
# Run the bot (local only)
.\run_slack_bot.ps1

# Run with ngrok tunnel for public access
.\run_slack_bot.ps1 -ngrok
```

### Option 2: Python Command

```bash
# Activate virtual environment first
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run the bot (local only)
python run_mcp_slack.py

# Run with ngrok tunnel for public access
python run_mcp_slack.py --ngrok
```

## 5. Configure Slack Events API

Once your bot is running with ngrok:

1. Copy the public ngrok URL from the console output
2. Go to your Slack app settings
3. In the left sidebar, click on "Event Subscriptions"
4. Toggle "Enable Events" to On
5. In the "Request URL" field, enter: `{your_ngrok_url}/slack/events`
6. Wait for Slack to verify the URL (it should turn green)
7. Under "Subscribe to bot events", click "Add Bot User Event"
8. Add the following events:
   - `message.channels` (to receive messages in channels)
   - `message.im` (to receive direct messages)
9. Click "Save Changes"
10. Reinstall your app to the workspace if prompted

## 6. Test the Integration

To test that everything is working:

1. Invite your bot to a channel:
   - Type `/invite @YourBotName` in the channel
2. Send a test message:
   - Type a simple analytical question like: "How many orders did we have last week?"
3. The bot should:
   - Respond with "Processing your question..."
   - Process the query through the MCP framework
   - Return results in a threaded reply

## Troubleshooting

### Missing Permissions

If you see errors like:
```
⚠️ Failed to add reaction: missing_scope
```

This indicates your Slack app is missing the required OAuth scopes. Follow these steps:

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Select your app from the list
3. In the left sidebar, click on "OAuth & Permissions"
4. Scroll down to the "Scopes" section
5. Under "Bot Token Scopes", verify all these scopes are present:
   - `chat:write` 
   - `reactions:write` 
   - `reactions:read` 
   - `channels:history` 
   - `im:history` 
   - `groups:history`
6. If any are missing, add them, then scroll to the top and click "Reinstall to Workspace"
7. Authorize the app with the new permissions
8. Once reinstalled, copy the new Bot User OAuth Token
9. Update your `.env` file with the new token
10. Restart your application

After updating permissions, you should no longer see the "missing_scope" errors in your logs.

### Bot Not Responding

Check:
1. Is the bot running? Look for errors in the console.
2. Is ngrok running? Check the URL is accessible.
3. Is the Request URL verified in Slack? It should have a green checkmark.
4. Is the bot invited to the channel?

### Session Errors

If you see "Session not found" errors, try:
1. Restart the bot
2. Check the database connection
3. Check that the session IDs are being passed correctly

### Challenge Verification Issues

If Slack events verification is failing:
1. Make sure the bot is running when you enter the URL in Slack
2. Check the console logs for "Received challenge verification request"
3. Verify your SLACK_SIGNING_SECRET is correct in .env
4. Make sure the ngrok tunnel is active and accessible 