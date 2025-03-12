from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from typing import Callable, Any, Dict
import json
from datetime import datetime

class SlackBot:
    def __init__(self, message_handler: Callable[[str, str], str]):
        """Initialize Slack bot with a message handler callback"""
        self.app = App(token=os.environ["SLACK_BOT_TOKEN"])
        self.message_handler = message_handler
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Set up Slack event handlers"""
        @self.app.event("message")
        def handle_message(event, say):
            # Ignore bot messages
            if event.get("bot_id"):
                return
                
            # Get the message text and channel
            text = event.get("text", "").strip()
            channel = event.get("channel")
            user = event.get("user")
            
            if not text or not channel or not user:
                return
                
            try:
                # Process the message and get response
                response = self.message_handler(user, text)
                
                # Send the response
                say(text=response, channel=channel)
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                say(text=error_msg, channel=channel)
                
    def start(self):
        """Start the Slack bot"""
        handler = SocketModeHandler(
            app=self.app,
            app_token=os.environ["SLACK_APP_TOKEN"]
        )
        print("Starting Slack bot...")
        handler.start()
        
    def send_message(self, channel: str, text: str):
        """Send a message to a Slack channel"""
        try:
            self.app.client.chat_postMessage(
                channel=channel,
                text=text
            )
        except Exception as e:
            print(f"Error sending message: {str(e)}")
            
    def update_message(self, channel: str, ts: str, text: str):
        """Update an existing message"""
        try:
            self.app.client.chat_update(
                channel=channel,
                ts=ts,
                text=text
            )
        except Exception as e:
            print(f"Error updating message: {str(e)}")
            
    def add_reaction(self, channel: str, ts: str, reaction: str):
        """Add a reaction to a message"""
        try:
            self.app.client.reactions_add(
                channel=channel,
                timestamp=ts,
                name=reaction
            )
        except Exception as e:
            print(f"Error adding reaction: {str(e)}") 